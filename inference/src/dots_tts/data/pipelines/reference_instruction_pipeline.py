"""Instruction-controlled TTS whose speaker anchor comes from a *different* utterance.

The manifest pairs two recordings of the same speaker::

    {"fid": "...", "audio": "<target>.flac", "text": "目标文本",
     "instruction": "请呈现“略带急躁的生气”的说话状态。",
     "reference_audio": "<reference>.flac", "duration": 4.12}

``audio`` is what the model learns to generate. ``reference_audio`` supplies the
CAM++ x-vector, and nothing else.

Why this split matters more than it looks. In ordinary training the x-vector is
extracted from the target audio itself, so the model is handed the ground-truth
timbre at every step and never has to construct a voice from anything else --
which is exactly why an instruction has so little work to do, and why the voice
is weak at inference where no target exists. Anchoring on a *different*
utterance of the same speaker removes that shortcut: timbre still arrives, but
prosody and emotion cannot be copied from it, because it is a different sentence
said a different way. The instruction becomes the only route to the target's
style, which is the dependency this task needs the model to learn.

No model change is involved. The x-vector reaches the model through the
precomputed feature cache, which ``prepare_training_inputs`` passes through
verbatim, and the speaker encoder uses a supplied ``fbank`` in preference to
recomputing one from the waveform.

The template deliberately carries no ``{voice}`` placeholder, so no voice tokens
are emitted, ``_build_voice_loss_masks`` returns nothing, and the four
voice-design losses are skipped entirely. A style instruction must not be
allowed to train the branch whose job is timbre.
"""

from __future__ import annotations

import torch

from dots_tts.data.audio_backends import load_waveform
from dots_tts.data.pipelines.tokenizing import build_tokenized_example
from dots_tts.data.pipelines.tts_pipeline import (
    BasicTtsPipeline,
    TTS_AUDIO_PREFIX,
    TTS_INSTRUCTION_TEXT_PREFIX,
    TTS_TEXT_PREFIX,
)
from dots_tts.modules.speaker.fbank import extract_speaker_fbank
from dots_tts.utils.audio import high_quality_resample

# Every marker here already exists in the released checkpoint:
# `[带指令文本]` means "text carrying an instruction", and `[文本]…[文本对应语音]`
# is the ordinary TTS body. Reusing them keeps this task in distribution instead
# of asking a finetune to learn a new format from scratch. `[声音描述]` is
# deliberately NOT used -- that marker means "a description of a voice" to the
# voice-design checkpoint, and feeding it a style instruction would teach it two
# contradictory jobs.
DEFAULT_REFERENCE_INSTRUCTION_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{instruction}}"
    f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)


def reference_audio_request(sample: dict, reference_audio_key: str) -> dict:
    """Build a backend-aware load request for the reference utterance.

    Target addressing uses the ordinary ``audio_*`` fields. Reference
    addressing is namespaced so both utterances can point at different members
    (or even different tar shards) in the same manifest row.
    """

    reference = sample.get(reference_audio_key)
    if reference is None:
        raise KeyError(
            "reference_instruction samples require a reference audio field "
            f"({reference_audio_key!r}); got keys {sorted(sample.keys())}."
        )
    request = {"audio": reference}
    backend = sample.get("reference_audio_backend")
    locator = sample.get("reference_audio_locator")
    if backend not in (None, ""):
        request["audio_backend"] = backend
    if locator not in (None, ""):
        request["audio_locator"] = locator
    for field in (
        "tar_path",
        "tar_dir",
        "tar_member",
        "tar_offset",
        "tar_size",
        "tar_root",
        "start_sec",
        "end_sec",
    ):
        value = sample.get(f"reference_{field}")
        if value not in (None, ""):
            request[field] = value
    return request


class ReferenceInstructionTtsPipeline(BasicTtsPipeline):
    """Target audio for the generation target, reference audio for the speaker."""

    template = DEFAULT_REFERENCE_INSTRUCTION_TEMPLATE

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        super().__init__(tokenizer, data_cfg, profiler=profiler)
        self.instruction_key = str(getattr(data_cfg, "instruction_key", "instruction"))
        self.reference_audio_key = str(
            getattr(data_cfg, "reference_audio_key", "reference_audio")
        )
        template_override = getattr(data_cfg, "reference_instruction_template", None)
        if template_override:
            self.template = str(template_override)

    def _tokenize(self, sample: dict, num_audio_tokens: int) -> dict:
        instruction = sample.get(self.instruction_key)
        if instruction is None:
            raise KeyError(
                "reference_instruction samples require an instruction field "
                f"({self.instruction_key!r}); got keys {sorted(sample.keys())}."
            )
        return build_tokenized_example(
            text=sample["text"],
            tokenizer=self.tokenizer,
            template=self.template,
            num_audio_tokens=num_audio_tokens,
            instruction=str(instruction),
        )

    @staticmethod
    def _match_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """Tile or trim ``waveform`` to exactly ``target_length`` samples.

        The speaker encoder derives its crop window from the *audio* tensor's
        lengths and then applies that same window to whatever ``fbank`` it was
        handed. Reference and target are different recordings of different
        durations, so handing over a mismatched fbank would let the crop run off
        the end. Matching the length first keeps every downstream shape
        identical to the ordinary path.

        Tiling rather than zero-padding is deliberate: CAM++ pools over time, so
        repeating the same voice is harmless, whereas silence would drag the
        embedding toward whatever the model has learned silence sounds like.
        """

        length = int(waveform.size(-1))
        if length == target_length:
            return waveform
        if length == 0:
            raise ValueError("Reference audio decoded to zero samples.")
        if length > target_length:
            return waveform[..., :target_length]
        repeats = -(-target_length // length)  # ceil
        return waveform.repeat(*([1] * (waveform.dim() - 1)), repeats)[
            ..., :target_length
        ]

    def _process_sample_impl(self, sample: dict) -> dict:
        sample = super()._process_sample_impl(sample)

        if sample.get("xvector") is not None:
            # The row came from the feature cache, which already carries the
            # reference-derived x-vector. Recomputing an fbank here would be
            # ignored at best and contradictory at worst.
            return sample

        reference_request = reference_audio_request(
            sample, self.reference_audio_key
        )
        reference = reference_request["audio"]

        with self.profiler.measure("worker.load_reference_audio"):
            try:
                waveform, sample_rate = load_waveform(reference_request)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                # A decoder error crossing the DataLoader worker boundary
                # arrives as `<exception str() failed>` with no path in it,
                # which is unusable. Name the file and the row here.
                raise RuntimeError(
                    "Failed to decode reference audio for fid="
                    f"{sample.get('fid')!r}: {reference}"
                ) from exc
        if int(sample_rate) != self.train_audio_sample_rate:
            with self.profiler.measure("worker.resample_reference_audio"):
                waveform = high_quality_resample(
                    waveform,
                    orig_sr=int(sample_rate),
                    target_sr=self.train_audio_sample_rate,
                )

        waveform = self._match_length(waveform, int(sample["sample_length"]))

        with self.profiler.measure("worker.extract_reference_fbank"):
            fbank = extract_speaker_fbank(
                waveform,
                sample_rate=self.train_audio_sample_rate,
            )
        sample["fbank"] = fbank
        sample["fbank_length"] = int(fbank.size(0))
        sample["reference_audio_path"] = str(reference)
        return sample


__all__ = [
    "DEFAULT_REFERENCE_INSTRUCTION_TEMPLATE",
    "ReferenceInstructionTtsPipeline",
    "reference_audio_request",
]
