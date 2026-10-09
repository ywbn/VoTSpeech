from __future__ import annotations

import torch

from dots_tts.data.audio_backends import load_waveform
from dots_tts.data.feature_cache import has_feature_cache, load_feature_cache
from dots_tts.data.pipelines.base import BaseSamplePipeline
from dots_tts.data.pipelines.preprocessing import (
    compute_num_audio_tokens,
    normalize_edge_silence_duration,
    pad_waveform_align_only,
)
from dots_tts.data.pipelines.tokenizing import build_tokenized_example
from dots_tts.utils.tokenizer import (
    PROSODY_THINK_TOKEN,
    VOICE_THINK_TOKEN,
    has_token,
    require_token_id,
)
from dots_tts.modules.speaker.fbank import extract_speaker_fbank
from dots_tts.utils.audio import high_quality_resample
from dots_tts.utils.profiling import ensure_data_profiler

TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"
TTS_INSTRUCTION_TEXT_PREFIX = "[带指令文本]"
TTA_TEXT_PREFIX = "[声音描述]"
TTA_AUDIO_PREFIX = "[描述对应声音]"
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"
DEFAULT_TRAIN_TEMPLATE = f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
DEFAULT_INSTRUCTION_TTS_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)
DEFAULT_TEXT_TO_AUDIO_TEMPLATE = f"{TTA_TEXT_PREFIX}{{text}}{TTA_AUDIO_PREFIX}{{audio}}"
DEFAULT_INTERLEAVE_TRAIN_TEMPLATE = f"{TTS_INTERLEAVE_PREFIX}{{interleave}}"
# Voice design reuses the pretrained section markers rather than inventing new
# ones: `[声音描述]` already means "a description of a voice" to this checkpoint,
# and `[文本]…[文本对应语音]` is the ordinary TTS body. The voice token pair sits
# between them, so causal attention lets both the text and every audio patch
# read the chosen voice while the voice itself can only read the instruction --
# that placement is what decouples timbre from sentence content.
DEFAULT_VOICE_DESIGN_TEMPLATE = (
    f"{TTA_TEXT_PREFIX}{{instruction}}{{voice}}"
    f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)


class BasicTtsPipeline(BaseSamplePipeline):
    """Fixed internal training pipeline for adapter-emitted samples."""

    template = DEFAULT_TRAIN_TEMPLATE

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        self.tokenizer = tokenizer
        self.train_audio_sample_rate = int(data_cfg.train_audio_sample_rate)
        self.audio_samples_per_llm_token = int(data_cfg.audio_samples_per_llm_token)
        self.profiler = ensure_data_profiler(profiler)

    def _tokenize(self, sample: dict, num_audio_tokens: int) -> dict:
        return build_tokenized_example(
            text=sample["text"],
            tokenizer=self.tokenizer,
            template=self.template,
            num_audio_tokens=num_audio_tokens,
        )

    @staticmethod
    def _load_waveform(sample: dict) -> tuple[torch.Tensor, int]:
        """Decode one sample, honouring audio_backend and segment bounds.

        Dispatching here rather than reading ``sample["audio"]`` directly is what
        lets a VoiceCrafter manifest - Parquet-backed rows, dataset references,
        ``start_sec``/``end_sec`` segments - train without being materialized to
        wav files first.
        """

        return load_waveform(sample)

    def _on_cached_sample(self, sample: dict) -> None:
        """Hook for subclasses to check a cache-loaded row. No-op by default."""

    @staticmethod
    def _validate_source_sample(sample: dict) -> None:
        missing = [field for field in ("fid", "text", "audio") if field not in sample]
        if missing:
            raise ValueError(
                "Source adapter must emit fid/text/audio. "
                f"Missing fields: {missing}. Sample keys: {sorted(sample.keys())}"
            )
        if not str(sample.get("text", "")).strip():
            raise ValueError(
                f"Sample {sample.get('fid')!r} has empty text. Filter these out "
                "during manifest preparation; an empty transcript trains the "
                "model to emit audio for nothing."
            )

    def process_sample(self, raw_sample: dict) -> dict:
        sample = dict(raw_sample)
        self._validate_source_sample(sample)
        sample["fid"] = str(sample["fid"])

        with self.profiler.measure("worker.process_sample_total"):
            return self._process_sample_impl(sample)

    def _process_sample_impl(self, sample: dict) -> dict:
        profiler = self.profiler
        if has_feature_cache(sample):
            with profiler.measure("worker.load_feature_cache"):
                sample.update(load_feature_cache(sample))
            self._on_cached_sample(sample)
            return sample

        with profiler.measure("worker.load_audio"):
            try:
                waveform, sample_rate = self._load_waveform(sample)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                # Decoder errors cross the DataLoader worker boundary stripped of
                # their message (`<exception str() failed>`), so the path has to
                # be attached here or there is nothing to debug from.
                raise RuntimeError(
                    "Failed to decode target audio for fid="
                    f"{sample.get('fid')!r}: {sample.get('audio')}"
                ) from exc
        with profiler.measure("worker.resample_audio"):
            waveform = high_quality_resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.train_audio_sample_rate,
            )
        with profiler.measure("worker.normalize_edge_silence"):
            waveform = normalize_edge_silence_duration(
                waveform,
                sample_rate=self.train_audio_sample_rate,
            )
        sample["sample"] = waveform
        sample["sample_rate"] = self.train_audio_sample_rate
        sample["unpadded_sample_length"] = int(waveform.size(-1))

        with profiler.measure("worker.pad_audio"):
            waveform = pad_waveform_align_only(
                waveform,
                multiple_of=self.audio_samples_per_llm_token,
            )
        sample["sample"] = waveform
        sample["sample_length"] = int(waveform.size(-1))

        num_audio_tokens = compute_num_audio_tokens(
            sample["sample_length"],
            audio_samples_per_llm_token=self.audio_samples_per_llm_token,
        )
        with profiler.measure("worker.tokenize"):
            tokenized = self._tokenize(sample, num_audio_tokens)
        sample["input_ids"] = tokenized["input_ids"]
        sample["labels"] = tokenized["labels"]
        sample["loss_mask"] = tokenized["loss_mask"]
        sample["input_ids_length"] = len(tokenized["input_ids"])
        sample["num_text_tokens"] = tokenized["text_token_count"]
        sample["num_audio_tokens"] = num_audio_tokens
        sample["num_total_tokens"] = sample["input_ids_length"]

        with profiler.measure("worker.extract_fbank"):
            fbank = extract_speaker_fbank(
                sample["sample"],
                sample_rate=sample["sample_rate"],
            )
        sample["fbank"] = fbank
        sample["fbank_length"] = int(fbank.size(0))
        return sample


class InterleaveTtsPipeline(BasicTtsPipeline):
    template = DEFAULT_INTERLEAVE_TRAIN_TEMPLATE


class VoiceDesignTtsPipeline(BasicTtsPipeline):
    """TTS samples that additionally carry a natural-language voice instruction.

    Manifest rows need one extra field on top of ``fid``/``text``/``audio``::

        {"fid": "...", "audio": "...", "text": "...",
         "instruction": "低沉、温暖、略带沙哑的成年男性声音"}

    No offline voice-latent extraction is required: the regression target is the
    frozen CAM++ x-vector, which the model already computes from the training
    audio itself during ``prepare_training_inputs``.
    """

    template = DEFAULT_VOICE_DESIGN_TEMPLATE

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        super().__init__(tokenizer, data_cfg, profiler=profiler)
        self.instruction_key = str(data_cfg.instruction_key)
        self.num_think_slots = int(getattr(data_cfg, "num_think_slots", 0) or 0)
        self.num_prosody_slots = int(
            getattr(data_cfg, "num_prosody_slots", 0) or 0
        )
        self._think_slots_verified = False
        template_override = getattr(data_cfg, "voice_design_template", None)
        if template_override:
            self.template = str(template_override)

    def _on_cached_sample(self, sample: dict) -> None:
        self._verify_cached_think_slots(sample)

    def _verify_cached_think_slots(self, sample: dict) -> None:
        """Check a cached row's token layout against the configured K.

        `_cache_signature` keeps caches built with different K in different
        directories, but nothing stops a *manifest* written for one K from being
        handed to a run configured for another -- and that failure is silent:
        training simply reads sequences shifted by K positions from what the
        model will produce at inference. Counting the think ids on the first
        cached row each worker sees costs one pass over one sequence.
        """

        if self._think_slots_verified:
            return
        self._think_slots_verified = True
        input_ids = sample.get("input_ids")
        if input_ids is None:
            return
        counts: dict[str, int] = {}
        for token in (VOICE_THINK_TOKEN, PROSODY_THINK_TOKEN):
            if not has_token(self.tokenizer, token):
                # An artifact predating this token cannot have emitted it,
                # and the run must not be configured to expect one.
                counts[token] = 0
                continue
            token_id = require_token_id(self.tokenizer, token)
            counts[token] = sum(
                1 for cached_id in input_ids if int(cached_id) == token_id
            )
        expected = {
            VOICE_THINK_TOKEN: (self.num_think_slots, "--num-think-slots"),
            PROSODY_THINK_TOKEN: (
                self.num_prosody_slots,
                "--num-prosody-slots",
            ),
        }
        for token, (wanted, flag) in expected.items():
            if counts[token] != wanted:
                raise ValueError(
                    f"Cached sample {sample.get('fid')!r} carries "
                    f"{counts[token]} {token} token(s) but this run is "
                    f"configured for {wanted}. The cache was built for a "
                    f"different layout; rebuild it with {flag} matching, or "
                    "point at the cache that matches."
                )

    def _tokenize(self, sample: dict, num_audio_tokens: int) -> dict:
        return build_tokenized_example(
            text=sample["text"],
            tokenizer=self.tokenizer,
            template=self.template,
            num_audio_tokens=num_audio_tokens,
        )

    @staticmethod
    def _load_waveform(sample: dict) -> tuple[torch.Tensor, int]:
        """Decode one sample, honouring audio_backend and segment bounds.

        Dispatching here rather than reading ``sample["audio"]`` directly is what
        lets a VoiceCrafter manifest - Parquet-backed rows, dataset references,
        ``start_sec``/``end_sec`` segments - train without being materialized to
        wav files first.
        """

        return load_waveform(sample)

    def _on_cached_sample(self, sample: dict) -> None:
        """Hook for subclasses to check a cache-loaded row. No-op by default."""

    @staticmethod
    def _validate_source_sample(sample: dict) -> None:
        missing = [field for field in ("fid", "text", "audio") if field not in sample]
        if missing:
            raise ValueError(
                "Source adapter must emit fid/text/audio. "
                f"Missing fields: {missing}. Sample keys: {sorted(sample.keys())}"
            )
        if not str(sample.get("text", "")).strip():
            raise ValueError(
                f"Sample {sample.get('fid')!r} has empty text. Filter these out "
                "during manifest preparation; an empty transcript trains the "
                "model to emit audio for nothing."
            )

    def process_sample(self, raw_sample: dict) -> dict:
        sample = dict(raw_sample)
        self._validate_source_sample(sample)
        sample["fid"] = str(sample["fid"])

        with self.profiler.measure("worker.process_sample_total"):
            return self._process_sample_impl(sample)

    def _process_sample_impl(self, sample: dict) -> dict:
        profiler = self.profiler
        if has_feature_cache(sample):
            with profiler.measure("worker.load_feature_cache"):
                sample.update(load_feature_cache(sample))
            self._on_cached_sample(sample)
            return sample

        with profiler.measure("worker.load_audio"):
            try:
                waveform, sample_rate = self._load_waveform(sample)
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                # Decoder errors cross the DataLoader worker boundary stripped of
                # their message (`<exception str() failed>`), so the path has to
                # be attached here or there is nothing to debug from.
                raise RuntimeError(
                    "Failed to decode target audio for fid="
                    f"{sample.get('fid')!r}: {sample.get('audio')}"
                ) from exc
        with profiler.measure("worker.resample_audio"):
            waveform = high_quality_resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.train_audio_sample_rate,
            )
        with profiler.measure("worker.normalize_edge_silence"):
            waveform = normalize_edge_silence_duration(
                waveform,
                sample_rate=self.train_audio_sample_rate,
            )
        sample["sample"] = waveform
        sample["sample_rate"] = self.train_audio_sample_rate
        sample["unpadded_sample_length"] = int(waveform.size(-1))

        with profiler.measure("worker.pad_audio"):
            waveform = pad_waveform_align_only(
                waveform,
                multiple_of=self.audio_samples_per_llm_token,
            )
        sample["sample"] = waveform
        sample["sample_length"] = int(waveform.size(-1))

        num_audio_tokens = compute_num_audio_tokens(
            sample["sample_length"],
            audio_samples_per_llm_token=self.audio_samples_per_llm_token,
        )
        with profiler.measure("worker.tokenize"):
            tokenized = self._tokenize(sample, num_audio_tokens)
        sample["input_ids"] = tokenized["input_ids"]
        sample["labels"] = tokenized["labels"]
        sample["loss_mask"] = tokenized["loss_mask"]
        sample["input_ids_length"] = len(tokenized["input_ids"])
        sample["num_text_tokens"] = tokenized["text_token_count"]
        sample["num_audio_tokens"] = num_audio_tokens
        sample["num_total_tokens"] = sample["input_ids_length"]

        with profiler.measure("worker.extract_fbank"):
            fbank = extract_speaker_fbank(
                sample["sample"],
                sample_rate=sample["sample_rate"],
            )
        sample["fbank"] = fbank
        sample["fbank_length"] = int(fbank.size(0))
        return sample


class InterleaveTtsPipeline(BasicTtsPipeline):
    template = DEFAULT_INTERLEAVE_TRAIN_TEMPLATE


class VoiceDesignTtsPipeline(BasicTtsPipeline):
    """TTS samples that additionally carry a natural-language voice instruction.

    Manifest rows need one extra field on top of ``fid``/``text``/``audio``::

        {"fid": "...", "audio": "...", "text": "...",
         "instruction": "低沉、温暖、略带沙哑的成年男性声音"}

    No offline voice-latent extraction is required: the regression target is the
    frozen CAM++ x-vector, which the model already computes from the training
    audio itself during ``prepare_training_inputs``.
    """

    template = DEFAULT_VOICE_DESIGN_TEMPLATE

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        super().__init__(tokenizer, data_cfg, profiler=profiler)
        self.instruction_key = str(data_cfg.instruction_key)
        self.num_think_slots = int(getattr(data_cfg, "num_think_slots", 0) or 0)
        self.num_prosody_slots = int(
            getattr(data_cfg, "num_prosody_slots", 0) or 0
        )
        self._think_slots_verified = False
        template_override = getattr(data_cfg, "voice_design_template", None)
        if template_override:
            self.template = str(template_override)

    def _on_cached_sample(self, sample: dict) -> None:
        self._verify_cached_think_slots(sample)

    def _verify_cached_think_slots(self, sample: dict) -> None:
        """Check a cached row's token layout against the configured K.

        `_cache_signature` keeps caches built with different K in different
        directories, but nothing stops a *manifest* written for one K from being
        handed to a run configured for another -- and that failure is silent:
        training simply reads sequences shifted by K positions from what the
        model will produce at inference. Counting the think ids on the first
        cached row each worker sees costs one pass over one sequence.
        """

        if self._think_slots_verified:
            return
        self._think_slots_verified = True
        if not has_token(self.tokenizer, VOICE_THINK_TOKEN):
            return
        input_ids = sample.get("input_ids")
        if input_ids is None:
            return
        for token, found_count, expected, flag in (
            (
                VOICE_THINK_TOKEN,
                sum(
                    1
                    for token_id in input_ids
                    if int(token_id) == require_token_id(
                        self.tokenizer, VOICE_THINK_TOKEN
                    )
                ),
                self.num_think_slots,
                "--num-think-slots",
            ),
            (
                PROSODY_THINK_TOKEN,
                sum(
                    1
                    for token_id in input_ids
                    if has_token(self.tokenizer, PROSODY_THINK_TOKEN)
                    and int(token_id)
                    == require_token_id(self.tokenizer, PROSODY_THINK_TOKEN)
                ),
                self.num_prosody_slots,
                "--num-prosody-slots",
            ),
        ):
            if found_count != expected:
                raise ValueError(
                    f"Cached sample {sample.get('fid')!r} carries "
                    f"{found_count} {token} token(s) but the run is configured "
                    f"for {expected}. The cache was built for a different "
                    f"layout; rebuild it with {flag} matching, or point at the "
                    "cache that matches."
                )

    def _tokenize(self, sample: dict, num_audio_tokens: int) -> dict:
        instruction = sample.get(self.instruction_key)
        if instruction is None:
            raise KeyError(
                "voice_design samples require an instruction field "
                f"({self.instruction_key!r}); got keys {sorted(sample.keys())}."
            )
        return build_tokenized_example(
            text=sample["text"],
            tokenizer=self.tokenizer,
            template=self.template,
            num_audio_tokens=num_audio_tokens,
            instruction=str(instruction),
            num_think_slots=self.num_think_slots,
            num_prosody_slots=self.num_prosody_slots,
        )
