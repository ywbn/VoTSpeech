from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from loguru import logger

from dots_tts.utils.tokenizer import (
    PROSODY_THINK_TOKEN,
    AUDIO_GEN_END_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    AUDIO_GEN_START_TOKEN,
    TEXT_COND_END_TOKEN,
    VOICE_GEN_START_TOKEN,
    VOICE_THINK_TOKEN,
    VOICE_PATCH_TOKEN,
    require_token_id,
)

TEMPLATE_PATTERN = re.compile(
    r"\{text\}|\{audio\}|\{interleave\}|\{instruction\}|\{voice\}|[^\{]+"
)


@dataclass(frozen=True)
class ParsedTemplate:
    parts: tuple[str, ...]
    has_audio_placeholder: bool
    has_interleave_placeholder: bool
    has_instruction_placeholder: bool = False
    has_voice_placeholder: bool = False


@dataclass(frozen=True)
class TokenizedTemplatePart:
    kind: str
    token_ids: tuple[int, ...] = ()
    raw_text: str | None = None


def parse_template(template: str) -> ParsedTemplate:
    parts = tuple(re.findall(TEMPLATE_PATTERN, template))
    has_audio_placeholder = "{audio}" in parts
    interleave_count = parts.count("{interleave}")
    if has_audio_placeholder and interleave_count:
        raise ValueError("Template cannot mix audio and interleave placeholders.")
    if interleave_count > 1:
        raise ValueError(
            "Interleave generation template must contain exactly one interleave placeholder."
        )
    voice_count = parts.count("{voice}")
    if voice_count > 1:
        raise ValueError("Template must contain at most one voice placeholder.")
    if voice_count and interleave_count:
        raise ValueError("Template cannot mix voice and interleave placeholders.")
    if voice_count and "{instruction}" not in parts:
        raise ValueError(
            "A voice placeholder needs an instruction placeholder before it: "
            "the voice patch is predicted from the instruction."
        )
    if voice_count and parts.index("{voice}") < parts.index("{instruction}"):
        raise ValueError(
            "The voice placeholder must follow the instruction placeholder; "
            "causal attention gives it nothing to read otherwise."
        )
    if voice_count and "{text}" in parts and parts.index("{voice}") > parts.index("{text}"):
        raise ValueError(
            "The voice placeholder must precede the text placeholder. Sampling "
            "the voice after the text lets sentence content pick the timbre, so "
            "one instruction would drift across sentences."
        )
    return ParsedTemplate(
        parts=parts,
        has_audio_placeholder=has_audio_placeholder,
        has_interleave_placeholder=interleave_count == 1,
        has_instruction_placeholder="{instruction}" in parts,
        has_voice_placeholder=voice_count == 1,
    )


def _prepare_template_tokens(
    *, text: str, tokenizer, template: str
) -> tuple[ParsedTemplate, list[int]]:
    return parse_template(template), tokenizer.encode(text, add_special_tokens=False)


def _build_voice_tokens(tokenizer, num_think_slots: int = 0) -> list[int]:
    """``<|voice_think|>`` x K, then the gen-start / patch pair.

    The think slots come first and carry nothing: no injected embedding, no loss
    of their own. They exist so the instruction has a full-width route to the
    text and audio positions, rather than only the 192-d summary the pair
    carries. K=0 reproduces the previous layout exactly.

    The pair stays adjacent, which is what `_find_voice_positions` asserts and
    what the two-chunk prefill relies on -- putting the think slots ahead of it
    leaves both untouched.
    """

    count = max(0, int(num_think_slots))
    think_tokens = (
        [require_token_id(tokenizer, VOICE_THINK_TOKEN)] * count
        if count > 0
        else []
    )
    return think_tokens + [
        require_token_id(tokenizer, VOICE_GEN_START_TOKEN),
        require_token_id(tokenizer, VOICE_PATCH_TOKEN),
    ]


def _build_prosody_tokens(tokenizer, num_prosody_slots: int = 0) -> list[int]:
    """``<|prosody_think|>`` x M, emitted just before the audio tokens.

    Separate from the pre-text think slots on purpose: different position,
    different target, different head. Sharing one token would also make the
    cached-layout check ambiguous, since it counts ids.
    """

    count = max(0, int(num_prosody_slots))
    if count == 0:
        return []
    return [require_token_id(tokenizer, PROSODY_THINK_TOKEN)] * count


def _iter_tokenized_template_parts(
    *,
    parsed_template: ParsedTemplate,
    tokenizer,
    text_tokens: list[int],
    instruction_tokens: list[int] | None = None,
):
    for part in parsed_template.parts:
        if part == "{text}":
            yield TokenizedTemplatePart(kind="text", token_ids=tuple(text_tokens))
            continue
        if part == "{instruction}":
            yield TokenizedTemplatePart(
                kind="instruction",
                token_ids=tuple(instruction_tokens or ()),
            )
            continue
        if part == "{voice}":
            yield TokenizedTemplatePart(kind="voice")
            continue
        if part == "{audio}":
            yield TokenizedTemplatePart(kind="audio")
            continue
        if part == "{interleave}":
            yield TokenizedTemplatePart(kind="interleave")
            continue
        yield TokenizedTemplatePart(
            kind="literal",
            token_ids=tuple(tokenizer.encode(part, add_special_tokens=False)),
            raw_text=part,
        )


def _extend_tokens_with_loss(
    *, full_ids: list[int], loss_mask: list[float], token_ids: tuple[int, ...], loss: float
) -> None:
    full_ids.extend(token_ids)
    loss_mask.extend([loss] * len(token_ids))


def build_tokenized_example(
    *,
    text: str,
    tokenizer,
    template: str,
    num_audio_tokens: int,
    instruction: str | None = None,
    num_think_slots: int = 0,
    num_prosody_slots: int = 0,
) -> dict[str, Any]:
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id is required for generation targets.")

    parsed_template, text_tokens = _prepare_template_tokens(
        text=text,
        tokenizer=tokenizer,
        template=template,
    )
    instruction_tokens: list[int] = []
    if parsed_template.has_instruction_placeholder:
        if instruction is None:
            raise ValueError(
                "Template has an instruction placeholder but no instruction was "
                "provided. Add the field to the manifest or use a template "
                "without {instruction}."
            )
        instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
    voice_tokens: list[int] = []
    if parsed_template.has_voice_placeholder:
        voice_tokens = _build_voice_tokens(tokenizer, num_think_slots)
    # Gated on the voice placeholder too: a plain-TTS row in a mixed batch
    # has no instruction to plan from, and giving it plan slots would train
    # the head on rows whose target is pure content.
    prosody_tokens: list[int] = []
    if parsed_template.has_voice_placeholder:
        prosody_tokens = _build_prosody_tokens(tokenizer, num_prosody_slots)

    full_ids: list[int] = []
    loss_mask: list[float] = []
    audio_tokens: list[int] | None = None
    if parsed_template.has_audio_placeholder:
        audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
        audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        audio_gen_end_id = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
        audio_tokens = (
            [audio_gen_start_id]
            + [audio_gen_span_id] * num_audio_tokens
            + [audio_gen_end_id]
        )
    elif parsed_template.has_interleave_placeholder:
        audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        audio_gen_end_id = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
        text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)

    for part in _iter_tokenized_template_parts(
        parsed_template=parsed_template,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
        instruction_tokens=instruction_tokens,
    ):
        if part.kind in {"text", "instruction"}:
            _extend_tokens_with_loss(
                full_ids=full_ids,
                loss_mask=loss_mask,
                token_ids=part.token_ids,
                loss=0.0,
            )
            continue

        if part.kind == "voice":
            # Loss 0 everywhere: the voice tokens are never a cross-entropy
            # target. Their supervision arrives through the voice branch, which
            # reads <|voice_gen_start|>'s hidden state and replaces
            # <|voice_patch|>'s input embedding -- neither goes through logits.
            _extend_tokens_with_loss(
                full_ids=full_ids,
                loss_mask=loss_mask,
                token_ids=tuple(voice_tokens),
                loss=0.0,
            )
            continue

        if part.kind == "audio":
            if audio_tokens is None:
                raise RuntimeError("Audio placeholder tokens were not initialized.")
            # Loss 0, like the voice tokens: these are never a cross-entropy
            # target. Their supervision is a regression off their hidden
            # states, and at inference they are part of the prefilled
            # schedule, so the LM is never asked to emit them.
            _extend_tokens_with_loss(
                full_ids=full_ids,
                loss_mask=loss_mask,
                token_ids=tuple(prosody_tokens),
                loss=0.0,
            )
            full_ids.extend(audio_tokens)
            loss_mask.extend([0.0])
            loss_mask.extend([1.0] * max(0, len(audio_tokens) - 2))
            loss_mask.append(0.0)
            continue

        if part.kind == "interleave":
            _append_interleave_generation_tokens(
                full_ids=full_ids,
                loss_mask=loss_mask,
                text_tokens=text_tokens,
                num_audio_tokens=num_audio_tokens,
                audio_span_id=audio_gen_span_id,
                audio_end_id=audio_gen_end_id,
                text_cond_end_id=text_cond_end_id,
            )
            continue

        _extend_tokens_with_loss(
            full_ids=full_ids,
            loss_mask=loss_mask,
            token_ids=part.token_ids,
            loss=0.0,
        )

    full_ids.append(tokenizer.eos_token_id)
    loss_mask.append(0.0)

    return {
        "input_ids": full_ids[:-1],
        "labels": full_ids[1:],
        "loss_mask": loss_mask[1:],
        # Instruction and voice tokens are counted here on purpose: the online
        # batcher budgets a batch by num_text_tokens, and a voice-design sample
        # is genuinely longer on the text side than a plain TTS sample.
        "text_token_count": (
            len(text_tokens)
            + len(instruction_tokens)
            + len(voice_tokens)
            + len(prosody_tokens)
        ),
        "instruction_token_count": len(instruction_tokens),
        "voice_token_count": len(voice_tokens),
    }


def build_generation_schedule(
    *,
    text: str,
    tokenizer,
    template: str,
    max_audio_tokens: int,
    instruction: str | None = None,
    num_think_slots: int = 0,
    num_prosody_slots: int = 0,
) -> dict[str, Any]:
    if max_audio_tokens <= 0:
        raise ValueError("max_audio_tokens must be positive for generation.")

    parsed_template, text_tokens = _prepare_template_tokens(
        text=text,
        tokenizer=tokenizer,
        template=template,
    )
    instruction_tokens: list[int] = []
    if parsed_template.has_instruction_placeholder:
        if instruction is None:
            raise ValueError(
                "Template has an instruction placeholder but no instruction was "
                "provided for generation."
            )
        instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False)
    voice_tokens: list[int] = []
    prosody_tokens: list[int] = []
    if parsed_template.has_voice_placeholder:
        voice_tokens = _build_voice_tokens(tokenizer, num_think_slots)
        prosody_tokens = _build_prosody_tokens(tokenizer, num_prosody_slots)
    schedule_ids: list[int] = []
    audio_gen_start_id = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
    audio_gen_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)

    if parsed_template.has_audio_placeholder:
        for part in _iter_tokenized_template_parts(
            parsed_template=parsed_template,
            tokenizer=tokenizer,
            text_tokens=text_tokens,
            instruction_tokens=instruction_tokens,
        ):
            if part.kind == "audio":
                # Same position as in training: after the text, before the
                # first audio token, and inside the prefilled prefix.
                schedule_ids.extend(prosody_tokens)
                schedule_ids.append(audio_gen_start_id)
                schedule_ids.extend([audio_gen_span_id] * max_audio_tokens)
                continue
            if part.kind == "voice":
                schedule_ids.extend(voice_tokens)
                continue
            schedule_ids.extend(part.token_ids)
        visible_schedule_ids = [
            token_id for token_id in schedule_ids if token_id != audio_gen_span_id
        ]
        decoded_schedule = (
            tokenizer.decode(
                visible_schedule_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if hasattr(tokenizer, "decode")
            else repr(visible_schedule_ids)
        )
        logger.info(
            "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
            False,
            int(max_audio_tokens),
            decoded_schedule,
        )
        return {
            "schedule_ids": schedule_ids,
            "interleave": False,
        }

    if not parsed_template.has_interleave_placeholder:
        raise ValueError(
            "Generation template must contain either {audio} or {interleave}."
        )
    text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
    if max_audio_tokens < len(text_tokens):
        raise ValueError(
            "Interleave generation requires at least one audio span per text token: "
            f"text_token_count={len(text_tokens)} "
            f"max_audio_patch_count={max_audio_tokens}."
        )

    interleave_started = False
    for part in _iter_tokenized_template_parts(
        parsed_template=parsed_template,
        tokenizer=tokenizer,
        text_tokens=text_tokens,
    ):
        if part.kind == "interleave":
            _append_interleave_schedule_tokens(
                schedule_ids=schedule_ids,
                text_tokens=text_tokens,
                max_audio_tokens=max_audio_tokens,
                audio_span_id=audio_gen_span_id,
                text_cond_end_id=text_cond_end_id,
            )
            interleave_started = True
            continue
        if part.kind == "text":
            raise ValueError(
                "Generation schedule does not support {text} inside an interleave template."
            )
        if part.kind == "audio":
            raise ValueError(
                "Generation schedule does not support {audio} inside an interleave template."
            )
        if interleave_started:
            if (part.raw_text or "").strip():
                raise ValueError(
                    "Generation schedule does not support non-empty suffix text after the interleave placeholder."
                )
            continue
        schedule_ids.extend(part.token_ids)

    visible_schedule_ids = [
        token_id for token_id in schedule_ids if token_id != audio_gen_span_id
    ]
    decoded_schedule = (
        tokenizer.decode(
            visible_schedule_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if hasattr(tokenizer, "decode")
        else repr(visible_schedule_ids)
    )
    logger.info(
        "Built generation schedule: interleave={} max_audio_tokens={} sequence={!r}",
        True,
        int(max_audio_tokens),
        decoded_schedule,
    )
    return {
        "schedule_ids": schedule_ids,
        "interleave": True,
    }


def build_edit_generation_schedule(
    *,
    source_text: str,
    target_text: str,
    instruction: str,
    tokenizer,
    source_text_prefix: str,
    source_audio_prefix: str,
    instruction_prefix: str,
    target_text_prefix: str,
    target_audio_prefix: str,
    source_num_audio_tokens: int,
    target_max_audio_tokens: int,
) -> dict[str, Any]:
    """Build the complete source-prefill and target-decode Edit schedule."""

    if source_num_audio_tokens <= 0:
        raise ValueError("source_num_audio_tokens must be positive.")
    if target_max_audio_tokens <= 0:
        raise ValueError("target_max_audio_tokens must be positive.")

    gen_start = require_token_id(tokenizer, AUDIO_GEN_START_TOKEN)
    gen_span = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
    gen_end = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
    schedule_ids: list[int] = []

    def append_text(prefix: str, value: str) -> None:
        schedule_ids.extend(tokenizer.encode(prefix, add_special_tokens=False))
        schedule_ids.extend(tokenizer.encode(value, add_special_tokens=False))

    append_text(source_text_prefix, source_text)
    schedule_ids.extend(
        tokenizer.encode(source_audio_prefix, add_special_tokens=False)
    )
    schedule_ids.extend(
        [gen_start, *([gen_span] * source_num_audio_tokens), gen_end]
    )
    append_text(instruction_prefix, instruction)
    append_text(target_text_prefix, target_text)
    schedule_ids.extend(
        tokenizer.encode(target_audio_prefix, add_special_tokens=False)
    )
    schedule_ids.append(gen_start)
    schedule_ids.extend([gen_span] * target_max_audio_tokens)
    schedule_ids.append(gen_end)
    return {"schedule_ids": schedule_ids, "interleave": False}


def _append_interleave_generation_tokens(
    *,
    full_ids: list[int],
    loss_mask: list[float],
    text_tokens: list[int],
    num_audio_tokens: int,
    audio_span_id: int,
    audio_end_id: int,
    text_cond_end_id: int,
) -> None:
    audio_tokens = [audio_span_id] * num_audio_tokens + [audio_end_id]
    text_index = 0
    audio_index = 0
    text_cond_end_added = False

    while text_index < len(text_tokens) or audio_index < len(audio_tokens):
        if text_index < len(text_tokens):
            full_ids.append(text_tokens[text_index])
            loss_mask.append(0.0)
            text_index += 1
        elif not text_cond_end_added:
            full_ids.append(text_cond_end_id)
            loss_mask.append(0.0)
            text_cond_end_added = True

        if audio_index < len(audio_tokens):
            full_ids.append(audio_tokens[audio_index])
            loss_mask.append(1.0 if audio_index < num_audio_tokens else 0.0)
            audio_index += 1

    if not text_cond_end_added:
        full_ids.append(text_cond_end_id)
        loss_mask.append(0.0)


def _append_interleave_schedule_tokens(
    *,
    schedule_ids: list[int],
    text_tokens: list[int],
    max_audio_tokens: int,
    audio_span_id: int,
    text_cond_end_id: int,
) -> None:
    for token_id in text_tokens:
        schedule_ids.append(token_id)
        schedule_ids.append(audio_span_id)
    schedule_ids.append(text_cond_end_id)
    remaining_audio_tokens = max_audio_tokens - len(text_tokens)
    if remaining_audio_tokens > 0:
        schedule_ids.extend([audio_span_id] * remaining_audio_tokens)
