from __future__ import annotations

AUDIO_COMP_START_TOKEN = "<|audio_comp_start|>"
AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"
AUDIO_COMP_END_TOKEN = "<|audio_comp_end|>"
AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"
AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"
AUDIO_GEN_END_TOKEN = "<|audio_gen_end|>"
TEXT_COND_END_TOKEN = "<|text_cond_end|>"

# Voice-design tokens. The voice is generated as the sequence's patch 0, using
# exactly the mechanism the audio patches use, which is why this pair mirrors
# AUDIO_GEN_START/AUDIO_GEN_SPAN rather than inventing a new shape:
#
#   [声音描述] instruction <|voice_gen_start|> <|voice_patch|> [文本] text ...
#
#   * <|voice_gen_start|> — its HIDDEN state conditions the voice flow DiT, the
#     same way <|audio_gen_start|>'s hidden predicts audio patch 0.
#   * <|voice_patch|> — its INPUT EMBEDDING is the voice latent (teacher-forced
#     in training, sampled at inference), the same way an audio span position's
#     input embedding is the encoded previous latent patch.
#
# The pair sits after the instruction and before the text on purpose. After the
# instruction because the voice is predicted *from* it; before the text because
# a sampling point placed after the text would let the model pick a voice from
# sentence content, so the same instruction would drift across sentences.
#
# `<|voice_think|>` repeated K times sits BEFORE the pair. Those positions carry
# no injected content and no loss of their own: they are scratch space, the way
# chain-of-thought tokens are, shaped only by whether the audio that follows
# comes out better. That distinction matters -- the pair is a 192-d summary
# regressed onto a speaker target, so anything the instruction says about
# prosody cannot survive it, and an instruction naming several acoustic
# parameters loses most of itself. The think slots give that information a
# full-width path to the text and audio positions instead.
VOICE_THINK_TOKEN = "<|voice_think|>"
VOICE_GEN_START_TOKEN = "<|voice_gen_start|>"
VOICE_PATCH_TOKEN = "<|voice_patch|>"
# Plan slots, placed AFTER the text and before the first audio token. The
# pre-text think slots cannot know what is about to be said, so the only
# thing they can carry is timbre. These can see the text, which is what lets
# them be supervised on how the utterance should unfold over time -- the
# component a single global g_cond cannot express at all.
PROSODY_THINK_TOKEN = "<|prosody_think|>"

# Order matters, and only here. `add_voice_design_tokens` appends whichever
# of these the artifact is missing, in THIS order, and HuggingFace assigns
# ids in the order given -- so the order decides the ids, and the ids are
# baked into every precomputed cache.
#
# <|voice_think|> comes LAST so the two older tokens keep the ids they were
# given before think slots existed. A bare artifact gains all three
# (gen_start, patch, think); a voice-design checkpoint already carries the
# first two and gains only think, at the same id the bare artifact would
# have used. Both paths therefore agree, which is what lets one cache serve
# a run started from either. Putting think first would silently shift
# gen_start and patch by one on the bare-artifact path only -- the model
# would look for <|voice_gen_start|> and find think tokens, with no error
# anywhere, just audio that never improves.
#
# The order here has nothing to do with where the tokens sit in a sequence;
# `_build_voice_tokens` decides that, and it emits think x K first.
VOICE_DESIGN_TOKENS = (
    VOICE_GEN_START_TOKEN,
    VOICE_PATCH_TOKEN,
    VOICE_THINK_TOKEN,
    PROSODY_THINK_TOKEN,
)


def require_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise ValueError(f"Artifact tokenizer is missing required special token: {token}")
    return int(token_id)


def has_token(tokenizer, token: str) -> bool:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        return False
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    return unk_token_id is None or int(token_id) != int(unk_token_id)


def has_voice_design_tokens(tokenizer) -> bool:
    return all(has_token(tokenizer, token) for token in VOICE_DESIGN_TOKENS)


def add_voice_design_tokens(tokenizer) -> int:
    """Register the voice-design tokens, returning how many were newly added.

    A released dots.tts artifact does not carry these tokens, so a voice-design
    run has to extend the tokenizer once and then resize the LLM embedding. The
    return value is what tells the caller whether a resize is needed at all —
    re-running on an already-extended artifact must be a no-op, otherwise every
    resume would grow the vocabulary again.
    """

    missing = [token for token in VOICE_DESIGN_TOKENS if not has_token(tokenizer, token)]
    if not missing:
        return 0
    return int(
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
    )


__all__ = [
    "AUDIO_COMP_END_TOKEN",
    "AUDIO_COMP_SPAN_TOKEN",
    "AUDIO_COMP_START_TOKEN",
    "AUDIO_GEN_END_TOKEN",
    "AUDIO_GEN_SPAN_TOKEN",
    "AUDIO_GEN_START_TOKEN",
    "TEXT_COND_END_TOKEN",
    "PROSODY_THINK_TOKEN",
    "VOICE_DESIGN_TOKENS",
    "VOICE_GEN_START_TOKEN",
    "VOICE_THINK_TOKEN",
    "VOICE_PATCH_TOKEN",
    "add_voice_design_tokens",
    "has_token",
    "has_voice_design_tokens",
    "require_token_id",
]
