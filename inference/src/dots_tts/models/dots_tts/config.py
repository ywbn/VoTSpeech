from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from dots_tts.config.base import ConfigBase, StrictConfigBase
from dots_tts.modules.vocoder.config import AudioVAEConfig


class _EncoderConfig(ConfigBase):
    num_layers: int = 6
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = False
    rotary_theta: float | None = 10000
    input_dim: int = 1024
    causal: bool = True


class _DiTConfig(ConfigBase):
    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "LayerNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float | None = 10000


class LossConfig(StrictConfigBase):
    ce_weight: float = 1.0
    fm_weight: float = 1.0
    eos_weight: float = 1.0
    # Voice-design terms. They are inert unless ModelConfig.voice_design is
    # enabled, because the branch then emits all-zero masks and the shared
    # aggregation in training/losses.py averages them to zero.
    voice_flow_weight: float = 1.0
    # The QFormer slice is trained to be speaker-discriminative but still picks
    # up channel and recording conditions, part of which no instruction can
    # predict. Weighted down rather than dropped: what it does carry (spectral
    # tilt, loudness, channel) is exactly what CAM++ throws away.
    voice_qformer_weight: float = 0.25
    voice_align_weight: float = 1.0
    voice_relation_weight: float = 0.5
    # Progressive refinement of the voice code across the think slots.
    # Zero leaves the chain unsupervised: it still exists and still reaches
    # g_cond through its zero-initialized projection, but nothing pushes its
    # content toward timbre, which is the whole reason for typing the slots.
    voice_think_weight: float = 0.0
    # Per-segment acoustic plan regressed off the post-text slots. This is
    # the only voice term whose target varies over time; every other one is
    # a single pooled vector, which is why none of them can teach anything
    # about speaking rate, stress placement or how a delivery develops.
    voice_prosody_weight: float = 0.0


class VoiceDesignConfig(ConfigBase):
    """Instruction -> voice latent, generated as the sequence's patch 0.

    Layout: ``instruction <|voice_gen_start|> <|voice_patch|> text audio``. The
    hidden state at ``<|voice_gen_start|>`` conditions the voice flow DiT; the
    sampled latent becomes the *input embedding* at ``<|voice_patch|>`` and,
    after decoding, ``g_cond``. Same mechanism as an audio patch, one position
    earlier, which is why no extra forward pass is needed at inference.
    """

    enabled: bool = False

    # The released VoTSpeech checkpoint uses pooled AudioVAE features. Keep
    # pooled as the artifact default so the public config does not need to
    # expose the training-time representation ablation switch.
    voice_source: Literal["qformer", "pooled", "none"] = "pooled"
    voice_dim: int = Field(default=192, ge=1)
    qformer_hidden_size: int = Field(default=512, ge=1)
    qformer_num_layers: int = Field(default=4, ge=1)
    qformer_num_heads: int = Field(default=8, ge=1)
    # Letting the acoustic loss move the teacher lets the model make the target
    # easier instead of making the prediction better: the representation drifts
    # toward whatever the flow head already emits, which is collapse with a
    # perfectly healthy-looking loss curve.
    freeze_extractor: bool = True
    # Stage-1 weights (scripts/train_voice_extractor.py). Null starts the
    # extractor from scratch, which only makes sense with freeze_extractor=false.
    extractor_checkpoint: str | None = None
    # Opt-in bounded correction, preserving the pooled latent's dimension.
    pooled_residual: bool = False
    pooled_residual_hidden: int = Field(default=128, ge=1)
    pooled_residual_alpha: float = Field(default=0.1, gt=0.0, le=0.5)

    flow_hidden_size: int = Field(default=512, ge=1)
    flow_num_layers: int = Field(default=4, ge=1)
    flow_num_heads: int = Field(default=8, ge=1)
    flow_mlp_ratio: float = Field(default=4.0, gt=0.0)
    flow_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    direct_head_intermediate_size: int | None = Field(default=None, ge=1)

    # Classifier-free guidance on the instruction, for the voice code only.
    condition_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    # Probability of blanking the <|voice_patch|> input embedding during
    # training. The patch is a compact, prosody-free summary of the
    # instruction sitting closer to the text than the instruction itself,
    # so the LM learns to read it instead of the instruction and style
    # adherence drops below a plain finetune. Dropping it keeps the direct
    # path alive. 1.0 removes the patch's influence on the LM entirely,
    # leaving g_cond as the voice's only route.
    voice_patch_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    # Optional direct instruction -> acoustic global-condition residual. Reuses
    # the existing voice_gen_start hidden state; no token/cache layout change.
    # Disabled for old artifacts. Only the final projection starts at zero.
    instruction_conditioning: bool = False
    instruction_condition_hidden_size: int = Field(default=256, ge=1)
    # Independent instruction residual in the existing LLM voice-patch input.
    # No new tokens or acoustic targets; the final projection starts at zero.
    # Fixed on for the released VoTSpeech dual-path checkpoint. Public
    # artifacts omit this historical ablation switch from config.json.
    semantic_patch_conditioning: bool = True
    # Eq. (3) ablation: e_v=P_h(h_c), g_v=G_h(h_c), no voice prior.
    instruction_residual_only: bool = False
    semantic_patch_hidden_size: int = Field(default=256, ge=1)
    # Scratch positions inserted between the instruction and the pair.
    # They carry no injected embedding and no loss of their own, so what
    # lands there is shaped only by the acoustic losses downstream -- the
    # chain-of-thought arrangement, not a second bottleneck. This value is
    # saved with the checkpoint so inference rebuilds the same layout.
    num_think_slots: int = Field(default=0, ge=0, le=64)
    # <|prosody_think|> slots, emitted AFTER the text and before the first
    # audio token. Unlike num_think_slots these can see what is about to be
    # said. The current target is the mean normalized VAE latent of segment m
    # (without subtracting the utterance mean). This is an acoustic summary,
    # not an explicitly disentangled prosody representation.
    num_prosody_slots: int = Field(default=0, ge=0, le=64)
    # Audio patches covered by one plan slot. Segments are an ABSOLUTE stride,
    # not a fraction of the utterance: a relative split needs the total patch
    # count, and at inference that number does not exist yet -- generation
    # stops on EOS while the slots are read during prefill. M * this value is
    # the span the plan covers at full resolution; longer utterances hold the
    # final slot's plan through the tail.
    prosody_patches_per_slot: int = Field(default=4, ge=1, le=64)
    # Where the DiT's global conditioning comes from.
    #   "xvector" -- the CAM++ anchor path, as it has always been.
    #   "plan"    -- every x-vector-derived term is scaled to zero and the
    #                per-position plan is the only conditioning, so the timbre
    #                is its time-average instead of a separately predicted
    #                vector competing with it. Cloning is unaffected: it
    #                supplies its own g_cond from the reference at inference.
    g_cond_source: Literal["xvector", "plan"] = "xvector"

    code_stats_momentum: float = Field(default=0.01, gt=0.0, le=1.0)
    code_stats_warmup_batches: int = Field(default=100, ge=0)

    # Fraction of voice-design rows whose g_cond comes from the PREDICTED code
    # instead of the teacher one. Without this the branch never trains against
    # its own output distribution and instruction-only inference is out of
    # distribution; ramped from zero because a freshly initialized flow head
    # emits noise, and feeding noise into g_cond early poisons the acoustic head.
    predicted_code_prob: float = Field(default=0.75, ge=0.0, le=1.0)
    predicted_code_start_step: int = Field(default=0, ge=0)
    predicted_code_warmup_steps: int = Field(default=2000, ge=0)

    # Training-only gradient boundary for the voice flow objective. The prior
    # learns to read LLM features without changing them through its FM loss.
    # Does not detach the semantic-patch/acoustic paths or change sampling.
    # False preserves existing training recipes and checkpoint behavior.
    detach_condition_for_flow: bool = False

    # Weight of the direct (cosine) alignment head, annealed away once the flow
    # head can carry the code on its own. Keeping it at full strength forever
    # pulls the <|voice_gen_start|> hidden state back toward the conditional mean.
    align_weight_start: float = Field(default=1.0, ge=0.0)
    align_weight_final: float = Field(default=0.1, ge=0.0)
    align_warmup_steps: int = Field(default=5000, ge=0)

    # Euler steps used by the scheduled-sampling prefix pass during training.
    # Kept small: it runs every step on the selected rows, and its job is to
    # produce something in the right distribution, not something good.
    train_sample_steps: int = Field(default=2, ge=1)
    infer_sample_steps: int = Field(default=16, ge=1)
    infer_guidance_scale: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_instruction_conditioning(self):
        if self.pooled_residual and (
            not self.enabled or self.voice_source != "pooled"
            or not self.freeze_extractor or self.instruction_residual_only
        ):
            raise ValueError("pooled_residual requires enabled pooled voice targets, a frozen extractor and the voice-code branch")
        if self.instruction_residual_only and (
            not self.enabled or not self.instruction_conditioning
            or not self.semantic_patch_conditioning or self.num_think_slots
            or self.num_prosody_slots or self.extractor_checkpoint
        ):
            raise ValueError("instruction_residual_only requires both instruction projections, enabled=True, no think/prosody slots or extractor checkpoint")
        if self.instruction_conditioning and self.g_cond_source != "xvector":
            raise ValueError("instruction_conditioning requires g_cond_source='xvector'")
        if self.semantic_patch_conditioning and (
            not self.instruction_conditioning or self.voice_source not in {"pooled", "none"}
            or self.num_think_slots or self.num_prosody_slots
        ):
            raise ValueError(
                "semantic_patch_conditioning requires pooled or speaker-only instruction_conditioning "
                "with zero extra think/prosody slots"
            )
        return self


class MeanFlowConfig(ConfigBase):
    enabled: bool = False
    use_duration_embedding: bool = True


class SamplingConfig(StrictConfigBase):
    solver: Literal["flow_matching", "scm"]
    ode_method: Literal["euler"] = "euler"
    num_steps: Literal[1, 2] = 2
    guidance_scale: Literal[0.0] = 0.0
    tau_mid: float = Field(default=1.3, gt=0.0, lt=math.pi / 2)

    @model_validator(mode="after")
    def _validate_solver_contract(self) -> "SamplingConfig":
        expected_num_steps = 1 if self.solver == "flow_matching" else 2
        if self.num_steps != expected_num_steps:
            raise ValueError(
                f"{self.solver} artifact requires num_steps={expected_num_steps}; "
                f"got num_steps={self.num_steps}."
            )
        return self

    def resolve(
        self,
        *,
        ode_method: str | None,
        num_steps: int | None,
        guidance_scale: float | None,
    ) -> tuple[str, int, float]:
        expected = (self.ode_method, self.num_steps, self.guidance_scale)
        resolved = (
            self.ode_method if ode_method is None else str(ode_method),
            self.num_steps if num_steps is None else int(num_steps),
            self.guidance_scale if guidance_scale is None else float(guidance_scale),
        )
        if resolved != expected:
            raise ValueError(
                f"{self.solver} artifact requires "
                f"ode_method={expected[0]!r}, num_steps={expected[1]}, "
                f"guidance_scale={expected[2]}; got "
                f"ode_method={resolved[0]!r}, num_steps={resolved[1]}, "
                f"guidance_scale={resolved[2]}."
            )
        return resolved


class StreamingConfig(StrictConfigBase):
    interleave_mode: Literal["one_to_one", "buffered_ratio"] = "one_to_one"
    initial_lookahead: int | None = None
    ta_per_tta: int = 2
    warmup_ta: int = 1

    @model_validator(mode="after")
    def _validate_streaming_contract(self) -> "StreamingConfig":
        if self.interleave_mode == "one_to_one":
            return self
        if self.initial_lookahead is not None and self.initial_lookahead <= 0:
            raise ValueError("initial_lookahead must be positive.")
        if self.ta_per_tta <= 0:
            raise ValueError("ta_per_tta must be positive for buffered_ratio.")
        if self.warmup_ta < 0:
            raise ValueError("warmup_ta must be non-negative.")
        return self


class ModelConfig(ConfigBase):
    model_type: str = "dots_tts"
    latent_dim: int
    patch_size: int
    cfg_droprate: float = 0.2
    PatchEncoder: _EncoderConfig
    DiT: _DiTConfig
    vocoder: AudioVAEConfig
    fm_sigma: float = 0.0
    xvec_drop_rate: float = 0.2
    campplus_embedding_size: int | None = 512
    xvec_max_audio_seconds: float = 10.0
    meanflow: MeanFlowConfig | None = None
    sampling: SamplingConfig | None = None
    streaming: StreamingConfig | None = None
    # Left as None on stock artifacts so `to_declared_dict()` omits the key and
    # released checkpoints keep validating byte-for-byte.
    voice_design: VoiceDesignConfig | None = None

    @model_validator(mode="after")
    def _validate_voice_design_contract(self) -> "ModelConfig":
        voice_design = self.voice_design
        if voice_design is None or not voice_design.enabled:
            return self
        if self.campplus_embedding_size is None:
            raise ValueError(
                "voice_design requires campplus_embedding_size; the branch "
                "predicts into the CAM++ x-vector space consumed by g_cond."
            )
        return self


__all__ = [
    "LossConfig",
    "MeanFlowConfig",
    "ModelConfig",
    "SamplingConfig",
    "StreamingConfig",
    "VoiceDesignConfig",
]
