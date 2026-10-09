from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from dots_tts.modules.voice_design.codec import VoiceCodeCodec
from dots_tts.modules.voice_design.extractor import VoiceQFormer
from dots_tts.modules.voice_design.robust_pooling import ResidualPool
from dots_tts.modules.voice_design.heads import VoiceDirectHead, VoiceFlowDiT
from dots_tts.training.voice_losses import (
    anneal,
    batch_relation_loss,
    cosine_distance,
    masked_mean,
    slice_mse,
)


@dataclass
class VoiceDesignOutput:
    """Per-sample, batch-first outputs of one voice-design training step.

    Every loss is ``[B, 1]``: the voice latent is a single token, so there is
    nothing to index along a second axis. The flow term is split by slice so the
    QFormer half can be weighted independently of the CAM++ anchor half.
    """

    flow_anchor_loss: torch.Tensor  # [B, 1]
    flow_voice_loss: torch.Tensor  # [B, 1]
    align_loss: torch.Tensor  # [B, 1]
    relation_loss: torch.Tensor  # [B, 1]
    row_mask: torch.Tensor  # [B, 1]
    offdiag_cosine: torch.Tensor  # [B, 1]
    align_weight: float
    predicted_code_prob: float


def gather_voice_hidden(
    hidden_states: torch.Tensor,
    voice_gen_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick the ``<|voice_gen_start|>`` hidden state, ``[B, 1, H]`` + ``[B]`` mask.

    Rows carrying no voice position (a plain TTS sample in a mixed batch) get
    zeros and a False mask rather than being dropped, so every tensor downstream
    keeps a fixed shape. Shape stability is not cosmetic: the run uses
    ``find_unused_parameters=False``, so a batch-dependent graph would change how
    many gradient all-reduces each rank issues and hang the job.
    """

    if hidden_states.dim() != 3:
        raise ValueError(
            f"Expected [batch, seq, hidden] hidden states, got {tuple(hidden_states.shape)}."
        )
    batch_size, _, hidden_size = hidden_states.shape
    voice_gen_mask = voice_gen_mask.to(device=hidden_states.device).bool()
    counts = voice_gen_mask.sum(dim=1)
    unexpected = counts[counts > 1]
    if unexpected.numel() > 0:
        raise ValueError(
            "Every sample must carry at most one <|voice_gen_start|>; got counts "
            f"{sorted(set(unexpected.tolist()))}."
        )

    packed = hidden_states.new_zeros((batch_size, 1, hidden_size))
    row_mask = counts.eq(1)
    for index in range(batch_size):
        if bool(row_mask[index]):
            packed[index, 0] = hidden_states[index][voice_gen_mask[index]][0]
    return packed, row_mask


class VoiceDesignBranch(nn.Module):
    """instruction -> voice latent (patch 0) -> LM context + ``g_cond``.

    The branch owns three things and nothing else: the teacher encoder that says
    what the voice latent of a given utterance *is*, the flow field that samples
    one from an instruction, and the running statistics that keep both in the
    same normalized space. Where the sampled latent then goes — into the
    ``<|voice_patch|>`` input embedding and into ``g_cond`` — is the caller's
    job, precisely because those two consumers must be handed the *same* draw.
    """

    def __init__(
        self,
        config,
        *,
        lm_hidden_size: int,
        anchor_dim: int,
        voice_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.anchor_dim = int(anchor_dim)
        self.voice_dim = int(voice_dim)
        self.latent_dim = int(latent_dim)
        # Public VoTSpeech artifacts omit the training-time representation
        # ablation field and are therefore pooled by construction.
        self.voice_source = str(getattr(config, "voice_source", "pooled"))
        if self.voice_source not in {"qformer", "pooled", "none"}:
            raise ValueError(
                "voice_source must be one of 'qformer' | 'pooled' | 'none'; got "
                f"{self.voice_source!r}."
            )
        if self.voice_source == "none":
            self.voice_dim = 0

        self.codec = VoiceCodeCodec(
            anchor_dim=self.anchor_dim,
            voice_dim=self.voice_dim,
            momentum=float(config.code_stats_momentum),
            warmup_batches=int(config.code_stats_warmup_batches),
        )
        self.code_dim = self.codec.code_dim

        # The teacher. Frozen by default: it was trained in Stage-1 against
        # speaker centroids, and letting the acoustic loss move it would let the
        # model make the target easier instead of making the prediction better —
        # the representation would drift toward whatever the flow head already
        # emits, which is collapse with a healthy-looking curve.
        self.extractor: nn.Module | None = None
        if getattr(config, "pooled_residual", False):
            self.extractor = ResidualPool(self.latent_dim, config.pooled_residual_hidden,
                                          config.pooled_residual_alpha).requires_grad_(False)
        if self.voice_source == "qformer":
            self.extractor = VoiceQFormer(
                self.latent_dim,
                self.voice_dim,
                num_queries=1,
                hidden_size=int(config.qformer_hidden_size),
                num_layers=int(config.qformer_num_layers),
                num_heads=int(config.qformer_num_heads),
            )
            if bool(config.freeze_extractor):
                for parameter in self.extractor.parameters():
                    parameter.requires_grad_(False)

        self.direct_head = VoiceDirectHead(
            lm_hidden_size,
            self.code_dim,
            intermediate_size=config.direct_head_intermediate_size,
        )
        self.flow_dit = VoiceFlowDiT(
            self.code_dim,
            lm_hidden_size,
            hidden_size=int(config.flow_hidden_size),
            num_layers=int(config.flow_num_layers),
            num_heads=int(config.flow_num_heads),
            mlp_ratio=float(config.flow_mlp_ratio),
            dropout=float(config.flow_dropout),
            max_positions=64,
        )
        self.register_buffer("train_steps", torch.zeros((), dtype=torch.long))

    # region schedules
    @property
    def current_step(self) -> int:
        return int(self.train_steps.item())

    def align_weight(self, step: int | None = None) -> float:
        step = self.current_step if step is None else int(step)
        return anneal(
            float(self.config.align_weight_start),
            float(self.config.align_weight_final),
            step,
            int(self.config.align_warmup_steps),
        )

    def predicted_code_prob(self, step: int | None = None) -> float:
        step = self.current_step if step is None else int(step)
        start_step = int(self.config.predicted_code_start_step)
        if step < start_step:
            return 0.0
        return anneal(
            0.0,
            float(self.config.predicted_code_prob),
            step - start_step,
            int(self.config.predicted_code_warmup_steps),
        )
    # endregion schedules

    # region teacher
    def voice_target(
        self,
        latents: torch.Tensor | None,
        latent_lengths: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """The voice slice of the teacher code, ``[B, voice_dim]``."""

        if self.voice_dim == 0 or latents is None:
            return None
        if self.extractor is not None:
            frozen = bool(self.config.freeze_extractor)
            with torch.set_grad_enabled(self.training and not frozen):
                if isinstance(self.extractor, ResidualPool):
                    if latent_lengths is None:
                        latent_lengths = torch.full((latents.size(0),), latents.size(1),
                                                    device=latents.device, dtype=torch.long)
                    return self.extractor(latents, latent_lengths)
                return self.extractor(latents, latent_lengths).squeeze(1)
        # Fallback for running Stage-2 before Stage-1 exists. Padding has to be
        # excluded or the mean drifts toward zero in proportion to how much a
        # sample was padded, turning batch composition into a learnable signal.
        return masked_mean(latents, latent_lengths)

    def encode_target(
        self,
        anchor: torch.Tensor,
        voice: torch.Tensor | None,
        *,
        update_stats: bool = True,
    ) -> torch.Tensor:
        """Teacher code in normalized space, ``[B, 1, code_dim]``."""

        return self.codec.encode(anchor, voice, update_stats=update_stats)
    # endregion teacher

    # region sampling
    def _velocity(self, noisy, timesteps, condition, guidance_scale):
        conditional = self.flow_dit(noisy, timesteps, condition)
        if guidance_scale == 1.0:
            return conditional
        unconditional = self.flow_dit(noisy, timesteps, torch.zeros_like(condition))
        return unconditional + guidance_scale * (conditional - unconditional)

    def sample_code(
        self,
        condition_hidden: torch.Tensor,
        *,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate the flow field from noise to a code, ``[B, 1, code_dim]``.

        The result is in the codec's *normalized* space, which is what both
        consumers want: ``voice_embed_proj`` because a unit-scale input is what
        a freshly initialized projection can learn from, and ``decode`` because
        that is the space it inverts.
        """

        num_steps = int(self.config.infer_sample_steps if num_steps is None else num_steps)
        if num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}.")
        guidance_scale = float(
            self.config.infer_guidance_scale if guidance_scale is None else guidance_scale
        )
        dtype = self.flow_dit.input_proj.weight.dtype
        device = condition_hidden.device
        sample = torch.randn(
            condition_hidden.size(0),
            1,
            self.code_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)
        for index in range(num_steps):
            current = grid[index].expand(condition_hidden.size(0))
            velocity = self._velocity(sample, current, condition_hidden, guidance_scale)
            sample = sample + (grid[index + 1] - grid[index]).to(sample) * velocity
        return sample

    def decode_code(self, code: torch.Tensor):
        return self.codec.decode(code)
    # endregion sampling

    # region training
    def forward(
        self,
        condition_hidden: torch.Tensor,
        target_code: torch.Tensor,
        row_mask: torch.Tensor,
        *,
        step: int | None = None,
    ) -> VoiceDesignOutput:
        """Losses only. Sampling for the scheduled-sampling draw happens in the
        caller's short prefix pass, so that one draw can feed both consumers."""

        batch_size = condition_hidden.size(0)
        device = condition_hidden.device
        row_weight = row_mask.to(condition_hidden.dtype).reshape(batch_size, 1)
        # Detached unconditionally. With freeze_extractor=false the teacher would
        # otherwise receive gradient from every voice term AND from the acoustic
        # loss, all of which it can reduce by collapsing its output to a
        # constant — the target chasing the prediction, which is collapse with a
        # perfectly healthy-looking curve.
        target = target_code.detach().to(condition_hidden.dtype)

        direct_code = self.direct_head(condition_hidden)
        anchor_slice = self.codec.anchor_slice
        # Alignment and anti-collapse look at the anchor slice only. It is the
        # speaker-invariant one, so "two different instructions should give two
        # different codes" is a well-posed statement about it.
        align_loss = cosine_distance(
            direct_code[..., anchor_slice], target[..., anchor_slice]
        )
        relation_loss, offdiag = batch_relation_loss(
            direct_code[..., anchor_slice],
            target[..., anchor_slice],
            row_weight=row_weight,
        )

        # Detach only the flow objective's input, not the caller's shared
        # hidden state: the acoustic and semantic-patch paths must remain live.
        condition = (
            condition_hidden.detach()
            if self.config.detach_condition_for_flow
            else condition_hidden
        )
        if self.training and self.config.condition_dropout > 0:
            keep = (
                torch.rand(batch_size, device=device) >= float(self.config.condition_dropout)
            ).to(condition_hidden.dtype)
            condition = condition * keep.reshape(-1, 1, 1)

        noise = torch.randn_like(target)
        times = torch.rand(batch_size, device=device, dtype=torch.float32)
        time_view = times.reshape(-1, 1, 1).to(target)
        noisy = (1.0 - time_view) * noise + time_view * target
        target_velocity = target - noise
        predicted_velocity = self.flow_dit(noisy, times, condition)
        flow_anchor_loss = slice_mse(predicted_velocity, target_velocity, anchor_slice)
        flow_voice_loss = (
            slice_mse(predicted_velocity, target_velocity, self.codec.voice_slice)
            if self.voice_dim
            else flow_anchor_loss.new_zeros(flow_anchor_loss.shape)
        )

        if self.training:
            self.train_steps.add_(1)

        return VoiceDesignOutput(
            flow_anchor_loss=flow_anchor_loss,
            flow_voice_loss=flow_voice_loss,
            align_loss=align_loss,
            relation_loss=relation_loss,
            row_mask=row_weight,
            offdiag_cosine=offdiag,
            align_weight=self.align_weight(step),
            predicted_code_prob=self.predicted_code_prob(step),
        )
    # endregion training


__all__ = [
    "VoiceDesignBranch",
    "VoiceDesignOutput",
    "gather_voice_hidden",
]
