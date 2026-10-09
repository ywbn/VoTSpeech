from __future__ import annotations

import torch
import torch.nn as nn


class _RunningStats(nn.Module):
    """Per-dimension mean/std tracked BatchNorm-voice, frozen at eval.

    Flow matching wants a roughly zero-mean unit-variance target. Neither a
    CAM++ x-vector nor a pooled AudioVAE latent is either, and their scales
    differ from each other by more than an order of magnitude, so the two halves
    of the code have to be normalized per dimension or the larger one dominates
    the velocity loss outright. Tracking the statistics online avoids requiring
    an offline pass over the corpus before the first training run; ``load`` can
    still install measured values when they exist.
    """

    def __init__(
        self,
        dim: int,
        *,
        momentum: float = 0.01,
        eps: float = 1e-5,
        warmup_batches: int = 100,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.warmup_batches = int(warmup_batches)
        self.register_buffer("running_mean", torch.zeros(self.dim))
        self.register_buffer("running_var", torch.ones(self.dim))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def load(self, mean: torch.Tensor, var: torch.Tensor, *, freeze: bool = True) -> None:
        mean = torch.as_tensor(mean).reshape(-1).to(self.running_mean)
        var = torch.as_tensor(var).reshape(-1).to(self.running_var)
        if mean.numel() != self.dim or var.numel() != self.dim:
            raise ValueError(
                f"Statistics must have {self.dim} entries; got mean={mean.numel()} "
                f"var={var.numel()}."
            )
        self.running_mean.copy_(mean)
        self.running_var.copy_(var.clamp_min(self.eps))
        self.frozen.fill_(bool(freeze))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if bool(self.frozen.item()) or values.size(0) < 2:
            # One row has zero unbiased variance; folding it in would collapse
            # the scale and blow up every normalized target afterwards.
            return
        values = values.detach().float()
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False).clamp_min(self.eps)
        tracked = int(self.num_batches_tracked.item())
        # Average the first few batches rather than applying the small EMA
        # momentum to the identity initialization: at momentum 0.01 it would take
        # thousands of steps to forget mean=0/var=1, and every target produced in
        # the meantime would be mis-scaled.
        momentum = 1.0 / float(tracked + 1) if tracked < self.warmup_batches else self.momentum
        self.running_mean.mul_(1.0 - momentum).add_(batch_mean, alpha=momentum)
        self.running_var.mul_(1.0 - momentum).add_(batch_var, alpha=momentum)
        self.num_batches_tracked.add_(1)

    def _scale(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.running_mean, self.running_var.clamp_min(self.eps).sqrt()

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, std = self._scale()
        return (values - mean.to(values)) / std.to(values)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, std = self._scale()
        return values * std.to(values) + mean.to(values)


class VoiceCodeCodec(nn.Module):
    """The voice latent that occupies patch 0: ``[CAM++ anchor | Voice-QFormer]``.

    Two quantities, deliberately kept as two slices of one vector rather than
    two independent heads:

    * **anchor** — the frozen CAM++ x-vector. Trained for speaker verification,
      so it is a speaker *invariant* by construction: the same voice reading a
      different sentence maps to nearly the same vector. That invariance is what
      makes ``instruction -> code`` a well-posed regression, and the pretrained
      ``xvec_proj`` already knows how to turn it into timbre.

    * **voice** — the Voice-QFormer output (or, as a fallback before Stage-1 has
      been run, the time-mean of the normalized AudioVAE latent). It carries
      what CAM++ deliberately discards: recording channel, loudness, long-term
      spectral tilt. The QFormer variant is trained against speaker centroids,
      so unlike the pooled fallback it is speaker-discriminative rather than
      content-dominated.

    One vector, one flow field: the two are correlated (a low warm voice has
    both a characteristic x-vector and a characteristic spectral tilt), and
    modelling them jointly lets the flow head use that correlation. The loss is
    still reported and weighted per slice, so the noisy half can be turned down
    without touching the anchor.
    """

    def __init__(
        self,
        *,
        anchor_dim: int,
        voice_dim: int,
        momentum: float = 0.01,
        warmup_batches: int = 100,
    ) -> None:
        super().__init__()
        if anchor_dim <= 0 or voice_dim < 0:
            raise ValueError(
                f"anchor_dim must be positive and voice_dim non-negative; got "
                f"anchor_dim={anchor_dim}, voice_dim={voice_dim}."
            )
        self.anchor_dim = int(anchor_dim)
        self.voice_dim = int(voice_dim)
        self.code_dim = self.anchor_dim + self.voice_dim
        self.stats = _RunningStats(
            self.code_dim, momentum=momentum, warmup_batches=warmup_batches
        )

    @property
    def anchor_slice(self) -> slice:
        return slice(0, self.anchor_dim)

    @property
    def voice_slice(self) -> slice:
        return slice(self.anchor_dim, self.code_dim)

    def encode(
        self,
        anchor: torch.Tensor,
        voice: torch.Tensor | None = None,
        *,
        update_stats: bool = False,
    ) -> torch.Tensor:
        """``[B, anchor_dim]`` (+ ``[B, voice_dim]``) -> normalized ``[B, 1, code_dim]``."""

        if anchor.dim() != 2 or anchor.size(-1) != self.anchor_dim:
            raise ValueError(
                f"anchor must be [batch, {self.anchor_dim}], got {tuple(anchor.shape)}."
            )
        if self.voice_dim == 0:
            combined = anchor
        else:
            if voice is None:
                voice = anchor.new_zeros((anchor.size(0), self.voice_dim))
            if voice.dim() != 2 or voice.size(-1) != self.voice_dim:
                raise ValueError(
                    f"voice must be [batch, {self.voice_dim}], got {tuple(voice.shape)}."
                )
            combined = torch.cat([anchor, voice.to(anchor)], dim=-1)
        if update_stats and self.training:
            self.stats.update(combined)
        return self.stats.normalize(combined).unsqueeze(1)

    def decode(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalized ``[B, 1, code_dim]`` -> ``(anchor, voice)`` in original units."""

        if code.dim() == 3:
            if code.size(1) != 1:
                raise ValueError(
                    f"Expected a single code token, got {tuple(code.shape)}."
                )
            code = code.squeeze(1)
        if code.size(-1) != self.code_dim:
            raise ValueError(
                f"code must end in {self.code_dim} dims, got {tuple(code.shape)}."
            )
        combined = self.stats.denormalize(code)
        anchor = combined[..., self.anchor_slice]
        voice = combined[..., self.voice_slice] if self.voice_dim else None
        return anchor, voice


__all__ = ["VoiceCodeCodec"]
