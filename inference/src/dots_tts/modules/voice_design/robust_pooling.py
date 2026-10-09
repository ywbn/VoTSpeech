"""Experimental bounded correction to pooled VAE features; not a TTS default."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def frame_mask(x, lengths):
    if x.ndim != 3 or lengths.shape != (x.shape[0],):
        raise ValueError('Expected latents [B,T,D] and lengths [B]')
    if bool(((lengths < 1) | (lengths > x.shape[1])).any()):
        raise ValueError('Lengths must be in [1,T]')
    return torch.arange(x.shape[1], device=x.device)[None, :] < lengths.to(x.device)[:, None]


def pool(x, lengths):
    mask = frame_mask(x, lengths)[..., None]
    return x.masked_fill(~mask, 0).sum(1) / lengths.to(x)[:, None]


class ResidualPool(nn.Module):
    """Identity-initialized temporal residual, bounded relative to input RMS.

    The scalar alpha is fixed, NOT zero-initialized/trainable: a zero gate plus
    zero output weights would block both gradients. Padding is removed at every
    temporal layer so batching does not leak padded frames into valid features.
    """
    def __init__(self, dim, hidden=128, alpha=0.1):
        super().__init__()
        if dim < 1 or hidden < 1 or not 0 < alpha <= 0.5:
            raise ValueError('Positive dimensions and 0 < alpha <= 0.5 required')
        self.dim, self.hidden, self.alpha = dim, hidden, alpha
        self.norm = nn.LayerNorm(dim)
        self.input = nn.Linear(dim, hidden)
        self.temporal = nn.Conv1d(hidden, hidden, 3, padding=1)
        self.output = nn.Linear(hidden, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latents, lengths):
        x = latents.detach().float()  # frozen target/extractor input, no VAE gradients
        mask = frame_mask(x, lengths)[..., None]
        x = x.masked_fill(~mask, 0)
        base = pool(x, lengths)
        h = F.silu(self.input(self.norm(x.to(self.input.weight.dtype)))).masked_fill(~mask, 0)
        h = F.silu(self.temporal(h.transpose(1, 2)).transpose(1, 2)).masked_fill(~mask, 0)
        delta = self.output(pool(h, lengths)).tanh().float()
        # Every coordinate's correction is bounded by alpha * input RMS.
        scale = pool(x.square(), lengths).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        return base + self.alpha * scale * delta


def paired_loss(clean_output, noisy_output, target, scale):
    """One regression objective, equal weights for identity and restoration.

    Fixed, detached targets prevent the teacher from chasing its own outputs.
    This preserves existing recording characteristics; it cannot purify them.
    """
    target = target.detach().float()
    scale = scale.detach().float().clamp_min(1e-6)
    identity = ((clean_output.float() - target) / scale).square().mean()
    restore = ((noisy_output.float() - target) / scale).square().mean()
    return 0.5 * (identity + restore)


def perturb_waveforms(waves, lengths, sample_rate, generator, mode='mixed'):
    """Mild synthetic additive noise/decaying echoes; NOT a measured room RIR.

    No pitch shift, formant manipulation, EQ or speed perturbation. No disk
    writes. Original waveforms and padding remain unchanged.
    """
    if mode not in ('noise', 'echo', 'mixed') or waves.ndim != 3 or waves.shape[1] != 1:
        raise ValueError('Expected [B,1,T] and noise/echo/mixed mode')
    output = torch.zeros_like(waves)
    for i, length in enumerate(lengths.tolist()):
        if not 0 < length <= waves.shape[-1]:
            raise ValueError('Invalid waveform length')
        x = waves[i, 0, :length]
        y = x.clone()
        if mode in ('echo', 'mixed'):
            # Short delayed copies approximate a mild room tail; do not call
            # this a physical room simulator or proof of real dereverberation.
            delay = int((0.012 + 0.028 * torch.rand((), generator=generator).item()) * sample_rate)
            strength = 0.04 + 0.12 * torch.rand((), generator=generator).item()
            for tap in range(1, 5):
                shift = delay * tap
                if 0 < shift < length:
                    y[shift:] += strength * (0.55 ** (tap - 1)) * x[:-shift]
        if mode in ('noise', 'mixed'):
            snr = 20 + 15 * torch.rand((), generator=generator).item()
            noise = torch.randn(x.shape, generator=generator, dtype=torch.float32).to(x)
            noise = noise / noise.square().mean().sqrt().clamp_min(1e-8)
            y += noise * x.square().mean().sqrt() * (10 ** (-snr / 20))
        output[i, 0, :length] = y
    return output
