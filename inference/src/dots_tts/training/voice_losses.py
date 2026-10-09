"""Unreduced, batch-first loss helpers for the voice-design branch.

Every function returns a tensor whose first dimension is the batch, so the
results drop straight into ``LossTerm(loss=..., mask=...)`` and inherit the
existing per-source statistics, cross-rank normalization and weighting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    """Mean over time of ``[B, T, D]``, ignoring padded frames -> ``[B, D]``."""

    if values.dim() != 3:
        raise ValueError(f"Expected [batch, time, dim], got {tuple(values.shape)}.")
    if lengths is None:
        return values.mean(dim=1)
    steps = torch.arange(values.size(1), device=values.device).unsqueeze(0)
    mask = (steps < lengths.to(values.device).reshape(-1, 1)).to(values.dtype).unsqueeze(-1)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def slice_mse(
    prediction: torch.Tensor, target: torch.Tensor, span: slice
) -> torch.Tensor:
    """Mean squared error over one slice of the code -> ``[B, 1]``."""

    error = (prediction[..., span].float() - target[..., span].float()).pow(2)
    return error.mean(dim=-1).reshape(prediction.size(0), -1).mean(dim=1, keepdim=True)


def cosine_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """``1 - cos`` between two ``[B, 1, D]`` codes -> ``[B, 1]``.

    Cosine rather than MSE on purpose: the direct head regresses a one-to-many
    mapping, and an MSE optimum is the conditional mean, i.e. an averaged voice.
    Constraining direction only leaves the magnitude of the shared hidden state
    free for the flow head to use.
    """

    similarity = F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1e-6)
    return (1.0 - similarity).reshape(prediction.size(0), -1).mean(dim=1, keepdim=True)


def batch_relation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    row_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the batch's pairwise geometry. Returns ``(loss[B, 1], offdiag[B, 1])``.

    Cosine and MSE only ask each prediction to sit near its own target, which a
    model can satisfy on average by mapping every instruction into the same
    region — the failure mode where output timbre stops responding to the
    prompt. This compares the ``[B, B]`` cosine matrix of the predictions with
    the target's: it does not care where the codes land, only that two different
    instructions stay as far apart as their targets are. That is the
    discriminability a voice-design model actually needs, and unlike a
    supervised contrastive term it needs neither speaker labels nor
    speaker-balanced batches — which matters because the dots.tts batcher
    buckets by sequence length and has no speaker concept at all.

    The second return value is a collapse detector: predictions whose mean
    off-diagonal cosine approaches 1 are all the same code.

    ``row_weight`` is applied as a weight rather than by slicing so the executed
    operations never depend on batch contents — a data-dependent graph makes
    ranks disagree on how many gradient collectives to run, which deadlocks DDP.
    """

    batch = prediction.size(0)
    pred_flat = prediction.reshape(batch, -1).float()
    target_flat = target.reshape(batch, -1).float()
    if row_weight is not None:
        weights = row_weight.reshape(batch, 1).to(pred_flat)
        pred_flat = pred_flat * weights
        target_flat = target_flat * weights

    pred_normalized = F.normalize(pred_flat, dim=1, eps=1e-8)
    target_normalized = F.normalize(target_flat, dim=1, eps=1e-8)
    pred_relations = pred_normalized @ pred_normalized.transpose(0, 1)
    target_relations = target_normalized @ target_normalized.transpose(0, 1)

    pair_mask = ~torch.eye(batch, device=prediction.device, dtype=torch.bool)
    pair_weights = pair_mask.to(pred_relations)
    if row_weight is not None:
        valid = row_weight.reshape(batch).to(pred_relations).gt(0).to(pred_relations)
        pair_weights = pair_weights * valid.unsqueeze(0) * valid.unsqueeze(1)

    denominator = pair_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    squared_error = (pred_relations - target_relations).pow(2)
    loss = (squared_error * pair_weights).sum(dim=1, keepdim=True) / denominator
    offdiag = (pred_relations.detach() * pair_weights).sum(
        dim=1, keepdim=True
    ) / denominator
    return loss, offdiag


def anneal(start: float, end: float, step: int, warmup_steps: int) -> float:
    """Linearly move a scalar from ``start`` to ``end``; no warmup keeps ``start``."""

    if warmup_steps <= 0:
        return float(start)
    progress = min(1.0, max(0.0, float(step) / float(warmup_steps)))
    return float(start) + (float(end) - float(start)) * progress


__all__ = [
    "anneal",
    "batch_relation_loss",
    "cosine_distance",
    "masked_mean",
    "slice_mse",
]
