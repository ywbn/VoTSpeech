from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    norm_cls = getattr(nn, "RMSNorm", nn.LayerNorm)
    return norm_cls(hidden_size, eps=eps)


class _QFormerBlock(nn.Module):
    """Self-attention among the queries, then cross-attention into the latents."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm_self = _norm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = _norm(hidden_size)
        self.norm_kv = _norm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = _norm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm_self(queries)
        queries = queries + self.self_attn(h, h, h, need_weights=False)[0]
        kv = self.norm_kv(memory)
        queries = queries + self.cross_attn(
            self.norm_q(queries),
            kv,
            kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return queries + self.mlp(self.norm_mlp(queries))


class VoiceQFormer(nn.Module):
    """AudioVAE latent sequence -> a fixed-size voice representation.

    Non-causal by construction: the queries cross-attend over the *whole*
    utterance. That is fine for what this is used for — a training-time teacher
    and an encoder for reference audio, both of which see a complete waveform —
    but it means this module can never sit inside the autoregressive loop. The
    model's own voice prediction comes from the flow head, not from here.

    Attentive pooling rather than a time-mean: a mean is dominated by whichever
    phonemes happen to be in the sentence, so it mostly measures content. Giving
    the pooling learned queries and training them against speaker centroids is
    what turns it into something speaker-discriminative.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        *,
        num_queries: int = 1,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "VoiceQFormer hidden_size must be divisible by num_heads: "
                f"hidden_size={hidden_size}, num_heads={num_heads}."
            )
        if num_queries <= 0:
            raise ValueError(f"num_queries must be positive, got {num_queries}.")
        self.latent_dim = int(latent_dim)
        self.out_dim = int(out_dim)
        self.num_queries = int(num_queries)
        self.hidden_size = int(hidden_size)

        self.input_proj = nn.Linear(self.latent_dim, self.hidden_size)
        self.queries = nn.Parameter(
            torch.randn(self.num_queries, self.hidden_size) * 0.02
        )
        self.blocks = nn.ModuleList(
            _QFormerBlock(self.hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        )
        self.out_norm = _norm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.out_dim)

    def forward(
        self,
        latents: torch.Tensor,
        latent_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, T, latent_dim]`` -> ``[B, num_queries, out_dim]``."""

        if latents.dim() != 3 or latents.size(-1) != self.latent_dim:
            raise ValueError(
                f"latents must be [batch, time, {self.latent_dim}], got "
                f"{tuple(latents.shape)}."
            )
        weight = self.input_proj.weight
        latents = latents.to(device=weight.device, dtype=weight.dtype)

        key_padding_mask: torch.Tensor | None = None
        if latent_lengths is not None:
            time = latents.size(1)
            positions = torch.arange(time, device=latents.device).unsqueeze(0)
            # True marks positions attention must ignore.
            key_padding_mask = positions >= latent_lengths.to(latents.device).reshape(-1, 1)
            # A row of all-True makes softmax produce NaN rather than an error,
            # which then silently poisons the centroid bank. Keep one position
            # alive; the row is meaningless anyway and gets masked downstream.
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        memory = self.input_proj(latents)
        hidden = self.queries.unsqueeze(0).expand(latents.size(0), -1, -1).to(memory)
        for block in self.blocks:
            hidden = block(hidden, memory, key_padding_mask)
        return self.out_proj(self.out_norm(hidden))


class SpeakerCentroidBank(nn.Module):
    """EMA centroid per speaker, plus the contrastive objective against them.

    Centroids instead of in-batch positives because the streaming dataset has no
    speaker concept: ``OnlineBatcher`` buckets by length, so a batch almost never
    contains two utterances from the same speaker and an in-batch supervised
    contrastive loss would have no positives to work with. A centroid bank moves
    the positive out of the batch entirely, which makes the objective independent
    of how the batcher happens to group things.

    The centroids are buffers, so they checkpoint and resume with the model.
    """

    def __init__(
        self,
        num_speakers: int,
        dim: int,
        *,
        momentum: float = 0.05,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if num_speakers <= 1:
            raise ValueError(
                "A centroid contrastive objective needs at least two speakers; "
                f"got num_speakers={num_speakers}."
            )
        self.num_speakers = int(num_speakers)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.register_buffer("centroids", torch.zeros(self.num_speakers, self.dim))
        self.register_buffer(
            "initialized", torch.zeros(self.num_speakers, dtype=torch.bool)
        )

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor) -> None:
        values = F.normalize(embeddings.detach().float(), dim=-1)
        speaker_ids = speaker_ids.to(device=self.centroids.device, dtype=torch.long)
        values = values.to(self.centroids)
        for row in range(values.size(0)):
            index = int(speaker_ids[row].item())
            if not 0 <= index < self.num_speakers:
                raise IndexError(
                    f"speaker index {index} outside the bank of {self.num_speakers}."
                )
            if not bool(self.initialized[index]):
                # Seeding with the first observation rather than EMA-ing away
                # from zero: at momentum 0.05 a zero centroid stays near zero for
                # dozens of updates, and every loss computed meanwhile is noise.
                self.centroids[index] = values[row]
                self.initialized[index] = True
                continue
            self.centroids[index].mul_(1.0 - self.momentum).add_(
                values[row], alpha=self.momentum
            )
        self.centroids.copy_(F.normalize(self.centroids, dim=-1))

    def forward(
        self, embeddings: torch.Tensor, speaker_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(loss [B], accuracy [])`` against the current centroids."""

        speaker_ids = speaker_ids.to(device=embeddings.device, dtype=torch.long)
        normalized = F.normalize(embeddings.float(), dim=-1)
        centroids = self.centroids.to(normalized)
        logits = normalized @ centroids.t() / self.temperature

        # An uninitialized centroid is a zero vector, so it scores 0 against
        # everything. Left in, it is a competitor that means nothing; masked out,
        # the softmax ranges over speakers actually observed so far. A large
        # finite penalty rather than -inf: an all-masked row (the very first
        # batch) would otherwise produce NaN, and torch.where propagates NaN
        # gradients from the branch it did not select.
        seen = self.initialized.to(embeddings.device)
        logits = logits.masked_fill(~seen.unsqueeze(0), -1e4)
        # A row whose own centroid has not been seeded yet has no target to
        # point at; zero it out rather than training against an arbitrary one.
        targets = speaker_ids.clamp(0, self.num_speakers - 1)
        valid = seen[targets]
        loss = F.cross_entropy(logits, targets, reduction="none")
        loss = torch.where(valid, loss, torch.zeros_like(loss))
        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            correct = (predicted == targets) & valid
            accuracy = correct.float().sum() / valid.float().sum().clamp_min(1.0)
        return loss, accuracy


__all__ = ["SpeakerCentroidBank", "VoiceQFormer"]
