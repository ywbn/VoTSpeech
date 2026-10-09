from __future__ import annotations

import math

import torch
import torch.nn as nn


def _norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    norm_cls = getattr(nn, "RMSNorm", nn.LayerNorm)
    return norm_cls(hidden_size, eps=eps)


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256) -> None:
        super().__init__()
        self.frequency_size = int(frequency_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float().reshape(-1, 1) * freqs.reshape(1, -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if embedding.size(-1) < self.frequency_size:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        first = self.mlp[0]
        embedding = embedding.to(device=first.weight.device, dtype=first.weight.dtype)
        return self.mlp(embedding)


class VoiceDirectHead(nn.Module):
    """Direct regression from the ``<|voice_gen_start|>`` hidden to the code.

    An auxiliary, not the generator: it gives the LM a fast, well-conditioned
    gradient while the flow field is still random, which is the difference
    between the queries learning something in the first thousand steps and
    learning nothing.

    Supervised with cosine and annealed away, because instruction -> timbre is
    one-to-many ("a low male voice" admits many speakers). An MSE-minimizing
    regressor converges to the conditional *mean* — an averaged, characterless
    voice — and would drag the shared hidden state there exactly when the flow
    head needs it to describe the full conditional distribution.
    """

    def __init__(
        self,
        hidden_size: int,
        code_dim: int,
        intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        intermediate_size = int(intermediate_size or max(hidden_size, code_dim))
        self.hidden_size = int(hidden_size)
        self.code_dim = int(code_dim)
        self.norm = _norm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, code_dim),
        )

    def forward(self, condition_hidden: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H]`` -> ``[B, 1, code_dim]``."""

        if condition_hidden.dim() != 3 or condition_hidden.size(1) != 1:
            raise ValueError(
                "VoiceDirectHead expects [batch, 1, hidden], got "
                f"{tuple(condition_hidden.shape)}."
            )
        weight = self.mlp[0].weight
        condition_hidden = condition_hidden.to(device=weight.device, dtype=weight.dtype)
        return self.mlp(self.norm(condition_hidden))


class _BidirectionalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = _norm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VoiceFlowDiT(nn.Module):
    """Conditional flow field over the single voice-code token.

    Layout is ``[time, condition_hidden, noisy_code]`` and only the code position
    is projected back to a velocity. The code is one token, so the sequence is
    three tokens long and this costs nothing next to the LM.

    Modelling a distribution rather than a point estimate is the whole reason
    for the flow head: it is what lets one instruction produce different
    plausible voices under different seeds instead of one averaged voice.
    """

    def __init__(
        self,
        code_dim: int,
        condition_dim: int,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_positions: int = 128,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                "VoiceFlowDiT hidden_size must be divisible by num_heads: "
                f"hidden_size={hidden_size}, num_heads={num_heads}."
            )
        self.code_dim = int(code_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_size = int(hidden_size)
        self.max_positions = int(max_positions)
        self.input_proj = nn.Linear(code_dim, hidden_size)
        self.condition_proj = nn.Linear(condition_dim, hidden_size)
        self.time_embed = TimestepEmbedding(hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_positions, hidden_size)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            _BidirectionalBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, dropout=dropout
            )
            for _ in range(num_layers)
        )
        self.out_norm = _norm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, code_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        noisy_code: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_code.dim() != 3 or condition.dim() != 3:
            raise ValueError(
                "VoiceFlowDiT expects rank-3 code and condition, got "
                f"{tuple(noisy_code.shape)} and {tuple(condition.shape)}."
            )
        dtype = self.input_proj.weight.dtype
        device = self.input_proj.weight.device
        noisy_code = noisy_code.to(device=device, dtype=dtype)
        condition = condition.to(device=device, dtype=dtype)

        x = self.input_proj(noisy_code)
        cond = self.condition_proj(condition)
        time = self.time_embed(timesteps).to(device=x.device, dtype=x.dtype).unsqueeze(1)
        x = torch.cat([time, cond, x], dim=1)
        if x.size(1) > self.max_positions:
            raise ValueError(
                f"VoiceFlowDiT sequence exceeds max_positions: {x.size(1)} > {self.max_positions}."
            )
        x = x + self.position_embedding[:, : x.size(1)].to(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(self.out_norm(x[:, -noisy_code.size(1) :]))


__all__ = [
    "TimestepEmbedding",
    "VoiceDirectHead",
    "VoiceFlowDiT",
]
