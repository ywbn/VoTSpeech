from __future__ import annotations

import atexit
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load as load_safetensors

CACHE_FIELD = "dots_tts_feature_cache"
CACHE_FORMAT = "safetensors-pread-v1"
_MAX_OPEN_SHARDS = 32
_OPEN_SHARDS: OrderedDict[str, int] = OrderedDict()


def _close_open_shards() -> None:
    while _OPEN_SHARDS:
        _path, fd = _OPEN_SHARDS.popitem(last=False)
        try:
            os.close(fd)
        except OSError:
            pass


atexit.register(_close_open_shards)


def _open_shard(path: str) -> int:
    normalized = str(Path(path).expanduser())
    fd = _OPEN_SHARDS.pop(normalized, None)
    if fd is not None:
        _OPEN_SHARDS[normalized] = fd
        return fd

    fd = os.open(normalized, os.O_RDONLY)
    _OPEN_SHARDS[normalized] = fd
    while len(_OPEN_SHARDS) > _MAX_OPEN_SHARDS:
        _old_path, old_fd = _OPEN_SHARDS.popitem(last=False)
        os.close(old_fd)
    return fd


def has_feature_cache(sample: dict[str, Any]) -> bool:
    return isinstance(sample.get(CACHE_FIELD), dict)


def load_feature_cache(sample: dict[str, Any]) -> dict[str, Any]:
    location = sample.get(CACHE_FIELD)
    if not isinstance(location, dict):
        raise KeyError(f"Sample has no {CACHE_FIELD!r} location.")
    if location.get("format") != CACHE_FORMAT:
        raise ValueError(
            f"Unsupported feature-cache format {location.get('format')!r}; "
            f"expected {CACHE_FORMAT!r}."
        )

    path = str(location["path"])
    offset = int(location["offset"])
    length = int(location["length"])
    if offset < 0 or length <= 0:
        raise ValueError(
            f"Invalid feature-cache location: path={path!r} "
            f"offset={offset} length={length}."
        )

    payload = os.pread(_open_shard(path), length, offset)
    if len(payload) != length:
        raise OSError(
            f"Short feature-cache read from {path}: wanted {length} bytes at "
            f"offset {offset}, got {len(payload)}."
        )
    tensors = load_safetensors(payload)
    required = {
        "token_ids",
        "loss_mask",
        "latent",
        "xvector",
        "sample_length",
        "num_text_tokens",
        "num_audio_tokens",
    }
    missing = sorted(required.difference(tensors))
    if missing:
        raise ValueError(f"Feature-cache record in {path} is missing {missing}.")

    token_ids = tensors["token_ids"].to(dtype=torch.long)
    loss_mask = tensors["loss_mask"].to(dtype=torch.float32)
    latent = tensors["latent"]
    xvector = tensors["xvector"]
    if token_ids.ndim != 1 or token_ids.numel() < 2:
        raise ValueError(
            f"Cached token_ids must be a 1D tensor with at least two items, got "
            f"{tuple(token_ids.shape)}."
        )
    if loss_mask.shape != token_ids[:-1].shape:
        raise ValueError(
            f"Cached loss_mask shape {tuple(loss_mask.shape)} does not match "
            f"shifted token shape {tuple(token_ids[:-1].shape)}."
        )
    if latent.ndim != 2:
        raise ValueError(
            f"Cached latent must have shape (frames, channels), got "
            f"{tuple(latent.shape)}."
        )
    if xvector.ndim != 1:
        raise ValueError(
            f"Cached xvector must be 1D, got {tuple(xvector.shape)}."
        )

    return {
        "input_ids": token_ids[:-1],
        "labels": token_ids[1:],
        "loss_mask": loss_mask,
        "input_ids_length": int(token_ids.numel() - 1),
        "num_text_tokens": int(tensors["num_text_tokens"].item()),
        "num_audio_tokens": int(tensors["num_audio_tokens"].item()),
        "num_total_tokens": int(token_ids.numel() - 1),
        "sample_length": int(tensors["sample_length"].item()),
        "latent_length": int(latent.size(0)),
        "latents_sampled": latent,
        "xvector": xvector,
        "features_precomputed": True,
    }


__all__ = [
    "CACHE_FIELD",
    "CACHE_FORMAT",
    "has_feature_cache",
    "load_feature_cache",
]
