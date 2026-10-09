"""Checkpoint helpers for distributed dots_tts training.

This module persists not only model/optimizer/scheduler state, but also
rank-local RNG state and data-loader progress so resumed training can continue
from the same point with minimal drift.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

_CHECKPOINT_FORMAT_FILE = "checkpoint_format.json"
_DEEPSPEED_BACKEND = "deepspeed"


def _uses_deepspeed(accelerator) -> bool:
    distributed_type = str(getattr(accelerator, "distributed_type", ""))
    return distributed_type.rsplit(".", maxsplit=1)[-1].lower() == _DEEPSPEED_BACKEND


def _checkpoint_backend(checkpoint_dir: Path) -> str:
    format_path = checkpoint_dir / _CHECKPOINT_FORMAT_FILE
    if not format_path.is_file():
        return "legacy"
    payload = json.loads(format_path.read_text(encoding="utf-8"))
    return str(payload.get("backend", "legacy")).lower()


def _checkpoint_dir(log_dir: str, step: int) -> Path:
    """Return the canonical directory name for a training step checkpoint."""
    return Path(log_dir) / f"checkpoint-{step:08d}"


def _checkpoint_entries(log_dir: str) -> list[tuple[int, Path]]:
    """List valid ``checkpoint-*`` directories sorted by step number."""
    entries = []
    for path in Path(log_dir).glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            entries.append((int(suffix), path))
    return sorted(entries)


def resolve_latest_train_checkpoint(log_dir: str) -> Path:
    """Resolve the checkpoint directory that should be used for resume.

    Preference order:
    1. ``<log_dir>/latest`` symlink, if present.
    2. The numerically largest ``checkpoint-*`` directory.
    """
    latest_path = Path(log_dir) / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        return latest_path.resolve(strict=True)

    entries = _checkpoint_entries(log_dir)
    if not entries:
        raise FileNotFoundError(
            f"No checkpoint found under {log_dir!s}; expected latest or checkpoint-*."
        )
    return entries[-1][1].resolve(strict=True)


def _rng_state() -> dict:
    """Capture Python/NumPy/PyTorch RNG state for deterministic resume."""
    numpy_state = np.random.get_state()
    state = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict) -> None:
    """Restore RNG state previously produced by :func:`_rng_state`."""
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _replace_latest_symlink(log_dir: str, save_dir: Path) -> None:
    """Atomically refresh the ``latest`` symlink to point at ``save_dir``."""
    log_path = Path(log_dir)
    link_path = log_path / "latest"
    tmp_link_path = log_path / "latest.tmp"

    if tmp_link_path.exists() or tmp_link_path.is_symlink():
        tmp_link_path.unlink()
    tmp_link_path.symlink_to(save_dir.name)

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    tmp_link_path.rename(link_path)


def _export_inference_model(log_dir: str, save_dir: Path, step: int) -> Path | None:
    """Hardlink the inference artifact out of a checkpoint before cleanup.

    ``max_checkpoints_to_keep`` deletes whole checkpoint directories -- optimizer
    shards and model weights alike -- so any step you meant to run inference on
    later disappears with it. A DeepSpeed ZeRO-2 checkpoint is dominated by the
    optimizer shards, while the ``model/`` subdirectory alone is everything
    inference needs. Hardlinking it into ``exports/`` costs no extra disk while
    the checkpoint still exists and keeps the weights alive once it is deleted.
    """

    source = save_dir / "model"
    if not source.is_dir():
        return None
    target = Path(log_dir) / "exports" / f"step-{step:08d}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        destination = target / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, destination)
        except OSError:
            # Cross-device, or a filesystem without hardlinks. A copy is
            # slower and costs real space, but it must not fail the run.
            shutil.copy2(item, destination)
    return target


_PENDING_DELETE: "threading.Thread | None" = None


def _await_pending_deletes(timeout: float | None = None) -> None:
    """Block until the previous save's background delete finished."""
    global _PENDING_DELETE
    thread = _PENDING_DELETE
    if thread is not None and thread.is_alive():
        thread.join(timeout)
    _PENDING_DELETE = None


def _cleanup_old_checkpoints(log_dir: str, keep_max: int) -> None:
    """Retire older checkpoints, keeping the newest ``keep_max``.

    The unlinking runs on a daemon thread. Every rank but rank 0 is parked in
    `wait_for_everyone()` during the main-process-only half of a save, and that
    barrier is a NCCL collective with a finite timeout; deleting tens of
    gigabytes is the slowest thing in the save and nothing waits on it.
    """
    global _PENDING_DELETE
    if keep_max <= 0:
        return
    _await_pending_deletes()
    doomed = [path for _, path in _checkpoint_entries(log_dir)[:-keep_max]]
    if not doomed:
        return

    def _worker() -> None:
        for path in doomed:
            shutil.rmtree(path, ignore_errors=True)

    thread = threading.Thread(target=_worker, name="checkpoint-cleanup", daemon=True)
    _PENDING_DELETE = thread
    thread.start()


def _pack_rank_payload(accelerator, payload: dict, *, payload_name: str) -> dict | None:
    """Collect rank-local payloads onto the main process for checkpointing.

    Some training state is intentionally local to each rank, for example RNG
    state or data-loader shard progress. We therefore gather a per-rank payload
    and store it in the checkpoint as ``{world_size, per_rank}``.
    """
    local_payload = {
        "rank": int(accelerator.process_index),
        "payload": payload,
    }
    if dist.is_available() and dist.is_initialized():
        gathered: list[dict | None] = [None] * int(accelerator.num_processes)
        dist.all_gather_object(gathered, local_payload)
    else:
        gathered = [local_payload]

    if not accelerator.is_main_process:
        return None

    per_rank = {}
    for item in gathered:
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Failed to gather rank-scoped {payload_name} for checkpointing."
            )
        per_rank[str(int(item["rank"]))] = item["payload"]
    return {
        "world_size": len(gathered),
        "per_rank": per_rank,
    }


def _extract_rank_payload(
    accelerator, payload: dict | None, *, payload_name: str
) -> dict:
    """Recover the payload for the current rank from a packed checkpoint blob."""
    if payload is None:
        return {}

    expected_world_size = int(accelerator.num_processes)
    if int(payload["world_size"]) != expected_world_size:
        raise RuntimeError(
            f"Checkpoint {payload_name} payload does not match the current world."
        )

    local_rank = str(int(accelerator.process_index))
    if local_rank not in payload["per_rank"]:
        raise RuntimeError(f"Checkpoint {payload_name} is missing rank {local_rank}.")
    return payload["per_rank"][local_rank]


def save_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    log_dir: str,
    keep_max: int,
    data_state: dict,
    scheduler_state: dict,
    export_inference_model: bool = False,
) -> None:
    """Save a full resumable training checkpoint.

    Stored artifacts include:
    - model weights in ``save_pretrained`` format
    - optimizer / scheduler / scaler state
    - training progress counters
    - rank-local RNG state
    - rank-local data pipeline state
    """
    if accelerator.is_main_process:
        # Never let a save overlap the previous save's delete.
        _await_pending_deletes()
    if _uses_deepspeed(accelerator):
        _save_deepspeed_train_checkpoint(
            accelerator,
            model,
            progress,
            log_dir,
            keep_max,
            data_state,
            scheduler_state,
            export_inference_model=export_inference_model,
        )
        return

    accelerator.wait_for_everyone()
    packed_data_state = _pack_rank_payload(
        accelerator,
        data_state,
        payload_name="data_state",
    )
    packed_rng_state = _pack_rank_payload(
        accelerator,
        _rng_state(),
        payload_name="rng_state",
    )

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_dir = _checkpoint_dir(log_dir, progress.global_step)
        tmp_dir = save_dir.with_name(f"{save_dir.name}.tmp")
        model_dir = tmp_dir / "model"
        scaler = getattr(accelerator, "scaler", None)

        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Write into a temporary directory first so interrupted saves never
            # leave behind a half-written checkpoint that looks valid.
            unwrapped_model.save_pretrained(model_dir)

            torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
            torch.save(scheduler_state, tmp_dir / "scheduler.pt")
            torch.save(
                {} if scaler is None else scaler.state_dict(),
                tmp_dir / "scaler.pt",
            )
            torch.save(packed_rng_state, tmp_dir / "rng_state.pt")
            torch.save(packed_data_state, tmp_dir / "data_state.pt")
            (tmp_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        field.name: int(getattr(progress, field.name))
                        for field in fields(progress)
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if save_dir.exists():
                shutil.rmtree(save_dir)
            tmp_dir.rename(save_dir)
            _replace_latest_symlink(log_dir, save_dir)
            if export_inference_model:
                exported = _export_inference_model(
                    log_dir, save_dir, int(progress.global_step)
                )
                if exported is not None:
                    accelerator.print(f"Inference weights exported: {exported}")
            _cleanup_old_checkpoints(log_dir, keep_max)
            accelerator.print(f"Checkpoint saved: {save_dir}")
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    accelerator.wait_for_everyone()


def _save_deepspeed_train_checkpoint(
    accelerator,
    model,
    progress,
    log_dir: str,
    keep_max: int,
    data_state: dict,
    scheduler_state: dict,
    export_inference_model: bool = False,
) -> None:
    """Save a ZeRO-compatible checkpoint through ``Accelerator.save_state``.

    DeepSpeed optimizer state is partitioned across ranks and cannot be safely
    serialized with a rank-0 ``optimizer.state_dict()``. Accelerate delegates
    this call to ``DeepSpeedEngine.save_checkpoint`` on every rank, while the
    project-specific progress and streaming-data state remain alongside it.
    """
    accelerator.wait_for_everyone()
    packed_data_state = _pack_rank_payload(
        accelerator,
        data_state,
        payload_name="data_state",
    )

    save_dir = _checkpoint_dir(log_dir, progress.global_step)
    tmp_dir = save_dir.with_name(f"{save_dir.name}.tmp")
    if accelerator.is_main_process:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # All ranks must participate: DeepSpeed writes one optimizer shard per rank.
    accelerator.save_state(str(tmp_dir), safe_serialization=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        model_dir = tmp_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        # ZeRO-2 keeps model parameters replicated, so rank 0 can also emit the
        # normal save_pretrained artifact used by inference and export tools.
        accelerator.unwrap_model(model).save_pretrained(model_dir)
        torch.save(packed_data_state, tmp_dir / "data_state.pt")
        torch.save(scheduler_state, tmp_dir / "scheduler_meta.pt")
        (tmp_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    field.name: int(getattr(progress, field.name))
                    for field in fields(progress)
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        (tmp_dir / _CHECKPOINT_FORMAT_FILE).write_text(
            json.dumps(
                {
                    "version": 1,
                    "backend": _DEEPSPEED_BACKEND,
                    "zero_stage": 2,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        tmp_dir.rename(save_dir)
        _replace_latest_symlink(log_dir, save_dir)
        if export_inference_model:
            exported = _export_inference_model(
                log_dir, save_dir, int(progress.global_step)
            )
            if exported is not None:
                accelerator.print(f"Inference weights exported: {exported}")
        _cleanup_old_checkpoints(log_dir, keep_max)
        accelerator.print(f"DeepSpeed checkpoint saved: {save_dir}")
    accelerator.wait_for_everyone()


def load_train_checkpoint(
    accelerator,
    model,
    optimizer,
    progress,
    checkpoint_dir: str | Path,
    scheduler,
) -> dict:
    """Restore a checkpoint previously written by :func:`save_train_checkpoint`.

    Returns auxiliary state that the caller usually needs to resume the input
    pipeline and scheduler bookkeeping.
    """
    checkpoint_dir = Path(checkpoint_dir)
    model_dir = checkpoint_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint model directory not found: {model_dir!s}")

    backend = _checkpoint_backend(checkpoint_dir)
    using_deepspeed = _uses_deepspeed(accelerator)
    if using_deepspeed and backend != _DEEPSPEED_BACKEND:
        raise RuntimeError(
            "Cannot resume a legacy DDP checkpoint with DeepSpeed optimizer "
            "partitioning. Use a fresh OUTPUT_DIR for the first DeepSpeed run."
        )
    if not using_deepspeed and backend == _DEEPSPEED_BACKEND:
        raise RuntimeError(
            "This checkpoint contains DeepSpeed optimizer shards. Resume it with "
            "USE_DEEPSPEED=1 and the same ZeRO stage/world size."
        )

    if using_deepspeed:
        accelerator.wait_for_everyone()
        accelerator.load_state(str(checkpoint_dir))
        trainer_state = json.loads(
            (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
        )
        for field in fields(progress):
            setattr(progress, field.name, int(trainer_state[field.name]))
        data_state_payload = torch.load(
            checkpoint_dir / "data_state.pt",
            map_location="cpu",
        )
        scheduler_payload = torch.load(
            checkpoint_dir / "scheduler_meta.pt",
            map_location="cpu",
        )
        accelerator.wait_for_everyone()
        return {
            "checkpoint_dir": checkpoint_dir,
            "data_state": _extract_rank_payload(
                accelerator,
                data_state_payload,
                payload_name="data_state",
            ),
            "scheduler_state": scheduler_payload,
        }

    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_pretrained_weights(model_dir)

    optimizer.load_state_dict(
        torch.load(checkpoint_dir / "optimizer.pt", map_location="cpu")
    )

    scheduler_payload = torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu")
    scheduler.load_state_dict(scheduler_payload["state_dict"])

    scaler = getattr(accelerator, "scaler", None)
    scaler_state = torch.load(checkpoint_dir / "scaler.pt", map_location="cpu")
    if scaler is not None and scaler_state:
        scaler.load_state_dict(scaler_state)

    rng_state_payload = torch.load(checkpoint_dir / "rng_state.pt", map_location="cpu")
    _restore_rng_state(
        _extract_rank_payload(
            accelerator,
            rng_state_payload,
            payload_name="rng_state",
        )
    )
    data_state_payload = torch.load(
        checkpoint_dir / "data_state.pt", map_location="cpu"
    )

    trainer_state = json.loads(
        (checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8")
    )
    for field in fields(progress):
        setattr(progress, field.name, int(trainer_state[field.name]))

    accelerator.wait_for_everyone()
    return {
        "checkpoint_dir": checkpoint_dir,
        "data_state": _extract_rank_payload(
            accelerator,
            data_state_payload,
            payload_name="data_state",
        ),
        "scheduler_state": scheduler_payload,
    }
