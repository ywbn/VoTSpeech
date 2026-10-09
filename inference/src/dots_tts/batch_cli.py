from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DEFAULT_VARIANTS = ("APS", "DSD", "RP")


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _gpu_ids(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be comma-separated integers"
        ) from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("GPU ids must be non-negative integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU ids must not contain duplicates")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU dots.tts voice-design batch inference. Each GPU owns one "
            "runtime process; no DDP or NCCL process group is used."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--results-manifest",
        default=None,
        help="Default: <output-dir>/results.jsonl",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpu-ids",
        type=_gpu_ids,
        default=None,
        help="Optional logical CUDA ids, e.g. 0,2,3. Overrides --num-gpus.",
    )
    parser.add_argument(
        "--variants",
        type=_csv_values,
        default=DEFAULT_VARIANTS,
        help=(
            "JSONL nested objects or Parquet string columns containing instructions. "
            "Default: APS,DSD,RP. A top-level JSONL instruction is also accepted."
        ),
    )
    parser.add_argument("--template-name", default="voice_design")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--voice-num-steps", type=int, default=None,
                        help="Voice prior steps; defaults to the checkpoint config.")
    parser.add_argument("--voice-guidance-scale", type=float, default=None,
                        help="Voice prior CFG; defaults to the checkpoint config.")
    parser.add_argument("--ode-method", default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-generate-length", type=int, default=500)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--model-load-timeout-seconds",
        type=float,
        default=900.0,
        help="Per-GPU model initialization timeout. GPUs are loaded sequentially.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize-text", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip non-empty output wav files. Enabled by default.",
    )
    parser.add_argument(
        "--worker-log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    parser.add_argument(
        "--output-subtype",
        default="PCM_16",
        help="libsndfile WAV subtype, e.g. PCM_16 or FLOAT.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit expanded requests for a smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize requests without loading a model or writing files.",
    )
    return parser.parse_args(argv)


def _clean_component(value: object, *, fallback: str) -> str:
    normalized = str(value).strip()
    normalized = re.sub(r"[^\w.-]+", "_", normalized, flags=re.UNICODE).strip("._")
    return normalized or fallback


def _input_format(input_path: Path) -> str:
    return "parquet" if input_path.suffix.lower() in {".parquet", ".pq"} else "jsonl"


def _iter_manifest_rows(
    input_path: Path, variants: tuple[str, ...]
) -> Iterator[tuple[int, dict[str, Any]]]:
    if _input_format(input_path) == "jsonl":
        with input_path.open("r", encoding="utf-8") as manifest_file:
            for line_number, raw_line in enumerate(manifest_file, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {input_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected an object at {input_path}:{line_number}, got "
                        f"{type(row).__name__}."
                    )
                yield line_number, row
        return

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pyarrow. Install it with: "
            "python -m pip install 'pyarrow>=18.0.0'"
        ) from exc

    parquet_file = pq.ParquetFile(input_path)
    columns = ["id", "text", *variants]
    available_columns = set(parquet_file.schema_arrow.names)
    missing_columns = [column for column in columns if column not in available_columns]
    if missing_columns:
        raise ValueError(
            f"Parquet input {input_path} is missing columns {missing_columns}; "
            f"available columns: {sorted(available_columns)}."
        )

    # Column projection is intentional: InstructTTSEval embeds reference audio bytes,
    # which are not needed for text-to-speech inference and can be very large.
    row_number = 0
    for batch in parquet_file.iter_batches(
        batch_size=256,
        columns=columns,
        use_threads=True,
    ):
        for row in batch.to_pylist():
            row_number += 1
            yield row_number, row


def _variant_request_id(row_id: str, variant: str) -> str:
    suffix = f"_{variant}"
    if row_id.casefold().endswith(suffix.casefold()):
        return row_id
    return f"{row_id}{suffix}"


def _extract_requests(args) -> list[dict[str, Any]]:
    input_path = Path(args.input_manifest).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    requests: list[dict[str, Any]] = []
    used_outputs: set[str] = set()

    for line_number, row in _iter_manifest_rows(input_path, args.variants):
        text_value = row.get("text")
        text = str(text_value).strip() if text_value is not None else ""
        if not text:
            raise ValueError(f"Missing text at {input_path}:{line_number}.")
        row_id = _clean_component(
            row.get("id", row.get("uid", line_number)),
            fallback=f"line-{line_number:08d}",
        )

        candidates: list[tuple[str, str, str | None, bool]] = []
        top_instruction = str(row.get("instruction", "")).strip()
        if top_instruction:
            variant = _clean_component(
                row.get("variant", "default"), fallback="default"
            )
            candidates.append((variant, top_instruction, None, False))
        else:
            for variant_name in args.variants:
                payload = row.get(variant_name)
                reference_gen_path = None
                append_variant_suffix = False
                if isinstance(payload, dict):
                    instruction = str(payload.get("instruction", "")).strip()
                    reference_gen_path = (
                        str(payload.get("gen_path", "")).strip() or None
                    )
                elif isinstance(payload, str):
                    instruction = payload.strip()
                    append_variant_suffix = True
                else:
                    continue
                if instruction:
                    candidates.append(
                        (
                            _clean_component(variant_name, fallback="default"),
                            instruction,
                            reference_gen_path,
                            append_variant_suffix,
                        )
                    )
        if not candidates:
            raise ValueError(
                f"Missing instruction at {input_path}:{line_number}; checked top-level "
                f"instruction and variants {list(args.variants)}."
            )

        for (
            variant,
            instruction,
            reference_gen_path,
            append_variant_suffix,
        ) in candidates:
            request_id = (
                _variant_request_id(row_id, variant)
                if append_variant_suffix
                else row_id
            )
            output_path = output_dir / variant / f"{request_id}.wav"
            output_key = str(output_path)
            if output_key in used_outputs:
                output_path = (
                    output_dir / variant / f"{request_id}__line-{line_number:08d}.wav"
                )
                output_key = str(output_path)
            if output_key in used_outputs:
                raise ValueError(
                    f"Duplicate output path after disambiguation: {output_path}"
                )
            used_outputs.add(output_key)

            request_index = len(requests)
            requests.append(
                {
                    "request_index": request_index,
                    "source_line": line_number,
                    "id": request_id,
                    "variant": variant,
                    "text": text,
                    "instruction": instruction,
                    "reference_gen_path": reference_gen_path,
                    "output_path": str(output_path),
                    "seed": int(args.seed) + request_index,
                }
            )

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        requests = requests[: args.max_samples]
    if not requests:
        raise ValueError(f"Input manifest produced no inference requests: {input_path}")
    return requests


def _worker_main(
    worker_id: int,
    gpu_id: int,
    args,
    task_queue,
    result_queue,
) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["DOTS_TTS_LOG_LEVEL"] = str(args.worker_log_level)

    try:
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)

        import soundfile as sf
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the worker process.")
        if gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Logical GPU id {gpu_id} is unavailable; visible device count is "
                f"{torch.cuda.device_count()}."
            )
        torch.cuda.set_device(gpu_id)

        from dots_tts.runtime import DotsTtsRuntime
        from dots_tts.utils.logging import configure_logging
        from dots_tts.utils.util import seed_everything

        configure_logging(level=args.worker_log_level)
        runtime = DotsTtsRuntime.from_pretrained(
            args.model_name_or_path,
            precision=args.precision,
            max_generate_length=args.max_generate_length,
            max_sequence_length=args.max_sequence_length,
        )
        result_queue.put({"kind": "ready", "worker_id": worker_id, "gpu_id": gpu_id})
    except Exception as exc:
        result_queue.put(
            {
                "kind": "worker_error",
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        task = task_queue.get()
        if task is None:
            break

        output_path = Path(task["output_path"])
        temporary_path = output_path.with_name(
            f".{output_path.stem}.worker-{worker_id}.pid-{os.getpid()}.tmp.wav"
        )
        started = time.perf_counter()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            seed_everything(int(task["seed"]))
            generated = runtime.generate(
                text=task["text"],
                language=args.language,
                template_name=args.template_name,
                ode_method=args.ode_method,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                speaker_scale=args.speaker_scale,
                normalize_text=args.normalize_text,
                instruction=task["instruction"],
                voice_num_steps=args.voice_num_steps,
                voice_guidance_scale=args.voice_guidance_scale,
            )
            audio = generated["audio"].float().cpu().squeeze().numpy()
            sf.write(
                str(temporary_path),
                audio,
                int(generated["sample_rate"]),
                format="WAV",
                subtype=args.output_subtype,
            )
            os.replace(temporary_path, output_path)
            samples = int(generated["audio"].shape[-1])
            sample_rate = int(generated["sample_rate"])
            result_queue.put(
                {
                    "kind": "result",
                    **task,
                    "status": "generated",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "request_id": generated["fid"],
                    "sample_rate": sample_rate,
                    "samples": samples,
                    "audio_seconds": samples / sample_rate,
                    "elapsed_seconds": float(generated["time_used"]),
                    "rtf": float(generated["rtf"]),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        except Exception as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            result_queue.put(
                {
                    "kind": "result",
                    **task,
                    "status": "error",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "wall_seconds": time.perf_counter() - started,
                }
            )

    result_queue.put({"kind": "done", "worker_id": worker_id, "gpu_id": gpu_id})


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "kind"}


def _write_results(results_path: Path, results: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = results_path.with_name(f".{results_path.name}.tmp")
    ordered = sorted(results, key=lambda item: int(item["request_index"]))
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for result in ordered:
            output_file.write(json.dumps(_public_result(result), ensure_ascii=False))
            output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, results_path)


def _select_gpu_ids(args, request_count: int) -> tuple[int, ...]:
    if args.gpu_ids is not None:
        selected = tuple(args.gpu_ids)
    else:
        if args.num_gpus <= 0:
            raise ValueError("--num-gpus must be positive.")
        selected = tuple(range(args.num_gpus))
    return selected[: min(len(selected), request_count)]


def _terminate_workers(workers) -> None:
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=10)
    for worker in workers:
        if worker.is_alive():
            worker.kill()
    for worker in workers:
        worker.join(timeout=10)


def _start_workers_sequentially(
    *,
    context,
    gpu_ids: tuple[int, ...],
    args,
    task_queue,
    result_queue,
    tqdm,
):
    if args.model_load_timeout_seconds <= 0:
        raise ValueError("--model-load-timeout-seconds must be positive.")

    workers = []
    try:
        with tqdm(
            total=len(gpu_ids),
            desc="loading models",
            unit="GPU",
            dynamic_ncols=True,
        ) as progress:
            for worker_id, gpu_id in enumerate(gpu_ids):
                worker = context.Process(
                    target=_worker_main,
                    args=(worker_id, gpu_id, args, task_queue, result_queue),
                    name=f"dots-tts-gpu-{gpu_id}",
                )
                worker.start()
                workers.append(worker)
                progress.set_postfix(gpu=f"cuda:{gpu_id}", refresh=True)

                deadline = time.monotonic() + float(args.model_load_timeout_seconds)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Timed out loading the model on cuda:{gpu_id} after "
                            f"{args.model_load_timeout_seconds:.0f} seconds."
                        )
                    try:
                        message = result_queue.get(timeout=min(2.0, remaining))
                    except queue.Empty:
                        if not worker.is_alive():
                            raise RuntimeError(
                                f"GPU worker {worker_id} on cuda:{gpu_id} exited "
                                f"during model initialization with code {worker.exitcode}."
                            )
                        continue

                    kind = message.get("kind")
                    if kind == "ready":
                        if int(message["worker_id"]) != worker_id:
                            raise RuntimeError(
                                f"Unexpected ready message while loading cuda:{gpu_id}: "
                                f"{message!r}"
                            )
                        progress.update(1)
                        break
                    if kind == "worker_error":
                        raise RuntimeError(
                            f"GPU worker {message['worker_id']} failed on "
                            f"cuda:{message['gpu_id']}: {message['error']}\n"
                            f"{message['traceback']}"
                        )
                    raise RuntimeError(
                        f"Unexpected worker message during model loading: {message!r}"
                    )
    except BaseException:
        _terminate_workers(workers)
        raise
    return workers


def main(argv=None) -> int:
    args = parse_args(argv)
    requests = _extract_requests(args)
    gpu_ids = _select_gpu_ids(args, len(requests))
    results_path = (
        Path(args.results_manifest).expanduser().resolve()
        if args.results_manifest
        else Path(args.output_dir).expanduser().resolve() / "results.jsonl"
    )

    variant_counts: dict[str, int] = {}
    for request in requests:
        variant = str(request["variant"])
        variant_counts[variant] = variant_counts.get(variant, 0) + 1
    summary = {
        "input_manifest": str(Path(args.input_manifest).expanduser().resolve()),
        "input_format": _input_format(Path(args.input_manifest)),
        "model": str(args.model_name_or_path),
        "requests": len(requests),
        "variants": variant_counts,
        "gpu_ids": list(gpu_ids),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "results_manifest": str(results_path),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    ready_results: list[dict[str, Any]] = []
    pending_requests: list[dict[str, Any]] = []
    for request in requests:
        output_path = Path(request["output_path"])
        if args.resume and output_path.is_file() and output_path.stat().st_size > 44:
            ready_results.append(
                {
                    "kind": "result",
                    **request,
                    "status": "skipped_existing",
                }
            )
        else:
            pending_requests.append(request)

    if not pending_requests:
        _write_results(results_path, ready_results)
        print(
            f"Batch inference complete: generated=0 skipped={len(ready_results)} "
            f"errors=0 results={results_path}",
            flush=True,
        )
        return 0

    gpu_ids = gpu_ids[: min(len(gpu_ids), len(pending_requests))]
    if not gpu_ids:
        raise RuntimeError("No GPU workers were selected.")

    from tqdm.auto import tqdm

    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    workers = _start_workers_sequentially(
        context=context,
        gpu_ids=gpu_ids,
        args=args,
        task_queue=task_queue,
        result_queue=result_queue,
        tqdm=tqdm,
    )
    for request in pending_requests:
        task_queue.put(request)
    for _ in workers:
        task_queue.put(None)

    all_results = list(ready_results)
    completed = 0
    ready_workers = len(workers)
    generated = 0
    errors = 0
    partial_path = results_path.with_name(f".{results_path.name}.partial")
    partial_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with (
            partial_path.open("w", encoding="utf-8") as partial_file,
            tqdm(
                total=len(requests),
                initial=len(ready_results),
                desc="batch inference",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            for result in ready_results:
                partial_file.write(
                    json.dumps(_public_result(result), ensure_ascii=False) + "\n"
                )
            partial_file.flush()

            while completed < len(pending_requests):
                try:
                    message = result_queue.get(timeout=2.0)
                except queue.Empty:
                    bad_workers = [
                        worker
                        for worker in workers
                        if not worker.is_alive() and worker.exitcode not in (None, 0)
                    ]
                    if bad_workers:
                        names = ", ".join(
                            f"{worker.name}(exit={worker.exitcode})"
                            for worker in bad_workers
                        )
                        raise RuntimeError(
                            f"Inference workers exited unexpectedly: {names}"
                        )
                    continue

                kind = message.get("kind")
                if kind == "worker_error":
                    raise RuntimeError(
                        f"GPU worker {message['worker_id']} failed on cuda:{message['gpu_id']}: "
                        f"{message['error']}\n{message['traceback']}"
                    )
                if kind == "done":
                    continue
                if kind != "result":
                    raise RuntimeError(f"Unknown worker message: {message!r}")

                completed += 1
                generated += int(message.get("status") == "generated")
                errors += int(message.get("status") == "error")
                all_results.append(message)
                partial_file.write(
                    json.dumps(_public_result(message), ensure_ascii=False) + "\n"
                )
                partial_file.flush()
                progress.update(1)
                progress.set_postfix(
                    workers=f"{ready_workers}/{len(workers)}",
                    generated=generated,
                    errors=errors,
                    refresh=False,
                )
    except BaseException:
        _terminate_workers(workers)
        raise
    finally:
        for worker in workers:
            worker.join(timeout=30)
        task_queue.close()
        result_queue.close()

    stuck_workers = [worker.name for worker in workers if worker.is_alive()]
    if stuck_workers:
        _terminate_workers(workers)
        raise RuntimeError(f"Inference workers did not exit: {stuck_workers}")
    bad_exit_codes = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if bad_exit_codes:
        raise RuntimeError(f"Inference worker exit codes: {bad_exit_codes}")

    _write_results(results_path, all_results)
    if partial_path.exists():
        partial_path.unlink()
    skipped = sum(result.get("status") == "skipped_existing" for result in all_results)
    print(
        f"Batch inference complete: generated={generated} skipped={skipped} "
        f"errors={errors} results={results_path}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
