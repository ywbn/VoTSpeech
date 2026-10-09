"""Segment helpers for VoiceCrafter-schema manifests.

Ported field-for-field from ``voicecrafter_v4/data/segments.py`` so a manifest
segmented by either project has identical bounds: same ``start_sec``/``end_sec``
rounding, same ``<source>_seg%06d`` id scheme, same tail-merge rule. Anything
else would make the two pipelines disagree about what "the same segment" means.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

DURATION_FIELDS = (
    "duration",
    "duration_sec",
    "audio_duration",
    "audio_duration_sec",
    "seconds",
)


def first_float(row: dict[str, Any], fields) -> float | None:
    for field in fields:
        value = row.get(field)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def probe_audio_duration(audio_path: str) -> float | None:
    """Best-effort duration without decoding the whole file."""

    path = Path(audio_path)
    if not path.exists():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:  # noqa: BLE001 - fall through to the next probe
        pass

    try:
        if path.suffix.lower() == ".wav":
            import wave

            with wave.open(str(path), "rb") as handle:
                sample_rate = handle.getframerate()
                if sample_rate > 0:
                    return float(handle.getnframes()) / float(sample_rate)
    except Exception:  # noqa: BLE001
        pass

    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:  # noqa: BLE001
        return None


def resolve_duration(row: dict[str, Any], *, probe_file: bool = True) -> float | None:
    duration = first_float(row, DURATION_FIELDS)
    if duration is not None:
        return max(0.0, duration)

    start_sec = first_float(row, ["start_sec", "start"])
    end_sec = first_float(row, ["end_sec", "end"])
    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        return end_sec - start_sec

    audio_path = row.get("audio_path") or row.get("audio")
    if probe_file and isinstance(audio_path, str) and audio_path:
        return probe_audio_duration(audio_path)
    return None


def segment_record(
    row: dict[str, Any],
    *,
    segment_seconds: float = 15.0,
    overlap_seconds: float = 0.0,
    min_segment_seconds: float = 1.0,
    probe_file_duration: bool = True,
) -> list[dict[str, Any]]:
    """Split one row into fixed-length segments when duration is known.

    Existing ``start_sec``/``end_sec`` bound the segmentable range. A row whose
    duration cannot be resolved is returned unchanged rather than guessed at.
    """

    max_seconds = float(segment_seconds)
    if max_seconds <= 0:
        return [row]

    existing_start = first_float(row, ["start_sec", "start"])
    existing_end = first_float(row, ["end_sec", "end"])
    duration = resolve_duration(row, probe_file=probe_file_duration)
    if duration is None:
        return [row]

    base_start = max(0.0, existing_start or 0.0)
    base_end = existing_end if existing_end is not None else base_start + duration
    base_end = max(base_start, float(base_end))
    total = base_end - base_start
    if total <= 0:
        return [row]

    step = max(1e-6, max_seconds - max(0.0, float(overlap_seconds)))
    segments: list[dict[str, Any]] = []
    current = base_start
    index = 0
    while current < base_end:
        end = min(current + max_seconds, base_end)
        # A runt tail is merged into the previous segment instead of becoming a
        # sub-minimum sample of its own.
        if end - current < float(min_segment_seconds) and segments:
            segments[-1]["end_sec"] = round(base_end, 6)
            segments[-1]["duration"] = round(
                base_end - float(segments[-1]["start_sec"]), 6
            )
            break

        item = dict(row)
        source_audio_id = str(
            row.get("source_audio_id") or row.get("audio_id") or "audio"
        )
        item["source_audio_id"] = source_audio_id
        item["segment_index"] = index
        item["start_sec"] = round(current, 6)
        item["end_sec"] = round(end, 6)
        item["duration"] = round(end - current, 6)
        item["audio_id"] = f"{source_audio_id}_seg{index:06d}"
        segments.append(item)

        index += 1
        if math.isclose(end, base_end) or end >= base_end:
            break
        current += step

    return segments or [row]


__all__ = [
    "DURATION_FIELDS",
    "first_float",
    "probe_audio_duration",
    "resolve_duration",
    "segment_record",
]
