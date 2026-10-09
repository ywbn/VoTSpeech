"""Load one training waveform from a VoiceCrafter-style manifest row.

VoiceCrafter manifests address audio three ways — a filesystem path, a row
inside a Parquet shard, or an index into a Hugging Face dataset — and may carry
``start_sec``/``end_sec`` bounds when a long recording was split into segments.
The stock dots.tts pipeline only understands filesystem paths, so this module is
what makes an existing VoiceCrafter manifest usable here unchanged.

Every handle cache below is process-local by design: these run inside DataLoader
workers, and a Parquet reader or a memory-mapped dataset cannot be shared across
processes. Caching matters a lot for the Parquet backend — a manifest with one
row per utterance would otherwise reopen and re-parse the same shard footer for
every sample in it.
"""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# torch is imported lazily, inside the decode helpers. Manifest preparation
# imports this module only for its schema constants and pure helpers, and that
# is a CPU metadata job that has no business dragging in a deep-learning stack.

_MAX_CACHED_PARQUET_FILES = 8
_MAX_CACHED_DATASETS = 4
_MAX_CACHED_LANCE_DATASETS = 24
_MAX_CACHED_TAR_HANDLES = 16

_parquet_cache: OrderedDict[str, Any] = OrderedDict()
_parquet_offsets_cache: dict[str, list[int]] = {}
_dataset_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
_lance_cache: OrderedDict[str, Any] = OrderedDict()
_tar_handle_cache: OrderedDict[str, Any] = OrderedDict()
_tar_member_index_cache: dict[str, dict[str, tuple[int, int]]] = {}
_tar_dir_index_cache: dict[str, dict[str, tuple[str, int, int]]] = {}

# Column names a Lance row may hold audio bytes under, in the order
# VoiceCrafter tries them (`--audio-columns`). Kept identical so a manifest
# written by either project resolves to the same column.
DEFAULT_LANCE_AUDIO_COLUMNS = (
    "bytes",
    "audio_binary_file",
    "audio_binary",
    "audio",
    "wav",
)


class AudioBackendError(RuntimeError):
    """Raised when a manifest row cannot be turned into a waveform."""


# region helpers
def _as_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_mono_tensor(array) -> "torch.Tensor":
    import torch

    waveform = torch.as_tensor(array, dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2:
        # soundfile returns [frames, channels]; the pipeline wants [channels, frames].
        if waveform.size(0) > waveform.size(1):
            waveform = waveform.transpose(0, 1)
    else:
        raise AudioBackendError(f"Unsupported audio array shape {tuple(waveform.shape)}.")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous()


def segment_frame_range(
    *,
    sample_rate: int,
    total_frames: int | None,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[int, int | None]:
    """Convert segment bounds in seconds to a ``(start, stop)`` frame range.

    Kept as a pure function so the arithmetic that decides which samples a
    segment covers can be tested without decoding audio. ``stop`` is None when
    the segment runs to the end of the file.
    """

    start = 0 if start_sec is None else max(0, int(math.floor(start_sec * sample_rate)))
    if end_sec is None:
        stop = None
    else:
        stop = int(math.ceil(end_sec * sample_rate))
        if total_frames is not None:
            stop = min(int(total_frames), stop)
    if stop is not None and stop <= start:
        raise AudioBackendError(
            f"Empty segment: start_sec={start_sec} end_sec={end_sec} "
            f"sample_rate={sample_rate} total_frames={total_frames}."
        )
    return start, stop


def _slice_segment(
    waveform: "torch.Tensor",
    sample_rate: int,
    start_sec: float | None,
    end_sec: float | None,
) -> "torch.Tensor":
    if start_sec is None and end_sec is None:
        return waveform
    total = waveform.size(-1)
    start, stop = segment_frame_range(
        sample_rate=sample_rate,
        total_frames=total,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    return waveform[..., start : (total if stop is None else stop)].contiguous()


def _cache_put(cache: OrderedDict, key, value, limit: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)
    return value
# endregion helpers


# region file backend
def _load_file(row: dict, start_sec: float | None, end_sec: float | None):
    import io

    import soundfile as sf

    audio_path = row.get("audio_path") or row.get("audio")
    if not isinstance(audio_path, str) or not audio_path:
        raise AudioBackendError(f"Row has no usable audio path: {row.get('audio_id')}")

    def _decode(target):
        # Seek to the segment instead of decoding the whole file and throwing
        # most of it away: VoiceCrafter splits long recordings into 15 s
        # segments, so a naive read would decode the same hour-long file once
        # per segment.
        start_frame = 0
        stop_frame = None
        if start_sec is not None or end_sec is not None:
            info = sf.info(target)
            if hasattr(target, "seek"):
                target.seek(0)
            start_frame, stop_frame = segment_frame_range(
                sample_rate=int(info.samplerate),
                total_frames=int(info.frames),
                start_sec=start_sec,
                end_sec=end_sec,
            )
        data, sample_rate = sf.read(
            target,
            dtype="float32",
            always_2d=True,
            start=start_frame,
            stop=stop_frame,
        )
        return _to_mono_tensor(data), int(sample_rate)

    try:
        return _decode(audio_path)
    except Exception as exc:  # noqa: BLE001 - retried, then re-raised with context
        # libsndfile reads a file by seeking around it. Object-store FUSE mounts
        # frequently cannot serve that access pattern and fail with an error
        # whose message will not even stringify, which is what
        # `<exception str() failed>` in a DataLoader worker looks like. Pulling
        # the file down in one sequential read and decoding from memory sidesteps
        # the mount entirely; on a real filesystem this path never runs.
        try:
            with open(audio_path, "rb") as handle:
                blob = handle.read()
        except OSError as read_error:
            raise AudioBackendError(
                f"Cannot open audio file: {audio_path}"
            ) from read_error
        try:
            return _decode(io.BytesIO(blob))
        except Exception as decode_error:  # noqa: BLE001
            raise AudioBackendError(
                f"Cannot decode audio file ({len(blob):,} bytes read): "
                f"{audio_path}"
            ) from decode_error
# endregion file backend


# region parquet backend
def _parquet_file(path: str):
    cached = _parquet_cache.get(path)
    if cached is not None:
        _parquet_cache.move_to_end(path)
        return cached
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise AudioBackendError(
            "audio_backend='parquet' needs pyarrow. Install with "
            "`pip install 'dots.tts[parquet]'` or `pip install pyarrow`."
        ) from error
    handle = pq.ParquetFile(path)
    offsets = [0]
    for group_index in range(handle.num_row_groups):
        offsets.append(offsets[-1] + handle.metadata.row_group(group_index).num_rows)
    _parquet_offsets_cache[path] = offsets
    return _cache_put(_parquet_cache, path, handle, _MAX_CACHED_PARQUET_FILES)


def _parquet_row(path: str, row_index: int, column: str):
    handle = _parquet_file(path)
    offsets = _parquet_offsets_cache[path]
    total = offsets[-1]
    if not 0 <= row_index < total:
        raise AudioBackendError(
            f"parquet_row {row_index} out of range for {path} ({total} rows)."
        )
    # Read only the row group holding this row, not the whole shard.
    group_index = 0
    for candidate in range(len(offsets) - 1):
        if offsets[candidate] <= row_index < offsets[candidate + 1]:
            group_index = candidate
            break
    table = handle.read_row_group(group_index, columns=[column])
    return table.column(column)[row_index - offsets[group_index]].as_py()


def _load_parquet(row: dict, start_sec: float | None, end_sec: float | None):
    path = row.get("parquet_path")
    row_index = row.get("parquet_row")
    column = row.get("audio_column") or "audio"
    if not path or row_index is None:
        raise AudioBackendError(
            f"parquet row needs parquet_path and parquet_row: {row.get('audio_id')}"
        )
    value = _parquet_row(str(path), int(row_index), str(column))
    waveform, sample_rate = _decode_audio_value(value)
    return _slice_segment(waveform, sample_rate, start_sec, end_sec), sample_rate
# endregion parquet backend


# region dataset backend
def _dataset(path: str, split: str):
    key = (path, split)
    cached = _dataset_cache.get(key)
    if cached is not None:
        _dataset_cache.move_to_end(key)
        return cached
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise AudioBackendError(
            "audio_backend='dataset' needs the `datasets` package."
        ) from error

    dataset_path = Path(path)
    if (dataset_path / "dataset_info.json").is_file() or (
        dataset_path / "dataset_dict.json"
    ).is_file():
        loaded = datasets.load_from_disk(str(dataset_path))
        if isinstance(loaded, datasets.DatasetDict):
            loaded = loaded[split]
    else:
        loaded = datasets.load_dataset(str(dataset_path), split=split)
    # Keep the raw bytes: letting `datasets` decode would resample and rebuild
    # arrays that this module decodes anyway, at meaningful CPU cost per sample.
    try:
        loaded = loaded.cast_column("audio", datasets.Audio(decode=False))
    except Exception:  # noqa: BLE001 - column may be absent or already raw
        pass
    return _cache_put(_dataset_cache, key, loaded, _MAX_CACHED_DATASETS)


def _load_dataset_row(row: dict, start_sec: float | None, end_sec: float | None):
    path = row.get("dataset_path")
    index = row.get("dataset_index")
    if not path or index is None:
        raise AudioBackendError(
            f"dataset row needs dataset_path and dataset_index: {row.get('audio_id')}"
        )
    split = str(row.get("dataset_split") or "train")
    column = str(row.get("audio_column") or "audio")
    dataset = _dataset(str(path), split)
    value = dataset[int(index)][column]
    waveform, sample_rate = _decode_audio_value(value)
    return _slice_segment(waveform, sample_rate, start_sec, end_sec), sample_rate
# endregion dataset backend


def _decode_audio_value(value: Any) -> tuple["torch.Tensor", int]:
    """Decode whatever a Parquet cell or dataset column hands back."""

    import soundfile as sf

    if isinstance(value, dict):
        if value.get("array") is not None and value.get("sampling_rate"):
            return _to_mono_tensor(value["array"]), int(value["sampling_rate"])
        payload = value.get("bytes")
        if payload:
            data, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
            return _to_mono_tensor(data), int(sample_rate)
        path = value.get("path")
        if path:
            data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
            return _to_mono_tensor(data), int(sample_rate)
        raise AudioBackendError(f"Audio dict has no array/bytes/path: {sorted(value)}")

    if isinstance(value, (bytes, bytearray)):
        data, sample_rate = sf.read(io.BytesIO(value), dtype="float32", always_2d=True)
        return _to_mono_tensor(data), int(sample_rate)

    if isinstance(value, str):
        data, sample_rate = sf.read(value, dtype="float32", always_2d=True)
        return _to_mono_tensor(data), int(sample_rate)

    raise AudioBackendError(f"Unsupported audio cell type: {type(value)!r}")


# region lance backend
def _lance_dataset(dburi: str):
    cached = _lance_cache.get(dburi)
    if cached is not None:
        _lance_cache.move_to_end(dburi)
        return cached
    try:
        import lance
    except ImportError as error:  # pragma: no cover - optional extra
        raise AudioBackendError(
            "audio_backend='lance' needs the `pylance` package: pip install pylance"
        ) from error
    return _cache_put(_lance_cache, dburi, lance.dataset(dburi), _MAX_CACHED_LANCE_DATASETS)


def _lance_audio_columns(row: dict) -> list[str]:
    declared = row.get("audio_columns") or row.get("audio_column")
    if isinstance(declared, str):
        declared = [part.strip() for part in declared.split(",") if part.strip()]
    columns = list(declared or ())
    for name in DEFAULT_LANCE_AUDIO_COLUMNS:
        if name not in columns:
            columns.append(name)
    return columns


def _lance_take(dburi: str, fragment_id: Any, row_id: int, columns: list[str]) -> dict:
    dataset = _lance_dataset(dburi)
    available = [name for name in columns if name in set(dataset.schema.names)]
    if not available:
        raise AudioBackendError(
            f"No audio column in {dburi}; tried={columns} "
            f"available={list(dataset.schema.names)}"
        )
    # A fragment id makes row_id fragment-local, which is how VoiceCrafter
    # records it; without one the id is a dataset-global row.
    if fragment_id is not None and int(fragment_id) >= 0:
        try:
            table = dataset.get_fragment(int(fragment_id)).take([int(row_id)], columns=available)
        except Exception:  # noqa: BLE001 - fall back to a global take
            table = dataset.take([int(row_id)], columns=available)
    else:
        table = dataset.take([int(row_id)], columns=available)
    records = table.to_pylist()
    if not records:
        raise AudioBackendError(
            f"Missing Lance row: dburi={dburi} fragment={fragment_id} row={row_id}"
        )
    return records[0]


def _load_lance(row: dict, start_sec: float | None, end_sec: float | None):
    columns = _lance_audio_columns(row)
    # VoiceCrafter stores lance addressing as a list, because one logical sample
    # can be several consecutive rows that concatenate into one utterance.
    sources = row.get("source_indices")
    if not sources:
        dburi = row.get("lance_dburi") or row.get("dburi")
        if not dburi or row.get("lance_row_id") is None:
            raise AudioBackendError(
                f"lance row needs source_indices or lance_dburi/lance_row_id: "
                f"{row.get('audio_id')}"
            )
        sources = [
            {
                "dburi": dburi,
                "fragment_id": row.get("lance_fragment_id"),
                "row_id": row["lance_row_id"],
            }
        ]

    waveforms = []
    sample_rate = None
    for source in sources:
        record = _lance_take(
            str(source["dburi"]),
            source.get("fragment_id"),
            int(source["row_id"]),
            columns,
        )
        payload = next(
            (record[name] for name in columns if record.get(name) not in (None, b"")),
            None,
        )
        if payload is None:
            raise AudioBackendError(
                f"Lance row has no audio in {columns}: {source}"
            )
        waveform, rate = _decode_audio_value(payload)
        if sample_rate is not None and rate != sample_rate:
            raise AudioBackendError(
                f"Concatenated Lance rows disagree on sample rate: {sample_rate} vs {rate}."
            )
        sample_rate = rate
        waveforms.append(waveform)

    if len(waveforms) == 1:
        combined = waveforms[0]
    else:
        import torch

        combined = torch.cat(waveforms, dim=-1)
    return _slice_segment(combined, int(sample_rate), start_sec, end_sec), int(sample_rate)
# endregion lance backend


# region tar backend
def _tar_fields(row: dict) -> dict:
    """Flatten VoiceCrafter's audio_locator into plain fields."""

    locator = row.get("audio_locator")
    locator = locator if isinstance(locator, dict) else {}

    def _pick(*names):
        for name in names:
            value = locator.get(name)
            if value not in (None, ""):
                return value
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    member = _pick("member", "tar_member")
    offset = _pick("offset", "tar_offset")
    size = _pick("size", "tar_size")
    return {
        "member": Path(str(member)).name if member else None,
        "tar_path": _pick("tar_path"),
        "tar_dir": _pick("tar_dir"),
        "offset": None if offset is None else int(offset),
        "size": None if size is None else int(size),
    }


def _resolve_tar_path(path: Any, tar_root: Any) -> str:
    candidate = Path(str(path)).expanduser()
    if not candidate.is_absolute() and tar_root:
        candidate = Path(str(tar_root)).expanduser() / candidate
    return str(candidate)


def _tar_handle(tar_path: str):
    """Keep tar files open across reads, closing what falls out of the cache.

    Shards hold thousands of members, so reopening per sample would dominate the
    read. The eviction has to close the handle explicitly — dropping it from the
    dict only defers the close to the garbage collector, and a worker that
    touches many shards would run out of file descriptors first.
    """

    cached = _tar_handle_cache.get(tar_path)
    if cached is not None:
        _tar_handle_cache.move_to_end(tar_path)
        return cached

    handle = open(tar_path, "rb")  # noqa: SIM115 - lifetime is the process cache
    _tar_handle_cache[tar_path] = handle
    _tar_handle_cache.move_to_end(tar_path)
    while len(_tar_handle_cache) > _MAX_CACHED_TAR_HANDLES:
        _, evicted = _tar_handle_cache.popitem(last=False)
        try:
            evicted.close()
        except Exception:  # noqa: BLE001 - a failed close must not fail the read
            pass
    return handle


def _tar_dir_index(tar_dir: str) -> dict[str, tuple[str, int, int]]:
    """Map member name -> (tar_path, offset, size) for every tar under a dir.

    Built once per directory per process. This is the slow path: a manifest that
    records tar_path/offset/size never reaches it, which is why the preparation
    script writes those through whenever the source provides them.
    """

    cached = _tar_dir_index_cache.get(tar_dir)
    if cached is not None:
        return cached
    import tarfile

    index: dict[str, tuple[str, int, int]] = {}
    for tar_path in sorted(Path(tar_dir).glob("*.tar")):
        with tarfile.open(tar_path, "r|") as archive:
            for member in archive:
                if member.isfile():
                    index.setdefault(
                        Path(member.name).name,
                        (str(tar_path), int(member.offset_data), int(member.size)),
                    )
    _tar_dir_index_cache[tar_dir] = index
    return index


def _tar_member_index(tar_path: str) -> dict[str, tuple[int, int]]:
    """Cache member payload ranges for one explicitly addressed tar shard.

    VoiceCrafter source rows often carry ``tar_path`` and ``tar_member`` but no
    byte range. Calling ``TarFile.getmembers()`` for every utterance rescans the
    same multi-gigabyte shard thousands of times. One process-local index turns
    all later reads from that shard into direct seeks, just like rows that came
    with offsets already populated.
    """

    cached = _tar_member_index_cache.get(tar_path)
    if cached is not None:
        return cached

    import tarfile

    index: dict[str, tuple[int, int]] = {}
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            if member.isfile():
                index.setdefault(
                    Path(member.name).name,
                    (int(member.offset_data), int(member.size)),
                )
    _tar_member_index_cache[tar_path] = index
    return index


def _load_tar(row: dict, start_sec: float | None, end_sec: float | None):
    fields = _tar_fields(row)
    if not fields["member"] and not fields["tar_path"]:
        raise AudioBackendError(
            f"tar row needs tar_member/audio_locator.member: {row.get('audio_id')}"
        )
    tar_root = row.get("tar_root")

    tar_path = fields["tar_path"]
    offset, size = fields["offset"], fields["size"]
    if tar_path and offset is None and fields["member"]:
        tar_path = _resolve_tar_path(tar_path, tar_root)
        entry = _tar_member_index(tar_path).get(fields["member"])
        if entry is None:
            raise AudioBackendError(
                f"tar member not found: member={fields['member']} tar={tar_path}"
            )
        offset, size = entry

    if not tar_path:
        if not fields["tar_dir"]:
            raise AudioBackendError(
                f"tar row needs tar_path or tar_dir: {row.get('audio_id')}"
            )
        entry = _tar_dir_index(_resolve_tar_path(fields["tar_dir"], tar_root)).get(
            fields["member"]
        )
        if entry is None:
            raise AudioBackendError(
                f"tar member not found: member={fields['member']} dir={fields['tar_dir']}"
            )
        tar_path, offset, size = entry
    else:
        tar_path = _resolve_tar_path(tar_path, tar_root)

    # Byte-range read: the member's payload is a contiguous slice of the tar, so
    # there is no reason to parse the archive to get at it.
    handle = _tar_handle(tar_path)
    handle.seek(int(offset))
    payload = handle.read(int(size))
    if len(payload) != int(size):
        raise AudioBackendError(
            f"Short read from {tar_path}: wanted {size} bytes, got {len(payload)}."
        )
    waveform, rate = _decode_audio_value(payload)
    return _slice_segment(waveform, rate, start_sec, end_sec), rate
# endregion tar backend


def resolve_audio_backend(row: dict) -> str:
    """Infer the backend when the manifest does not state one explicitly."""

    backend = row.get("audio_backend")
    if backend:
        return str(backend)
    if row.get("parquet_path") is not None and row.get("parquet_row") is not None:
        return "parquet"
    if row.get("dataset_path") is not None and row.get("dataset_index") is not None:
        return "dataset"
    if row.get("source_indices") or row.get("lance_dburi"):
        return "lance"
    locator = row.get("audio_locator")
    if (isinstance(locator, dict) and locator.get("member")) or row.get("tar_member"):
        return "tar"
    return "file"


def load_waveform(row: dict) -> tuple["torch.Tensor", int]:
    """Return ``([1, frames] float32, sample_rate)`` for one manifest row."""

    backend = resolve_audio_backend(row)
    start_sec = _as_float(row.get("start_sec"))
    end_sec = _as_float(row.get("end_sec"))

    if backend == "file":
        return _load_file(row, start_sec, end_sec)
    if backend == "parquet":
        return _load_parquet(row, start_sec, end_sec)
    if backend == "dataset":
        return _load_dataset_row(row, start_sec, end_sec)
    if backend in {"lance", "lancedb"}:
        return _load_lance(row, start_sec, end_sec)
    if backend == "tar":
        return _load_tar(row, start_sec, end_sec)
    raise AudioBackendError(
        f"Unknown audio_backend={backend!r} for {row.get('audio_id') or row.get('fid')}. "
        "Expected 'file', 'parquet', 'dataset', 'lance' or 'tar'."
    )


__all__ = [
    "DEFAULT_LANCE_AUDIO_COLUMNS",
    "AudioBackendError",
    "load_waveform",
    "resolve_audio_backend",
    "segment_frame_range",
]
