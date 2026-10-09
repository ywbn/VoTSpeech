from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from dots_tts.data.source_adapters.base_adapter import (
    BaseSourceAdapter,
    ShardableSourceAdapter,
    SourceContext,
)


class JsonlManifestSourceAdapter(ShardableSourceAdapter, BaseSourceAdapter):
    """Finite adapter for line-delimited JSON manifests."""

    def __init__(
        self,
        *,
        manifest_path: str,
        fid_key: str = "fid",
        text_key: str = "text",
        audio_key: str = "audio",
        shuffle: bool = False,
        streaming: bool = False,
        shuffle_buffer_size: int = 4096,
        encoding: str = "utf-8",
    ):
        self.manifest_path = Path(manifest_path)
        self.fid_key = fid_key
        self.text_key = text_key
        self.audio_key = audio_key
        self.shuffle = shuffle
        self.streaming = bool(streaming)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1.")
        if self.streaming and encoding.lower().replace("_", "-") not in {
            "utf-8",
            "utf8",
        }:
            raise ValueError("streaming JSONL reads currently require UTF-8 encoding.")
        self.encoding = encoding
        self._records: list[dict[str, Any]] | None = None

    def initial_state(self) -> dict[str, Any]:
        return {"cycle": 0, "cursor": 0}

    def is_cycle_start_state(self, state: dict[str, Any] | None) -> bool:
        normalized = self.normalize_state(state)
        return int(normalized["cursor"]) == 0

    def advance_cycle(self, state: dict[str, Any] | None) -> dict[str, Any]:
        normalized = self.normalize_state(state)
        return {"cycle": int(normalized["cycle"]) + 1, "cursor": 0}

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path!s}")
        with self.manifest_path.open("r", encoding=self.encoding) as fin:
            for line_no, raw_line in enumerate(fin, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {self.manifest_path}:{line_no}"
                    ) from exc

    def _base_records(self) -> list[dict[str, Any]]:
        if self._records is None:
            self._records = list(self._iter_records())
        return self._records

    def _stream_shard_records(
        self,
        context: SourceContext,
        *,
        cycle: int,
    ) -> Iterator[dict[str, Any]]:
        """Read only this worker's contiguous byte range with bounded memory.

        Loading a multi-million-row JSONL into a Python list in every DataLoader
        worker multiplies both memory and startup I/O by world_size*num_workers.
        Byte-range sharding lets all workers collectively scan the file once per
        cycle.  When shuffle is enabled, workers are assigned a deterministic
        permutation of the ranges and each range uses a bounded shuffle buffer.
        """

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path!s}")

        worker_count = int(context.global_worker_count)
        worker_id = int(context.global_worker_id)
        shard_ids = list(range(worker_count))
        if self.shuffle:
            random.Random(context.seed + context.epoch + 1009 * int(cycle)).shuffle(
                shard_ids
            )
        shard_id = shard_ids[worker_id]
        file_size = int(self.manifest_path.stat().st_size)
        start = file_size * shard_id // worker_count
        end = file_size * (shard_id + 1) // worker_count

        def iter_range() -> Iterator[dict[str, Any]]:
            with self.manifest_path.open("rb") as fin:
                if start:
                    fin.seek(start - 1)
                    if fin.read(1) != b"\n":
                        fin.seek(start)
                        fin.readline()
                    else:
                        fin.seek(start)

                while True:
                    byte_offset = fin.tell()
                    if byte_offset >= end:
                        return
                    raw_line = fin.readline()
                    if not raw_line:
                        return
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line.decode(self.encoding))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError(
                            f"Invalid JSON at {self.manifest_path}:byte {byte_offset}"
                        ) from exc

        records = iter_range()
        if not self.shuffle or self.shuffle_buffer_size == 1:
            yield from records
            return

        rng = random.Random(
            context.seed
            + context.epoch
            + 1009 * int(cycle)
            + 9176 * int(context.global_worker_id)
        )
        buffer: list[dict[str, Any]] = []
        for record in records:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(record)
                continue
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = record
        rng.shuffle(buffer)
        yield from buffer

    def _build_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        missing = [
            key
            for key in (self.fid_key, self.text_key, self.audio_key)
            if key not in record
        ]
        if missing:
            raise KeyError(
                f"Manifest record is missing required keys {missing}: {record}"
            )

        sample = {
            "fid": str(record[self.fid_key]),
            "text": record[self.text_key],
            "audio": record[self.audio_key],
        }
        for key, value in record.items():
            if key in {self.fid_key, self.text_key, self.audio_key}:
                continue
            sample[key] = value
        return sample

    def _indices_for_cycle(
        self,
        context: SourceContext,
        *,
        cycle: int,
    ) -> list[int]:
        indices = list(range(len(self._base_records())))
        if self.shuffle:
            random.Random(context.seed + context.epoch + 1009 * int(cycle)).shuffle(
                indices
            )
            indices = [
                record_index
                for shuffled_index, record_index in enumerate(indices)
                if self.is_assigned_index(shuffled_index, context)
            ]
        else:
            indices = [
                record_index
                for record_index in indices
                if self.is_assigned_index(record_index, context)
            ]
        return indices

    def iter_samples(
        self,
        context: SourceContext,
        *,
        state: dict[str, Any] | None = None,
    ) -> Iterable[dict[str, Any]]:
        live_state = self.normalize_state(state)
        cycle = int(live_state["cycle"])
        cursor = int(live_state["cursor"])

        if self.streaming:
            records = self._stream_shard_records(context, cycle=cycle)
            for position, record in enumerate(records):
                if position < cursor:
                    continue
                sample = self._build_sample(record)
                sample["_adapter_state"] = {
                    "cycle": cycle,
                    "cursor": position + 1,
                }
                yield sample
            return

        records = self._base_records()
        indices = self._indices_for_cycle(context, cycle=cycle)

        for position in range(cursor, len(indices)):
            sample = self._build_sample(records[indices[position]])
            sample["_adapter_state"] = {
                "cycle": cycle,
                "cursor": position + 1,
            }
            yield sample
