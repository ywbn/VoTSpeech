from __future__ import annotations

from pathlib import Path
from typing import Any

from dots_tts.data.source_adapters.jsonl_manifest_adapter import (
    JsonlManifestSourceAdapter,
)

# Fields that carry addressing or segment information through to
# dots_tts.data.audio_backends. They are copied verbatim rather than
# interpreted here: the adapter's job is naming, the backend's job is loading.
_PASSTHROUGH_FIELDS = (
    "audio_backend",
    "audio_path",
    "audio_column",
    "parquet_path",
    "parquet_row",
    "dataset_path",
    "dataset_split",
    "dataset_index",
    # Lance / tar addressing, emitted by voicecrafter_v3-style preparation.
    "source_indices",
    "lance_dburi",
    "lance_fragment_id",
    "lance_row_id",
    "audio_columns",
    "audio_locator",
    "tar_path",
    "tar_dir",
    "tar_member",
    "tar_offset",
    "tar_size",
    "tar_partition",
    "tar_root",
    "uid",
    "sample_id",
    "start_sec",
    "end_sec",
    "duration",
    "segment_index",
    "source_audio_id",
    "speaker_id",
    "speaker",
    "language",
    "language_tag",
    "semantic_label",
)


def _reference_label(backend: str, record: dict[str, Any]) -> str:
    """A human-readable stand-in for the `audio` key on non-file backends.

    The pipeline contract requires an `audio` entry and it is what shows up in
    error messages, but for these backends it is never opened — the addressing
    fields are.
    """

    if backend == "parquet":
        return f"parquet:{record.get('parquet_path')}#{record.get('parquet_row')}"
    if backend == "dataset":
        return f"dataset:{record.get('dataset_path')}#{record.get('dataset_index')}"
    if backend == "lance":
        sources = record.get("source_indices") or []
        first = sources[0] if sources else {}
        dburi = first.get("dburi") or record.get("lance_dburi")
        row_id = first.get("row_id", record.get("lance_row_id"))
        return f"lance:{dburi}#{row_id}"
    if backend == "tar":
        locator = record.get("audio_locator")
        locator = locator if isinstance(locator, dict) else {}
        member = locator.get("member") or record.get("tar_member")
        container = locator.get("tar_path") or record.get("tar_path") or record.get("tar_dir")
        return f"tar:{container}#{member}"
    return f"{backend}:unknown"


class VoiceCrafterManifestSourceAdapter(JsonlManifestSourceAdapter):
    """Read a VoiceCrafter-schema JSONL manifest.

    VoiceCrafter names things differently from dots.tts — ``audio_id`` instead of
    ``fid``, ``audio_path`` instead of ``audio`` — and addresses audio three ways
    (``file`` / ``parquet`` / ``dataset``). This adapter renames the required
    fields and passes the addressing fields through untouched, so a manifest
    produced for VoiceCrafter trains here without being rewritten or
    materialized to wav files.

    Only the naming layer lives here. Everything about shuffling, sharding across
    workers and resume state is inherited, so the two manifest formats share one
    implementation of the parts that are easy to get subtly wrong.
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        fid_key: str = "audio_id",
        text_key: str = "text",
        audio_key: str = "audio_path",
        instruction_key: str = "instruction",
        audio_root: str | None = None,
        shuffle: bool = False,
        streaming: bool = False,
        shuffle_buffer_size: int = 4096,
        encoding: str = "utf-8",
    ):
        super().__init__(
            manifest_path=manifest_path,
            fid_key=fid_key,
            text_key=text_key,
            audio_key=audio_key,
            shuffle=shuffle,
            streaming=streaming,
            shuffle_buffer_size=shuffle_buffer_size,
            encoding=encoding,
        )
        self.instruction_key = instruction_key
        self.audio_root = Path(audio_root).expanduser() if audio_root else None

    def _resolve_audio_path(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        path = Path(str(value)).expanduser()
        if not path.is_absolute() and self.audio_root is not None:
            path = self.audio_root / path
        return str(path)

    def _build_sample(self, record: dict[str, Any]) -> dict[str, Any]:
        fid = record.get(self.fid_key) or record.get("fid")
        if fid in (None, ""):
            raise KeyError(
                f"Manifest record has no {self.fid_key!r} or 'fid': {record}"
            )

        audio_path = self._resolve_audio_path(
            record.get(self.audio_key) or record.get("audio")
        )
        backend = record.get("audio_backend")
        if not backend:
            if record.get("parquet_path") and record.get("parquet_row") is not None:
                backend = "parquet"
            elif record.get("dataset_path") and record.get("dataset_index") is not None:
                backend = "dataset"
            elif record.get("source_indices") or record.get("lance_dburi"):
                backend = "lance"
            elif record.get("tar_member") or isinstance(record.get("audio_locator"), dict):
                backend = "tar"
            else:
                backend = "file"
        if backend == "file" and not audio_path:
            raise KeyError(
                f"Row {fid!r} declares audio_backend='file' but has no audio path."
            )

        sample: dict[str, Any] = {
            "fid": str(fid),
            "text": record.get(self.text_key, ""),
            # `audio` stays in the sample because the pipeline contract requires
            # it and because it is what shows up in error messages. For non-file
            # backends it is a human-readable reference, never opened: the
            # backend dispatch reads the addressing fields instead.
            "audio": audio_path if backend == "file" else _reference_label(backend, record),
            "audio_backend": backend,
            "instruction": str(
                record.get(self.instruction_key)
                or record.get("instruction")
                or ""
            ),
        }
        if audio_path:
            sample["audio_path"] = audio_path

        for field in _PASSTHROUGH_FIELDS:
            if field in {"audio_path", "audio_backend"}:
                continue
            if record.get(field) not in (None, ""):
                sample[field] = record[field]

        for key, value in record.items():
            if key in sample or key in {self.fid_key, self.text_key, self.audio_key}:
                continue
            sample[key] = value
        return sample


__all__ = ["VoiceCrafterManifestSourceAdapter"]
