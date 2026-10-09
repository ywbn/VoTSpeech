from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from dots_tts.config.base import StrictConfigBase

DEFAULT_SOURCE_ADAPTER_CLASS_NAME = "JsonlManifestSourceAdapter"


class SourceAdapterConfig(StrictConfigBase):
    class_name: Literal[
        "JsonlManifestSourceAdapter",
        "VoiceCrafterManifestSourceAdapter",
    ] = DEFAULT_SOURCE_ADAPTER_CLASS_NAME
    params: dict[str, Any] = Field(default_factory=dict)


class DataSourceConfig(StrictConfigBase):
    name: str
    weight: float = Field(default=1.0, gt=0.0)
    pipeline: Literal[
        "basic", "interleave", "voice_design", "reference_instruction"
    ] = "basic"
    adapter: SourceAdapterConfig = Field(default_factory=SourceAdapterConfig)


class DataConfig(StrictConfigBase):
    sources: list[DataSourceConfig]
    train_audio_sample_rate: int = Field(ge=1)
    audio_samples_per_llm_token: int = Field(ge=1)
    num_tokens_per_epoch: int | None = Field(
        default=None,
        ge=1,
        description="Global token budget across all ranks for one training epoch.",
    )
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    prefetch_factor: int = Field(
        default=2,
        ge=1,
        description="Samples prefetched by each DataLoader worker.",
    )
    max_audio_seconds_in_batch: float = Field(gt=0.0)
    max_text_tokens_in_batch: int = Field(ge=1)
    max_samples_per_batch: int | None = Field(default=None, ge=1)
    # Hard cap on a SINGLE sample, separate from the batch budget above.
    # Peak attention memory is quadratic in one sample's length, so one long
    # utterance admitted as a batch of one can OOM even though the batch
    # budget is satisfied. null keeps the historical behaviour, where the
    # batch budget is also the per-sample limit.
    max_audio_seconds_per_sample: float | None = Field(default=None, gt=0.0)
    bucketing_pool_size: int = Field(default=64, ge=1)

    # voice_design pipeline settings. The voice occupies exactly two tokens
    # (<|voice_gen_start|> then <|voice_patch|>), so there is no count to
    # configure and nothing that can silently disagree with the model.
    instruction_key: str = "instruction"
    voice_design_template: str | None = None

    # reference_instruction pipeline: the speaker anchor is taken from a
    # different utterance by the same speaker, so timbre still arrives
    # while prosody and emotion cannot be copied from it.
    # <|voice_think|> slots emitted before the gen-start/patch pair. Must
    # match train.voice_design.num_think_slots -- the token ids are baked
    # into the precomputed cache, so a mismatch trains on a layout the
    # model does not expect.
    num_think_slots: int = Field(default=0, ge=0, le=64)
    # <|prosody_think|> slots emitted after the text and before the first
    # audio token. Same cache-consistency rule as num_think_slots: the ids
    # are baked in, so this must match train.voice_design.num_prosody_slots.
    num_prosody_slots: int = Field(default=0, ge=0, le=64)
    reference_audio_key: str = "reference_audio"
    reference_instruction_template: str | None = None

    @model_validator(mode="after")
    def _validate_unique_source_names(self) -> "DataConfig":
        counts: dict[str, int] = {}
        for source in self.sources:
            counts[source.name] = counts.get(source.name, 0) + 1
        duplicated = [name for name, count in counts.items() if count > 1]
        if duplicated:
            raise ValueError(f"Source names must be unique: {duplicated}")
        return self


__all__ = [
    "DEFAULT_SOURCE_ADAPTER_CLASS_NAME",
    "DataConfig",
    "DataSourceConfig",
    "SourceAdapterConfig",
]
