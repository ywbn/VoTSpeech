from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import yaml

# PyYAML implements YAML 1.1, where a float in scientific notation must carry a
# decimal point and a signed exponent: "2.0e-05" parses as a float but "2e-5"
# parses as a plain string. Learning rates are written the short way by every
# human and every shell script, so close that specific gap and nothing else --
# the pattern requires digits, an exponent marker and more digits, which no
# realistic run name or path matches.
_BARE_SCIENTIFIC = re.compile(r"^[+-]?\d+(?:\.\d*)?[eE][+-]?\d+$")


def _parse_override_value(raw_value: str) -> Any:
    value = yaml.safe_load(raw_value)
    if isinstance(value, str) and _BARE_SCIENTIFIC.match(value.strip()):
        return float(value)
    return value

from dots_tts.config.base import StrictConfigBase
from dots_tts.config.data import DataConfig
from dots_tts.config.train import TrainConfig
from dots_tts.models.dots_tts.config import LossConfig

DEFAULT_CONFIG_PATH = "configs/dots_tts.yaml"


def apply_overrides(
    payload: dict[str, Any], overrides: Iterable[str] | None
) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides to a raw config mapping in place.

    Values are parsed as YAML scalars, so ``2e-5`` becomes a float, ``null``
    becomes None and ``['a','b']`` becomes a list. Numeric path segments index
    into lists, which is what makes per-source overrides like
    ``train_data.sources.0.weight`` expressible.

    A path that does not already exist is an error rather than a silent
    creation: a typo in a launcher script must fail loudly, not quietly train
    something other than what was asked for. This lives here, with no torch or
    accelerate imports above it, so config tooling stays cheap to run.
    """

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override expects DOTTED.KEY=VALUE, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.strip().split(".") if part]
        if not keys:
            raise ValueError(f"Override has an empty key: {override!r}")
        value = _parse_override_value(raw_value)

        cursor: Any = payload
        for depth, key in enumerate(keys[:-1]):
            index = int(key) if key.isdigit() else None
            try:
                cursor = cursor[index if index is not None else key]
            except (KeyError, IndexError, TypeError) as error:
                path = ".".join(keys[: depth + 1])
                raise KeyError(
                    f"Override path {dotted_key!r} does not exist at {path!r}"
                ) from error
        last = keys[-1]
        if last.isdigit() and isinstance(cursor, list):
            cursor[int(last)] = value
        elif isinstance(cursor, dict):
            cursor[last] = value
        else:
            raise KeyError(
                f"Override path {dotted_key!r} does not address a mapping entry."
            )
    return payload


class AppConfig(StrictConfigBase):
    train_data: DataConfig
    val_data: DataConfig | None = None
    loss: LossConfig
    train: TrainConfig

    @classmethod
    def from_yaml(
        cls,
        config_path: str = DEFAULT_CONFIG_PATH,
        overrides: Iterable[str] | None = None,
    ) -> AppConfig:
        with Path(config_path).open(encoding="utf-8") as fin:
            raw_config = yaml.safe_load(fin)
        return cls.model_validate(apply_overrides(raw_config, overrides))


def load_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    overrides: Iterable[str] | None = None,
) -> AppConfig:
    return AppConfig.from_yaml(config_path, overrides)


__all__ = [
    "AppConfig",
    "DEFAULT_CONFIG_PATH",
    "apply_overrides",
    "load_config",
]
