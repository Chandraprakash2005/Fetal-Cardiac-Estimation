"""src/fcac/config.py

Loads configs/default.yaml, merges an optional dataset override
(configs/datasets/<name>.yaml) and an optional experiment override
(configs/experiments/<name>.yaml) on top, and returns a single frozen,
attribute-accessible `Config` object. Nothing elsewhere in fcac should read YAML directly
or hardcode a hyperparameter -- everything comes through this Config.

Implements: CLAUDE.md §8 item 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


class Config:
    """Immutable, dot-accessible view over a nested dict loaded from YAML.

    `cfg.preprocessing.window.size_samples` reads like a dataclass but the shape is
    whatever configs/default.yaml declares -- no need to hand-write a dataclass field
    for every hyperparameter as the config grows.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        object.__setattr__(self, "_data", {k: _freeze(v) for k, v in data.items()})

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"Config has no key '{name}'. Add it to configs/default.yaml first "
                "(CLAUDE.md §6: no hardcoded numbers)."
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Config is frozen; edit the YAML files instead")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"

    def to_dict(self) -> dict:
        return _unfreeze(self._data)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _unfreeze(value: Any) -> Any:
    if isinstance(value, Config):
        return _unfreeze(value._data)
    if isinstance(value, dict):
        return {k: _unfreeze(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_unfreeze(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base`; `override` wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_config(
    dataset: Optional[str] = None,
    experiment: Optional[str] = None,
    configs_dir: Path = CONFIGS_DIR,
) -> Config:
    """Load default.yaml, optionally merge datasets/<dataset>.yaml and
    experiments/<experiment>.yaml on top, and return a frozen Config.

    Implements: CLAUDE.md §8 item 2.
    """
    merged = _load_yaml(configs_dir / "default.yaml")

    if dataset is not None:
        dataset_path = configs_dir / "datasets" / f"{dataset}.yaml"
        merged = _deep_merge(merged, _load_yaml(dataset_path))

    if experiment is not None:
        experiment_path = configs_dir / "experiments" / f"{experiment}.yaml"
        merged = _deep_merge(merged, _load_yaml(experiment_path))

    return Config(merged)
