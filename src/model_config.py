from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_CONFIG_PATH = Path("config/model_config.yaml")


def load_model_config(config_path: Path | str = DEFAULT_MODEL_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config YAML not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "enabled_models" not in data:
        raise KeyError("model_config.yaml must contain enabled_models")
    if "models" not in data:
        raise KeyError("model_config.yaml must contain models")
    return data


def get_enabled_models(config: dict[str, Any], task: str) -> list[str]:
    enabled = config.get("enabled_models", {}).get(task)
    if not enabled:
        raise ValueError(f"No enabled models configured for task: {task}")
    return [str(model_name) for model_name in enabled]


def get_model_params(config: dict[str, Any], model_name: str, task: str) -> dict[str, Any]:
    model_block = config.get("models", {}).get(model_name)
    if model_block is None:
        raise KeyError(f"Model '{model_name}' is enabled but not configured under models")
    params = model_block.get(task, {})
    return deepcopy(params) if params is not None else {}


def get_selection_metric(config: dict[str, Any], task: str, default: str) -> str:
    return str(config.get("selection", {}).get(f"{task}_metric", default))
