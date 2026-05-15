from __future__ import annotations

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.runner import run_experiment


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config files require PyYAML. Use JSON or install PyYAML.") from exc
        config = yaml.safe_load(text)
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("Experiment config must be a JSON/YAML object.")
    config["_config_path"] = str(config_path)
    return config


def expand_experiment_configs(config: dict) -> list[dict]:
    """Expand one config file into concrete experiments for the shared runner."""
    base = _base_config(config)

    if "experiments" in config:
        experiments = config["experiments"]
        if not isinstance(experiments, list):
            raise ValueError("config['experiments'] must be a list.")
        return [_standardize_config(_deep_merge(base, item)) for item in experiments]

    if "grid" in config:
        return [_standardize_config(item) for item in _expand_grid(base, config["grid"])]

    return [_standardize_config(base)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch/config modular LLM experiments.")
    parser.add_argument("--config", default="configs/example_experiment.json", help="Path to JSON or YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    experiments = expand_experiment_configs(config)
    for idx, experiment_config in enumerate(experiments, 1):
        print(f"[experiment {idx}/{len(experiments)}] {experiment_config.get('run_name', 'unnamed')}")
        run_experiment(experiment_config)


def _base_config(config: dict) -> dict:
    base = deepcopy(config.get("base", {}))
    for key, value in config.items():
        if key not in {"base", "experiments", "grid"}:
            base[key] = deepcopy(value)
    return base


def _expand_grid(base: dict, grid: dict) -> list[dict]:
    if not isinstance(grid, dict):
        raise ValueError("config['grid'] must be a dictionary.")
    keys = list(grid)
    value_lists = [_as_list(grid[key]) for key in keys]
    experiments = []
    for values in itertools.product(*value_lists):
        experiment = deepcopy(base)
        for key, value in zip(keys, values):
            experiment = _deep_merge(experiment, _grid_override(key, value))
        experiment.setdefault("run_name", _grid_run_name(experiment))
        experiments.append(experiment)
    return experiments


def _grid_override(key: str, value: Any) -> dict:
    normalized = key.strip()
    lowered = normalized.lower()

    if "." in normalized:
        return _dotted_override(normalized, value)

    if lowered in {"dataset", "dataset_name"}:
        return {"dataset": value if isinstance(value, dict) else {"name": value}}
    if lowered in {"input", "input_type", "input_method"}:
        return {"input": value if isinstance(value, dict) else {"type": value}}
    if lowered in {"lm", "lm_usage", "lm_usage_type", "lm_method"}:
        return {"lm_usage": value if isinstance(value, dict) else {"type": value}}
    if lowered in {"output", "output_type", "output_format"}:
        return {"output": value if isinstance(value, dict) else {"type": value}}
    return {normalized: value}


def _dotted_override(key: str, value: Any) -> dict:
    parts = [part for part in key.split(".") if part]
    if not parts:
        raise ValueError("Grid key cannot be empty.")
    result: Any = value
    for part in reversed(parts):
        result = {part: result}
    return result


def _standardize_config(config: dict) -> dict:
    standardized = deepcopy(config)
    if _dataset_name(standardized) is None:
        # Existing JSON configs in this repository were WESAD-only and did not
        # carry a dataset block. Keep them usable while new configs stay explicit.
        input_data_dir = (standardized.get("input") or {}).get("data_dir", ".")
        standardized["dataset"] = {"name": "WESAD", "data_dir": input_data_dir}
    _carry_legacy_input_loader_settings(standardized)
    return standardized


def _carry_legacy_input_loader_settings(config: dict) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        return
    input_config = config.get("input") or {}
    if "data_dir" not in dataset and input_config.get("data_dir"):
        dataset["data_dir"] = input_config["data_dir"]
    loader_kwargs = dict(dataset.get("loader_kwargs") or {})
    for key in ("window_sec", "stride_sec", "window_size", "stride_size", "label_map"):
        if key in input_config:
            loader_kwargs.setdefault(key, input_config[key])
    if loader_kwargs:
        dataset["loader_kwargs"] = loader_kwargs


def _dataset_name(config: dict) -> str | None:
    dataset = config.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        return dataset
    if isinstance(dataset, dict) and str(dataset.get("name", "")).strip():
        return str(dataset["name"])
    input_dataset = (config.get("input") or {}).get("dataset")
    if input_dataset and str(input_dataset).strip():
        return str(input_dataset)
    return None


def _grid_run_name(config: dict) -> str:
    dataset = _dataset_name(config) or "UNKNOWN"
    input_type = (config.get("input") or {}).get("type", "feature_description")
    lm_type = (config.get("lm_usage") or {}).get("type", "direct")
    output_type = (config.get("output") or {}).get("type", "label_only")
    return f"{dataset}_{input_type}_{lm_type}_{output_type}"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _deep_merge(base: dict, override: dict) -> dict:
    if not isinstance(override, dict):
        raise ValueError("Experiment overrides must be dictionaries.")
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


if __name__ == "__main__":
    main()
