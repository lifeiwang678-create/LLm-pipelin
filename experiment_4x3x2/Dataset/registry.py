from __future__ import annotations

from .dreamt_loader import DREAMTLoader
from .hhar_loader import HHARLoader
from .wesad_loader import WESADLoader


DATASET_REGISTRY = {
    "WESAD": {
        "loader": WESADLoader,
        "data_dir": "..",
        "labels": [0, 1],
        "subjects": ["S2", "S3"],
        "train_subjects": ["S2", "S3", "S4", "S5", "S6"],
        "test_subjects": ["S7", "S8"],
        "loader_kwargs": {
            "physiology_window_sec": 60.0,
            "acc_window_sec": 5.0,
            "stride_sec": 0.25,
        },
    },
    "HHAR": {
        "loader": HHARLoader,
        "data_dir": "Dataset/HHAR",
        "labels": [0, 1],
        "subjects": None,
        "train_subjects": None,
        "test_subjects": None,
        "loader_kwargs": {
            "window_size": 128,
            "stride_size": 64,
            "sampling_rate": 64.0,
            "min_samples_per_window": 10,
            "max_gap_sec": 5.0,
        },
    },
    "DREAMT": {
        "loader": DREAMTLoader,
        "data_dir": "Dataset/DREAMT",
        "labels": [0, 1],
        "subjects": None,
        "train_subjects": None,
        "test_subjects": None,
        "loader_kwargs": {
            "sampling_rate": 64,
            "epoch_seconds": 30.0,
            "stride_seconds": 30.0,
            "min_epoch_fraction": 0.5,
        },
    },
}


def get_dataset_config(name: str) -> dict:
    try:
        return DATASET_REGISTRY[name].copy()
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")


def build_dataset_loader(config: dict):
    name = str(config.get("name", "WESAD"))
    dataset_cfg = get_dataset_config(name)
    loader_cls = dataset_cfg["loader"]
    kwargs = dict(dataset_cfg.get("loader_kwargs", {}))
    kwargs.update(config.get("loader_kwargs", {}))

    loader_keys = (
        "physiology_window_sec",
        "acc_window_sec",
        "stride_sec",
        "window_sec",
        "window_size",
        "stride_size",
        "sampling_rate",
        "epoch_seconds",
        "stride_seconds",
        "min_epoch_fraction",
        "min_samples_per_window",
        "max_gap_sec",
        "label_map",
        "skip_artifact_epochs",
        "include_gyroscope",
        "max_rows",
    )
    for key in loader_keys:
        if key in config:
            kwargs[key] = config[key]

    return loader_cls(data_dir=config.get("data_dir", dataset_cfg["data_dir"]), **kwargs)
