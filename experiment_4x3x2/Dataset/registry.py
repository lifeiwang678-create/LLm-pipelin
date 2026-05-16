from __future__ import annotations

from .dreamt_loader import DREAMTLoader
from .hhar_loader import HHARLoader
from .wesad_loader import WESADLoader


DATASET_REGISTRY = {
    "WESAD": {
        "loader": WESADLoader,
        "data_dir": "..",
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
        "subjects": None,
        "train_subjects": None,
        "test_subjects": None,
        "loader_kwargs": {
            "window_size": 128,
            "stride_size": 64,
        },
    },
    "DREAMT": {
        "loader": DREAMTLoader,
        "data_dir": "Dataset/DREAMT",
        "subjects": None,
        "train_subjects": None,
        "test_subjects": None,
        "loader_kwargs": {
            "window_size": 128,
            "stride_size": 64,
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
        "label_map",
    )
    for key in loader_keys:
        if key in config:
            kwargs[key] = config[key]

    return loader_cls(data_dir=config.get("data_dir", dataset_cfg["data_dir"]), **kwargs)
