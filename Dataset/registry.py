from __future__ import annotations


DATASET_REGISTRY = {
    "WESAD": {
        "data_dir": ".",
        "feature_pattern": "*_features_paperstyle.csv",
        "subjects": ["S2", "S3"],
        "train_subjects": ["S2", "S3", "S4", "S5", "S6"],
        "test_subjects": ["S7", "S8"],
    },
    "HHAR": {
        "data_dir": "Dataset/HHAR",
        "feature_pattern": "*_features.csv",
    },
    "DREAMT": {
        "data_dir": "Dataset/DREAMT",
        "feature_pattern": "*_features.csv",
    },
}


def get_dataset_config(name: str) -> dict:
    try:
        return DATASET_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {available}")

