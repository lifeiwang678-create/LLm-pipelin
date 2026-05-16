from __future__ import annotations

from .basic_feature_description import BaseFeatureDescriptionInput


def get_feature_description_builder(dataset: str | None = None) -> BaseFeatureDescriptionInput:
    if dataset is None or str(dataset).strip() == "":
        raise ValueError("Feature description input requires config['dataset'] or config['input']['dataset'].")

    key = _normalize_dataset_name(dataset)
    if key == "WESAD":
        from .wesad_feature_description import WESADFeatureDescriptionInput

        return WESADFeatureDescriptionInput()
    if key == "HHAR":
        from .hhar_feature_description import HHARFeatureDescriptionInput

        return HHARFeatureDescriptionInput()
    if key == "DREAMT":
        from .dreamt_feature_description import DreaMTFeatureDescriptionInput

        return DreaMTFeatureDescriptionInput()
    raise ValueError(f"Unknown feature description dataset: {dataset}")


build_feature_description_input = get_feature_description_builder


def _normalize_dataset_name(dataset: str) -> str:
    return str(dataset).replace("-", "").replace("_", "").strip().upper()


__all__ = [
    "build_feature_description_input",
    "get_feature_description_builder",
]
