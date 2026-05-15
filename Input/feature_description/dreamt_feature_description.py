from __future__ import annotations

from core.schema import SensorSample

from .basic_feature_description import BaseFeatureDescriptionInput
from .feature_functions import FeatureDict, extract_signal_features


class DreaMTFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "DREAMT"
    title = "Input feature description for DreaMT:"
    sections = {
        "Electrodermal activity": ["eda", "gsr"],
        "Pulse and heart-rate signals": ["bvp", "ppg", "pulse", "hr", "heart_rate", "ibi"],
        "Temperature": ["temp", "temperature", "skin_temp"],
        "Activity and movement": ["acc", "actigraphy", "activity", "motion"],
    }

    def extract_features(self, sample: SensorSample) -> FeatureDict:
        dreamt_signals = {
            name: values
            for name, values in sample.signals.items()
            if _is_dreamt_signal(name)
        }
        return extract_signal_features(dreamt_signals)


# Backward-compatible alias for earlier all-caps imports.
DREAMTFeatureDescriptionInput = DreaMTFeatureDescriptionInput


def _is_dreamt_signal(name: str) -> bool:
    lower = name.lower()
    return any(
        token in lower
        for token in [
            "eda",
            "gsr",
            "bvp",
            "ppg",
            "pulse",
            "hr",
            "heart_rate",
            "ibi",
            "temp",
            "temperature",
            "skin_temp",
            "acc",
            "actigraphy",
            "activity",
            "motion",
        ]
    )


def _demo() -> None:
    import numpy as np

    from core.schema import SensorSample

    sample = SensorSample(
        dataset="DREAMT",
        subject="mock",
        label=1,
        signals={
            "eda": np.linspace(0.1, 0.5, 32),
            "bvp": np.sin(np.linspace(0, 4, 32)),
            "heart_rate": np.linspace(65, 70, 32),
            "skin_temp": np.linspace(32, 33, 32),
            "actigraphy_x": np.cos(np.linspace(0, 3, 32)),
        },
    )
    print(DreaMTFeatureDescriptionInput().build_input(sample))


__all__ = ["DreaMTFeatureDescriptionInput", "DREAMTFeatureDescriptionInput"]


if __name__ == "__main__":
    _demo()
