from __future__ import annotations

from core.schema import SensorSample

from .feature_description import BaseFeatureDescriptionInput
from .feature_functions import FeatureDict, extract_signal_features


class HHARFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "HHAR"
    title = "Input feature description for HHAR:"
    sections = {
        "Accelerometer": ["acc", "accelerometer", "x", "y", "z"],
        "Gyroscope": ["gyro", "gyroscope"],
    }

    def extract_features(self, sample: SensorSample) -> FeatureDict:
        motion_signals = {
            name: values
            for name, values in sample.signals.items()
            if _is_hhar_motion_signal(name)
        }
        return extract_signal_features(motion_signals)


def _is_hhar_motion_signal(name: str) -> bool:
    lower = name.lower()
    if any(token in lower for token in ["acc", "accelerometer", "gyro", "gyroscope"]):
        return True
    return lower in {"x", "y", "z"}


def _demo() -> None:
    import numpy as np

    from core.schema import SensorSample

    sample = SensorSample(
        dataset="HHAR",
        subject="mock",
        label=1,
        signals={
            "acc_x": np.linspace(0, 1, 32),
            "acc_y": np.linspace(1, 0, 32),
            "gyro_z": np.sin(np.linspace(0, 2, 32)),
        },
    )
    print(HHARFeatureDescriptionInput().build_input(sample))


__all__ = ["HHARFeatureDescriptionInput"]


if __name__ == "__main__":
    _demo()
