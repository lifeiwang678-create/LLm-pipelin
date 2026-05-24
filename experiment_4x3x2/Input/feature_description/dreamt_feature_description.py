from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.schema import SensorSample

try:
    from .basic_feature_description import BaseFeatureDescriptionInput
    from .feature_functions import FeatureDict, extract_signal_features, format_feature_block
except ImportError:
    from Input.feature_description.basic_feature_description import BaseFeatureDescriptionInput
    from Input.feature_description.feature_functions import FeatureDict, extract_signal_features, format_feature_block


class DREAMTFeatureDescriptionInput(BaseFeatureDescriptionInput):
    dataset_name = "DREAMT"
    title = "Input feature description for DREAMT:"
    sections = {
        "Electrodermal activity": ["eda", "gsr"],
        "Pulse and heart-rate signals": ["bvp", "ppg", "pulse", "hr", "heart_rate", "ibi"],
        "Temperature": ["temp", "temperature", "skin_temp"],
        "Activity and movement": ["acc", "actigraphy", "activity", "motion"],
        "DREAMT sleep/wake wearable features": ["sleep_"],
    }

    def extract_features(self, sample: SensorSample) -> dict:
        # DREAMT-specific sleep/wake-oriented wearable features for the
        # prompt-based feature_description input. They are not a trained sleep
        # staging model.
        dreamt_signals = {
            name: values
            for name, values in sample.signals.items()
            if _is_dreamt_signal(name)
        }
        if not dreamt_signals:
            raise ValueError(
                "No DREAMT wearable signals found in sample.signals. "
                f"Available signals: {list(sample.signals.keys())}"
            )
        generic_features = extract_signal_features(dreamt_signals)
        sleep_features = {
            f"sleep_{name}": value
            for name, value in extract_dreamt_sleep_features(dreamt_signals).items()
        }
        return {**generic_features, **sleep_features}

    def format_features(self, features: dict) -> str:
        generic_features: FeatureDict = {
            name: stats
            for name, stats in features.items()
            if isinstance(stats, dict)
        }
        sleep_features = {
            name: value
            for name, value in features.items()
            if not isinstance(value, dict)
        }
        lines = [format_feature_block(generic_features, title=self.title, sections=self.sections)]
        if sleep_features:
            lines.append("")
            lines.append("DREAMT sleep/wake wearable features:")
            for name, value in sorted(sleep_features.items()):
                lines.append(f"- {name}: {float(value):.3f}")
        return "\n".join(lines)


# Backward-compatible alias for earlier mixed-case imports.
DreaMTFeatureDescriptionInput = DREAMTFeatureDescriptionInput


def _is_dreamt_signal(name: str) -> bool:
    key = _normalize_key(name)
    exact_names = {
        "eda",
        "gsr",
        "bvp",
        "ppg",
        "pulse",
        "hr",
        "ibi",
        "temp",
        "acc",
        "acc_x",
        "acc_y",
        "acc_z",
    }
    if key in exact_names:
        return True
    long_tokens = [
        "heart_rate",
        "skin_temp",
        "temperature",
        "actigraphy",
        "activity",
        "motion",
        "accelerometer",
    ]
    return any(token in key for token in long_tokens)


def extract_dreamt_sleep_features(signals: dict[str, Any]) -> dict[str, float]:
    """Extract DREAMT-specific sleep/wake wearable features.

    These features are prompt-side cues for binary sleep/wake classification.
    They are not a trained sleep staging model and do not use labels or
    answer-related fields.
    """
    features: dict[str, float] = {}

    acc_mag = _acc_magnitude(signals)
    if acc_mag is not None:
        features.update(_movement_features(acc_mag))

    heart_rate = _first_signal(signals, ["heart_rate", "hr"])
    if heart_rate is not None:
        features.update(_basic_series_features("heart_rate", heart_rate, include_slope=True))

    ibi = _first_signal(signals, ["ibi"])
    if ibi is not None:
        features.update(_basic_series_features("ibi", ibi, include_slope=False))
        rmssd = _rmssd(ibi)
        if rmssd is not None:
            features["ibi_rmssd"] = rmssd

    bvp = _first_signal(signals, ["bvp", "ppg", "pulse"])
    if bvp is not None:
        features.update(_basic_series_features("bvp", bvp, include_slope=True))
        features["bvp_peak_count"] = float(_peak_count(bvp))

    eda = _first_signal(signals, ["eda", "gsr"])
    if eda is not None:
        features.update(_basic_series_features("eda", eda, include_slope=True))
        features["eda_peak_count"] = float(_peak_count(eda))

    temp = _first_signal(signals, ["skin_temp", "temperature", "temp"])
    if temp is not None:
        features.update(_basic_series_features("skin_temperature", temp, include_slope=True))

    return features


def _movement_features(acc_magnitude: np.ndarray) -> dict[str, float]:
    features = _basic_series_features("acc_magnitude", acc_magnitude, include_slope=False)
    if len(acc_magnitude) < 2:
        features["movement_count"] = 0.0
        features["stillness_ratio"] = 1.0
        return features

    diff = np.abs(np.diff(acc_magnitude))
    if len(diff) == 0:
        features["movement_count"] = 0.0
        features["stillness_ratio"] = 1.0
        return features

    threshold = float(np.nanmedian(diff) + np.nanstd(diff))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = 1e-8
    moving = diff > threshold
    features["movement_count"] = float(np.sum(moving))
    features["stillness_ratio"] = float(np.mean(~moving))
    return features


def _basic_series_features(prefix: str, values: np.ndarray, include_slope: bool) -> dict[str, float]:
    arr = _to_finite_1d(values)
    if len(arr) == 0:
        return {}

    features = {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_range": float(np.max(arr) - np.min(arr)),
    }
    if include_slope:
        slope = _slope(arr)
        if slope is not None:
            features[f"{prefix}_slope"] = slope
    return features


def _acc_magnitude(signals: dict[str, Any]) -> np.ndarray | None:
    acc = _first_signal(signals, ["acc", "accelerometer"])
    if acc is not None:
        arr = np.asarray(acc, dtype=float)
        if arr.ndim == 2:
            if arr.shape[1] >= 3:
                return _to_finite_1d(np.linalg.norm(arr[:, :3], axis=1))
            if arr.shape[0] >= 3:
                return _to_finite_1d(np.linalg.norm(arr[:3, :], axis=0))

    x = _first_signal(signals, ["acc_x", "accelerometer_x", "actigraphy_x"])
    y = _first_signal(signals, ["acc_y", "accelerometer_y", "actigraphy_y"])
    z = _first_signal(signals, ["acc_z", "accelerometer_z", "actigraphy_z"])
    if x is None or y is None or z is None:
        return _first_signal(signals, ["actigraphy", "activity", "motion"])

    x_arr = _to_finite_1d(x)
    y_arr = _to_finite_1d(y)
    z_arr = _to_finite_1d(z)
    n = min(len(x_arr), len(y_arr), len(z_arr))
    if n == 0:
        return None
    return _to_finite_1d(np.sqrt(x_arr[:n] ** 2 + y_arr[:n] ** 2 + z_arr[:n] ** 2))


def _first_signal(signals: dict[str, Any], candidates: list[str]) -> np.ndarray | None:
    normalized = {_normalize_key(key): value for key, value in signals.items()}
    for candidate in candidates:
        key = _normalize_key(candidate)
        if key in normalized:
            arr = _to_finite_1d(normalized[key])
            if len(arr):
                return arr
    for raw_key, value in signals.items():
        normalized_key = _normalize_key(raw_key)
        if any(_normalize_key(candidate) in normalized_key for candidate in candidates):
            arr = _to_finite_1d(value)
            if len(arr):
                return arr
    return None


def _to_finite_1d(values: Any) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr.astype(float)


def _slope(values: np.ndarray) -> float | None:
    arr = _to_finite_1d(values)
    if len(arr) < 2:
        return None
    x = np.arange(len(arr), dtype=float)
    try:
        return float(np.polyfit(x, arr, 1)[0])
    except Exception:
        return None


def _rmssd(values: np.ndarray) -> float | None:
    arr = _to_finite_1d(values)
    if len(arr) < 2:
        return None
    diff = np.diff(arr)
    return float(np.sqrt(np.mean(diff**2)))


def _peak_count(values: np.ndarray) -> int:
    arr = _to_finite_1d(values)
    if len(arr) < 3:
        return 0
    threshold = float(np.mean(arr) + np.std(arr))
    peaks = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]) & (arr[1:-1] > threshold)
    return int(np.sum(peaks))


def _normalize_key(key: str) -> str:
    return str(key).replace("-", "_").replace(" ", "_").strip().lower()


def _demo() -> None:
    from core.schema import SensorSample

    sample = SensorSample(
        dataset="DREAMT",
        subject="mock",
        label=1,
        signals={
            "eda": np.linspace(0.1, 0.5, 32),
            "bvp": np.sin(np.linspace(0, 4, 32)),
            "heart_rate": np.linspace(65, 70, 32),
            "ibi": np.linspace(0.8, 0.9, 32),
            "skin_temp": np.linspace(32, 33, 32),
            "acc_x": np.cos(np.linspace(0, 3, 32)),
            "acc_y": np.sin(np.linspace(0, 3, 32)),
            "acc_z": np.linspace(0.1, 0.2, 32),
        },
    )
    print(DREAMTFeatureDescriptionInput().build_input(sample))


__all__ = [
    "DreaMTFeatureDescriptionInput",
    "DREAMTFeatureDescriptionInput",
    "extract_dreamt_sleep_features",
]


if __name__ == "__main__":
    _demo()
