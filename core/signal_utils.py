from __future__ import annotations

import numpy as np


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def downsample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x)
    if len(x) <= target_len:
        return x
    idx = np.linspace(0, len(x) - 1, target_len).astype(int)
    return x[idx]


def round_list(x: np.ndarray, decimals: int = 3) -> list[float]:
    return [round(float(v), decimals) for v in x]


def get_segment(sig: np.ndarray, center_idx: int, window_sec: float, fs: int) -> np.ndarray:
    half = int(window_sec * fs / 2)
    start = max(0, center_idx - half)
    end = min(len(sig), center_idx + half)
    return sig[start:end]


def is_pure_window(labels: np.ndarray, center_idx: int, fs: int, window_sec: float) -> bool:
    half = int(window_sec * fs / 2)
    start = max(0, center_idx - half)
    end = min(len(labels), center_idx + half)
    seg = labels[start:end]
    return len(seg) > 0 and len(np.unique(seg)) == 1


def pack_1d(sig: np.ndarray, target_len: int = 60) -> list[float]:
    sig = np.asarray(sig).flatten()
    sig = downsample_to_length(zscore_safe(sig), target_len)
    return round_list(sig)


def pack_acc_xyz(acc_seg: np.ndarray, target_len: int = 40) -> dict[str, list[float]]:
    acc_seg = np.asarray(acc_seg)
    if acc_seg.ndim != 2 or acc_seg.shape[1] != 3:
        return {"x": [], "y": [], "z": []}
    return {
        "x": round_list(downsample_to_length(zscore_safe(acc_seg[:, 0]), target_len)),
        "y": round_list(downsample_to_length(zscore_safe(acc_seg[:, 1]), target_len)),
        "z": round_list(downsample_to_length(zscore_safe(acc_seg[:, 2]), target_len)),
    }


def describe_1d(sig: np.ndarray) -> dict[str, float | str | None]:
    x = np.asarray(sig, dtype=np.float32).flatten()
    if len(x) == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
            "trend": "empty",
        }

    trend = "relatively stable"
    if len(x) >= 2:
        delta = float(x[-1] - x[0])
        std = float(np.std(x))
        if abs(delta) > max(std * 0.25, 1e-6):
            trend = "increasing" if delta > 0 else "decreasing"

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "trend": trend,
    }
