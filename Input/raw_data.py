from __future__ import annotations

import json

import numpy as np

from core.schema import LLMSample, SensorSample
from core.signal_utils import pack_1d, pack_acc_xyz


RAW_TARGET_LENGTHS = {
    "chest_ecg": 80,
    "chest_eda": 40,
    "chest_resp": 60,
    "chest_emg": 40,
    "chest_temp": 20,
    "wrist_bvp": 60,
    "wrist_eda": 40,
    "wrist_temp": 20,
}


def format_raw_block(signals: dict) -> str:
    """Backward-compatible WESAD raw-data formatter."""
    return format_wesad_raw_block(signals)


def format_wesad_raw_block(signals: dict) -> str:
    raw = {
        name: pack_1d(signals.get(name, []), target_len)
        for name, target_len in RAW_TARGET_LENGTHS.items()
    }
    raw["chest_acc"] = pack_acc_xyz(signals.get("chest_acc", []), 40)
    raw["wrist_acc"] = pack_acc_xyz(signals.get("wrist_acc", []), 40)

    return f"""Input raw sequences:

chest_ecg = {raw["chest_ecg"]}
chest_eda = {raw["chest_eda"]}
chest_resp = {raw["chest_resp"]}
chest_emg = {raw["chest_emg"]}
chest_temp = {raw["chest_temp"]}
chest_acc = {json.dumps(raw["chest_acc"], ensure_ascii=False)}

wrist_bvp = {raw["wrist_bvp"]}
wrist_eda = {raw["wrist_eda"]}
wrist_temp = {raw["wrist_temp"]}
wrist_acc = {json.dumps(raw["wrist_acc"], ensure_ascii=False)}"""


def format_generic_raw_block(signals: dict, dataset: str | None = None, target_len: int = 80) -> str:
    lines = [
        "Input raw sequences:",
        f"Dataset: {dataset or 'UNKNOWN'}",
        "Channels are packed from the numeric signals provided by the dataset loader.",
        "",
    ]
    emitted = 0
    for name in sorted(signals):
        arr = _numeric_array(signals.get(name))
        if arr.size == 0:
            lines.append(f"{name} = []")
            continue
        if arr.ndim == 1:
            lines.append(f"{name} = {pack_1d(arr, target_len)}")
            emitted += 1
            continue
        lines.append(f"{name} = {json.dumps(_pack_matrix(arr, target_len), ensure_ascii=False)}")
        emitted += 1
    if emitted == 0:
        lines.append("No numeric raw signal channels were available.")
    return "\n".join(lines)


class RawDataInput:
    name = "raw_data"

    def transform(self, sample: SensorSample) -> LLMSample:
        meta = dict(sample.meta)
        meta["input_type"] = self.name
        input_text = (
            format_wesad_raw_block(sample.signals)
            if str(sample.dataset).upper() == "WESAD"
            else format_generic_raw_block(sample.signals, sample.dataset)
        )
        return LLMSample(
            dataset=sample.dataset,
            subject=sample.subject,
            label=sample.label,
            input_text=input_text,
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]


def _numeric_array(value) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr if np.any(np.isfinite(arr)) else np.asarray([], dtype=float)


def _pack_matrix(arr: np.ndarray, target_len: int) -> dict[str, list[float]]:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        return {"values": pack_1d(arr, target_len)}
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[0] <= 6 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    max_cols = min(arr.shape[1], 6)
    return {f"axis_{idx}": pack_1d(arr[:, idx], target_len) for idx in range(max_cols)}


__all__ = ["RawDataInput", "format_generic_raw_block", "format_raw_block", "format_wesad_raw_block"]
