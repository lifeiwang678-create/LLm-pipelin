from __future__ import annotations

import json

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

GENERIC_TARGET_LENGTH = 48


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


def format_generic_raw_block(signals: dict, max_channels: int = 12) -> str:
    lines = ["Input raw sequences:"]
    used = 0
    for name in sorted(signals):
        if used >= max_channels:
            lines.append(f"... {len(signals) - used} additional channel(s) omitted for prompt length.")
            break
        value = signals.get(name, [])
        if _looks_like_acc_matrix(value):
            packed = pack_acc_xyz(value, GENERIC_TARGET_LENGTH)
            if packed:
                lines.append(f"{name} = {json.dumps(packed, ensure_ascii=False)}")
                used += 1
            continue
        packed = pack_1d(value, GENERIC_TARGET_LENGTH)
        if packed:
            lines.append(f"{name} = {packed}")
            used += 1
    if used == 0:
        lines.append("No numeric raw sensor channel was available.")
    return "\n".join(lines)


def _looks_like_acc_matrix(value) -> bool:
    try:
        rows = list(value)
    except TypeError:
        return False
    if not rows:
        return False
    first = rows[0]
    try:
        return len(first) >= 3
    except TypeError:
        return False


class RawDataInput:
    name = "raw_data"

    def transform(self, sample: SensorSample) -> LLMSample:
        meta = dict(sample.meta)
        meta["input_type"] = self.name
        input_text = (
            format_wesad_raw_block(sample.signals)
            if str(sample.dataset).upper() == "WESAD"
            else format_generic_raw_block(sample.signals)
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


__all__ = ["RawDataInput", "format_raw_block", "format_wesad_raw_block", "format_generic_raw_block"]
