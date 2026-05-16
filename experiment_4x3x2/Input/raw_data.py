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


class RawDataInput:
    name = "raw_data"

    def transform(self, sample: SensorSample) -> LLMSample:
        if str(sample.dataset).upper() != "WESAD":
            raise ValueError(
                "RawDataInput is currently WESAD-specific. "
                "HHAR and DREAMT require dataset-aware raw signal formatting before raw_data can be used."
            )
        meta = dict(sample.meta)
        meta["input_type"] = self.name
        return LLMSample(
            dataset=sample.dataset,
            subject=sample.subject,
            label=sample.label,
            input_text=format_wesad_raw_block(sample.signals),
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]


__all__ = ["RawDataInput", "format_raw_block", "format_wesad_raw_block"]
