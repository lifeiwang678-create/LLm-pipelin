from __future__ import annotations

import numpy as np

from core.schema import LLMSample, SensorSample
from core.signal_utils import describe_1d


SIGNAL_ORDER = [
    "chest_ecg",
    "chest_eda",
    "chest_resp",
    "chest_emg",
    "chest_temp",
    "chest_acc",
    "wrist_bvp",
    "wrist_eda",
    "wrist_temp",
    "wrist_acc",
]


def extract_feature_dict(signals: dict) -> dict[str, dict[str, float | str]]:
    features: dict[str, dict[str, float | str]] = {}
    for name in SIGNAL_ORDER:
        if name not in signals:
            continue
        arr = np.asarray(signals[name])
        if arr.ndim == 2 and arr.shape[1] == 3:
            features[f"{name}_x"] = describe_1d(arr[:, 0])
            features[f"{name}_y"] = describe_1d(arr[:, 1])
            features[f"{name}_z"] = describe_1d(arr[:, 2])
            features[f"{name}_magnitude"] = describe_1d(np.linalg.norm(arr, axis=1))
        else:
            features[name] = describe_1d(arr)
    return features


def format_feature_block(features: dict[str, dict[str, float | str]]) -> str:
    lines = ["Input feature description:"]
    for name, stats in features.items():
        lines.append(
            "- "
            f"{name}: "
            f"mean={_fmt(stats['mean'])}, "
            f"std={_fmt(stats['std'])}, "
            f"min={_fmt(stats['min'])}, "
            f"max={_fmt(stats['max'])}, "
            f"p25={_fmt(stats['p25'])}, "
            f"p75={_fmt(stats['p75'])}, "
            f"trend={stats['trend']}"
        )
    if len(lines) == 1:
        lines.append("- no numeric signals available")
    return "\n".join(lines)


def _fmt(value: float | str) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


class FeatureDescriptionInput:
    name = "feature_description"

    def transform(self, sample: SensorSample) -> LLMSample:
        meta = dict(sample.meta)
        meta["input_type"] = self.name
        features = extract_feature_dict(sample.signals)
        return LLMSample(
            dataset=sample.dataset,
            subject=sample.subject,
            label=sample.label,
            input_text=format_feature_block(features),
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]


__all__ = ["FeatureDescriptionInput", "extract_feature_dict", "format_feature_block"]
