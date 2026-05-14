from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from core.schema import Sample


FEATURE_COLUMNS = [
    "chest_scl_mean",
    "chest_scr_num",
    "chest_scr_amp_sum",
    "wrist_scl_mean",
    "wrist_scr_num",
    "wrist_scr_amp_sum",
    "chest_hr_mean",
    "chest_hr_std",
    "chest_hrv_rmssd",
    "chest_hrv_abs_lf",
    "chest_hrv_abs_hf",
    "chest_resp_rate",
    "chest_resp_rate_mean",
    "chest_resp_range",
    "chest_resp_ie_ratio",
    "chest_acc_3d_std",
    "wrist_acc_3d_std",
    "chest_emg_std",
    "chest_emg_peaks_num",
    "chest_temp_mean",
    "chest_temp_slope",
    "wrist_temp_mean",
    "wrist_temp_slope",
]


def extract_feature_dict(row: pd.Series) -> dict[str, float]:
    features = {}
    for name in FEATURE_COLUMNS:
        if name == "chest_resp_rate" and name not in row:
            value = row.get("chest_resp_rate_mean", 0.0)
        else:
            value = row.get(name, 0.0)
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            features[name] = 0.0
    return features


def format_feature_block(features: dict[str, float]) -> str:
    resp_rate = features.get("chest_resp_rate", features.get("chest_resp_rate_mean", 0.0))
    return f"""Input feature description:

EDA_chest:
- scl_mean: {features["chest_scl_mean"]:.3f}
- scr_num: {features["chest_scr_num"]:.3f}
- scr_amp_sum: {features["chest_scr_amp_sum"]:.3f}

EDA_wrist:
- scl_mean: {features["wrist_scl_mean"]:.3f}
- scr_num: {features["wrist_scr_num"]:.3f}
- scr_amp_sum: {features["wrist_scr_amp_sum"]:.3f}

Cardio:
- hr_mean: {features["chest_hr_mean"]:.3f}
- hr_std: {features["chest_hr_std"]:.3f}
- hrv_rmssd: {features["chest_hrv_rmssd"]:.3f}
- hrv_lf: {features["chest_hrv_abs_lf"]:.3f}
- hrv_hf: {features["chest_hrv_abs_hf"]:.3f}

Resp:
- resp_rate: {resp_rate:.3f}
- resp_range: {features["chest_resp_range"]:.3f}
- resp_ie_ratio: {features["chest_resp_ie_ratio"]:.3f}

Motion:
- chest_acc_3d_std: {features["chest_acc_3d_std"]:.3f}
- wrist_acc_3d_std: {features["wrist_acc_3d_std"]:.3f}

EMG:
- emg_std: {features["chest_emg_std"]:.3f}
- emg_peaks_num: {features["chest_emg_peaks_num"]:.3f}

Temp:
- chest_temp_mean: {features["chest_temp_mean"]:.3f}
- chest_temp_slope: {features["chest_temp_slope"]:.3f}
- wrist_temp_mean: {features["wrist_temp_mean"]:.3f}
- wrist_temp_slope: {features["wrist_temp_slope"]:.3f}"""


class FeatureDescriptionInput:
    name = "feature_description"

    def __init__(self, data_dir: str | Path, pattern: str = "*_features_paperstyle.csv") -> None:
        self.data_dir = Path(data_dir)
        self.pattern = pattern

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        subject_filter = set(subjects or [])
        samples = []
        for csv_path in sorted(self.data_dir.glob(self.pattern)):
            subject = csv_path.stem.split("_")[0]
            if subject_filter and subject not in subject_filter:
                continue
            df = pd.read_csv(csv_path)
            if "subject" not in df.columns:
                df["subject"] = subject
            df = df[df["label"].isin(labels)].copy()
            for row_idx, row in df.iterrows():
                features = extract_feature_dict(row)
                samples.append(
                    Sample(
                        subject=str(row.get("subject", subject)),
                        label=int(row["label"]),
                        input_text=format_feature_block(features),
                        meta={"row_index": int(row_idx), "source": str(csv_path)},
                    )
                )
        return samples


__all__ = ["FeatureDescriptionInput", "extract_feature_dict", "format_feature_block"]
