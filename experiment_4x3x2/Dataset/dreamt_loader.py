from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from core.schema import SensorSample


DEFAULT_LABEL_MAP = {
    "baseline": 1,
    "neutral": 1,
    "relax": 1,
    "stress": 2,
    "stressed": 2,
    "amusement": 3,
    "happy": 3,
}


class DREAMTLoader:
    name = "DREAMT"

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 128,
        stride_size: int = 64,
        label_map: dict | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride_size = stride_size
        self.label_map = label_map or DEFAULT_LABEL_MAP

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[SensorSample]:
        csv_paths = sorted(self.data_dir.glob("*.csv"))
        if not csv_paths:
            print(f"[skip] No DREAMT CSV files found in {self.data_dir}")
            return []

        samples: list[SensorSample] = []
        for csv_path in csv_paths:
            samples.extend(self._load_csv(csv_path, set(subjects or []), set(labels)))
        return samples

    def _load_csv(self, csv_path: Path, subject_filter: set[str], labels_to_keep: set[int]) -> list[SensorSample]:
        df = pd.read_csv(csv_path)
        subject_col = _first_existing(df, ["subject", "participant", "participant_id", "subj_id"])
        label_col = _first_existing(df, ["label", "state", "condition", "class"])
        if label_col is None:
            raise ValueError(f"DREAMT file {csv_path} does not contain a label/state column.")

        time_col = _first_existing(df, ["timestamp", "time", "datetime"])
        if time_col:
            df = df.sort_values(time_col)

        if subject_col is None:
            df["_subject"] = csv_path.stem
            subject_col = "_subject"

        sensor_cols = _numeric_sensor_columns(df, exclude={subject_col, label_col, time_col})
        samples: list[SensorSample] = []
        for subject, subject_df in df.groupby(subject_col, sort=True):
            subject_name = str(subject)
            if subject_filter and subject_name not in subject_filter:
                continue
            samples.extend(self._slice_subject(subject_df, subject_name, label_col, sensor_cols, labels_to_keep, csv_path))
        return samples

    def _slice_subject(
        self,
        df: pd.DataFrame,
        subject: str,
        label_col: str,
        sensor_cols: list[str],
        labels_to_keep: set[int],
        source: Path,
    ) -> list[SensorSample]:
        samples: list[SensorSample] = []
        for start in range(0, max(len(df) - self.window_size + 1, 0), self.stride_size):
            window = df.iloc[start : start + self.window_size]
            mapped_labels = [_map_label(value, self.label_map) for value in window[label_col].tolist()]
            mapped_labels = [value for value in mapped_labels if value is not None]
            if not mapped_labels:
                continue
            label = Counter(mapped_labels).most_common(1)[0][0]
            if label not in labels_to_keep:
                continue
            signals = {col: window[col].to_numpy(dtype=np.float32) for col in sensor_cols}
            samples.append(
                SensorSample(
                    dataset=self.name,
                    subject=subject,
                    label=int(label),
                    signals=signals,
                    meta={"source": str(source), "row_start": int(start), "row_end": int(start + self.window_size)},
                )
            )
        return samples


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _numeric_sensor_columns(df: pd.DataFrame, exclude: set[str | None]) -> list[str]:
    excluded = {col for col in exclude if col is not None}
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _map_label(value, label_map: dict) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    return label_map.get(str(value).strip().lower())


__all__ = ["DREAMTLoader"]
