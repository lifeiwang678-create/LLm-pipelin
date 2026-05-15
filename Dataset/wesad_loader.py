from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from core.schema import SensorSample
from core.signal_utils import get_segment, is_pure_window


class WESADLoader:
    name = "WESAD"

    def __init__(
        self,
        data_dir: str | Path,
        window_sec: float = 10.0,
        stride_sec: float = 15.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_sec = window_sec
        self.stride_sec = stride_sec

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[SensorSample]:
        subject_list = list(subjects or self._discover_subjects())
        samples: list[SensorSample] = []
        for subject in subject_list:
            samples.extend(self._load_subject(subject, labels))
        return samples

    def _discover_subjects(self) -> list[str]:
        return sorted(path.name for path in self.data_dir.glob("S*") if (path / f"{path.name}.pkl").exists())

    def _load_subject(self, subject: str, labels_to_keep: list[int]) -> list[SensorSample]:
        pkl_path = self.data_dir / subject / f"{subject}.pkl"
        if not pkl_path.exists():
            print(f"[skip] Cannot find {pkl_path}")
            return []

        with pkl_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")

        chest = data["signal"]["chest"]
        wrist = data["signal"]["wrist"]
        label_series = np.asarray(data["label"]).flatten()

        fs_chest = 700
        fs_wrist_bvp = 64
        fs_wrist_acc = 32
        fs_wrist_eda = 4
        fs_wrist_temp = 4

        step = int(self.stride_sec * fs_chest)
        half = int(self.window_sec * fs_chest / 2)
        samples: list[SensorSample] = []

        for center_idx in range(half, len(label_series) - half, step):
            label = int(label_series[center_idx])
            if label not in labels_to_keep:
                continue
            if not is_pure_window(label_series, center_idx, fs_chest, self.window_sec):
                continue

            time_sec = center_idx / fs_chest
            signals = {
                "chest_ecg": get_segment(chest["ECG"].flatten(), center_idx, self.window_sec, fs_chest),
                "chest_eda": get_segment(chest["EDA"].flatten(), center_idx, self.window_sec, fs_chest),
                "chest_resp": get_segment(chest["Resp"].flatten(), center_idx, self.window_sec, fs_chest),
                "chest_acc": get_segment(chest["ACC"], center_idx, self.window_sec, fs_chest),
                "chest_emg": get_segment(chest["EMG"].flatten(), center_idx, self.window_sec, fs_chest),
                "chest_temp": get_segment(chest["Temp"].flatten(), center_idx, self.window_sec, fs_chest),
                "wrist_bvp": get_segment(
                    wrist["BVP"].flatten(),
                    int(time_sec * fs_wrist_bvp),
                    self.window_sec,
                    fs_wrist_bvp,
                ),
                "wrist_eda": get_segment(
                    wrist["EDA"].flatten(),
                    int(time_sec * fs_wrist_eda),
                    self.window_sec,
                    fs_wrist_eda,
                ),
                "wrist_temp": get_segment(
                    wrist["TEMP"].flatten(),
                    int(time_sec * fs_wrist_temp),
                    self.window_sec,
                    fs_wrist_temp,
                ),
                "wrist_acc": get_segment(
                    wrist["ACC"],
                    int(time_sec * fs_wrist_acc),
                    self.window_sec,
                    fs_wrist_acc,
                ),
            }
            samples.append(
                SensorSample(
                    dataset=self.name,
                    subject=subject,
                    label=label,
                    signals=signals,
                    meta={
                        "time_sec": round(float(time_sec), 3),
                        "center_index": int(center_idx),
                        "source": str(pkl_path),
                        "window_sec": float(self.window_sec),
                        "stride_sec": float(self.stride_sec),
                    },
                )
            )
        return samples


__all__ = ["WESADLoader"]
