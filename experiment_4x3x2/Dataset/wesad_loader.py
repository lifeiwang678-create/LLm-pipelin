from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np

from core.schema import SensorSample
from core.signal_utils import get_segment


class WESADLoader:
    name = "WESAD"

    def __init__(
        self,
        data_dir: str | Path,
        physiology_window_sec: float = 60.0,
        acc_window_sec: float = 5.0,
        stride_sec: float = 0.25,
        window_sec: float | None = None,
    ) -> None:
        """
        Paper-style WESAD windowing protocol.

        - Physiology window: 60 s
          ECG / EDA / EMG / RESP / TEMP / BVP

        - ACC window: 5 s
          chest ACC / wrist ACC

        - Stride: 0.25 s

        - A physiology window must be fully contained in one contiguous
          label segment.

        The window_sec argument is kept only for backward compatibility.
        If window_sec is provided, it overrides physiology_window_sec.
        """
        self.data_dir = Path(data_dir)
        self.physiology_window_sec = float(window_sec if window_sec is not None else physiology_window_sec)
        self.acc_window_sec = float(acc_window_sec)
        self.stride_sec = float(stride_sec)

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[SensorSample]:
        subject_list = list(subjects or self._discover_subjects())
        samples: list[SensorSample] = []

        for subject in subject_list:
            samples.extend(self._load_subject(subject, labels))

        return samples

    def _discover_subjects(self) -> list[str]:
        return sorted(
            path.name
            for path in self.data_dir.glob("S*")
            if (path / f"{path.name}.pkl").exists()
        )

    def _load_subject(self, subject: str, labels_to_keep: list[int]) -> list[SensorSample]:
        pkl_path = self.data_dir / subject / f"{subject}.pkl"
        if not pkl_path.exists():
            print(f"[skip] Cannot find {pkl_path}")
            return []

        with pkl_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")

        chest = data["signal"]["chest"]
        wrist = data["signal"]["wrist"]
        label_series = np.asarray(data["label"]).ravel()

        chest_ecg = np.asarray(chest["ECG"]).ravel()
        chest_eda = np.asarray(chest["EDA"]).ravel()
        chest_resp = np.asarray(chest["Resp"]).ravel()
        chest_acc = np.asarray(chest["ACC"])
        chest_emg = np.asarray(chest["EMG"]).ravel()
        chest_temp = np.asarray(chest["Temp"]).ravel()

        wrist_bvp = np.asarray(wrist["BVP"]).ravel()
        wrist_eda = np.asarray(wrist["EDA"]).ravel()
        wrist_temp = np.asarray(wrist["TEMP"]).ravel()
        wrist_acc = np.asarray(wrist["ACC"])

        fs_chest = 700
        fs_wrist_bvp = 64
        fs_wrist_acc = 32
        fs_wrist_eda = 4
        fs_wrist_temp = 4

        step_samples_chest = int(self.stride_sec * fs_chest)
        if step_samples_chest <= 0:
            raise ValueError("stride_sec is too small; step_samples_chest must be positive.")

        half_phys_chest = int(self.physiology_window_sec * fs_chest / 2.0)

        samples: list[SensorSample] = []
        valid_labels = tuple(int(label) for label in labels_to_keep)

        for seg_start, seg_end, label_value in self._iter_contiguous_label_segments(
            label_series,
            valid_labels=valid_labels,
        ):
            # Paper-style rule:
            # the full physiology window must stay inside one contiguous label segment.
            centers = range(
                seg_start + half_phys_chest,
                seg_end - half_phys_chest,
                step_samples_chest,
            )

            for center_idx in centers:
                time_sec = center_idx / fs_chest

                chest_acc_center = center_idx
                chest_phys_center = center_idx

                wrist_acc_center = int(time_sec * fs_wrist_acc)
                wrist_bvp_center = int(time_sec * fs_wrist_bvp)
                wrist_eda_center = int(time_sec * fs_wrist_eda)
                wrist_temp_center = int(time_sec * fs_wrist_temp)

                signals = {
                    # ACC: 5 s paper-style window
                    "chest_acc": get_segment(
                        chest_acc,
                        chest_acc_center,
                        self.acc_window_sec,
                        fs_chest,
                    ).copy(),
                    "wrist_acc": get_segment(
                        wrist_acc,
                        wrist_acc_center,
                        self.acc_window_sec,
                        fs_wrist_acc,
                    ).copy(),

                    # Physiology: 60 s paper-style window
                    "chest_ecg": get_segment(
                        chest_ecg,
                        chest_phys_center,
                        self.physiology_window_sec,
                        fs_chest,
                    ).copy(),
                    "chest_eda": get_segment(
                        chest_eda,
                        chest_phys_center,
                        self.physiology_window_sec,
                        fs_chest,
                    ).copy(),
                    "chest_resp": get_segment(
                        chest_resp,
                        chest_phys_center,
                        self.physiology_window_sec,
                        fs_chest,
                    ).copy(),
                    "chest_emg": get_segment(
                        chest_emg,
                        chest_phys_center,
                        self.physiology_window_sec,
                        fs_chest,
                    ).copy(),
                    "chest_temp": get_segment(
                        chest_temp,
                        chest_phys_center,
                        self.physiology_window_sec,
                        fs_chest,
                    ).copy(),
                    "wrist_bvp": get_segment(
                        wrist_bvp,
                        wrist_bvp_center,
                        self.physiology_window_sec,
                        fs_wrist_bvp,
                    ).copy(),
                    "wrist_eda": get_segment(
                        wrist_eda,
                        wrist_eda_center,
                        self.physiology_window_sec,
                        fs_wrist_eda,
                    ).copy(),
                    "wrist_temp": get_segment(
                        wrist_temp,
                        wrist_temp_center,
                        self.physiology_window_sec,
                        fs_wrist_temp,
                    ).copy(),
                }

                samples.append(
                    SensorSample(
                        dataset=self.name,
                        subject=subject,
                        label=int(label_value),
                        signals=signals,
                        meta={
                            "time_sec": round(float(time_sec), 3),
                            "center_index": int(center_idx),
                            "source": str(pkl_path),
                            "protocol": "paper_style",
                            "physiology_window_sec": float(self.physiology_window_sec),
                            "acc_window_sec": float(self.acc_window_sec),
                            "stride_sec": float(self.stride_sec),
                        },
                    )
                )

        return samples

    @staticmethod
    def _iter_contiguous_label_segments(
        labels: np.ndarray,
        valid_labels: tuple[int, ...],
    ):
        """
        Yield contiguous label segments.

        This follows the paper-style script:
        use continuous baseline / stress / amusement segments,
        without 30 s margin and without baseline normalization.
        """
        labels = np.asarray(labels).ravel()

        start = None
        current_label = None

        for index, label in enumerate(labels):
            label = int(label)

            if label in valid_labels:
                if start is None:
                    start = index
                    current_label = label
                elif label != current_label:
                    yield start, index, int(current_label)
                    start = index
                    current_label = label
            else:
                if start is not None:
                    yield start, index, int(current_label)
                    start = None
                    current_label = None

        if start is not None:
            yield start, len(labels), int(current_label)


__all__ = ["WESADLoader"]
