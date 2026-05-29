from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import iqr, kurtosis, skew

from core.schema import SensorSample


ACC_FILE = "Phones_accelerometer.csv"
GYRO_FILE = "Phones_gyroscope.csv"

TARGET_ACTIVITIES = [
    "stairsdown",
    "stairsup",
]

ORIGINAL_ACTIVITY_TO_INT = {
    "bike": 0,
    "sit": 1,
    "stand": 2,
    "walk": 3,
    "stairsup": 4,
    "stairsdown": 5,
}

DEFAULT_ACTIVITY_TO_BINARY = {
    "stairsdown": 0,
    "stairsup": 1,
}

MIN_SAMPLES_PER_WINDOW = 10


def normalize_activity_label(label):
    """Normalize HHAR raw activity labels."""
    if pd.isna(label):
        return None

    text = str(label).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    label_map = {
        "bike": "bike",
        "biking": "bike",
        "sit": "sit",
        "sitting": "sit",
        "stand": "stand",
        "standing": "stand",
        "walk": "walk",
        "walking": "walk",
        "stairsup": "stairsup",
        "upstairs": "stairsup",
        "up": "stairsup",
        "walkingupstairs": "stairsup",
        "walkupstairs": "stairsup",
        "walkingup": "stairsup",
        "stairsascending": "stairsup",
        "ascendingstairs": "stairsup",
        "stairsdown": "stairsdown",
        "downstairs": "stairsdown",
        "down": "stairsdown",
        "walkingdownstairs": "stairsdown",
        "walkdownstairs": "stairsdown",
        "walkingdown": "stairsdown",
        "stairsdescending": "stairsdown",
        "descendingstairs": "stairsdown",
        "null": None,
        "nan": None,
        "none": None,
    }
    return label_map.get(text, text)


def infer_time_unit_and_convert_to_seconds(time_values):
    """Convert HHAR Creation_Time to seconds by inferring its unit."""
    t = np.asarray(time_values, dtype=np.float64)
    valid_t = np.sort(t[~np.isnan(t)])
    if len(valid_t) < 2:
        return t

    diffs = np.diff(valid_t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return t

    median_diff = np.median(diffs)
    if median_diff > 1e6:
        return t / 1e9
    if median_diff > 1:
        return t / 1e3
    return t


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    value = np.corrcoef(a, b)[0, 1]
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


def extract_time_domain_features(window_df):
    """Keep the previous HHAR RF-style time-domain feature extraction."""
    features = {}

    x = window_df["x"].values.astype(np.float64)
    y = window_df["y"].values.astype(np.float64)
    z = window_df["z"].values.astype(np.float64)
    mag = np.sqrt(x**2 + y**2 + z**2)

    for name, sig in {
        "acc_x": x,
        "acc_y": y,
        "acc_z": z,
        "acc_mag": mag,
    }.items():
        features[f"{name}_mean"] = float(np.mean(sig))
        features[f"{name}_std"] = float(np.std(sig))
        features[f"{name}_min"] = float(np.min(sig))
        features[f"{name}_max"] = float(np.max(sig))
        features[f"{name}_median"] = float(np.median(sig))
        features[f"{name}_range"] = float(np.max(sig) - np.min(sig))
        features[f"{name}_iqr"] = float(iqr(sig))
        features[f"{name}_rms"] = float(np.sqrt(np.mean(sig**2)))
        features[f"{name}_energy"] = float(np.mean(sig**2))

        if len(sig) >= 3 and np.std(sig) > 0:
            features[f"{name}_skew"] = float(skew(sig))
            features[f"{name}_kurtosis"] = float(kurtosis(sig))
        else:
            features[f"{name}_skew"] = 0.0
            features[f"{name}_kurtosis"] = 0.0

    if len(window_df) >= 3:
        features["corr_xy"] = safe_corr(x, y)
        features["corr_xz"] = safe_corr(x, z)
        features["corr_yz"] = safe_corr(y, z)
    else:
        features["corr_xy"] = 0.0
        features["corr_xz"] = 0.0
        features["corr_yz"] = 0.0

    return features


def split_by_time_gap(group_df, max_gap_sec=5.0):
    """Avoid creating windows across discontinuous recording segments."""
    group_df = group_df.sort_values("time_sec").copy()
    time_diff = group_df["time_sec"].diff().fillna(0)
    group_df["continuous_segment_id"] = (time_diff > max_gap_sec).cumsum()
    return group_df


class HHARLoader:
    """HHAR phone IMU loader for the modular 4x3x2 framework.

    This keeps the original processing basis:
    HHAR phone accelerometer/gyroscope CSV -> label normalization -> timestamp
    unit inference -> continuous-segment split -> downsample to 10 Hz tokens ->
    2-second windows with 1-second stride.

    The HHAR task is a binary HARGPT-style stair task:
    0 = walking downstairs, 1 = walking upstairs.
    """

    name = "HHAR"

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 20,
        stride_size: int = 10,
        sampling_rate: float = 10.0,
        min_samples_per_window: int = MIN_SAMPLES_PER_WINDOW,
        max_gap_sec: float = 5.0,
        label_map: dict | None = None,
        include_gyroscope: bool = True,
        max_rows: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_size = int(window_size)
        self.stride_size = int(stride_size)
        self.sampling_rate = float(sampling_rate)
        self.window_sec = self.window_size / self.sampling_rate
        self.stride_sec = self.stride_size / self.sampling_rate
        self.min_samples_per_window = int(min_samples_per_window)
        self.max_gap_sec = float(max_gap_sec)
        self.label_map = label_map or DEFAULT_ACTIVITY_TO_BINARY
        self.include_gyroscope = bool(include_gyroscope)
        self.max_rows = int(max_rows) if max_rows is not None else None

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[SensorSample]:
        df = self._load_clean_accelerometer()
        gyro_df = self._load_clean_gyroscope() if self.include_gyroscope else None
        gyro_groups = self._build_gyro_group_index(gyro_df)
        subject_filter = {str(subject) for subject in subjects} if subjects else None
        labels_to_keep = {int(label) for label in labels}

        samples: list[SensorSample] = []
        group_cols = ["user_id", "model", "device", "activity_label"]
        for group_key, group in df.groupby(group_cols, sort=True):
            subject = str(group["user_id"].iloc[0])
            if subject_filter and subject not in subject_filter:
                continue

            group = split_by_time_gap(group, max_gap_sec=self.max_gap_sec)
            gyro_group = gyro_groups.get(tuple(group_key)) if gyro_groups else None
            if gyro_group is not None:
                gyro_group = split_by_time_gap(gyro_group, max_gap_sec=self.max_gap_sec)
            for _, continuous_group in group.groupby("continuous_segment_id", sort=True):
                segment_id = int(continuous_group["continuous_segment_id"].iloc[0])
                continuous_gyro = None
                if gyro_group is not None and "continuous_segment_id" in gyro_group:
                    matching = gyro_group[gyro_group["continuous_segment_id"] == segment_id]
                    continuous_gyro = matching if not matching.empty else gyro_group
                samples.extend(
                    self._segment_continuous_group(
                        continuous_group,
                        labels_to_keep=labels_to_keep,
                        gyro_df=continuous_gyro,
                    )
                )
        return samples

    def _discover_subjects(self) -> list[str]:
        file_path = self._find_file(ACC_FILE)
        df = pd.read_csv(file_path, nrows=self.max_rows, usecols=["User", "gt"])
        df = df.dropna(subset=["User", "gt"]).copy()
        df["activity_label"] = df["gt"].apply(normalize_activity_label)
        df = df[df["activity_label"].isin(TARGET_ACTIVITIES)]
        return sorted(str(subject) for subject in df["User"].dropna().unique())

    def _build_gyro_group_index(self, gyro_df: pd.DataFrame | None) -> dict[tuple, pd.DataFrame]:
        """Pre-index and downsample gyro streams once per HHAR group."""
        if gyro_df is None or gyro_df.empty:
            return {}

        group_cols = ["user_id", "model", "device", "activity_label"]
        grouped: dict[tuple, pd.DataFrame] = {}
        for group_key, group in gyro_df.groupby(group_cols, sort=False):
            group = split_by_time_gap(group, max_gap_sec=self.max_gap_sec)
            parts = [
                self._downsample_motion_group(continuous_group)
                for _, continuous_group in group.groupby("continuous_segment_id", sort=True)
            ]
            parts = [part for part in parts if not part.empty]
            if not parts:
                continue
            grouped[group_key] = (
                pd.concat(parts, ignore_index=True)
                .sort_values("time_sec")
                .reset_index(drop=True)
            )
        return grouped

    def _load_clean_accelerometer(self) -> pd.DataFrame:
        file_path = self._find_file(ACC_FILE)
        df = pd.read_csv(file_path, nrows=self.max_rows)
        return self._clean_motion_dataframe(df, file_path=file_path, log_distribution=True)

    def _load_clean_gyroscope(self) -> pd.DataFrame | None:
        try:
            file_path = self._find_file(GYRO_FILE)
        except FileNotFoundError:
            return None
        df = pd.read_csv(file_path, nrows=self.max_rows)
        return self._clean_motion_dataframe(df, file_path=file_path, log_distribution=False)

    def _find_file(self, filename: str) -> Path:
        if self.data_dir.is_file():
            return self.data_dir

        direct = self.data_dir / filename
        if direct.exists():
            return direct

        matches = sorted(self.data_dir.rglob(filename)) if self.data_dir.exists() else []
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Cannot find HHAR file {filename} under {self.data_dir}")

    def _clean_motion_dataframe(
        self,
        df: pd.DataFrame,
        file_path: Path,
        log_distribution: bool = False,
    ) -> pd.DataFrame:
        required_cols = ["Creation_Time", "x", "y", "z", "User", "Model", "Device", "gt"]
        missing_cols = [column for column in required_cols if column not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {file_path.name}: {missing_cols}")

        out = df[required_cols].copy()
        out.rename(
            columns={
                "User": "user_id",
                "Model": "model",
                "Device": "device",
                "gt": "activity_label",
            },
            inplace=True,
        )
        raw_distribution = df["gt"].value_counts(dropna=False).to_dict()
        out["activity_label"] = out["activity_label"].apply(normalize_activity_label)
        out = out[out["activity_label"].isin(TARGET_ACTIVITIES)].copy()

        for column in ["Creation_Time", "x", "y", "z"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")

        out.dropna(
            subset=["Creation_Time", "x", "y", "z", "user_id", "model", "device", "activity_label"],
            inplace=True,
        )
        out["time_sec_raw"] = infer_time_unit_and_convert_to_seconds(out["Creation_Time"].values)
        # Keep an absolute, unit-normalized timeline. Accelerometer and gyroscope
        # are loaded from separate CSV files; subtracting each file's own minimum
        # timestamp can shift the two streams relative to each other and break
        # window matching.
        out["time_sec"] = out["time_sec_raw"]
        out["label_int"] = out["activity_label"].apply(self._map_activity_to_label)
        out["original_activity_id"] = out["activity_label"].map(ORIGINAL_ACTIVITY_TO_INT)
        out.dropna(subset=["label_int"], inplace=True)
        out["label_int"] = out["label_int"].astype(int)
        if log_distribution:
            print(f"HHAR label distribution before filtering: {raw_distribution}")
            print(
                "HHAR label distribution after binary mapping: "
                f"{out['label_int'].value_counts().sort_index().to_dict()}"
            )
        out.sort_values(by=["user_id", "model", "device", "activity_label", "time_sec"], inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out

    def _map_activity_to_label(self, activity_label) -> int | None:
        normalized = normalize_activity_label(activity_label)
        if normalized is None:
            return None
        try:
            return int(self.label_map[normalized])
        except KeyError:
            return None

    def _segment_continuous_group(
        self,
        group_df: pd.DataFrame,
        labels_to_keep: set[int],
        gyro_df: pd.DataFrame | None,
    ) -> list[SensorSample]:
        group_df = self._downsample_motion_group(group_df)
        if group_df.empty:
            return []

        gyro_downsampled = gyro_df if gyro_df is not None and not gyro_df.empty else None
        times = group_df["time_sec"].to_numpy(dtype=float)
        current_start = float(times[0])
        end_time = float(times[-1])
        samples: list[SensorSample] = []
        segment_id = int(group_df["continuous_segment_id"].iloc[0]) if "continuous_segment_id" in group_df else 0
        window_index = 0

        while current_start + self.window_sec <= end_time:
            current_end = current_start + self.window_sec
            start_idx = int(np.searchsorted(times, current_start, side="left"))
            end_idx = int(np.searchsorted(times, current_end, side="left"))
            window_df = group_df.iloc[start_idx:end_idx]
            if len(window_df) >= self.min_samples_per_window:
                label = int(window_df["label_int"].iloc[0])
                if label in labels_to_keep:
                    samples.append(
                        self._build_sample(
                            window_df=window_df,
                            label=label,
                            window_start=current_start,
                            window_end=current_end,
                            segment_id=segment_id,
                            window_index=window_index,
                            gyro_df=gyro_downsampled,
                        )
                    )
            current_start += self.stride_sec
            window_index += 1

        return samples

    def _downsample_motion_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        """Downsample one continuous HHAR IMU segment to the LLM token rate.

        The original HHAR phone streams can be around 100/200 Hz. For prompt
        inputs we follow the HARGPT-style setup and convert them to 10 Hz tokens
        before windowing. Downsampling is done by time-bin averaging within each
        continuous user/device/activity segment.
        """
        group_df = group_df.sort_values("time_sec").reset_index(drop=True)
        if group_df.empty:
            return group_df

        start_time = float(group_df["time_sec"].min())
        out = group_df.copy()
        if "continuous_segment_id" not in out.columns:
            out["continuous_segment_id"] = 0
        out["downsample_bin"] = np.floor((out["time_sec"] - start_time) * self.sampling_rate).astype(int)

        grouped = out.groupby("downsample_bin", sort=True, as_index=True)
        means = grouped[["Creation_Time", "x", "y", "z", "time_sec_raw"]].mean()
        first = grouped[
            [
                "user_id",
                "model",
                "device",
                "activity_label",
                "label_int",
                "original_activity_id",
                "continuous_segment_id",
            ]
        ].first()
        downsampled = pd.concat([means, first], axis=1)
        bins = downsampled.index.to_numpy(dtype=np.float64)
        downsampled["time_sec"] = start_time + (bins / self.sampling_rate)
        downsampled["label_int"] = downsampled["label_int"].astype(int)
        downsampled["original_activity_id"] = downsampled["original_activity_id"].astype(int)
        downsampled["continuous_segment_id"] = downsampled["continuous_segment_id"].astype(int)
        return downsampled.reset_index(drop=True)[
            [
                "Creation_Time",
                "x",
                "y",
                "z",
                "user_id",
                "model",
                "device",
                "activity_label",
                "time_sec_raw",
                "time_sec",
                "label_int",
                "original_activity_id",
                "continuous_segment_id",
            ]
        ]

    def _build_sample(
        self,
        window_df: pd.DataFrame,
        label: int,
        window_start: float,
        window_end: float,
        segment_id: int,
        window_index: int,
        gyro_df: pd.DataFrame | None,
    ) -> SensorSample:
        x = window_df["x"].to_numpy(dtype=float)
        y = window_df["y"].to_numpy(dtype=float)
        z = window_df["z"].to_numpy(dtype=float)
        acc_mag = np.sqrt(x**2 + y**2 + z**2)
        # ===== 修改: 只保留 2D "acc" + 1D "acc_mag", 不再同时存 1D acc_x/y/z =====
        # 旧实现把同一三轴信号塞了两份 (2D acc 和 3 个 1D 轴), 导致 Input/raw_data.py
        # 与 Input/embedding_alignment.py 在 HHAR 上把 ACC 三轴各打印两次, 既拉长 prompt
        # 又让 LLM 看到重复"通道"。下游 feature_description 仍能从 2D acc 自动派生出
        # x / y / z / magnitude 特征, 不受影响。
        signals = {
            "acc": np.column_stack([x, y, z]),
            "acc_mag": acc_mag,
        }

        gyro_window = self._matching_gyro_window(window_start, window_end, gyro_df)
        if gyro_window is not None and len(gyro_window) >= self.min_samples_per_window:
            gx = gyro_window["x"].to_numpy(dtype=float)
            gy = gyro_window["y"].to_numpy(dtype=float)
            gz = gyro_window["z"].to_numpy(dtype=float)
            # ===== 修改: 陀螺仪同样只保留 2D "gyro" + 1D "gyro_mag", 去掉 1D gyro_x/y/z =====
            signals.update(
                {
                    "gyro": np.column_stack([gx, gy, gz]),
                    "gyro_mag": np.sqrt(gx**2 + gy**2 + gz**2),
                }
            )

        features = extract_time_domain_features(window_df)
        sample_id = (
            f"HHAR_{window_df['user_id'].iloc[0]}_"
            f"{window_df['model'].iloc[0]}_"
            f"{window_df['device'].iloc[0]}_"
            f"{window_df['activity_label'].iloc[0]}_"
            f"seg{segment_id}_"
            f"{window_index}"
        )
        return SensorSample(
            dataset=self.name,
            subject=str(window_df["user_id"].iloc[0]),
            label=label,
            signals=signals,
            meta={
                "sample_id": str(sample_id),
                "true_label": int(label),
                "model": str(window_df["model"].iloc[0]),
                "device": str(window_df["device"].iloc[0]),
                "activity_label": str(window_df["activity_label"].iloc[0]),
                "original_activity_id": int(window_df["original_activity_id"].mode().iloc[0]),
                "window_start_sec": float(window_start),
                "window_end_sec": float(window_end),
                "continuous_segment_id": int(segment_id),
                "window_index": int(window_index),
                "n_samples": int(len(window_df)),
                "window_features": features,
            },
        )

    def _matching_gyro_window(
        self,
        window_start: float,
        window_end: float,
        gyro_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if gyro_df is None:
            return None

        times = gyro_df["time_sec"]
        start_idx = int(times.searchsorted(window_start, side="left"))
        end_idx = int(times.searchsorted(window_end, side="left"))
        if end_idx <= start_idx:
            return None
        return gyro_df.iloc[start_idx:end_idx]


__all__ = [
    "HHARLoader",
    "normalize_activity_label",
    "infer_time_unit_and_convert_to_seconds",
    "extract_time_domain_features",
    "split_by_time_gap",
]
