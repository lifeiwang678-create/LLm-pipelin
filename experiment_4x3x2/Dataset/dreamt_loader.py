from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import signal

from core.schema import SensorSample


warnings.filterwarnings("ignore")


def norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def detect_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {column: norm_name(column) for column in df.columns}
    for pattern in candidates:
        for column, name in normalized.items():
            if re.search(pattern, name):
                return column
    return None


def subject_id_from_path(path: Path) -> str:
    match = re.search(r"S\d+", path.stem, flags=re.IGNORECASE)
    return match.group(0).upper() if match else path.stem


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def mode_label(values: pd.Series):
    valid = values.dropna()
    if valid.empty:
        return np.nan
    return valid.mode().iloc[0]


def butter_bandpass(x: np.ndarray, low: float, high: float, fs: int, order: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.sum(~np.isnan(x)) < max(20, order * 5):
        return x
    x = pd.Series(x).interpolate(limit_direction="both").ffill().bfill().values
    nyquist = 0.5 * fs
    try:
        b, a = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass")
        return signal.filtfilt(b, a, x)
    except Exception:
        return x


def butter_lowpass(x: np.ndarray, cutoff: float, fs: int, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.sum(~np.isnan(x)) < max(20, order * 5):
        return x
    x = pd.Series(x).interpolate(limit_direction="both").ffill().bfill().values
    try:
        b, a = signal.butter(order, cutoff / (0.5 * fs), btype="lowpass")
        return signal.filtfilt(b, a, x)
    except Exception:
        return x


def cheby2_bandpass(x: np.ndarray, low: float, high: float, fs: int, order: int = 4, rs: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.sum(~np.isnan(x)) < max(20, order * 5):
        return x
    x = pd.Series(x).interpolate(limit_direction="both").ffill().bfill().values
    nyquist = 0.5 * fs
    try:
        b, a = signal.cheby2(order, rs, [low / nyquist, high / nyquist], btype="bandpass")
        return signal.filtfilt(b, a, x)
    except Exception:
        return x


def estimate_snr_db(x: np.ndarray, fs: int, low: float = 0.5, high: float = 20.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < fs * 5:
        return np.nan
    try:
        f, pxx = signal.welch(x, fs=fs, nperseg=min(len(x), fs * 8))
        in_band = (f >= low) & (f <= high)
        signal_power = np.trapz(pxx[in_band], f[in_band])
        noise_power = np.trapz(pxx[~in_band], f[~in_band])
        if signal_power <= 0 or noise_power <= 0:
            return np.nan
        return float(10 * np.log10(signal_power / noise_power))
    except Exception:
        return np.nan


class DREAMTLoader:
    """DREAMT raw 64 Hz CSV loader for the modular LLM experiment framework.

    This keeps the previous DREAMT preprocessing basis: raw 64 Hz files,
    30-second epoch segmentation, Sleep/Wake label mapping, optional filtering,
    and lightweight artifact metadata. It intentionally does not run the old
    LightGBM/SMOTE/5-fold baseline; model training belongs outside the 4x3x2
    LLM input pipeline.
    """

    name = "DREAMT"

    def __init__(
        self,
        data_dir: str | Path,
        sampling_rate: int = 64,
        epoch_seconds: float = 30.0,
        stride_seconds: float | None = None,
        window_size: int | None = None,
        stride_size: int | None = None,
        min_epoch_fraction: float = 0.5,
        label_map: dict[str, int] | None = None,
        skip_artifact_epochs: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sampling_rate = int(sampling_rate)
        self.window_size = int(window_size or round(self.sampling_rate * float(epoch_seconds)))
        if stride_size is not None:
            self.stride_size = int(stride_size)
        elif stride_seconds is not None:
            self.stride_size = int(round(self.sampling_rate * float(stride_seconds)))
        else:
            self.stride_size = self.window_size
        self.min_samples = max(1, int(self.window_size * float(min_epoch_fraction)))
        self.label_map = label_map or {
            "wake": 0,
            "w": 0,
            "awake": 0,
            "wakefulness": 0,
            "sleep": 1,
            "r": 1,
            "rem": 1,
            "remsleep": 1,
            "nrem": 1,
            "nonrem": 1,
            "nonremsleep": 1,
            "n1": 1,
            "n2": 1,
            "n3": 1,
            "stage1": 1,
            "stage2": 1,
            "stage3": 1,
            "s1": 1,
            "s2": 1,
            "s3": 1,
            "0": 0,
            "1": 1,
        }
        self.skip_artifact_epochs = bool(skip_artifact_epochs)

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[SensorSample]:
        files = self._select_files(subjects)
        if not files:
            raise FileNotFoundError(
                f"No DREAMT raw CSV files found under {self.data_dir}. "
                "Expected files like data_64Hz/S099_whole_df.csv."
            )

        labels_to_keep = {int(label) for label in labels}
        samples: list[SensorSample] = []
        for file_path in files:
            samples.extend(self._load_subject_file(file_path, labels_to_keep))
        return samples

    def _select_files(self, subjects: Iterable[str] | None) -> list[Path]:
        files_by_subject = {subject_id_from_path(path): path for path in self._discover_files()}
        if subjects:
            selected = []
            for subject in subjects:
                key = str(subject).upper()
                if key in files_by_subject:
                    selected.append(files_by_subject[key])
                else:
                    print(f"[skip] Cannot find DREAMT subject {subject} under {self.data_dir}")
            return selected
        return [files_by_subject[key] for key in sorted(files_by_subject)]

    def _discover_files(self) -> list[Path]:
        if self.data_dir.is_file():
            return [self.data_dir]

        search_roots = [self.data_dir]
        data_64hz = self.data_dir / "data_64Hz"
        if data_64hz.exists():
            search_roots.insert(0, data_64hz)

        files: list[Path] = []
        for root in search_roots:
            files.extend(sorted(root.glob("S*_whole_df.csv")))
            files.extend(sorted(root.glob("*_whole_df.csv")))
        if not files and self.data_dir.exists():
            files.extend(sorted(self.data_dir.rglob("*_whole_df.csv")))
        return sorted(set(files))

    def _load_subject_file(self, file_path: Path, labels_to_keep: set[int]) -> list[SensorSample]:
        subject = subject_id_from_path(file_path)
        df = pd.read_csv(file_path)
        cols = self._detect_columns(df)
        if cols["label"] is None:
            raise ValueError(f"Cannot find DREAMT sleep-stage label column in {file_path.name}.")

        df = self._prepare_dataframe(df, cols)
        samples: list[SensorSample] = []

        for epoch_id, group in df.groupby("epoch_id", sort=True):
            if pd.isna(epoch_id) or len(group) < self.min_samples:
                continue

            label = mode_label(group["label_mapped"])
            if pd.isna(label):
                continue
            label = int(label)
            if label not in labels_to_keep:
                continue

            artifact = self._artifact_flag(group)
            if self.skip_artifact_epochs and artifact:
                continue

            signals = self._signals_from_epoch(group)
            if not signals:
                continue

            samples.append(
                SensorSample(
                    dataset=self.name,
                    subject=subject,
                    label=label,
                    signals=signals,
                    meta={
                        "source_file": str(file_path),
                        "epoch_id": int(epoch_id),
                        "sampling_rate": self.sampling_rate,
                        "window_size": self.window_size,
                        "stride_size": self.stride_size,
                        "n_samples": int(len(group)),
                        "artifact_epoch": int(artifact),
                    },
                )
            )

        return samples

    def _detect_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        return {
            "timestamp": detect_col(df, [r"^timestamp$", r"^time$"]),
            "label": detect_col(df, [r"sleep.*stage", r"stage", r"label", r"annotation"]),
            "bvp": detect_col(df, [r"^bvp$", r"blood.*volume"]),
            "ibi": detect_col(df, [r"^ibi$", r"inter.*beat"]),
            "eda": detect_col(df, [r"^eda$", r"electrodermal", r"gsr"]),
            "temp": detect_col(df, [r"^temp$", r"temperature", r"skin.*temp"]),
            "hr": detect_col(df, [r"^hr$", r"heart.*rate"]),
            "acc_x": detect_col(df, [r"acc.*x", r"^x$"]),
            "acc_y": detect_col(df, [r"acc.*y", r"^y$"]),
            "acc_z": detect_col(df, [r"acc.*z", r"^z$"]),
        }

    def _prepare_dataframe(self, df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
        out = df.copy()
        if cols["timestamp"] is not None:
            timestamp = safe_numeric(out[cols["timestamp"]])
            out["epoch_id"] = np.floor(timestamp / float(self.window_size / self.sampling_rate)).astype("Int64")
        else:
            out["epoch_id"] = np.arange(len(out)) // self.window_size

        for key, column in cols.items():
            if key in {"timestamp", "label"} or column is None:
                continue
            out[key] = safe_numeric(out[column])

        out["label_mapped"] = out[cols["label"]].apply(self._map_sleep_wake_label)

        for axis in ["acc_x", "acc_y", "acc_z"]:
            if axis in out:
                out[f"{axis}_filt"] = butter_bandpass(
                    out[axis].values,
                    low=3.0,
                    high=11.0,
                    fs=self.sampling_rate,
                    order=5,
                )

        if "temp" in out:
            out["skin_temp"] = out["temp"].clip(lower=31, upper=40)
        if "bvp" in out:
            out["bvp_filt"] = cheby2_bandpass(out["bvp"].values, low=0.5, high=20.0, fs=self.sampling_rate)
        if "eda" in out:
            out["eda_lowpass"] = butter_lowpass(out["eda"].values, cutoff=1.0, fs=self.sampling_rate, order=4)
        if all(axis in out for axis in ["acc_x", "acc_y", "acc_z"]):
            out["actigraphy"] = np.sqrt(out["acc_x"] ** 2 + out["acc_y"] ** 2 + out["acc_z"] ** 2)

        return out

    def _map_sleep_wake_label(self, value) -> float:
        if pd.isna(value):
            return np.nan
        key = str(value).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if key in {"", "nan", "none", "null", "missing", "nolabel", "unlabeled", "?"}:
            return np.nan
        return self.label_map.get(key, np.nan)

    def _signals_from_epoch(self, group: pd.DataFrame) -> dict[str, np.ndarray]:
        signal_names = [
            "bvp",
            "bvp_filt",
            "ibi",
            "eda",
            "eda_lowpass",
            "hr",
            "skin_temp",
            "temp",
            "acc_x",
            "acc_y",
            "acc_z",
            "acc_x_filt",
            "acc_y_filt",
            "acc_z_filt",
            "actigraphy",
        ]
        signals = {}
        for name in signal_names:
            if name not in group:
                continue
            values = pd.to_numeric(group[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not values.empty:
                signals[name] = values.to_numpy(dtype=float)
        if all(axis in signals for axis in ["acc_x", "acc_y", "acc_z"]):
            signals["acc"] = np.column_stack([signals["acc_x"], signals["acc_y"], signals["acc_z"]])
        return signals

    def _artifact_flag(self, group: pd.DataFrame) -> bool:
        activity_bad = False
        if "actigraphy" in group:
            activity_index = float(np.nanvar(group["actigraphy"].values))
            activity_bad = activity_index > 0.4125

        bvp_bad_range = False
        snr_bad = False
        if "bvp" in group:
            bvp = pd.to_numeric(group["bvp"], errors="coerce")
            bvp_bad_range = bool(((bvp < -500) | (bvp > 500)).any())
            snr = estimate_snr_db(bvp.values, fs=self.sampling_rate)
            snr_bad = False if pd.isna(snr) else bool(snr < 10)

        return bool(activity_bad or bvp_bad_range or snr_bad)


__all__ = ["DREAMTLoader"]
