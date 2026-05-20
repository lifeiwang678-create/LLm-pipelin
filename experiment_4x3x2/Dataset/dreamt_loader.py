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


# ===== 修改: 新增 DREAMT timestamp 单位推断 helper, 修复"时间戳一律按秒处理"的隐患 =====
# 旧实现 _prepare_dataframe 直接对 timestamp 列做 floor(t / window_seconds), 假定 t
# 一定是秒。如果 DREAMT csv 用毫秒 / 纳秒 / 采样点索引, 会得到完全错误的 epoch 划分且不报错。
# 这里参考 HHAR loader 中的同名函数, 用相邻样本时间差的中位数推断单位。
def _infer_time_unit_and_convert_to_seconds(time_values: np.ndarray) -> np.ndarray:
    t = np.asarray(time_values, dtype=np.float64)
    valid_t = np.sort(t[~np.isnan(t)])
    if len(valid_t) < 2:
        return t
    diffs = np.diff(valid_t)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return t
    median_diff = float(np.median(diffs))
    if median_diff > 1e6:
        # 中位数差 > 1e6 视为纳秒 (典型 64Hz 数据相邻样本约 1.5e7 ns)
        return t / 1e9
    if median_diff > 1.0:
        # 1 ~ 1e6 视为毫秒 (典型 64Hz 数据相邻样本约 15.6 ms)
        return t / 1e3
    # 否则视为秒 (典型 64Hz 数据相邻样本约 0.0156 s)
    return t


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
            "p": 0,
            "prep": 0,
            "preparation": 0,
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

        raw_distribution = df[cols["label"]].value_counts(dropna=False).to_dict()
        df = self._prepare_dataframe(df, cols)
        mapped_distribution = (
            df["label_mapped"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )
        print(f"DREAMT label distribution before binary mapping ({subject}): {raw_distribution}")
        print(f"DREAMT label distribution after binary mapping ({subject}): {mapped_distribution}")
        samples: list[SensorSample] = []

        # ===== 修改: 旧实现按 floor(timestamp / window_seconds) 做非重叠 epoch,
        # stride_seconds / stride_size 配置形同虚设, 任何 overlap 设置都被静默忽略。
        # 改为按 time_sec 做真正的滑窗扫描, 让 stride 真实生效, 同时复用 _prepare_dataframe
        # 里推断好的 time_sec (已减去最小值, 从 0 开始)。 =====
        df = df.sort_values("time_sec").reset_index(drop=True)
        if df.empty:
            return samples

        window_sec = float(self.window_size) / float(self.sampling_rate)
        stride_sec = float(self.stride_size) / float(self.sampling_rate)
        if stride_sec <= 0:
            raise ValueError(
                f"DREAMT stride_size={self.stride_size} 与 sampling_rate={self.sampling_rate} "
                "推出的 stride_sec 非正, 请检查 loader_kwargs。"
            )

        start_time = float(df["time_sec"].iloc[0])
        end_time = float(df["time_sec"].iloc[-1])

        epoch_index = 0
        current_start = start_time
        # 加一个极小 epsilon 防止浮点截断把最后一个完整窗丢掉
        while current_start + window_sec <= end_time + 1e-9:
            current_end = current_start + window_sec
            group = df.loc[
                (df["time_sec"] >= current_start) & (df["time_sec"] < current_end)
            ]
            if len(group) >= self.min_samples:
                label = mode_label(group["label_mapped"])
                if not pd.isna(label):
                    label = int(label)
                    if label in labels_to_keep:
                        artifact = self._artifact_flag(group)
                        if not (self.skip_artifact_epochs and artifact):
                            signals = self._signals_from_epoch(group)
                            if signals:
                                sample_id = f"DREAMT_{subject}_{int(epoch_index)}"
                                samples.append(
                                    SensorSample(
                                        dataset=self.name,
                                        subject=subject,
                                        label=label,
                                        signals=signals,
                                        meta={
                                            "sample_id": sample_id,
                                            "true_label": int(label),
                                            "source_file": str(file_path),
                                            # epoch_id 现在是滑窗计数, 不再是 floor(t / 30)
                                            "epoch_id": int(epoch_index),
                                            "window_start": float(current_start),
                                            "window_end": float(current_end),
                                            "window_start_sec": float(current_start),
                                            "window_end_sec": float(current_end),
                                            "sampling_rate": self.sampling_rate,
                                            "window_size": self.window_size,
                                            "stride_size": self.stride_size,
                                            "n_samples": int(len(group)),
                                            "artifact_epoch": int(artifact),
                                        },
                                    )
                                )
            current_start += stride_sec
            epoch_index += 1

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
        # ===== 修改: 旧实现把 timestamp 强行当秒处理, 并用 floor 算 epoch_id,
        # 这里改为先用 _infer_time_unit_and_convert_to_seconds 推断单位 (s / ms / ns),
        # 再生成 time_sec (减去最小值, 从 0 开始) 给上层滑窗使用。
        # epoch_id 不再在这里生成, 由 _load_subject_file 的滑窗计数赋值。 =====
        if cols["timestamp"] is not None:
            raw_t = safe_numeric(out[cols["timestamp"]]).to_numpy(dtype=float)
            time_sec_raw = _infer_time_unit_and_convert_to_seconds(raw_t)
            finite_mask = np.isfinite(time_sec_raw)
            baseline = (
                float(np.nanmin(time_sec_raw[finite_mask])) if finite_mask.any() else 0.0
            )
            out["time_sec"] = time_sec_raw - baseline
        else:
            # 没有 timestamp 列时, 用行号 / sampling_rate 当兜底, 同时打印 warning 提醒数据异常
            print(
                "[DREAMT] timestamp column not found, falling back to row_index / sampling_rate. "
                "Window timing may be inaccurate if sampling is not uniform."
            )
            out["time_sec"] = np.arange(len(out), dtype=float) / float(self.sampling_rate)

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
        # ===== 修改 (Fix 5): 旧实现同时把 1D acc_x/y/z(以及 acc_x_filt/y_filt/z_filt)
        # 和 2D acc 一起塞进 signals, 导致 Input/raw_data.py 与
        # Input/embedding_alignment.py 在 DREAMT 上把 ACC 三轴各打印两次。
        # 这里只输出 2D "acc" + 1D "actigraphy" 作为运动通道, 与 HHAR 风格一致。
        # 下游 feature_description 仍能从 2D acc 自动派生 x/y/z/magnitude 特征。 =====
        signal_names = [
            "bvp",
            "bvp_filt",
            "ibi",
            "eda",
            "eda_lowpass",
            "hr",
            "skin_temp",
            "temp",
            "actigraphy",
        ]
        signals: dict[str, np.ndarray] = {}
        for name in signal_names:
            if name not in group:
                continue
            values = (
                pd.to_numeric(group[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not values.empty:
                signals[name] = values.to_numpy(dtype=float)

        # ===== 修改 (Fix 2): 旧实现对 acc_x/y/z 各自单独 dropna 后直接
        # np.column_stack, 三轴长度不等时会抛 ValueError 让整个 run 崩。
        # 这里改为只在三个轴 dropna 之后长度对齐时才拼 2D acc, 长度不齐则
        # 放弃拼接 (上游 actigraphy 仍可用), 避免运行时崩溃。 =====
        acc_axes: dict[str, np.ndarray] = {}
        for axis in ("acc_x", "acc_y", "acc_z"):
            if axis not in group:
                continue
            vals = (
                pd.to_numeric(group[axis], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not vals.empty:
                acc_axes[axis] = vals.to_numpy(dtype=float)
        if len(acc_axes) == 3:
            lengths = {len(values) for values in acc_axes.values()}
            if len(lengths) == 1:
                signals["acc"] = np.column_stack(
                    [acc_axes["acc_x"], acc_axes["acc_y"], acc_axes["acc_z"]]
                )
            # else: 三轴长度不一致, 不强行拼接 2D acc, 避免 column_stack 抛错。
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
