from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from Input.embedding_alignment import EmbeddingAlignmentInput as PromptEmbeddingAlignmentInput

from .schema import Sample


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


def zscore_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    std = np.std(x)
    if std < 1e-8:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def downsample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x)
    if len(x) <= target_len:
        return x
    idx = np.linspace(0, len(x) - 1, target_len).astype(int)
    return x[idx]


def round_list(x: np.ndarray, decimals: int = 3) -> list[float]:
    return [round(float(v), decimals) for v in x]


def get_segment(sig: np.ndarray, center_idx: int, window_sec: float, fs: int) -> np.ndarray:
    half = int(window_sec * fs / 2)
    start = max(0, center_idx - half)
    end = min(len(sig), center_idx + half)
    return sig[start:end]


def is_pure_window(labels: np.ndarray, center_idx: int, fs: int, window_sec: float) -> bool:
    half = int(window_sec * fs / 2)
    start = max(0, center_idx - half)
    end = min(len(labels), center_idx + half)
    seg = labels[start:end]
    return len(seg) > 0 and len(np.unique(seg)) == 1


def pack_1d(sig: np.ndarray, target_len: int = 60) -> list[float]:
    sig = np.asarray(sig).flatten()
    sig = downsample_to_length(zscore_safe(sig), target_len)
    return round_list(sig)


def pack_acc_xyz(acc_seg: np.ndarray, target_len: int = 40) -> dict[str, list[float]]:
    acc_seg = np.asarray(acc_seg)
    if acc_seg.ndim != 2 or acc_seg.shape[1] != 3:
        return {"x": [], "y": [], "z": []}
    return {
        "x": round_list(downsample_to_length(zscore_safe(acc_seg[:, 0]), target_len)),
        "y": round_list(downsample_to_length(zscore_safe(acc_seg[:, 1]), target_len)),
        "z": round_list(downsample_to_length(zscore_safe(acc_seg[:, 2]), target_len)),
    }


def format_raw_block(raw: dict) -> str:
    return f"""Input raw sequences:

chest_ecg = {raw["chest_ecg"]}
chest_eda = {raw["chest_eda"]}
chest_resp = {raw["chest_resp"]}
chest_acc = {json.dumps(raw["chest_acc"], ensure_ascii=False)}

wrist_bvp = {raw["wrist_bvp"]}
wrist_eda = {raw["wrist_eda"]}
wrist_temp = {raw["wrist_temp"]}
wrist_acc = {json.dumps(raw["wrist_acc"], ensure_ascii=False)}"""


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


class InputProvider(ABC):
    """Base interface for all input representations."""

    name: str

    @abstractmethod
    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        """Return samples whose input_text matches this representation."""


class NotImplementedInput(InputProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        raise NotImplementedError(
            f"Input interface '{self.name}' is reserved but not implemented yet. "
            "Implement load() to convert data into Sample objects."
        )


EmbeddingAlignmentInput = PromptEmbeddingAlignmentInput


class FeatureDescriptionInput(InputProvider):
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


class RawDataInput(InputProvider):
    name = "raw_data"

    def __init__(
        self,
        data_dir: str | Path,
        window_sec: float = 10.0,
        stride_sec: float = 15.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.window_sec = window_sec
        self.stride_sec = stride_sec

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        subject_list = list(subjects or self._discover_subjects())
        samples = []
        for subject in subject_list:
            samples.extend(self._load_subject(subject, labels))
        return samples

    def _discover_subjects(self) -> list[str]:
        return sorted(path.name for path in self.data_dir.glob("S*") if (path / f"{path.name}.pkl").exists())

    def _load_subject(self, subject: str, labels_to_keep: list[int]) -> list[Sample]:
        pkl_path = self.data_dir / subject / f"{subject}.pkl"
        if not pkl_path.exists():
            print(f"[skip] Cannot find {pkl_path}")
            return []

        with pkl_path.open("rb") as f:
            data = pickle.load(f, encoding="latin1")

        chest = data["signal"]["chest"]
        wrist = data["signal"]["wrist"]
        labels = data["label"].flatten()

        fs_c = 700
        fs_w_bvp = 64
        fs_w_acc = 32
        fs_w_eda = 4
        fs_w_temp = 4
        step = int(self.stride_sec * fs_c)
        half = int(self.window_sec * fs_c / 2)

        samples = []
        for center_idx in range(half, len(labels) - half, step):
            label_val = int(labels[center_idx])
            if label_val not in labels_to_keep:
                continue
            if not is_pure_window(labels, center_idx, fs_c, self.window_sec):
                continue

            t = center_idx / fs_c
            raw = {
                "chest_ecg": pack_1d(get_segment(chest["ECG"].flatten(), center_idx, self.window_sec, fs_c), 80),
                "chest_eda": pack_1d(get_segment(chest["EDA"].flatten(), center_idx, self.window_sec, fs_c), 40),
                "chest_resp": pack_1d(get_segment(chest["Resp"].flatten(), center_idx, self.window_sec, fs_c), 60),
                "chest_acc": pack_acc_xyz(get_segment(chest["ACC"], center_idx, self.window_sec, fs_c), 40),
                "wrist_bvp": pack_1d(
                    get_segment(wrist["BVP"].flatten(), int(t * fs_w_bvp), self.window_sec, fs_w_bvp),
                    60,
                ),
                "wrist_eda": pack_1d(
                    get_segment(wrist["EDA"].flatten(), int(t * fs_w_eda), self.window_sec, fs_w_eda),
                    40,
                ),
                "wrist_temp": pack_1d(
                    get_segment(wrist["TEMP"].flatten(), int(t * fs_w_temp), self.window_sec, fs_w_temp),
                    20,
                ),
                "wrist_acc": pack_acc_xyz(
                    get_segment(wrist["ACC"], int(t * fs_w_acc), self.window_sec, fs_w_acc),
                    40,
                ),
            }
            samples.append(
                Sample(
                    subject=subject,
                    label=label_val,
                    input_text=format_raw_block(raw),
                    meta={"time_sec": round(float(t), 3), "source": str(pkl_path)},
                )
            )
        return samples


def build_input_provider(config: dict):
    kind = str(config.get("type", "feature_description")).strip().lower()
    data_dir = config.get("data_dir", ".")
    if kind in {"feature", "feature_description", "description"}:
        return FeatureDescriptionInput(data_dir=data_dir, pattern=config.get("pattern", "*_features_paperstyle.csv"))
    if kind in {"raw", "raw_data"}:
        return RawDataInput(
            data_dir=data_dir,
            window_sec=float(config.get("window_sec", 10.0)),
            stride_sec=float(config.get("stride_sec", 15.0)),
        )
    if kind in {
        "embedding",
        "alignment",
        "embedding_alignment",
        "embedding / alignment",
        "encoded_time_series",
        "encoded-time-series",
        "encoded time series",
    }:
        return PromptEmbeddingAlignmentInput(
            dataset=config.get("dataset", config.get("dataset_name", "WESAD")),
            data_path=config.get("data_path", "sensorllm_wesad_binary_loso/fold_S2/eval_data.pkl"),
            qa_path=config.get("qa_path", "sensorllm_wesad_binary_loso/fold_S2/eval_qa.json"),
            label_map=config.get("label_map"),
            max_points=int(config.get("max_points", 256)),
            num_segments=int(config.get("num_segments", 4)),
            name=kind,
        )
    if kind in {"extra_knowledge", "knowledge", "extra knowledge"}:
        return NotImplementedInput("extra_knowledge")
    raise ValueError(f"Unknown input type: {kind}")
