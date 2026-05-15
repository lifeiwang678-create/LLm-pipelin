from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.schema import LLMSample, Sample, SensorSample


@dataclass(frozen=True)
class ChannelMetadata:
    channel: str
    sensor_type: str
    body_location: str
    sampling_rate: float | None = None
    axis_index: int | None = None


class EmbeddingAlignmentInput:
    """SensorLLM-inspired prompt-compatible encoded time-series input.

    This is not a full embedding-level SensorLLM implementation. It does not
    train projectors, replace token embeddings, load Chronos, or modify the LLM
    forward pass. It adapts SensorLLM's channel-aware alignment idea into plain
    text blocks that can be used by the existing prompt pipeline.
    """

    name = "embedding_alignment"
    aliases = {"embedding_alignment", "encoded_time_series"}

    def __init__(
        self,
        dataset: str | None = "WESAD",
        data_path: str | Path | None = None,
        qa_path: str | Path | None = None,
        label_map: dict[str, int] | None = None,
        channel_metadata: dict[str, ChannelMetadata | dict] | None = None,
        max_points: int = 256,
        num_segments: int = 4,
        trend_threshold: float = 0.25,
        fluctuation_low_threshold: float = 0.08,
        fluctuation_high_threshold: float = 0.35,
        periodicity_threshold: float = 0.35,
        strong_periodicity_threshold: float = 0.60,
        peak_z_threshold: float = 2.5,
        sudden_change_z_threshold: float = 2.5,
        include_qa: bool = True,
        include_supporting_stats: bool = True,
        name: str | None = None,
    ) -> None:
        self.name = self._canonical_input_name(name or "embedding_alignment")
        self.canonical_name = "embedding_alignment"
        self.dataset = dataset or "WESAD"
        self.data_path = Path(data_path) if data_path else None
        self.qa_path = Path(qa_path) if qa_path else None
        self.label_map = label_map or {
            "Non-stress": 1,
            "Stress": 2,
            "Baseline": 1,
            "Amusement": 3,
        }
        self.channel_metadata = self._load_channel_metadata(self.dataset, channel_metadata)
        self.max_points = int(max_points)
        self.num_segments = max(1, int(num_segments))
        self.trend_threshold = float(trend_threshold)
        self.fluctuation_low_threshold = float(fluctuation_low_threshold)
        self.fluctuation_high_threshold = float(fluctuation_high_threshold)
        self.periodicity_threshold = float(periodicity_threshold)
        self.strong_periodicity_threshold = float(strong_periodicity_threshold)
        self.peak_z_threshold = float(peak_z_threshold)
        self.sudden_change_z_threshold = float(sudden_change_z_threshold)
        self.include_qa = bool(include_qa)
        self.include_supporting_stats = bool(include_supporting_stats)

    def build_input(self, sample: SensorSample | dict) -> str:
        dataset = self._sample_dataset(sample)
        metadata = self._load_channel_metadata(dataset, None)
        signals = self._extract_signal_container(sample)
        blocks = []

        for raw_name, values in signals.items():
            for channel_name, channel_values, channel_meta in self._expand_channel(raw_name, values, metadata):
                arr = self._to_1d_numeric_array(channel_values)
                if len(arr) == 0:
                    continue
                blocks.append(self._format_channel_block(channel_name, arr, channel_meta))

        header = [
            "Input representation: Encoded Time-series",
            "This input describes channel-aware and segment-level temporal patterns.",
            "The ground-truth label is not included.",
            "",
        ]
        if not blocks:
            blocks = ["No valid numeric sensor channel was available for encoded time-series analysis."]
        footer = [
            "",
            "Task:",
            "Based only on the encoded time-series representation above, predict the class label.",
        ]
        return "\n".join([*header, *blocks, *footer])

    def transform(self, sample: SensorSample | dict) -> LLMSample:
        meta = dict(getattr(sample, "meta", {}) if not isinstance(sample, dict) else sample.get("meta", {}))
        meta["input_type"] = self.name
        meta["input_canonical_type"] = self.canonical_name
        return LLMSample(
            dataset=self._sample_dataset(sample),
            subject=self._sample_subject(sample),
            label=self._sample_label(sample),
            input_text=self.build_input(sample),
            meta=meta,
        )

    def transform_all(self, samples: list[SensorSample]) -> list[LLMSample]:
        return [self.transform(sample) for sample in samples]

    def load(self, subjects: Iterable[str] | None, labels: list[int]) -> list[Sample]:
        """Compatibility path for older config-style runners.

        New experiments should use Dataset loaders plus transform_all(). This
        method remains so older JSON configs that reference data_path/qa_path do
        not break.
        """
        if self.qa_path is None or not self.qa_path.exists():
            raise FileNotFoundError("EmbeddingAlignmentInput.load() requires an existing qa_path.")

        legacy_data = self._load_legacy_data()
        subject_filter = set(subjects or [])
        with self.qa_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        qa_items = payload.get("dataset", payload if isinstance(payload, list) else [])

        samples = []
        for item in qa_items:
            subject = str(item.get("subject", ""))
            if subject_filter and subject not in subject_filter:
                continue

            qa_pair = item.get("qa_pair", {})
            label_text = qa_pair.get("A", item.get("binary_label", item.get("original_3class_label", "")))
            label = int(self.label_map.get(label_text, item.get("majority_label_original", 0)))
            if label not in labels:
                continue

            signals = self._legacy_signals_for_item(item, legacy_data)
            sensor_sample = SensorSample(
                dataset=self.dataset,
                subject=subject,
                label=label,
                signals=signals,
                meta={
                    "data_path": str(self.data_path) if self.data_path else "",
                    "qa_path": str(self.qa_path),
                    "data_index": item.get("index"),
                    "question": qa_pair.get("Q", ""),
                    "source": str(self.qa_path),
                    "local_index": item.get("local_index"),
                    "start_index": item.get("start_index"),
                    "end_index": item.get("end_index"),
                },
            )
            samples.append(self.transform(sensor_sample))
        return samples

    def _load_channel_metadata(
        self,
        dataset: str | None,
        custom_metadata: dict[str, ChannelMetadata | dict] | None,
    ) -> dict[str, ChannelMetadata]:
        normalized = self._normalize_dataset_name(dataset)
        if normalized != "WESAD":
            raise ValueError(f"Unsupported dataset for encoded time-series input: {dataset}")

        metadata = {
            "chest_acc_x": ChannelMetadata("ACC_X", "accelerometer", "chest", 700.0, 0),
            "chest_acc_y": ChannelMetadata("ACC_Y", "accelerometer", "chest", 700.0, 1),
            "chest_acc_z": ChannelMetadata("ACC_Z", "accelerometer", "chest", 700.0, 2),
            "chest_eda": ChannelMetadata("EDA", "electrodermal activity", "chest", 700.0),
            "chest_temp": ChannelMetadata("TEMP", "temperature", "chest", 700.0),
            "chest_ecg": ChannelMetadata("ECG", "electrocardiography", "chest", 700.0),
            "chest_emg": ChannelMetadata("EMG", "electromyography", "chest", 700.0),
            "chest_resp": ChannelMetadata("RESP", "respiration", "chest", 700.0),
            "wrist_acc_x": ChannelMetadata("ACC_X", "accelerometer", "wrist", 32.0, 0),
            "wrist_acc_y": ChannelMetadata("ACC_Y", "accelerometer", "wrist", 32.0, 1),
            "wrist_acc_z": ChannelMetadata("ACC_Z", "accelerometer", "wrist", 32.0, 2),
            "wrist_eda": ChannelMetadata("EDA", "electrodermal activity", "wrist", 4.0),
            "wrist_temp": ChannelMetadata("TEMP", "temperature", "wrist", 4.0),
            "wrist_bvp": ChannelMetadata("BVP", "blood volume pulse", "wrist", 64.0),
        }
        if custom_metadata:
            for key, value in custom_metadata.items():
                metadata[self._normalize_key(key)] = self._coerce_metadata(value)
        return metadata

    def _expand_channel(
        self,
        raw_name: str,
        values: Any,
        metadata: dict[str, ChannelMetadata],
    ) -> list[tuple[str, Any, ChannelMetadata]]:
        key = self._normalize_key(raw_name)
        arr = np.asarray(values, dtype=object)

        if key in {"chest_acc", "wrist_acc", "acc"} and arr.ndim == 2 and arr.shape[1] >= 3:
            location = "chest" if key.startswith("chest") else "wrist" if key.startswith("wrist") else "unknown"
            prefix = f"{location}_acc" if location != "unknown" else "acc"
            expanded = []
            for axis, idx in zip(["x", "y", "z"], [0, 1, 2]):
                meta_key = f"{prefix}_{axis}"
                meta = metadata.get(meta_key) or ChannelMetadata(f"ACC_{axis.upper()}", "accelerometer", location, None, idx)
                expanded.append((meta.channel, arr[:, idx], meta))
            return expanded

        meta = metadata.get(key) or self._infer_channel_metadata(raw_name)
        if meta is None:
            return []
        return [(meta.channel, values, meta)]

    def _format_channel_block(self, channel_name: str, values: np.ndarray, meta: ChannelMetadata) -> str:
        downsampled = self._downsample(values)
        normalized = self._normalize(downsampled)
        global_trend = self._detect_global_trend(normalized)
        segment_trends = self._detect_segment_trends(normalized)
        fluctuation = self._estimate_fluctuation(normalized)
        periodicity = self._estimate_periodicity(normalized)
        peaks = self._detect_peaks_and_changes(normalized)
        cue = self._generate_dynamic_cue(global_trend, fluctuation, periodicity, peaks)
        tag = self._channel_tag(meta)
        answer = self._natural_language_answer(global_trend, segment_trends, fluctuation, periodicity, peaks)

        lines = [
            f"<{tag}>",
            f"Channel: {channel_name}",
            f"Sensor type: {meta.sensor_type}",
            f"Body location: {meta.body_location}",
            f"Sampling rate: {self._format_sampling_rate(meta.sampling_rate)}",
            "",
        ]
        if self.include_qa:
            lines.extend([
                "Question: What is the temporal pattern of this sensor signal?",
                "Answer:",
                answer,
                "",
            ])
        lines.extend([
            "Encoded temporal pattern:",
            f"- Global trend: {global_trend}",
            f"- Segment-level trend: {self._format_segment_trends(segment_trends)}",
            f"- Fluctuation level: {fluctuation}",
            f"- Periodicity: {periodicity}",
            f"- Sudden changes or peaks: {peaks['description']}",
            f"- Dynamic cue: {cue}",
        ])
        if self.include_supporting_stats:
            stats = self._supporting_stats(values)
            lines.extend([
                "",
                "Supporting statistics:",
                (
                    f"mean={stats['mean']:.4g}, std={stats['std']:.4g}, "
                    f"min={stats['min']:.4g}, max={stats['max']:.4g}"
                ),
            ])
        lines.append(f"</{tag}>")
        return "\n".join(lines)

    def _to_1d_numeric_array(self, values: Any) -> np.ndarray:
        if values is None:
            return np.asarray([], dtype=float)
        try:
            arr = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            flattened = np.asarray(values, dtype=object).ravel()
            numeric = []
            for item in flattened:
                try:
                    numeric.append(float(item))
                except (TypeError, ValueError):
                    continue
            arr = np.asarray(numeric, dtype=float)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        arr = arr[np.isfinite(arr)]
        return arr.astype(float)

    def _downsample(self, values: np.ndarray) -> np.ndarray:
        if len(values) <= self.max_points:
            return values
        idx = np.linspace(0, len(values) - 1, self.max_points).astype(int)
        return values[idx]

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values
        std = float(np.std(values))
        if std < 1e-8:
            return np.zeros_like(values, dtype=float)
        return (values - float(np.mean(values))) / std

    def _split_segments(self, values: np.ndarray) -> list[np.ndarray]:
        if len(values) == 0:
            return []
        return [segment for segment in np.array_split(values, min(self.num_segments, len(values))) if len(segment)]

    def _detect_global_trend(self, values: np.ndarray) -> str:
        if len(values) < 3 or float(np.std(values)) < 1e-8:
            return "stable"
        edge = max(1, len(values) // 5)
        diff = float(np.mean(values[-edge:]) - np.mean(values[:edge]))
        if diff > self.trend_threshold:
            return "increasing"
        if diff < -self.trend_threshold:
            return "decreasing"
        return "stable"

    def _detect_segment_trends(self, values: np.ndarray) -> list[str]:
        return [self._detect_global_trend(segment) for segment in self._split_segments(values)]

    def _estimate_fluctuation(self, values: np.ndarray) -> str:
        if len(values) < 3:
            return "unclear"
        score = float(np.std(np.diff(values)))
        if score < self.fluctuation_low_threshold:
            return "low"
        if score > self.fluctuation_high_threshold:
            return "high"
        return "moderate"

    def _estimate_periodicity(self, values: np.ndarray) -> str:
        if len(values) < 8 or float(np.std(values)) < 1e-8:
            return "weak or unclear"
        t = np.arange(len(values), dtype=float)
        try:
            slope, intercept = np.polyfit(t, values, 1)
            x = values - (slope * t + intercept)
        except Exception:
            x = values - np.mean(values)
        if float(np.std(x)) < 1e-8:
            return "weak or unclear"
        corr = np.correlate(x, x, mode="full")[len(x) - 1 :]
        if corr[0] == 0:
            return "weak or unclear"
        corr = corr / corr[0]
        max_lag = max(3, len(values) // 2)
        candidates = corr[2:max_lag]
        if len(candidates) == 0:
            return "weak or unclear"
        best = float(np.max(candidates))
        if best >= self.strong_periodicity_threshold:
            return "strong"
        if best >= self.periodicity_threshold:
            return "moderate"
        return "weak or unclear"

    def _detect_peaks_and_changes(self, values: np.ndarray) -> dict[str, Any]:
        if len(values) < 3:
            return {"peak_count": 0, "change_count": 0, "description": "insufficient evidence"}
        local_peak_mask = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]) & (values[1:-1] > self.peak_z_threshold)
        local_valley_mask = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]) & (values[1:-1] < -self.peak_z_threshold)
        peak_count = int(np.sum(local_peak_mask) + np.sum(local_valley_mask))
        diff = np.diff(values)
        if len(diff) and float(np.std(diff)) > 1e-8:
            change_z = np.abs((diff - np.mean(diff)) / np.std(diff))
            change_count = int(np.sum(change_z > self.sudden_change_z_threshold))
        else:
            change_count = 0

        if peak_count == 0 and change_count == 0:
            description = "no strong sudden change detected"
        elif peak_count > 0 and change_count > 0:
            description = f"{peak_count} prominent peak/valley event(s) with {change_count} abrupt change point(s)"
        elif peak_count > 0:
            description = f"{peak_count} prominent peak/valley event(s)"
        else:
            description = f"{change_count} abrupt change point(s)"
        return {"peak_count": peak_count, "change_count": change_count, "description": description}

    def _generate_dynamic_cue(
        self,
        global_trend: str,
        fluctuation: str,
        periodicity: str,
        peaks: dict[str, Any],
    ) -> str:
        if peaks["peak_count"] or peaks["change_count"]:
            return "transient event-driven temporal variation"
        if periodicity in {"strong", "moderate"}:
            return "repeating oscillatory temporal pattern"
        if global_trend == "increasing":
            return "gradual upward temporal shift"
        if global_trend == "decreasing":
            return "gradual downward temporal shift"
        if fluctuation == "high":
            return "irregular high-variation temporal activity"
        return "mostly stable temporal profile"

    def _natural_language_answer(
        self,
        global_trend: str,
        segment_trends: list[str],
        fluctuation: str,
        periodicity: str,
        peaks: dict[str, Any],
    ) -> str:
        segment_text = self._format_segment_trends(segment_trends)
        return (
            f"The signal shows {self._trend_article(global_trend)} {global_trend} temporal pattern. "
            f"Across temporal segments, {segment_text}. "
            f"The fluctuation level is {fluctuation}, periodicity is {periodicity}, "
            f"and {peaks['description']}."
        )

    def _trend_article(self, trend: str) -> str:
        return "an" if trend[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

    def _supporting_stats(self, values: np.ndarray) -> dict[str, float]:
        arr = self._to_1d_numeric_array(values)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _format_segment_trends(self, trends: list[str]) -> str:
        if not trends:
            return "insufficient temporal evidence"
        return ", ".join(f"segment {idx + 1} is {trend}" for idx, trend in enumerate(trends))

    def _format_sampling_rate(self, sampling_rate: float | None) -> str:
        if sampling_rate is None:
            return "unknown"
        return f"{sampling_rate:g} Hz"

    def _channel_tag(self, meta: ChannelMetadata) -> str:
        location = self._normalize_key(meta.body_location).upper()
        channel = self._normalize_key(meta.channel).upper()
        if location and location != "UNKNOWN":
            return f"{location}_{channel}_CHANNEL"
        return f"{channel}_CHANNEL"

    def _extract_signal_container(self, sample: SensorSample | dict) -> dict:
        if isinstance(sample, SensorSample):
            return sample.signals or {}
        if isinstance(sample, dict):
            for key in ("signals", "data", "features"):
                value = sample.get(key)
                if isinstance(value, dict):
                    return value
            return {key: value for key, value in sample.items() if key not in {"dataset", "subject", "label", "meta"}}
        signals = getattr(sample, "signals", None)
        return signals if isinstance(signals, dict) else {}

    def _sample_dataset(self, sample: SensorSample | dict) -> str:
        if isinstance(sample, SensorSample):
            return sample.dataset or self.dataset
        if isinstance(sample, dict):
            return str(sample.get("dataset") or sample.get("dataset_name") or self.dataset)
        return str(getattr(sample, "dataset", self.dataset) or self.dataset)

    def _sample_subject(self, sample: SensorSample | dict) -> str:
        if isinstance(sample, SensorSample):
            return sample.subject
        if isinstance(sample, dict):
            return str(sample.get("subject", ""))
        return str(getattr(sample, "subject", ""))

    def _sample_label(self, sample: SensorSample | dict) -> int:
        if isinstance(sample, SensorSample):
            return int(sample.label)
        if isinstance(sample, dict):
            return int(sample.get("label", 0))
        return int(getattr(sample, "label", 0))

    def _legacy_signals_for_item(self, item: dict, legacy_data: Any) -> dict:
        for key in ("signals", "data", "features"):
            if isinstance(item.get(key), dict):
                return item[key]
        if legacy_data is None:
            return {}
        index = item.get("index", item.get("data_index", item.get("local_index")))
        try:
            if isinstance(legacy_data, list) and index is not None:
                candidate = legacy_data[int(index)]
            elif isinstance(legacy_data, dict) and index is not None:
                candidate = legacy_data.get(index, legacy_data.get(str(index), {}))
            else:
                candidate = legacy_data
        except (IndexError, TypeError, ValueError):
            candidate = {}
        if isinstance(candidate, dict):
            for key in ("signals", "data", "features"):
                if isinstance(candidate.get(key), dict):
                    return candidate[key]
            return candidate
        return {}

    def _load_legacy_data(self) -> Any:
        if self.data_path is None or not self.data_path.exists():
            return None
        try:
            with self.data_path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _coerce_metadata(self, value: ChannelMetadata | dict) -> ChannelMetadata:
        if isinstance(value, ChannelMetadata):
            return value
        return ChannelMetadata(
            channel=str(value.get("channel", value.get("name", "UNKNOWN"))),
            sensor_type=str(value.get("sensor_type", "unknown")),
            body_location=str(value.get("body_location", value.get("location", "unknown"))),
            sampling_rate=value.get("sampling_rate"),
            axis_index=value.get("axis_index"),
        )

    def _infer_channel_metadata(self, raw_name: str) -> ChannelMetadata | None:
        key = self._normalize_key(raw_name)
        location = "chest" if "chest" in key else "wrist" if "wrist" in key else "unknown"
        if "eda" in key:
            return ChannelMetadata("EDA", "electrodermal activity", location)
        if "temp" in key:
            return ChannelMetadata("TEMP", "temperature", location)
        if "bvp" in key:
            return ChannelMetadata("BVP", "blood volume pulse", location)
        if "ecg" in key:
            return ChannelMetadata("ECG", "electrocardiography", location)
        if "emg" in key:
            return ChannelMetadata("EMG", "electromyography", location)
        if "resp" in key:
            return ChannelMetadata("RESP", "respiration", location)
        if "acc_x" in key or key.endswith("_x"):
            return ChannelMetadata("ACC_X", "accelerometer", location)
        if "acc_y" in key or key.endswith("_y"):
            return ChannelMetadata("ACC_Y", "accelerometer", location)
        if "acc_z" in key or key.endswith("_z"):
            return ChannelMetadata("ACC_Z", "accelerometer", location)
        return None

    def _normalize_dataset_name(self, dataset: str | None) -> str:
        return str(dataset or "").replace("-", "").replace("_", "").strip().upper()

    def _normalize_key(self, key: str) -> str:
        return str(key).replace("-", "_").replace(" ", "_").strip().lower()

    def _canonical_input_name(self, name: str) -> str:
        normalized = self._normalize_key(name)
        if normalized in {"encoded_time_series", "encodedtimeseries"}:
            return "encoded_time_series"
        return "embedding_alignment"


__all__ = ["ChannelMetadata", "EmbeddingAlignmentInput"]
