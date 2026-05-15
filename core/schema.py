from dataclasses import dataclass, field
from typing import Any


DEFAULT_LABEL_DATASET = "WESAD"

LABEL_NAMES_BY_DATASET = {
    "WESAD": {
        1: "Baseline",
        2: "Stress",
        3: "Amusement",
    },
    "HHAR": {
        1: "Static activity",
        2: "Dynamic activity",
        3: "Stairs activity",
    },
    "DREAMT": {
        1: "Baseline/Neutral/Relax",
        2: "Stress",
        3: "Amusement/Happy",
    },
}


@dataclass
class SensorSample:
    dataset: str
    subject: str
    label: int
    signals: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMSample:
    subject: str
    label: int
    input_text: str
    dataset: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# Backward-compatible alias for the legacy config runner.
Sample = LLMSample


def label_names_for_dataset(dataset: str | None = None) -> dict[int, str]:
    normalized = _normalize_dataset_name(dataset or DEFAULT_LABEL_DATASET)
    return LABEL_NAMES_BY_DATASET.get(normalized, {})


def label_block(labels: list[int], dataset: str | None = None) -> str:
    names = label_names_for_dataset(dataset)
    return "\n".join(f"- {label} = {names.get(label, str(label))}" for label in labels)


def target_names(labels: list[int], dataset: str | None = None) -> list[str]:
    names = label_names_for_dataset(dataset)
    return [f"{names.get(label, str(label))} ({label})" for label in labels]


def _normalize_dataset_name(dataset: str | None) -> str:
    return str(dataset or "").replace("-", "").replace("_", "").strip().upper()
