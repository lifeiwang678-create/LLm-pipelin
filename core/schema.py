from dataclasses import dataclass, field
from typing import Any


LABEL_NAMES = {
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
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


def label_block(labels: list[int]) -> str:
    return "\n".join(f"- {label} = {LABEL_NAMES.get(label, str(label))}" for label in labels)


def target_names(labels: list[int]) -> list[str]:
    return [f"{LABEL_NAMES.get(label, str(label))} ({label})" for label in labels]
