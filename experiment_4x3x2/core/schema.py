from dataclasses import dataclass, field
from typing import Any


DEFAULT_LABEL_DATASET = "WESAD"

LABEL_NAMES_BY_DATASET = {
    "WESAD": {
        0: "no stress",
        1: "stress",
    },
    "HHAR": {
        0: "walking downstairs",
        1: "walking upstairs",
    },
    "DREAMT": {
        0: "wake",
        1: "sleep",
    },
}

LABEL_RULES_BY_DATASET = {
    "WESAD": [
        "Label 0 (no stress) includes baseline, amusement, meditation, and recovery-like non-stress segments.",
        "Label 1 (stress) means the dedicated stress-task segment only.",
        "Physiological arousal, movement, and subject-specific baselines can appear in either label, so compare the full pattern for both labels.",
        "Choose label 0 when the overall multi-channel pattern is more consistent with baseline, amusement, meditation, or recovery-like no-stress segments.",
        "Choose label 1 when the overall multi-channel pattern is more consistent with the dedicated stress-task segment.",
        "Do not use either label as a default fallback; decide by the stronger full-window evidence.",
    ],
    "HHAR": [
        "Label 0 is walking downstairs.",
        "Label 1 is walking upstairs.",
        "Compare upstairs and downstairs motion evidence symmetrically.",
        "Choose label 0 when the temporal motion pattern is more consistent with downstairs.",
        "Choose label 1 when the temporal motion pattern is more consistent with upstairs.",
        "Do not use either stair direction as a default fallback.",
    ],
    "DREAMT": [
        "Label 0 is wake.",
        "Label 1 is sleep, including REM and NREM stages.",
        "Compare wake-like and sleep-like evidence symmetrically over the full window.",
        "Choose label 0 when the full-window physiology and movement pattern is more consistent with wake.",
        "Choose label 1 when the full-window physiology and movement pattern is more consistent with sleep.",
        "Do not use either wake or sleep as a default fallback.",
    ],
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


def label_rules_block(dataset: str | None = None) -> str:
    rules = LABEL_RULES_BY_DATASET.get(_normalize_dataset_name(dataset), [])
    if not rules:
        return "- Apply the label definitions exactly as written."
    return "\n".join(f"- {rule}" for rule in rules)


def target_names(labels: list[int], dataset: str | None = None) -> list[str]:
    names = label_names_for_dataset(dataset)
    return [f"{names.get(label, str(label))} ({label})" for label in labels]


def _normalize_dataset_name(dataset: str | None) -> str:
    return str(dataset or "").replace("-", "").replace("_", "").strip().upper()
