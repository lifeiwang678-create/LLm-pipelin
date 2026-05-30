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
        "High absolute EDA, SCR, heart-rate, HRV, EMG, or ACC values are not sufficient by themselves to predict stress.",
        "Some no-stress WESAD windows can show physiological arousal, movement, or subject-specific high baselines.",
        "Choose label 1 when the overall multi-channel pattern is more consistent with stress than no-stress after checking movement, artifacts, and subject-specific baseline explanations.",
        "Choose label 0 when stress-like cues are isolated, ambiguous, or better explained by no-stress arousal, movement, artifacts, or subject variation.",
        "Do not use label 0 as a safe fallback when stress evidence is present but imperfect.",
    ],
    "HHAR": [
        "Label 0 is walking downstairs.",
        "Label 1 is walking upstairs.",
        "Use temporal acceleration and gyroscope patterns rather than a single large magnitude value.",
        "Do not treat downstairs as the default stair label; compare upstairs and downstairs motion evidence symmetrically.",
        "Choose label 0 when the temporal motion pattern is more consistent with downstairs than upstairs.",
    ],
    "DREAMT": [
        "Label 0 is wake.",
        "Label 1 is sleep, including REM and NREM stages.",
        "Use the full window context; do not decide from one isolated noisy physiological value.",
        "Do not treat wake as the default label; sleep evidence can be present even when some physiological channels are noisy.",
        "Choose label 0 when wake-like evidence is stronger than sleep-like evidence over the full window.",
    ],
}

DECISION_GUIDANCE_BY_DATASET = {
    "WESAD": [
        "Stress evidence should be evaluated as a pattern across channels or features, not as one isolated high value.",
        "Predict label 1 only when stress-supporting cues are stronger than no-stress explanations across the full provided input.",
    ],
    "HHAR": [
        "For upstairs vs downstairs, focus on directional and temporal motion structure rather than a generic activity intensity.",
        "Predict label 1 only when the motion pattern better matches upstairs than downstairs; do not require unusually large magnitudes.",
    ],
    "DREAMT": [
        "For wake vs sleep, compare sustained window-level evidence rather than brief artifacts.",
        "Predict label 1 only when full-window physiology is more sleep-like than wake-like; do not require all channels to be quiet.",
    ],
}

GENERAL_DECISION_GUIDANCE = [
    "Treat all allowed labels symmetrically; neither label 0 nor label 1 is a default or safer answer.",
    "Compare the strongest evidence for label 0 against the strongest evidence for label 1 before deciding.",
    "Do not predict either label from one isolated extreme value alone.",
    "Choose label 1 only when label 1 support clearly outweighs label 0 support in the provided input.",
    "Choose label 0 when label 0 support clearly outweighs label 1 support, or when apparent label 1 cues are isolated or better explained by artifacts, movement, noise, or subject variation.",
    "If evidence is mixed, choose the label with the stronger overall support in the provided input.",
]


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


def decision_guidance_block(dataset: str | None = None) -> str:
    guidance = [
        *GENERAL_DECISION_GUIDANCE,
        *DECISION_GUIDANCE_BY_DATASET.get(_normalize_dataset_name(dataset), []),
    ]
    return "\n".join(f"- {rule}" for rule in guidance)


def target_names(labels: list[int], dataset: str | None = None) -> list[str]:
    names = label_names_for_dataset(dataset)
    return [f"{names.get(label, str(label))} ({label})" for label in labels]


def _normalize_dataset_name(dataset: str | None) -> str:
    return str(dataset or "").replace("-", "").replace("_", "").strip().upper()
