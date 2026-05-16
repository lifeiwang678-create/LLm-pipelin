from __future__ import annotations

from .label_only import LabelOnlyOutput
from .label_explanation import LabelExplanationOutput


OUTPUT_REGISTRY = {
    "label_only": LabelOnlyOutput,
    "label_explanation": LabelExplanationOutput,
}


def build_output_handler(config: dict, labels: list[int]):
    kind = str(config.get("type", "label_only")).strip().lower()

    if kind not in OUTPUT_REGISTRY:
        raise ValueError(f"Unknown output type: {kind}")

    return OUTPUT_REGISTRY[kind](labels)


__all__ = [
    "OUTPUT_REGISTRY",
    "LabelOnlyOutput",
    "LabelExplanationOutput",
    "build_output_handler",
]
