from .label_only import LabelOnlyOutput
from .label_explanation import LabelExplanationOutput


def build_output_handler(config: dict, labels: list[int]):
    kind = str(config.get("type", "label_only")).lower()

    if kind == "label_only":
        return LabelOnlyOutput(labels=labels, fallback_label=config.get("fallback_label"))

    if kind == "label_explanation":
        return LabelExplanationOutput(labels=labels, fallback_label=config.get("fallback_label"))

    raise ValueError(f"Unknown output type: {kind}")


__all__ = [
    "LabelOnlyOutput",
    "LabelExplanationOutput",
    "build_output_handler",
]
