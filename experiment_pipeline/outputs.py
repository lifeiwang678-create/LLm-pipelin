from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod


class OutputHandler(ABC):
    """Base interface for parsing model responses."""

    name: str

    @abstractmethod
    def instructions(self, labels: list[int]) -> str:
        """Return the JSON schema instruction for prompts."""

    @abstractmethod
    def parse(self, text: str) -> dict:
        """Return at least {'label': int, 'explanation': str}."""


class LabelOnlyOutput(OutputHandler):
    name = "label_only"

    def __init__(self, labels: list[int], fallback_label: int | None = None) -> None:
        self.labels = labels
        self.fallback_label = fallback_label if fallback_label is not None else labels[0]

    def instructions(self, labels: list[int]) -> str:
        return f"""Output format:
Return STRICT JSON only.

{{
  "predicted_state": {" or ".join(str(x) for x in labels)}
}}"""

    def parse(self, text: str) -> dict:
        label = self._parse_label(text)
        return {"label": label, "explanation": ""}

    def _parse_label(self, text: str) -> int:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                pred = int(obj["predicted_state"])
                if pred in self.labels:
                    return pred
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        label_pattern = "|".join(str(label) for label in self.labels)
        loose_match = re.search(rf"\b({label_pattern})\b", text)
        if loose_match:
            return int(loose_match.group(1))
        return int(self.fallback_label)


class LabelExplanationOutput(LabelOnlyOutput):
    name = "label_explanation"

    def instructions(self, labels: list[int]) -> str:
        return f"""Output format:
Return STRICT JSON only.

{{
  "predicted_state": {" or ".join(str(x) for x in labels)},
  "explanation": "one short sentence explaining the decision using only the provided input"
}}"""

    def parse(self, text: str) -> dict:
        label = self._parse_label(text)
        explanation = ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                explanation = str(obj.get("explanation", "")).strip()
            except json.JSONDecodeError:
                explanation = ""
        return {"label": label, "explanation": explanation}


def build_output_handler(config: dict, labels: list[int]):
    kind = str(config.get("type", "label_only")).lower()
    if kind in {"label", "label_only", "label-only"}:
        return LabelOnlyOutput(labels=labels, fallback_label=config.get("fallback_label"))
    if kind in {"label_explanation", "label+explanation", "label-and-explanation", "label_explanation"}:
        return LabelExplanationOutput(labels=labels, fallback_label=config.get("fallback_label"))
    raise ValueError(f"Unknown output type: {kind}")
