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
        if fallback_label is not None:
            raise ValueError("fallback_label is disabled. Parse failures are recorded as invalid.")

    def instructions(self, labels: list[int]) -> str:
        return f"""Output format:
Return STRICT JSON only.

{{
  "predicted_state": {" or ".join(str(x) for x in labels)}
}}"""

    def parse(self, text: str) -> dict:
        label, error = self._parse_label(text)
        return {
            "label": label,
            "explanation": "",
            "valid": label is not None,
            "parse_error": error,
        }

    def _parse_label(self, text: str) -> tuple[int | None, str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, "no_json_object"

        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, "invalid_json"

        if "predicted_state" not in obj:
            return None, "missing_predicted_state"

        try:
            pred = int(obj["predicted_state"])
        except (TypeError, ValueError):
            return None, "non_integer_predicted_state"

        if pred not in self.labels:
            return None, f"out_of_label_space:{pred}"
        return pred, ""


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
        label, error = self._parse_label(text)
        explanation = ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                explanation = str(obj.get("explanation", "")).strip()
            except json.JSONDecodeError:
                explanation = ""
        return {
            "label": label,
            "explanation": explanation,
            "valid": label is not None,
            "parse_error": error,
        }


def build_output_handler(config: dict, labels: list[int]):
    kind = str(config.get("type", "label_only")).lower()
    if kind in {"label", "label_only", "label-only"}:
        return LabelOnlyOutput(labels=labels, fallback_label=config.get("fallback_label"))
    if kind in {"label_explanation", "label+explanation", "label-and-explanation", "label_explanation"}:
        return LabelExplanationOutput(labels=labels, fallback_label=config.get("fallback_label"))
    raise ValueError(f"Unknown output type: {kind}")
