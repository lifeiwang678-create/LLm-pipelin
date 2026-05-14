from __future__ import annotations

import json
import re


class LabelOnlyOutput:
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


__all__ = ["LabelOnlyOutput"]

