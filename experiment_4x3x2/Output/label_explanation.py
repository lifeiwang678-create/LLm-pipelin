from __future__ import annotations

from .label_only import LabelOnlyOutput


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
        has_explanation = False

        obj, obj_error = self._parse_json_object(text)
        if not obj_error and obj is not None:
            explanation = str(obj.get("explanation", "")).strip()
            has_explanation = explanation != ""

        return {
            "label": label,
            "explanation": explanation,
            "valid": label is not None,
            "has_explanation": has_explanation,
            "parse_error": error,
        }


__all__ = ["LabelExplanationOutput"]
