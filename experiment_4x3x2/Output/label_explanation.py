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
        # 旧版调用了两次 _parse_json_object (一次在 _parse_label 里、一次单独取 explanation),
        # 这里改为只解析一次,避免重复 JSON 解析开销和回退路径不一致的隐患。
        obj, obj_error = self._parse_json_object(text)

        label: int | None = None
        label_error = obj_error
        explanation = ""
        has_explanation = False

        if not obj_error and isinstance(obj, dict):
            if "predicted_state" not in obj:
                label_error = "missing_predicted_state"
            else:
                try:
                    pred = int(obj["predicted_state"])
                    if pred in self.labels:
                        label = pred
                        label_error = ""
                    else:
                        label_error = f"out_of_label_space:{pred}"
                except (TypeError, ValueError):
                    label_error = "non_integer_predicted_state"

            explanation = str(obj.get("explanation", "")).strip()
            has_explanation = explanation != ""

        return {
            "label": label,
            "explanation": explanation,
            "valid": label is not None,
            "has_explanation": has_explanation,
            "parse_error": label_error,
        }


__all__ = ["LabelExplanationOutput"]
