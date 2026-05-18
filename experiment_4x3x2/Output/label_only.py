from __future__ import annotations

import json
import re


# qwen2.5-14b 比较听话,直接 json.loads 还能跑;但换成 GPT-4 / Claude / Llama 后,
# 模型常会输出 ```json ... ``` 或 "Here is the answer: {...}" 这类包装,严格
# json.loads 会一律记为 invalid。这里在出错时再做一次容错回退,扫描第一个 {...} 块。
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LabelOnlyOutput:
    name = "label_only"

    def __init__(self, labels: list[int]) -> None:
        # 旧版本接受 fallback_label 仅用于在调用时报错,实际功能已废弃,这里直接移除参数。
        self.labels = labels

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
        obj, error = self._parse_json_object(text)
        if error:
            return None, error

        if "predicted_state" not in obj:
            return None, "missing_predicted_state"

        try:
            pred = int(obj["predicted_state"])
        except (TypeError, ValueError):
            return None, "non_integer_predicted_state"

        if pred not in self.labels:
            return None, f"out_of_label_space:{pred}"
        return pred, ""

    def _parse_json_object(self, text: str) -> tuple[dict | None, str]:
        stripped = text.strip()
        if not stripped:
            return None, "empty_response"

        # 1) 严格路径:大多数听话模型直接命中。
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            obj = None

        # 2) markdown 代码块回退:```json\n{...}\n``` 或 ``` {...} ```。
        if obj is None:
            fence = _CODE_FENCE_RE.search(stripped)
            if fence:
                inner = fence.group(1).strip()
                try:
                    obj = json.loads(inner)
                except json.JSONDecodeError:
                    obj = None

        # 3) 通用回退:抓第一个 {...} 大括号块,容忍 "Here is the answer: {...}" 这种包装。
        if obj is None:
            match = _JSON_OBJECT_RE.search(stripped)
            if match:
                try:
                    obj = json.loads(match.group(0))
                except json.JSONDecodeError:
                    obj = None

        if obj is None:
            return None, "invalid_json"
        if not isinstance(obj, dict):
            return None, "json_not_object"
        return obj, ""


__all__ = ["LabelOnlyOutput"]
