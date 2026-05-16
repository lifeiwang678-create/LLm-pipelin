from __future__ import annotations

import time

import requests


class LMStudioClient:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "lm-studio",
        model: str = "qwen2.5-14b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 600,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.last_usage: dict = {}
        self.usage_records: list[dict] = []

    def complete(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        start = time.perf_counter()
        response = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        elapsed_time_sec = time.perf_counter() - start
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            self._record_usage(prompt, {}, elapsed_time_sec)
            detail = response.text.strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "..."
            raise RuntimeError(
                f"LM Studio request failed with HTTP {response.status_code}. "
                f"Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc

        response_json = response.json()
        self._record_usage(prompt, response_json.get("usage", {}) or {}, elapsed_time_sec)
        return response_json["choices"][0]["message"]["content"].strip()

    def _record_usage(self, prompt: str, usage: dict, elapsed_time_sec: float) -> None:
        record = {
            "model": self.model,
            "prompt_characters": len(prompt),
            "prompt_tokens": _optional_int(usage.get("prompt_tokens")),
            "completion_tokens": _optional_int(usage.get("completion_tokens")),
            "total_tokens": _optional_int(usage.get("total_tokens")),
            "elapsed_time_sec": float(elapsed_time_sec),
        }
        self.last_usage = record
        self.usage_records.append(record)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
