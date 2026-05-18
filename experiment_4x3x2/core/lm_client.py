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
        system_message: str | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 新增 system_message:换到 GPT-4 / Claude / Llama 时,加 system 角色 prompt
        # 能显著稳定 JSON 输出。默认 None 保持对 qwen2.5-14b 的旧行为不变。
        self.system_message = system_message
        self.last_usage: dict = {}
        self.usage_records: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        # 新增 max_tokens / temperature 关键字覆盖:multi_agent 的前两个 agent
        # 要输出长结构化 JSON,需要比 client 默认更大的 token 上限;为此提供 per-call 覆盖。
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": int(max_tokens) if max_tokens is not None else self.max_tokens,
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
            self._record_usage(prompt, "", {}, elapsed_time_sec)
            detail = response.text.strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "..."
            raise RuntimeError(
                f"LM Studio request failed with HTTP {response.status_code}. "
                f"Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc

        response_json = response.json()
        completion = response_json["choices"][0]["message"]["content"].strip()
        self._record_usage(prompt, completion, response_json.get("usage", {}) or {}, elapsed_time_sec)
        return completion

    def _record_usage(self, prompt: str, completion: str, usage: dict, elapsed_time_sec: float) -> None:
        record = {
            "model": self.model,
            "prompt_chars": len(prompt),
            "completion_chars": len(completion),
            "total_chars": len(prompt) + len(completion),
            "prompt_characters": len(prompt),
            "completion_characters": len(completion),
            "total_characters": len(prompt) + len(completion),
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
