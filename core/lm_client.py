from __future__ import annotations

import requests


class LMStudioClient:
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "lm-studio",
        model: str = "qwen2.5-14b-instruct",
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 30,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

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
        response = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if len(detail) > 1200:
                detail = detail[:1200] + "..."
            raise RuntimeError(
                f"LM Studio request failed with HTTP {response.status_code}. "
                f"Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc
        return response.json()["choices"][0]["message"]["content"].strip()
