from __future__ import annotations

import os
import time
from typing import Any

import requests


OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
GEMINI_PROVIDER = "gemini"


def build_lm_client(config: dict | None = None):
    """Build a completion client from a normalized lm_client config."""
    config = dict(config or {})
    provider = normalize_lm_provider(config.pop("provider", config.pop("type", OPENAI_COMPATIBLE_PROVIDER)))
    if "max_tokens" in config and "max_completion_tokens" not in config:
        config["max_completion_tokens"] = config.pop("max_tokens")

    if provider == GEMINI_PROVIDER:
        allowed = {
            "api_key",
            "model",
            "temperature",
            "max_completion_tokens",
            "timeout",
            "system_message",
            "extra_config",
        }
        kwargs = {key: value for key, value in config.items() if key in allowed}
        return GeminiClient(**kwargs)

    allowed = {
        "api_url",
        "api_key",
        "model",
        "temperature",
        "max_completion_tokens",
        "timeout",
        "system_message",
        "chat_template_kwargs",
        "extra_body",
    }
    kwargs = {key: value for key, value in config.items() if key in allowed}
    return OpenAICompatibleClient(**kwargs)


def normalize_lm_provider(provider: str | None) -> str:
    normalized = str(provider or OPENAI_COMPATIBLE_PROVIDER).strip().lower().replace("-", "_")
    aliases = {
        "openai": OPENAI_COMPATIBLE_PROVIDER,
        "openai_compatible": OPENAI_COMPATIBLE_PROVIDER,
        "openai_compat": OPENAI_COMPATIBLE_PROVIDER,
        "vllm": OPENAI_COMPATIBLE_PROVIDER,
        "lm_studio": OPENAI_COMPATIBLE_PROVIDER,
        "gemini": GEMINI_PROVIDER,
        "google": GEMINI_PROVIDER,
        "google_genai": GEMINI_PROVIDER,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported lm_client provider {provider!r}. "
            f"Expected one of: {OPENAI_COMPATIBLE_PROVIDER}, {GEMINI_PROVIDER}."
        )
    return aliases[normalized]


class OpenAICompatibleClient:
    """Client for OpenAI-compatible chat/completions servers.

    Works with local servers such as vLLM and LM Studio, and with hosted
    providers that expose the same /v1/chat/completions shape.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "lm-studio",
        model: str = "qwen2.5-14b-instruct",
        temperature: float = 0.0,
        max_completion_tokens: int = 128,
        timeout: int = 600,
        system_message: str | None = None,
        chat_template_kwargs: dict | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        # 新增 system_message:换到 GPT-4 / Claude / Llama 时,加 system 角色 prompt
        # 能显著稳定 JSON 输出。默认 None 保持对 qwen2.5-14b 的旧行为不变。
        self.system_message = system_message
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.last_usage: dict = {}
        self.usage_records: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        # 新增 max_completion_tokens / temperature 关键字覆盖:multi_agent 的前两个 agent
        # 要输出长结构化 JSON,需要比 client 默认更大的 token 上限;为此提供 per-call 覆盖。
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict] = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})

        output_limit = (
            int(max_completion_tokens)
            if max_completion_tokens is not None
            else int(max_tokens)
            if max_tokens is not None
            else self.max_completion_tokens
        )

        model_name = str(self.model).lower()
        api_url = str(self.api_url).lower()
        is_deepseek = "deepseek" in model_name or "deepseek" in api_url
        uses_openai_completion_limit = (
            "api.openai.com" in api_url
            or model_name.startswith(("gpt-5", "o1", "o3", "o4"))
        )

        if is_deepseek:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": output_limit,
            }

            if "deepseek-v4-pro" in model_name:
                payload["thinking"] = {
                    "type": "disabled"
                }
        else:
            token_limit_key = "max_completion_tokens" if uses_openai_completion_limit else "max_tokens"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else float(temperature),
                token_limit_key: output_limit,
            }

            if self.chat_template_kwargs:
                payload["chat_template_kwargs"] = self.chat_template_kwargs

        if self.extra_body:
            payload.update(self.extra_body)

        start = time.perf_counter()
        endpoint = f"{self.api_url}/chat/completions"
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            elapsed_time_sec = time.perf_counter() - start
            self._record_usage(prompt, "", {}, elapsed_time_sec)
            raise RuntimeError(
                f"OpenAI-compatible chat completion request failed before receiving a response. "
                f"Endpoint: {endpoint}. Prompt characters: {len(prompt)}. Error: {exc}"
            ) from exc

        elapsed_time_sec = time.perf_counter() - start
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            self._record_usage(prompt, "", {}, elapsed_time_sec)
            detail = _truncate_detail(response.text)
            raise RuntimeError(
                f"OpenAI-compatible chat completion request failed with HTTP {response.status_code}. "
                f"Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc

        try:
            response_json = response.json()
        except ValueError as exc:
            self._record_usage(prompt, "", {}, elapsed_time_sec)
            detail = _truncate_detail(response.text)
            raise RuntimeError(
                "OpenAI-compatible chat completion response was not valid JSON. "
                f"Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc

        try:
            content = response_json["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
            self._record_usage(prompt, "", usage or {}, elapsed_time_sec)
            detail = _truncate_detail(str(response_json))
            raise RuntimeError(
                "OpenAI-compatible chat completion response did not contain "
                f"choices[0].message.content. Prompt characters: {len(prompt)}. Response: {detail}"
            ) from exc

        if not isinstance(content, str):
            usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
            self._record_usage(prompt, "", usage or {}, elapsed_time_sec)
            raise RuntimeError(
                "OpenAI-compatible chat completion response content was not text. "
                f"Prompt characters: {len(prompt)}. Content type: {type(content).__name__}."
            )

        completion = content.strip()
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        self._record_usage(prompt, completion, usage or {}, elapsed_time_sec)
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


class GeminiClient:
    """Client for Google's official google-genai SDK.

    When api_key is omitted, google-genai reads GEMINI_API_KEY from the
    environment, which keeps secrets out of repository configs and code.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-3.5-flash",
        temperature: float = 0.0,
        max_completion_tokens: int = 128,
        timeout: int = 600,
        system_message: str | None = None,
        extra_config: dict | None = None,
    ) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "Gemini provider requires the official google-genai SDK. "
                "Install it with: pip install google-genai"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.timeout = timeout
        self.system_message = system_message
        self.extra_config = dict(extra_config or {})
        self.last_usage: dict = {}
        self.usage_records: list[dict] = []

        clean_api_key = str(api_key).strip() if api_key is not None else ""
        if clean_api_key:
            self.client = genai.Client(api_key=clean_api_key)
        else:
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError(
                    "Gemini provider requires an API key. Set GEMINI_API_KEY in the environment "
                    "or pass lm_client.api_key / --api-key for a temporary test."
                )
            self.client = genai.Client()

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini provider requires the official google-genai SDK. "
                "Install it with: pip install google-genai"
            ) from exc

        config_values = dict(self.extra_config)
        config_values.setdefault("temperature", self.temperature if temperature is None else float(temperature))
        config_values.setdefault(
            "max_output_tokens",
            int(max_completion_tokens)
            if max_completion_tokens is not None
            else int(max_tokens)
            if max_tokens is not None
            else self.max_completion_tokens,
        )
        config_values.setdefault("response_mime_type", "application/json")
        if "thinking_config" not in config_values and "thinkingConfig" not in config_values:
            config_values["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        if self.system_message and "system_instruction" not in config_values:
            config_values["system_instruction"] = self.system_message

        start = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_values),
            )
        except Exception as exc:
            elapsed_time_sec = time.perf_counter() - start
            self._record_usage(prompt, "", {}, elapsed_time_sec)
            raise RuntimeError(
                "Gemini generate_content request failed before receiving text. "
                f"Model: {self.model}. Prompt characters: {len(prompt)}. Error: {exc}"
            ) from exc

        elapsed_time_sec = time.perf_counter() - start
        usage = _gemini_usage(response)
        try:
            completion = response.text
        except Exception as exc:
            self._record_usage(prompt, "", usage, elapsed_time_sec)
            raise RuntimeError(
                "Gemini generate_content response text could not be read. "
                f"Model: {self.model}. Prompt characters: {len(prompt)}. Error: {exc}"
            ) from exc
        if completion is None:
            self._record_usage(prompt, "", usage, elapsed_time_sec)
            raise RuntimeError(
                "Gemini generate_content response did not contain text. "
                f"Model: {self.model}. Prompt characters: {len(prompt)}."
            )
        if not isinstance(completion, str):
            self._record_usage(prompt, "", usage, elapsed_time_sec)
            raise RuntimeError(
                "Gemini generate_content response text was not a string. "
                f"Model: {self.model}. Text type: {type(completion).__name__}."
            )

        completion = completion.strip()
        self._record_usage(prompt, completion, usage, elapsed_time_sec)
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


def _gemini_usage(response: Any) -> dict:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return {}
    return {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", None),
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", None),
        "total_tokens": getattr(usage_metadata, "total_token_count", None),
    }


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truncate_detail(text: str, limit: int = 1200) -> str:
    detail = str(text).strip()
    if len(detail) > limit:
        return detail[:limit] + "..."
    return detail


# Backward-compatible alias for older imports/config references.
LMStudioClient = OpenAICompatibleClient
