"""LLM client with OpenAI-compatible, Gemini, Grok, DeepSeek, Doubao, and Ollama backends."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import time
from pathlib import Path
from uuid import uuid4
from typing import Any
from urllib import parse
from urllib import error, request

from common_utils import utc_now


DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
OLLAMA_MIN_TIMEOUT_SECONDS = 900
LLM_LOG_FILENAME = "llm_interactions.jsonl"
SENSITIVE_CONFIG_KEYS = {"api_key", "api_base", "model_name", "model"}


def _coerce_llm_log_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _resolve_log_path(config: dict[str, Any]) -> Path | None:
    project_path = str(config.get("project_path") or "").strip()
    if not project_path:
        return None
    path = Path(project_path)
    if not path.is_absolute():
        return None
    return path / "llm_logs" / LLM_LOG_FILENAME


def _mask_config_for_log(config: dict[str, Any]) -> dict[str, Any]:
    def mask_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "***" if key in SENSITIVE_CONFIG_KEYS else mask_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [mask_value(item) for item in value]
        return value

    masked = mask_value(config)
    masked.pop("project_path", None)
    return masked


def _append_log_entry(
    config: dict[str, Any],
    *,
    phase: str,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None = None,
    response_text: str = "",
    request_attempts: int,
    log_context: dict[str, Any] | None,
) -> None:
    log_path = _resolve_log_path(config)
    if log_path is None:
        return

    entry = {
        "request_id": uuid4().hex,
        "created_at": utc_now(),
        "phase": phase,
        "provider": config.get("model_provider", ""),
        "model": config.get("model") or config.get("model_name"),
        "attempts": request_attempts,
        "request": request_payload,
        "response": response_payload,
        "response_text": response_text,
        "config": _mask_config_for_log(config),
    }
    if log_context:
        entry["log_context"] = log_context

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False))
        handle.write("\n")


def _normalize_chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith(("/v1", "/v2", "/v3", "/api/v1", "/api/v2", "/api/v3")):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _resolve_timeout(config: dict[str, Any], provider: str) -> int:
    default_timeout = OLLAMA_MIN_TIMEOUT_SECONDS if provider == "ollama" else DEFAULT_REQUEST_TIMEOUT_SECONDS
    raw_timeout = config.get("timeout", default_timeout)
    try:
        timeout = int(raw_timeout)
    except (TypeError, ValueError):
        timeout = default_timeout

    if timeout <= 0:
        timeout = default_timeout
    if provider == "ollama":
        return max(timeout, OLLAMA_MIN_TIMEOUT_SECONDS)
    return timeout


def _request_json(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], int]:
    data = json.dumps(body).encode("utf-8")

    last_error: Exception | None = None
    retryable_errors = (
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionResetError,
        TimeoutError,
        socket.timeout,
        ssl.SSLEOFError,
        ssl.SSLZeroReturnError,
    )

    for attempt in range(4):
        try:
            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8")), attempt + 1
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except retryable_errors as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
            continue
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, retryable_errors):
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Failed to connect to LLM endpoint: {exc}") from exc

    raise RuntimeError(
        "LLM connection was closed before a response was returned after multiple retries. "
        f"Endpoint: {endpoint}. Last error: {last_error}"
    )


def _build_openai_compatible_headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_openai_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response missing choices: {payload}")

    message = choices[0].get("message", {})
    content: Any = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        text = "".join(text_parts).strip()
        if text:
            return text

    fallback_text = choices[0].get("text")
    if isinstance(fallback_text, str):
        return fallback_text.strip()
    raise RuntimeError(f"Unsupported LLM response format: {payload}")


def _extract_openai_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "cached_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
        "thought_tokens": 0,
    }


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        prompt_feedback = payload.get("promptFeedback")
        block_reason = ""
        if isinstance(prompt_feedback, dict):
            block_reason = str(prompt_feedback.get("blockReason", "") or "").strip()
        if block_reason:
            raise RuntimeError(
                f"Gemini blocked the request. blockReason={block_reason}, promptFeedback={prompt_feedback}"
            )
        raise RuntimeError(
            f"Gemini response missing candidates. promptFeedback={prompt_feedback}"
        )

    content = candidates[0].get("content", {})
    parts = content.get("parts") or []
    text_parts = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])

    text = "".join(text_parts).strip()
    if text:
        return text
    raise RuntimeError(f"Gemini response missing text parts: {payload}")


def _extract_gemini_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usageMetadata") or {}
    return {
        "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
        "completion_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
        "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
        "cached_tokens": int(usage.get("cachedContentTokenCount", 0) or 0),
        "reasoning_tokens": 0,
        "thought_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
    }

def generate_text_with_metadata(
    prompt: str,
    config: dict,
    log_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate text and return normalized usage metadata."""
    provider = (config.get("model_provider") or "openai_compatible").strip().lower()
    should_log = _coerce_llm_log_enabled(config.get("log_llm_payload"))
    phase = str((log_context or {}).get("phase", "")).strip() or "llm"

    if provider == "openai_compatible":
        api_base = config.get("api_base", "").strip()
        api_key = config.get("api_key", "")
        model = config.get("model") or config.get("model_name")
        temperature = config.get("temperature", 0.8)
        max_tokens = config.get("max_tokens", 4000)
        timeout = _resolve_timeout(config, "openai_compatible")

        if not api_base:
            raise ValueError("Missing 'api_base' in config.")
        if not model:
            raise ValueError("Missing 'model' or 'model_name' in config.")

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        endpoint = _normalize_chat_url(api_base)
        headers = _build_openai_compatible_headers(api_key)
        payload, request_attempts = _request_json(endpoint, headers, body, timeout)
        response_text = _extract_openai_text(payload)
        metadata = {
            "provider": provider,
            "model": model,
            "usage": _extract_openai_usage(payload),
        }
        if should_log:
            _append_log_entry(
                config,
                phase=phase,
                request_payload=body,
                response_payload=payload,
                response_text=response_text,
                request_attempts=request_attempts,
                log_context=log_context,
            )
        return response_text, metadata

    if provider == "gemini":
        api_key = config.get("api_key", "")
        model = config.get("model") or config.get("model_name")
        temperature = config.get("temperature", 1.0)
        max_tokens = config.get("max_tokens", 4000)
        timeout = _resolve_timeout(config, "gemini")
        api_base = (
            config.get("api_base", "").strip()
            or "https://generativelanguage.googleapis.com/v1beta"
        )

        if not api_key:
            raise ValueError("Missing 'api_key' in config.")
        if not model:
            raise ValueError("Missing 'model' or 'model_name' in config.")

        response_mime_type = "application/json" if "输出 JSON" in prompt else "text/plain"
        endpoint = (
            f"{api_base.rstrip('/')}/models/{model}:generateContent"
            f"?{parse.urlencode({'key': api_key})}"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
            "x-goog-api-client": "novel-writer/1.0",
            "Connection": "close",
        }
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": response_mime_type,
            },
        }
        payload, request_attempts = _request_json(endpoint, headers, body, timeout)
        response_text = _extract_gemini_text(payload)
        metadata = {
            "provider": provider,
            "model": model,
            "usage": _extract_gemini_usage(payload),
        }
        if should_log:
            _append_log_entry(
                config,
                phase=phase,
                request_payload=body,
                response_payload=payload,
                response_text=response_text,
                request_attempts=request_attempts,
                log_context=log_context,
            )
        return response_text, metadata

    if provider == "grok":
        grok_config = dict(config)
        grok_config["api_base"] = grok_config.get("api_base", "").strip() or "https://api.x.ai/v1"
        text, metadata = generate_text_with_metadata(
            prompt,
            {**grok_config, "model_provider": "openai_compatible"},
            log_context=log_context,
        )
        metadata["provider"] = "grok"
        return text, metadata

    if provider == "deepseek":
        deepseek_config = dict(config)
        deepseek_config["api_base"] = (
            deepseek_config.get("api_base", "").strip() or "https://api.deepseek.com/v1"
        )
        text, metadata = generate_text_with_metadata(
            prompt,
            {**deepseek_config, "model_provider": "openai_compatible"},
            log_context=log_context,
        )
        metadata["provider"] = "deepseek"
        return text, metadata

    if provider == "doubao":
        doubao_config = dict(config)
        doubao_config["api_base"] = (
            doubao_config.get("api_base", "").strip()
            or "https://ark.cn-beijing.volces.com/api/v3"
        )
        text, metadata = generate_text_with_metadata(
            prompt,
            {**doubao_config, "model_provider": "openai_compatible"},
            log_context=log_context,
        )
        metadata["provider"] = "doubao"
        return text, metadata

    if provider == "ollama":
        ollama_config = dict(config)
        ollama_config["api_base"] = (
            ollama_config.get("api_base", "").strip() or "http://127.0.0.1:11434/v1"
        )
        ollama_config["timeout"] = _resolve_timeout(ollama_config, "ollama")
        text, metadata = generate_text_with_metadata(
            prompt,
            {**ollama_config, "model_provider": "openai_compatible"},
            log_context=log_context,
        )
        metadata["provider"] = "ollama"
        return text, metadata

    raise ValueError(
        "Unsupported model_provider. Expected one of: "
        "'openai_compatible', 'gemini', 'grok', 'deepseek', 'doubao', 'ollama'."
    )


def generate_text(prompt: str, config: dict) -> str:
    """Generate text from the configured backend."""
    return generate_text_with_metadata(prompt, config)[0]
