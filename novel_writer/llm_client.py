"""LLM client with OpenAI-compatible, Gemini, Grok, DeepSeek, and Doubao backends."""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import time
from typing import Any
from urllib import parse
from urllib import error, request


def _normalize_chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith(("/v1", "/v2", "/v3", "/api/v1", "/api/v2", "/api/v3")):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _request_json(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
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
                return json.loads(response.read().decode("utf-8"))
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


def _generate_openai_compatible(prompt: str, config: dict) -> str:
    api_base = config.get("api_base", "").strip()
    api_key = config.get("api_key", "")
    model = config.get("model") or config.get("model_name")
    temperature = config.get("temperature", 0.8)
    max_tokens = config.get("max_tokens", 4000)
    timeout = config.get("timeout", 120)

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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = _request_json(endpoint, headers, body, timeout)
    return _extract_openai_text(payload)


def _generate_gemini(prompt: str, config: dict) -> str:
    api_key = config.get("api_key", "")
    model = config.get("model") or config.get("model_name")
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 4000)
    timeout = config.get("timeout", 120)
    thinking_level = config.get("thinking_level")
    thinking_budget = config.get("thinking_budget")
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
    if thinking_level is not None:
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": str(thinking_level)
        }
    elif thinking_budget is not None:
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingBudget": int(thinking_budget)
        }
    payload = _request_json(endpoint, headers, body, timeout)
    return _extract_gemini_text(payload)


def _generate_grok(prompt: str, config: dict) -> str:
    grok_config = dict(config)
    grok_config["api_base"] = grok_config.get("api_base", "").strip() or "https://api.x.ai/v1"
    return _generate_openai_compatible(prompt, grok_config)


def _generate_deepseek(prompt: str, config: dict) -> str:
    deepseek_config = dict(config)
    deepseek_config["api_base"] = (
        deepseek_config.get("api_base", "").strip() or "https://api.deepseek.com/v1"
    )
    return _generate_openai_compatible(prompt, deepseek_config)


def _generate_doubao(prompt: str, config: dict) -> str:
    doubao_config = dict(config)
    doubao_config["api_base"] = (
        doubao_config.get("api_base", "").strip()
        or "https://ark.cn-beijing.volces.com/api/v3"
    )
    return _generate_openai_compatible(prompt, doubao_config)


def generate_text_with_metadata(prompt: str, config: dict) -> tuple[str, dict[str, Any]]:
    """Generate text and return normalized usage metadata."""
    provider = (config.get("model_provider") or "openai_compatible").strip().lower()

    if provider == "openai_compatible":
        api_base = config.get("api_base", "").strip()
        api_key = config.get("api_key", "")
        model = config.get("model") or config.get("model_name")
        temperature = config.get("temperature", 0.8)
        max_tokens = config.get("max_tokens", 4000)
        timeout = config.get("timeout", 120)

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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = _request_json(endpoint, headers, body, timeout)
        return _extract_openai_text(payload), {
            "provider": provider,
            "model": model,
            "usage": _extract_openai_usage(payload),
        }

    if provider == "gemini":
        api_key = config.get("api_key", "")
        model = config.get("model") or config.get("model_name")
        temperature = config.get("temperature", 1.0)
        max_tokens = config.get("max_tokens", 4000)
        timeout = config.get("timeout", 120)
        thinking_level = config.get("thinking_level")
        thinking_budget = config.get("thinking_budget")
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
        if thinking_level is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": str(thinking_level)
            }
        elif thinking_budget is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingBudget": int(thinking_budget)
            }
        payload = _request_json(endpoint, headers, body, timeout)
        return _extract_gemini_text(payload), {
            "provider": provider,
            "model": model,
            "usage": _extract_gemini_usage(payload),
        }

    if provider == "grok":
        grok_config = dict(config)
        grok_config["api_base"] = grok_config.get("api_base", "").strip() or "https://api.x.ai/v1"
        text, metadata = generate_text_with_metadata(prompt, {**grok_config, "model_provider": "openai_compatible"})
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
        )
        metadata["provider"] = "doubao"
        return text, metadata

    raise ValueError(
        "Unsupported model_provider. Expected one of: "
        "'openai_compatible', 'gemini', 'grok', 'deepseek', 'doubao'."
    )


def generate_text(prompt: str, config: dict) -> str:
    """Generate text from the configured backend."""
    return generate_text_with_metadata(prompt, config)[0]
