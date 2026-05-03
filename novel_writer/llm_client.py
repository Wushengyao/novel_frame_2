"""LLM client with OpenAI-compatible, Gemini, Grok, DeepSeek, Doubao, Ollama, and llama.cpp backends."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import ssl
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
from typing import Any
from urllib import parse
from urllib import error, request

from common_utils import utc_now


DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
LOCAL_LLM_MIN_TIMEOUT_SECONDS = 900
LOCAL_OPENAI_COMPATIBLE_PROVIDERS = {"ollama", "llama_cpp"}
LLM_LOG_FILENAME = "llm_interactions.jsonl"
LLM_ACTIVITY_DIR_ENV = "NOVEL_LLM_ACTIVITY_DIR"
LLM_ACTIVITY_DEFAULT_DIR = Path.home() / ".cache" / "novel_writer" / "llm_activity"
SENSITIVE_CONFIG_KEYS = {"api_key", "api_base", "model_name", "model"}
SENSITIVE_URL_QUERY_RE = re.compile(r"([?&](?:key|api_key|access_token|token)=)[^&\s]+", re.IGNORECASE)
LLM_REQUEST_MAX_ATTEMPTS = 4
RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
GEMINI_LOCATION_ERROR_SNIPPETS = (
    "User location is not supported for the API use",
    "FAILED_PRECONDITION",
)
CREATIVE_PHASES = {
    "writer",
    "rewrite",
    "polish",
    "illustration_prompt",
}
STRUCTURED_PHASES = {
    "init",
    "outline",
    "craft_brief",
    "quality_review",
    "summary",
}
PROVIDER_CREATIVE_TEMPERATURES = {
    "gemini": {
        "writer": 1.0,
        "rewrite": 0.8,
        "polish": 0.8,
        "illustration_prompt": 0.7,
    },
    "grok": {
        "writer": 1.0,
        "rewrite": 0.9,
        "polish": 0.9,
        "illustration_prompt": 0.8,
    },
    "doubao": {
        "writer": 0.9,
        "rewrite": 0.8,
        "polish": 0.8,
        "illustration_prompt": 0.7,
    },
    "ollama": {
        "writer": 0.9,
        "rewrite": 0.7,
        "polish": 0.8,
        "illustration_prompt": 0.7,
    },
    "llama_cpp": {
        "writer": 0.9,
        "rewrite": 0.7,
        "polish": 0.8,
        "illustration_prompt": 0.7,
    },
}
PROVIDER_STRUCTURED_TEMPERATURES = {
    "gemini": 0.2,
    "grok": 0.2,
    "doubao": 0.2,
    "ollama": 0.2,
    "llama_cpp": 0.2,
}
LEGACY_DEFAULT_TEMPERATURES = {0.8, 0.9, 1.0}
GEMINI_MODEL_ALIASES = {
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
}
GEMINI_31_PRO_CREATIVE_TEMPERATURES = {
    "writer": 0.8,
    "rewrite": 0.7,
    "polish": 0.65,
    "illustration_prompt": 0.6,
}
DEEPSEEK_V4_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}
DEEPSEEK_NON_THINKING_TEMPERATURES_BY_MODEL = {
    "deepseek-v4-flash": {
        "writer": 0.8,
        "rewrite": 0.7,
        "polish": 0.65,
        "illustration_prompt": 0.6,
    },
    "deepseek-v4-pro": {
        "writer": 0.8,
        "rewrite": 0.75,
        "polish": 0.65,
        "illustration_prompt": 0.6,
    },
}
DEEPSEEK_MAX_REASONING_PHASES = {"quality_review"}
DEEPSEEK_THINKING_MIN_TIMEOUT_SECONDS = 300
DEEPSEEK_THINKING_INACTIVE_FIELDS = (
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
)
INPUT_TOKEN_LIMIT_CONFIG_KEYS = (
    "input_token_limit",
    "max_input_tokens",
    "prompt_token_limit",
    "max_prompt_tokens",
)
CONTEXT_WINDOW_CONFIG_KEYS = (
    "context_window_tokens",
    "max_context_tokens",
    "model_context_tokens",
    "context_window",
)


def _coerce_llm_log_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _sanitize_error_text(value: object) -> str:
    return SENSITIVE_URL_QUERY_RE.sub(r"\1***", str(value or ""))


def _llm_activity_dir() -> Path:
    raw = os.environ.get(LLM_ACTIVITY_DIR_ENV, "")
    if raw.strip():
        return Path(raw).expanduser()
    return LLM_ACTIVITY_DEFAULT_DIR


@contextmanager
def _llm_activity_marker(config: dict[str, Any], phase: str, log_context: dict[str, Any] | None):
    marker_path: Path | None = None
    try:
        marker_dir = _llm_activity_dir()
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker_path = marker_dir / f"{int(time.time())}_{os.getpid()}_{uuid4().hex}.json"
        marker = {
            "pid": os.getpid(),
            "created_at": utc_now(),
            "phase": phase,
            "provider": config.get("model_provider", ""),
            "model": config.get("model") or config.get("model_name"),
            "workflow_id": (log_context or {}).get("workflow_id", ""),
        }
        marker_path.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
    except Exception:
        marker_path = None
    try:
        yield
    finally:
        if marker_path is not None:
            try:
                marker_path.unlink(missing_ok=True)
            except Exception:
                pass


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
    status: str = "succeeded",
    error_summary: str = "",
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
        "status": status,
        "attempts": request_attempts,
        "request": request_payload,
        "response": response_payload,
        "response_text": response_text,
        "config": _mask_config_for_log(config),
    }
    if error_summary:
        entry["error"] = _sanitize_error_text(error_summary)[:1200]
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
    default_timeout = (
        LOCAL_LLM_MIN_TIMEOUT_SECONDS
        if provider in LOCAL_OPENAI_COMPATIBLE_PROVIDERS
        else DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    raw_timeout = config.get("timeout", default_timeout)
    try:
        timeout = int(raw_timeout)
    except (TypeError, ValueError):
        timeout = default_timeout

    if timeout <= 0:
        timeout = default_timeout
    if provider in LOCAL_OPENAI_COMPATIBLE_PROVIDERS:
        return max(timeout, LOCAL_LLM_MIN_TIMEOUT_SECONDS)
    return timeout


def _is_gemini_location_error(status_code: int, detail: str) -> bool:
    return status_code == 400 and all(snippet in detail for snippet in GEMINI_LOCATION_ERROR_SNIPPETS)


def _retry_delay(attempt: int, *, slow: bool = False, retry_after: float = 0.0) -> float:
    if slow:
        return max(retry_after, 20.0 * (attempt + 1))
    return max(retry_after, 1.5 * (attempt + 1))


def _is_retryable_network_error(exc: BaseException) -> bool:
    retryable_errors = (
        http.client.RemoteDisconnected,
        http.client.IncompleteRead,
        ConnectionAbortedError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
        TimeoutError,
        socket.timeout,
        ssl.SSLEOFError,
        ssl.SSLZeroReturnError,
    )
    return isinstance(exc, retryable_errors)


def _request_json(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
) -> tuple[dict[str, Any], int]:
    data = json.dumps(body).encode("utf-8")

    last_error: Exception | None = None

    for attempt in range(LLM_REQUEST_MAX_ATTEMPTS):
        try:
            req = request.Request(endpoint, data=data, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout) as response:
                raw_response = response.read().decode("utf-8")
                try:
                    return json.loads(raw_response), attempt + 1
                except json.JSONDecodeError as exc:
                    last_error = exc
                    if attempt < LLM_REQUEST_MAX_ATTEMPTS - 1:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"LLM endpoint returned invalid JSON: {raw_response[:500]}") from exc
        except error.HTTPError as exc:
            detail = _sanitize_error_text(exc.read().decode("utf-8", errors="replace"))
            retryable_location_error = _is_gemini_location_error(exc.code, detail)
            if (
                exc.code in RETRYABLE_HTTP_STATUS_CODES or retryable_location_error
            ) and attempt < LLM_REQUEST_MAX_ATTEMPTS - 1:
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
                try:
                    retry_after = float(exc.headers.get("Retry-After", "0") or 0)
                except (AttributeError, TypeError, ValueError):
                    retry_after = 0.0
                time.sleep(_retry_delay(attempt, slow=retryable_location_error, retry_after=retry_after))
                continue
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            ConnectionAbortedError,
            ConnectionRefusedError,
            ConnectionResetError,
            BrokenPipeError,
            TimeoutError,
            socket.timeout,
            ssl.SSLEOFError,
            ssl.SSLZeroReturnError,
        ) as exc:
            last_error = exc
            if attempt < LLM_REQUEST_MAX_ATTEMPTS - 1:
                time.sleep(_retry_delay(attempt))
            continue
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if _is_retryable_network_error(reason):
                last_error = exc
                if attempt < LLM_REQUEST_MAX_ATTEMPTS - 1:
                    time.sleep(_retry_delay(attempt))
                continue
            raise RuntimeError(f"Failed to connect to LLM endpoint: {_sanitize_error_text(exc)}") from exc

    raise RuntimeError(
        "LLM connection was closed before a response was returned after multiple retries. "
        f"Endpoint: {_sanitize_error_text(endpoint)}. Last error: {_sanitize_error_text(last_error)}"
    )


def _request_json_with_activity(
    endpoint: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int,
    config: dict[str, Any],
    phase: str,
    log_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], int]:
    with _llm_activity_marker(config, phase, log_context):
        return _request_json(endpoint, headers, body, timeout)


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


def _extract_openai_finish_reason(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", "") or "").strip()


def _is_truncated_finish_reason(provider: str, finish_reason: object) -> bool:
    normalized = str(finish_reason or "").strip().lower()
    if not normalized:
        return False
    if provider == "gemini":
        return normalized in {"max_tokens", "max_output_tokens", "length"}
    return normalized in {"length", "max_tokens", "max_output_tokens"}


def llm_response_was_truncated(metadata: dict[str, Any] | None) -> bool:
    if not isinstance(metadata, dict):
        return False
    if bool(metadata.get("truncated")):
        return True
    return _is_truncated_finish_reason(
        str(metadata.get("provider", "") or ""),
        metadata.get("finish_reason"),
    )


def raise_if_llm_response_truncated(metadata: dict[str, Any] | None, *, phase: str) -> None:
    if not llm_response_was_truncated(metadata):
        return
    finish_reason = str((metadata or {}).get("finish_reason", "") or "").strip() or "unknown"
    raise RuntimeError(f"{phase} response was truncated by the model output token limit (finish_reason={finish_reason}).")


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


def _extract_gemini_finish_reason(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        return ""
    return str(candidates[0].get("finishReason", "") or "").strip()


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


def _normalize_optional_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_model_id(value: object) -> str:
    return str(value or "").strip().lower()


def _strip_model_prefix(model: str) -> str:
    normalized = _normalize_model_id(model)
    if normalized.startswith("models/"):
        return normalized.removeprefix("models/")
    return normalized


def _canonical_gemini_model_id(value: object) -> str:
    normalized = _strip_model_prefix(str(value or ""))
    return GEMINI_MODEL_ALIASES.get(normalized, normalized)


def _is_nonempty(value: object) -> bool:
    return value not in (None, "")


def _is_legacy_default_temperature(value: object) -> bool:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(temperature - default) < 0.000001 for default in LEGACY_DEFAULT_TEMPERATURES)


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_positive_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _configured_token_limit(config: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = _coerce_positive_int(config.get(key))
        if value > 0:
            return value
    return 0


def _request_limit_metadata(config: dict[str, Any], max_output_tokens: object) -> dict[str, int]:
    output_limit = _coerce_positive_int(max_output_tokens)
    input_limit = _configured_token_limit(config, INPUT_TOKEN_LIMIT_CONFIG_KEYS)
    context_window = _configured_token_limit(config, CONTEXT_WINDOW_CONFIG_KEYS)
    if input_limit <= 0 and context_window > 0:
        input_limit = max(0, context_window - output_limit) if output_limit > 0 else context_window
    return {
        "input_token_limit": input_limit,
        "max_output_tokens": output_limit,
    }


def _is_json_response_format(response_format: str = "") -> bool:
    return _normalize_optional_text(response_format).lower() in {
        "json",
        "json_object",
        "application/json",
    }


def _phase_uses_structured_defaults(phase: str, response_format: str = "") -> bool:
    return str(phase or "").strip().lower() in STRUCTURED_PHASES or _is_json_response_format(response_format)


def _phase_uses_creative_defaults(phase: str) -> bool:
    return str(phase or "").strip().lower() in CREATIVE_PHASES


def _apply_temperature_defaults(
    config: dict[str, Any],
    *,
    provider: str,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    raw_temperature = config.get("temperature")
    if _is_nonempty(raw_temperature) and not _is_legacy_default_temperature(raw_temperature):
        return config

    normalized_phase = str(phase or "").strip().lower()
    next_temperature: float | None = None
    if _phase_uses_structured_defaults(normalized_phase, response_format):
        next_temperature = PROVIDER_STRUCTURED_TEMPERATURES.get(provider)
    elif _phase_uses_creative_defaults(normalized_phase):
        next_temperature = PROVIDER_CREATIVE_TEMPERATURES.get(provider, {}).get(normalized_phase)

    if next_temperature is None:
        return config
    optimized = dict(config)
    optimized["temperature"] = next_temperature
    return optimized


def _apply_gemini_temperature_defaults(
    config: dict[str, Any],
    *,
    model: str,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    raw_temperature = config.get("temperature")
    if _is_nonempty(raw_temperature) and not _is_legacy_default_temperature(raw_temperature):
        return config

    normalized_phase = str(phase or "").strip().lower()
    next_temperature: float | None = None
    if _phase_uses_structured_defaults(normalized_phase, response_format):
        next_temperature = PROVIDER_STRUCTURED_TEMPERATURES["gemini"]
    elif _phase_uses_creative_defaults(normalized_phase):
        if _canonical_gemini_model_id(model) == "gemini-3.1-pro-preview":
            next_temperature = GEMINI_31_PRO_CREATIVE_TEMPERATURES.get(normalized_phase)
        else:
            next_temperature = PROVIDER_CREATIVE_TEMPERATURES["gemini"].get(normalized_phase)

    if next_temperature is None:
        return config
    optimized = dict(config)
    optimized["temperature"] = next_temperature
    return optimized


def _coerce_thinking(value: object) -> dict[str, str] | None:
    if isinstance(value, dict):
        thinking_type = str(value.get("type") or "").strip().lower()
    elif isinstance(value, bool):
        thinking_type = "enabled" if value else "disabled"
    else:
        thinking_type = str(value or "").strip().lower()
    if thinking_type in {"enabled", "disabled"}:
        return {"type": thinking_type}
    return None


def _merge_request_options(body: dict[str, Any], options: object) -> None:
    if not isinstance(options, dict):
        return
    for key, value in options.items():
        if not _is_nonempty(value):
            continue
        body[str(key)] = value


def _request_fields_to_omit(config: dict[str, Any]) -> set[str]:
    raw_fields = config.get("omit_request_fields")
    if isinstance(raw_fields, (list, tuple, set)):
        return {str(field).strip() for field in raw_fields if str(field).strip()}
    return set()


def _merge_generation_config(config: dict[str, Any], generation_config: object) -> None:
    if not isinstance(generation_config, dict):
        return
    for key, value in generation_config.items():
        if not _is_nonempty(value):
            continue
        config[str(key)] = value


def _gemini_model_family(model: str) -> str:
    normalized = _canonical_gemini_model_id(model)
    if normalized.startswith("gemini-3"):
        return "gemini-3"
    if normalized.startswith("gemini-2.5"):
        return "gemini-2.5"
    return ""


def _apply_gemini_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    model = _canonical_gemini_model_id(config.get("model") or config.get("model_name"))
    optimized = _apply_gemini_temperature_defaults(
        config,
        model=model,
        phase=phase,
        response_format=response_format,
    )
    generation_config = (
        dict(optimized.get("generation_config") or {})
        if isinstance(optimized.get("generation_config"), dict)
        else {}
    )
    if _is_nonempty(generation_config.get("thinkingConfig")):
        optimized = dict(optimized)
        optimized["generation_config"] = generation_config
        return optimized

    family = _gemini_model_family(model)
    normalized_phase = str(phase or "").strip().lower()
    if family == "gemini-3":
        if _phase_uses_creative_defaults(normalized_phase):
            thinking_level = "minimal" if "flash" in model else "low"
        elif _phase_uses_structured_defaults(normalized_phase, response_format):
            thinking_level = "high"
        else:
            thinking_level = "low"
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    elif family == "gemini-2.5":
        if _phase_uses_creative_defaults(normalized_phase) and "flash" in model:
            thinking_budget = 0
        elif _phase_uses_structured_defaults(normalized_phase, response_format):
            thinking_budget = -1
        else:
            thinking_budget = 0 if "flash" in model else -1
        generation_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    if generation_config:
        optimized = dict(optimized)
        optimized["generation_config"] = generation_config
    return optimized


def _is_grok_reasoning_model(model: str) -> bool:
    normalized = _normalize_model_id(model)
    if "non-reasoning" in normalized:
        return False
    return "reasoning" in normalized or normalized.endswith("-reasoning")


def _apply_grok_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    optimized = _apply_temperature_defaults(
        config,
        provider="grok",
        phase=phase,
        response_format=response_format,
    )
    model = _normalize_model_id(optimized.get("model") or optimized.get("model_name"))
    request_options = (
        dict(optimized.get("request_options") or {})
        if isinstance(optimized.get("request_options"), dict)
        else {}
    )
    request_options.pop("reasoning_effort", None)
    if _is_json_response_format(response_format):
        request_options.setdefault("response_format", {"type": "json_object"})
    optimized = dict(optimized)
    optimized["request_options"] = request_options
    if _is_grok_reasoning_model(model):
        optimized["timeout"] = max(_resolve_timeout(optimized, "grok"), 3600)
    return optimized


def _apply_doubao_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    optimized = _apply_temperature_defaults(
        config,
        provider="doubao",
        phase=phase,
        response_format=response_format,
    )
    model = _normalize_model_id(optimized.get("model") or optimized.get("model_name"))
    if "doubao-seed" not in model:
        return optimized

    request_options = (
        dict(optimized.get("request_options") or {})
        if isinstance(optimized.get("request_options"), dict)
        else {}
    )
    if _coerce_thinking(request_options.get("thinking")) is None:
        thinking_type = "enabled" if _phase_uses_structured_defaults(phase, response_format) else "disabled"
        request_options["thinking"] = {"type": thinking_type}
    optimized = dict(optimized)
    optimized["request_options"] = request_options
    return optimized


def _apply_ollama_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    optimized = _apply_temperature_defaults(
        config,
        provider="ollama",
        phase=phase,
        response_format=response_format,
    )
    request_options = (
        dict(optimized.get("request_options") or {})
        if isinstance(optimized.get("request_options"), dict)
        else {}
    )
    if _is_json_response_format(response_format):
        request_options.setdefault("response_format", {"type": "json_object"})
    optimized = dict(optimized)
    optimized["request_options"] = request_options
    return optimized


def _apply_llama_cpp_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    optimized = _apply_temperature_defaults(
        config,
        provider="llama_cpp",
        phase=phase,
        response_format=response_format,
    )
    request_options = (
        dict(optimized.get("request_options") or {})
        if isinstance(optimized.get("request_options"), dict)
        else {}
    )
    if _is_json_response_format(response_format):
        request_options.setdefault("response_format", {"type": "json_object"})
    optimized = dict(optimized)
    optimized["request_options"] = request_options
    return optimized


def _apply_deepseek_v4_defaults(
    config: dict[str, Any],
    *,
    phase: str,
    response_format: str,
) -> dict[str, Any]:
    model = _normalize_model_id(config.get("model") or config.get("model_name"))
    if model not in DEEPSEEK_V4_MODELS:
        return config

    optimized = dict(config)
    request_options = (
        dict(config.get("request_options") or {})
        if isinstance(config.get("request_options"), dict)
        else {}
    )
    omit_fields = _request_fields_to_omit(config)

    normalized_phase = str(phase or "").strip().lower()
    explicit_thinking = _coerce_thinking(request_options.get("thinking"))
    if explicit_thinking is None:
        explicit_thinking = _coerce_thinking(config.get("thinking"))

    if explicit_thinking is None:
        thinking = {"type": "enabled"}
    else:
        thinking = explicit_thinking
    request_options["thinking"] = thinking

    if thinking["type"] == "enabled":
        json_task = _is_json_response_format(response_format)
        if json_task:
            request_options.setdefault("response_format", {"type": "json_object"})
        if not _is_nonempty(request_options.get("reasoning_effort")) and not _is_nonempty(
            config.get("reasoning_effort")
        ):
            request_options["reasoning_effort"] = (
                "max" if normalized_phase in DEEPSEEK_MAX_REASONING_PHASES else "high"
            )
        elif _is_nonempty(config.get("reasoning_effort")) and not _is_nonempty(
            request_options.get("reasoning_effort")
        ):
            request_options["reasoning_effort"] = str(config.get("reasoning_effort")).strip()
        omit_fields.update(DEEPSEEK_THINKING_INACTIVE_FIELDS)
        for field in DEEPSEEK_THINKING_INACTIVE_FIELDS:
            request_options.pop(field, None)
        optimized["timeout"] = max(_resolve_timeout(optimized, "deepseek"), DEEPSEEK_THINKING_MIN_TIMEOUT_SECONDS)
    else:
        request_options.pop("reasoning_effort", None)
        model_temperatures = DEEPSEEK_NON_THINKING_TEMPERATURES_BY_MODEL.get(model, {})
        default_temperature = model_temperatures.get(normalized_phase, 1.0)
        custom_temperature = _coerce_float(config.get("temperature"))
        if (
            custom_temperature is not None
            and not _is_legacy_default_temperature(config.get("temperature"))
            and 0.0 <= custom_temperature <= 1.0
        ):
            optimized["temperature"] = custom_temperature
        else:
            optimized["temperature"] = default_temperature

    optimized["request_options"] = request_options
    optimized["omit_request_fields"] = sorted(omit_fields)
    return optimized


def _build_openai_messages(prompt: str, system_prompt: str = "") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    normalized_system_prompt = _normalize_optional_text(system_prompt)
    if normalized_system_prompt:
        messages.append({"role": "system", "content": normalized_system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _is_json_response_requested(prompt: str, response_format: str = "") -> bool:
    if _is_json_response_format(response_format):
        return True
    return "输出 JSON" in prompt or "输出必须是合法 JSON" in prompt


def generate_text_with_metadata(
    prompt: str,
    config: dict,
    log_context: dict[str, Any] | None = None,
    system_prompt: str = "",
    response_format: str = "",
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
            "messages": _build_openai_messages(prompt, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for field in _request_fields_to_omit(config):
            body.pop(field, None)
        _merge_request_options(body, config.get("request_options"))
        endpoint = _normalize_chat_url(api_base)
        headers = _build_openai_compatible_headers(api_key)
        payload = None
        request_attempts = 0
        try:
            payload, request_attempts = _request_json_with_activity(
                endpoint,
                headers,
                body,
                timeout,
                config,
                phase,
                log_context,
            )
            response_text = _extract_openai_text(payload)
        except Exception as exc:
            if should_log:
                _append_log_entry(
                    config,
                    phase=phase,
                    request_payload=body,
                    response_payload=payload,
                    response_text="",
                    request_attempts=request_attempts or LLM_REQUEST_MAX_ATTEMPTS,
                    log_context=log_context,
                    status="failed",
                    error_summary=str(exc),
                )
            raise
        metadata = {
            "provider": provider,
            "model": model,
            "finish_reason": _extract_openai_finish_reason(payload or {}),
            "truncated": _is_truncated_finish_reason(provider, _extract_openai_finish_reason(payload or {})),
            "usage": _extract_openai_usage(payload),
            **_request_limit_metadata(config, body.get("max_tokens")),
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
        raw_model = config.get("model") or config.get("model_name")
        canonical_model = _canonical_gemini_model_id(raw_model)
        if canonical_model and canonical_model != str(raw_model or "").strip():
            config = dict(config)
            config["model"] = canonical_model
            config["model_name"] = canonical_model
        config = _apply_gemini_defaults(
            config,
            phase=phase,
            response_format=response_format,
        )
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

        response_mime_type = "application/json" if _is_json_response_requested(prompt, response_format) else "text/plain"
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
        _merge_generation_config(body["generationConfig"], config.get("generation_config"))
        normalized_system_prompt = _normalize_optional_text(system_prompt)
        if normalized_system_prompt:
            body["systemInstruction"] = {
                "parts": [{"text": normalized_system_prompt}],
            }
        payload = None
        request_attempts = 0
        try:
            payload, request_attempts = _request_json_with_activity(
                endpoint,
                headers,
                body,
                timeout,
                config,
                phase,
                log_context,
            )
            response_text = _extract_gemini_text(payload)
        except Exception as exc:
            if should_log:
                _append_log_entry(
                    config,
                    phase=phase,
                    request_payload=body,
                    response_payload=payload,
                    response_text="",
                    request_attempts=request_attempts or LLM_REQUEST_MAX_ATTEMPTS,
                    log_context=log_context,
                    status="failed",
                    error_summary=str(exc),
                )
            raise
        metadata = {
            "provider": provider,
            "model": model,
            "finish_reason": _extract_gemini_finish_reason(payload or {}),
            "truncated": _is_truncated_finish_reason(provider, _extract_gemini_finish_reason(payload or {})),
            "usage": _extract_gemini_usage(payload),
            **_request_limit_metadata(config, body.get("generationConfig", {}).get("maxOutputTokens")),
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
        grok_config = _apply_grok_defaults(
            grok_config,
            phase=phase,
            response_format=response_format,
        )
        text, metadata = generate_text_with_metadata(
            prompt,
            {**grok_config, "model_provider": "openai_compatible"},
            log_context=log_context,
            system_prompt=system_prompt,
            response_format=response_format,
        )
        metadata["provider"] = "grok"
        return text, metadata

    if provider == "deepseek":
        deepseek_config = dict(config)
        deepseek_config["api_base"] = (
            deepseek_config.get("api_base", "").strip() or "https://api.deepseek.com/v1"
        )
        deepseek_config = _apply_deepseek_v4_defaults(
            deepseek_config,
            phase=phase,
            response_format=response_format,
        )
        text, metadata = generate_text_with_metadata(
            prompt,
            {**deepseek_config, "model_provider": "openai_compatible"},
            log_context=log_context,
            system_prompt=system_prompt,
            response_format=response_format,
        )
        metadata["provider"] = "deepseek"
        return text, metadata

    if provider == "doubao":
        doubao_config = dict(config)
        doubao_config["api_base"] = (
            doubao_config.get("api_base", "").strip()
            or "https://ark.cn-beijing.volces.com/api/v3"
        )
        doubao_config = _apply_doubao_defaults(
            doubao_config,
            phase=phase,
            response_format=response_format,
        )
        text, metadata = generate_text_with_metadata(
            prompt,
            {**doubao_config, "model_provider": "openai_compatible"},
            log_context=log_context,
            system_prompt=system_prompt,
            response_format=response_format,
        )
        metadata["provider"] = "doubao"
        return text, metadata

    if provider == "ollama":
        ollama_config = dict(config)
        ollama_config["api_base"] = (
            ollama_config.get("api_base", "").strip() or "http://127.0.0.1:11434/v1"
        )
        ollama_config = _apply_ollama_defaults(
            ollama_config,
            phase=phase,
            response_format=response_format,
        )
        ollama_config["timeout"] = _resolve_timeout(ollama_config, "ollama")
        text, metadata = generate_text_with_metadata(
            prompt,
            {**ollama_config, "model_provider": "openai_compatible"},
            log_context=log_context,
            system_prompt=system_prompt,
            response_format=response_format,
        )
        metadata["provider"] = "ollama"
        return text, metadata

    if provider == "llama_cpp":
        llama_cpp_config = dict(config)
        llama_cpp_config["api_base"] = (
            llama_cpp_config.get("api_base", "").strip() or "http://127.0.0.1:8080/v1"
        )
        llama_cpp_config = _apply_llama_cpp_defaults(
            llama_cpp_config,
            phase=phase,
            response_format=response_format,
        )
        llama_cpp_config["timeout"] = _resolve_timeout(llama_cpp_config, "llama_cpp")
        text, metadata = generate_text_with_metadata(
            prompt,
            {**llama_cpp_config, "model_provider": "openai_compatible"},
            log_context=log_context,
            system_prompt=system_prompt,
            response_format=response_format,
        )
        metadata["provider"] = "llama_cpp"
        return text, metadata

    raise ValueError(
        "Unsupported model_provider. Expected one of: "
        "'openai_compatible', 'gemini', 'grok', 'deepseek', 'doubao', 'ollama', 'llama_cpp'."
    )


def generate_text(prompt: str, config: dict, system_prompt: str = "", response_format: str = "") -> str:
    """Generate text from the configured backend."""
    return generate_text_with_metadata(
        prompt,
        config,
        system_prompt=system_prompt,
        response_format=response_format,
    )[0]
