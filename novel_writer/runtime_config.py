"""Shared runtime-config helpers for CLI and Web flows."""

from __future__ import annotations

import json
from pathlib import Path

from project_manager import DEFAULT_PLANNING_MODE, load_json, normalize_planning_mode


SUPPORTED_PROVIDERS = {
    "gemini",
    "grok",
    "deepseek",
    "doubao",
    "openai_compatible",
    "ollama",
}
WEB_SELECTABLE_PROVIDERS = {
    "gemini",
    "grok",
    "deepseek",
    "doubao",
    "ollama",
}
API_KEY_PROVIDERS = {"gemini", "grok", "deepseek", "doubao"}
DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite-preview",
    "grok": "grok-4.20-beta-latest-non-reasoning",
    "deepseek": "deepseek-chat",
    "doubao": "doubao-seed-1-8-251228",
    "openai_compatible": "",
    "ollama": "llama3.2",
}
DEFAULT_API_BASES = {
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "ollama": "http://127.0.0.1:11434/v1",
}
DEFAULT_TIMEOUTS = {
    "ollama": 900,
}
MODEL_PRESETS_PATH = Path(__file__).resolve().parent / "model_presets.json"
DEFAULT_MODEL_PRESETS = {
    "gemini": [
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "grok": [
        "grok-4.20-beta-latest-non-reasoning",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "doubao": [
        "doubao-seed-1-8-251228",
    ],
    "openai_compatible": [],
    "ollama": [
        "llama3.2",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "mistral:7b",
    ],
}
RUNTIME_OVERRIDE_KEYS = (
    "provider",
    "model_name",
    "api_base",
    "temperature",
    "max_tokens",
    "timeout",
    "planning_mode",
    "log_llm_payload",
)


def _coerce_bool(raw_value: object, default: bool = False) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default

    raw = str(raw_value).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def normalize_provider(provider: object, default: str = "gemini") -> str:
    normalized = str(provider or "").strip().lower()
    if normalized in SUPPORTED_PROVIDERS:
        return normalized
    return default


def provider_requires_api_key(provider: str) -> bool:
    return normalize_provider(provider) in API_KEY_PROVIDERS


def default_model_for_provider(provider: str) -> str:
    return DEFAULT_MODELS.get(normalize_provider(provider, default="openai_compatible"), "")


def default_api_base_for_provider(provider: str) -> str:
    return DEFAULT_API_BASES.get(normalize_provider(provider, default="openai_compatible"), "")


def default_timeout_for_provider(provider: str) -> int:
    normalized = normalize_provider(provider, default="openai_compatible")
    return DEFAULT_TIMEOUTS.get(normalized, 120)


def _normalize_model_preset_entry(entry: object) -> dict[str, str] | None:
    if isinstance(entry, str):
        value = entry.strip()
        label = value
    elif isinstance(entry, dict):
        value = str(entry.get("value") or entry.get("model") or "").strip()
        label = str(entry.get("label") or value).strip()
    else:
        return None
    if not value:
        return None
    return {
        "value": value,
        "label": label or value,
    }


def load_model_presets() -> dict[str, list[dict[str, str]]]:
    raw_presets: object = DEFAULT_MODEL_PRESETS
    if MODEL_PRESETS_PATH.exists():
        try:
            parsed = json.loads(MODEL_PRESETS_PATH.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw_presets = parsed
        except (OSError, ValueError, TypeError):
            raw_presets = DEFAULT_MODEL_PRESETS

    source = raw_presets if isinstance(raw_presets, dict) else DEFAULT_MODEL_PRESETS
    normalized: dict[str, list[dict[str, str]]] = {}
    for provider in sorted(SUPPORTED_PROVIDERS):
        entries = source.get(provider)
        items: list[dict[str, str]] = []
        seen_values: set[str] = set()
        if isinstance(entries, list):
            for entry in entries:
                item = _normalize_model_preset_entry(entry)
                if item is None or item["value"] in seen_values:
                    continue
                items.append(item)
                seen_values.add(item["value"])

        default_model = default_model_for_provider(provider)
        if default_model and default_model not in seen_values:
            items.insert(0, {"value": default_model, "label": default_model})
        normalized[provider] = items
    return normalized


def resolve_timeout_for_provider(provider: str, raw_value: object) -> int:
    default_timeout = default_timeout_for_provider(provider)
    try:
        timeout = int(raw_value)
    except (TypeError, ValueError):
        timeout = default_timeout
    if timeout <= 0:
        timeout = default_timeout
    if normalize_provider(provider) == "ollama":
        return max(timeout, default_timeout)
    return timeout


def api_key_for_provider(provider: str, api_keys: dict[str, str]) -> str:
    mapping = {
        "gemini": api_keys.get("GEMINI_API_KEY", ""),
        "grok": api_keys.get("GROK_API_KEY", ""),
        "deepseek": api_keys.get("DEEPSEEK_API_KEY", ""),
        "doubao": api_keys.get("DOUBAO_API_KEY", ""),
        "ollama": api_keys.get("OLLAMA_API_KEY", ""),
        "openai_compatible": api_keys.get("OPENAI_API_KEY", ""),
    }
    return mapping.get(normalize_provider(provider, default="openai_compatible"), "")


def sanitize_runtime_overrides(overrides: dict | None) -> dict[str, str]:
    raw = overrides if isinstance(overrides, dict) else {}
    sanitized: dict[str, str] = {}
    for key in RUNTIME_OVERRIDE_KEYS:
        value = raw.get(key)
        if value in (None, ""):
            continue
        if key == "provider":
            normalized = normalize_provider(value, default="")
            if normalized:
                sanitized[key] = normalized
            continue
        if key == "planning_mode":
            sanitized[key] = normalize_planning_mode(value, default=DEFAULT_PLANNING_MODE)
            continue
        sanitized[key] = str(value).strip()
    return sanitized


def _normalized_llm_config(raw: dict) -> dict:
    provider = normalize_provider(raw.get("model_provider"), default="openai_compatible")
    model = str(raw.get("model") or raw.get("model_name") or "").strip()
    config = {
        "model_provider": provider,
        "model": model,
        "model_name": model,
        "api_base": str(raw.get("api_base", "") or "").strip(),
        "api_key": str(raw.get("api_key", "") or "").strip(),
        "temperature": raw.get("temperature", 0.8),
        "max_tokens": raw.get("max_tokens", 4000),
        "timeout": resolve_timeout_for_provider(provider, raw.get("timeout", default_timeout_for_provider(provider))),
        "planning_mode": normalize_planning_mode(raw.get("planning_mode"), default=DEFAULT_PLANNING_MODE),
        "log_llm_payload": _coerce_bool(raw.get("log_llm_payload")),
    }
    return config


def extract_llm_config(config_path: str) -> dict:
    return _normalized_llm_config(load_json(str(Path(config_path).resolve())))


def load_runtime_config(project_path: str) -> dict:
    project = load_json(str(Path(project_path) / "project.json"))
    return {
        **_normalized_llm_config(project.get("llm_config") or {}),
        "project_path": str(Path(project_path).resolve()),
        "planning_mode": normalize_planning_mode(project.get("planning_mode"), default=DEFAULT_PLANNING_MODE),
    }


def build_runtime_config(project_path: str | Path, overrides: dict[str, object], api_keys: dict[str, str]) -> dict:
    project_file = Path(project_path) / "project.json"
    project = load_json(str(project_file))
    saved = project.get("llm_config") or {}
    runtime_overrides = sanitize_runtime_overrides(overrides)

    saved_provider = normalize_provider(saved.get("model_provider"), default="gemini")
    provider = normalize_provider(runtime_overrides.get("provider"), default=saved_provider)
    saved_model_name = str(saved.get("model_name") or saved.get("model") or "").strip()
    saved_api_base = str(saved.get("api_base", "") or "").strip()

    model_name = (
        runtime_overrides.get("model_name")
        or (
            default_model_for_provider(provider)
            if provider != saved_provider
            else saved_model_name
        )
        or default_model_for_provider(provider)
    )
    api_base = runtime_overrides.get("api_base") or (
        default_api_base_for_provider(provider)
        if provider != saved_provider
        else (saved_api_base or default_api_base_for_provider(provider))
    )
    runtime = {
        "model_provider": provider,
        "model_name": model_name,
        "model": model_name,
        "api_base": api_base,
        "api_key": api_key_for_provider(provider, api_keys) or str(runtime_overrides.get("api_key", "") or ""),
        "temperature": float(runtime_overrides.get("temperature") or saved.get("temperature", 0.8)),
        "max_tokens": int(runtime_overrides.get("max_tokens") or saved.get("max_tokens", 4000)),
        "timeout": resolve_timeout_for_provider(
            provider,
            runtime_overrides.get("timeout") or saved.get("timeout", default_timeout_for_provider(provider)),
        ),
        "planning_mode": normalize_planning_mode(
            runtime_overrides.get("planning_mode") or project.get("planning_mode"),
            default=DEFAULT_PLANNING_MODE,
        ),
        "log_llm_payload": _coerce_bool(
            runtime_overrides.get("log_llm_payload"),
            default=_coerce_bool(saved.get("log_llm_payload")),
        ),
        "project_path": str(Path(project_path).resolve()),
    }
    if provider_requires_api_key(provider) and not runtime["api_key"]:
        raise RuntimeError(f"provider={provider} missing API key, please fill api_keys.sh")
    return runtime
