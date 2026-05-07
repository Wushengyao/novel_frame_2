"""Shared runtime-config helpers for CLI and Web flows."""

from __future__ import annotations

import json
from pathlib import Path

from project_manager import DEFAULT_PLANNING_MODE, load_json, normalize_planning_mode
from workflow_modes import DEFAULT_WORKFLOW_MODE, normalize_workflow_mode


SUPPORTED_PROVIDERS = {
    "gemini",
    "grok",
    "deepseek",
    "doubao",
    "openai_compatible",
    "ollama",
    "llama_cpp",
}
WEB_SELECTABLE_PROVIDERS = {
    "gemini",
    "grok",
    "deepseek",
    "doubao",
    "ollama",
    "llama_cpp",
}
API_KEY_PROVIDERS = {"gemini", "grok", "deepseek", "doubao"}
PROVIDER_ALIASES = {
    "llama.cpp": "llama_cpp",
    "llama-cpp": "llama_cpp",
    "llamacpp": "llama_cpp",
}
DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite-preview",
    "grok": "grok-4.20-beta-latest-non-reasoning",
    "deepseek": "deepseek-v4-flash",
    "doubao": "doubao-seed-1-8-251228",
    "openai_compatible": "",
    "ollama": "llama3.2",
    "llama_cpp": "local-model",
}
DEFAULT_API_BASES = {
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "ollama": "http://127.0.0.1:11434/v1",
    "llama_cpp": "http://127.0.0.1:8080/v1",
}
DEFAULT_TIMEOUTS = {
    "ollama": 900,
    "llama_cpp": 900,
}
WRITING_QUALITY_LIGHT = "light"
WRITING_QUALITY_BALANCED = "balanced"
WRITING_QUALITY_HIGH = "high"
DEFAULT_WRITING_QUALITY_MODE = WRITING_QUALITY_BALANCED
WRITING_QUALITY_MODES = {
    WRITING_QUALITY_LIGHT,
    WRITING_QUALITY_BALANCED,
    WRITING_QUALITY_HIGH,
}
REVIEW_MODE_AUTO = "auto"
REVIEW_MODE_MANUAL = "manual"
DEFAULT_REVIEW_MODE = REVIEW_MODE_AUTO
REVIEW_MODES = {
    REVIEW_MODE_AUTO,
    REVIEW_MODE_MANUAL,
}
MODEL_PRESETS_PATH = Path(__file__).resolve().parent / "model_presets.json"
DEFAULT_MODEL_PRESETS = {
    "gemini": [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    "grok": [
        "grok-4.20-beta-latest-non-reasoning",
    ],
    "deepseek": [
        "deepseek-v4-flash",
        "deepseek-v4-pro",
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
    "llama_cpp": [
        "local-model",
    ],
}
RUNTIME_OVERRIDE_KEYS = (
    "provider",
    "model_name",
    "api_base",
    "max_tokens",
    "timeout",
    "planning_mode",
    "workflow_mode",
    "writing_quality_mode",
    "review_mode",
    "log_llm_payload",
)
QUALITY_MODEL_OVERRIDE_KEYS = {
    "quality_provider": "model_provider",
    "quality_model_name": "model_name",
    "quality_api_base": "api_base",
    "quality_max_tokens": "max_tokens",
    "quality_timeout": "timeout",
}
AUDIOBOOK_SEGMENT_MODEL_OVERRIDE_KEYS = {
    "audiobook_segment_provider": "model_provider",
    "audiobook_segment_model_name": "model_name",
    "audiobook_segment_api_base": "api_base",
    "audiobook_segment_max_tokens": "max_tokens",
    "audiobook_segment_timeout": "timeout",
}
EXPERT_MODE_MODEL_LIMIT = 3


def _coerce_bool(raw_value: object, default: bool = False) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None:
        return default

    raw = str(raw_value).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def normalize_provider(provider: object, default: str = "gemini") -> str:
    normalized = str(provider or "").strip().lower()
    normalized = PROVIDER_ALIASES.get(normalized, normalized)
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


def normalize_writing_quality_mode(mode: object, default: str = DEFAULT_WRITING_QUALITY_MODE) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in WRITING_QUALITY_MODES:
        return normalized
    return default


def normalize_review_mode(mode: object, default: str = DEFAULT_REVIEW_MODE) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in REVIEW_MODES:
        return normalized
    return default


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
    if normalize_provider(provider) in {"ollama", "llama_cpp"}:
        return max(timeout, default_timeout)
    return timeout


def api_key_for_provider(provider: str, api_keys: dict[str, str]) -> str:
    mapping = {
        "gemini": api_keys.get("GEMINI_API_KEY", ""),
        "grok": api_keys.get("GROK_API_KEY", ""),
        "deepseek": api_keys.get("DEEPSEEK_API_KEY", ""),
        "doubao": api_keys.get("DOUBAO_API_KEY", ""),
        "ollama": api_keys.get("OLLAMA_API_KEY", ""),
        "llama_cpp": api_keys.get("LLAMA_CPP_API_KEY", ""),
        "openai_compatible": api_keys.get("OPENAI_API_KEY", ""),
    }
    return mapping.get(normalize_provider(provider, default="openai_compatible"), "")


def _is_nonempty(value: object) -> bool:
    return value not in (None, "")


def _clean_quality_model_config(raw: object, *, include_api_key: bool = True) -> dict:
    source = raw if isinstance(raw, dict) else {}
    quality_model: dict[str, object] = {}

    provider = normalize_provider(source.get("model_provider") or source.get("provider"), default="")
    if provider:
        quality_model["model_provider"] = provider

    model = str(source.get("model_name") or source.get("model") or "").strip()
    if model:
        quality_model["model_name"] = model
        quality_model["model"] = model

    api_base = str(source.get("api_base", "") or "").strip()
    if api_base:
        quality_model["api_base"] = api_base

    if include_api_key:
        api_key = str(source.get("api_key", "") or "").strip()
        if api_key:
            quality_model["api_key"] = api_key

    for key in ("temperature", "max_tokens", "timeout"):
        value = source.get(key)
        if _is_nonempty(value):
            quality_model[key] = value

    return quality_model


def quality_model_configured(raw: object) -> bool:
    return bool(_clean_quality_model_config(raw, include_api_key=False))


def merge_quality_model_configs(base: object, override: object) -> dict:
    merged = _clean_quality_model_config(base)
    override_clean = _clean_quality_model_config(override)
    if override_clean.get("model_provider") and not (override_clean.get("model_name") or override_clean.get("model")):
        merged.pop("model_name", None)
        merged.pop("model", None)
    merged.update(override_clean)
    if "model_name" in merged:
        merged["model"] = merged["model_name"]
    elif "model" in merged:
        merged["model_name"] = merged["model"]
    return merged


def _parse_expert_models_json(value: object) -> list[object]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _clean_expert_mode_config(raw: object, *, include_api_key: bool = True) -> dict:
    if isinstance(raw, bool):
        return {"enabled": raw}
    source = raw if isinstance(raw, dict) else {}
    expert_mode: dict[str, object] = {}

    if "enabled" in source:
        expert_mode["enabled"] = _coerce_bool(source.get("enabled"))

    raw_models = source.get("models")
    if isinstance(raw_models, str):
        raw_models = _parse_expert_models_json(raw_models)
    elif raw_models is None and any(
        key in source
        for key in (
            "model_provider",
            "provider",
            "model_name",
            "model",
            "api_base",
            "api_key",
            "max_tokens",
            "timeout",
        )
    ):
        raw_models = [source]

    models: list[dict] = []
    if isinstance(raw_models, list):
        for item in raw_models:
            cleaned = _clean_quality_model_config(item, include_api_key=include_api_key)
            if not cleaned:
                continue
            models.append(cleaned)
            if len(models) >= EXPERT_MODE_MODEL_LIMIT:
                break
    if models:
        expert_mode["models"] = models
    return expert_mode


def expert_mode_configured(raw: object) -> bool:
    return bool(_clean_expert_mode_config(raw, include_api_key=False))


def expert_mode_enabled(config: dict) -> bool:
    expert_mode = _clean_expert_mode_config(config.get("expert_mode"), include_api_key=False)
    return _coerce_bool(expert_mode.get("enabled"), default=False)


def merge_expert_mode_configs(base: object, override: object) -> dict:
    merged = _clean_expert_mode_config(base)
    override_clean = _clean_expert_mode_config(override)
    if not override_clean:
        return merged
    if "enabled" in override_clean:
        merged["enabled"] = override_clean["enabled"]
    if "models" in override_clean:
        merged["models"] = override_clean["models"]
    return merged


def _resolve_auxiliary_model_config(
    config: dict,
    raw_model_config: object,
    *,
    default_temperature: float,
) -> dict:
    raw_model = _clean_quality_model_config(raw_model_config)
    base_provider = normalize_provider(config.get("model_provider"), default="openai_compatible")
    raw_provider = str(raw_model.get("model_provider", "") or "").strip()
    provider = normalize_provider(raw_provider, default=base_provider)
    provider_changed = provider != base_provider

    base_model = str(config.get("model_name") or config.get("model") or "").strip()
    raw_model_name = str(raw_model.get("model_name") or raw_model.get("model") or "").strip()
    model = raw_model_name or (default_model_for_provider(provider) if raw_provider else base_model)

    raw_api_base = str(raw_model.get("api_base", "") or "").strip()
    api_base = raw_api_base or (
        default_api_base_for_provider(provider)
        if provider_changed
        else str(config.get("api_base", "") or "").strip()
    )

    raw_api_key = str(raw_model.get("api_key", "") or "").strip()
    api_key = raw_api_key or (
        str(config.get("api_key", "") or "").strip()
        if provider == base_provider
        else ""
    )

    timeout_value = raw_model.get("timeout")
    if not _is_nonempty(timeout_value):
        timeout_value = default_timeout_for_provider(provider) if provider_changed else config.get("timeout")
    temperature_value = raw_model.get("temperature")
    if not _is_nonempty(temperature_value):
        temperature_value = default_temperature
    max_tokens_value = raw_model.get("max_tokens")
    if not _is_nonempty(max_tokens_value):
        max_tokens_value = config.get("max_tokens", 4000)

    resolved = dict(config)
    resolved.pop("quality_model", None)
    resolved.pop("expert_mode", None)
    resolved.pop("audiobook_segment_model", None)
    resolved.update(
        {
            "model_provider": provider,
            "model_name": model,
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "temperature": float(temperature_value),
            "max_tokens": int(max_tokens_value),
            "timeout": resolve_timeout_for_provider(
                provider,
                timeout_value or default_timeout_for_provider(provider),
            ),
            "log_llm_payload": True,
        }
    )
    return resolved


def resolve_expert_model_configs(config: dict) -> list[dict]:
    expert_mode = _clean_expert_mode_config(config.get("expert_mode"))
    if not _coerce_bool(expert_mode.get("enabled"), default=False):
        return []

    raw_models = expert_mode.get("models") if isinstance(expert_mode.get("models"), list) else []
    if not raw_models:
        quality_model = _clean_quality_model_config(config.get("quality_model"))
        raw_models = [quality_model] if quality_model_configured(quality_model) else [{}]

    return [
        _resolve_auxiliary_model_config(config, raw_model, default_temperature=0.2)
        for raw_model in raw_models
    ]


def resolve_quality_model_config(config: dict) -> tuple[dict, bool]:
    raw_quality_model = _clean_quality_model_config(config.get("quality_model"))
    if not quality_model_configured(raw_quality_model):
        resolved = dict(config)
        resolved.pop("quality_model", None)
        return resolved, False

    resolved = _resolve_auxiliary_model_config(config, raw_quality_model, default_temperature=float(config.get("temperature", 0.8)))
    resolved["log_llm_payload"] = config.get("log_llm_payload", False)
    return resolved, True


def resolve_audiobook_segment_model_config(config: dict) -> tuple[dict, bool]:
    raw_segment_model = _clean_quality_model_config(config.get("audiobook_segment_model"))
    if not quality_model_configured(raw_segment_model):
        resolved = dict(config)
        resolved.pop("audiobook_segment_model", None)
        return resolved, False

    resolved = _resolve_auxiliary_model_config(config, raw_segment_model, default_temperature=0.2)
    resolved["log_llm_payload"] = config.get("log_llm_payload", False)
    return resolved, True


def sanitize_runtime_overrides(overrides: dict | None) -> dict[str, object]:
    raw = overrides if isinstance(overrides, dict) else {}
    sanitized: dict[str, object] = {}
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
        if key == "workflow_mode":
            sanitized[key] = normalize_workflow_mode(value)
            continue
        if key == "writing_quality_mode":
            sanitized[key] = normalize_writing_quality_mode(value)
            continue
        if key == "review_mode":
            sanitized[key] = normalize_review_mode(value)
            continue
        sanitized[key] = str(value).strip()
    quality_model = _clean_quality_model_config(raw.get("quality_model"))
    for raw_key, quality_key in QUALITY_MODEL_OVERRIDE_KEYS.items():
        value = raw.get(raw_key)
        if value in (None, ""):
            continue
        if quality_key == "model_provider":
            normalized = normalize_provider(value, default="")
            if normalized:
                quality_model[quality_key] = normalized
            continue
        quality_model[quality_key] = str(value).strip()
        if quality_key == "model_name":
            quality_model["model"] = str(value).strip()
    if quality_model:
        sanitized["quality_model"] = quality_model

    audiobook_segment_model = _clean_quality_model_config(raw.get("audiobook_segment_model"))
    for raw_key, model_key in AUDIOBOOK_SEGMENT_MODEL_OVERRIDE_KEYS.items():
        value = raw.get(raw_key)
        if value in (None, ""):
            continue
        if model_key == "model_provider":
            normalized = normalize_provider(value, default="")
            if normalized:
                audiobook_segment_model[model_key] = normalized
            continue
        audiobook_segment_model[model_key] = str(value).strip()
        if model_key == "model_name":
            audiobook_segment_model["model"] = str(value).strip()
    if audiobook_segment_model:
        sanitized["audiobook_segment_model"] = audiobook_segment_model

    expert_mode = _clean_expert_mode_config(raw.get("expert_mode"))
    for enabled_key in ("expert_mode_enabled", "expert_enabled"):
        if enabled_key in raw and raw.get(enabled_key) not in (None, ""):
            expert_mode["enabled"] = _coerce_bool(raw.get(enabled_key))
    raw_expert_models = raw.get("expert_models")
    if raw_expert_models not in (None, ""):
        if isinstance(raw_expert_models, str):
            raw_expert_models = _parse_expert_models_json(raw_expert_models)
        if isinstance(raw_expert_models, list):
            models = []
            for item in raw_expert_models:
                cleaned = _clean_quality_model_config(item)
                if cleaned:
                    models.append(cleaned)
                if len(models) >= EXPERT_MODE_MODEL_LIMIT:
                    break
            expert_mode["models"] = models
    raw_expert_models_json = raw.get("expert_models_json")
    if raw_expert_models_json not in (None, ""):
        models = []
        for item in _parse_expert_models_json(raw_expert_models_json):
            cleaned = _clean_quality_model_config(item)
            if cleaned:
                models.append(cleaned)
            if len(models) >= EXPERT_MODE_MODEL_LIMIT:
                break
        expert_mode["models"] = models
    if expert_mode:
        sanitized["expert_mode"] = expert_mode
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
        "workflow_mode": normalize_workflow_mode(raw.get("workflow_mode")),
        "writing_quality_mode": normalize_writing_quality_mode(raw.get("writing_quality_mode")),
        "review_mode": normalize_review_mode(raw.get("review_mode")),
        "log_llm_payload": _coerce_bool(raw.get("log_llm_payload")),
    }
    quality_model = _clean_quality_model_config(raw.get("quality_model"))
    if quality_model:
        config["quality_model"] = quality_model
    audiobook_segment_model = _clean_quality_model_config(raw.get("audiobook_segment_model"))
    if audiobook_segment_model:
        config["audiobook_segment_model"] = audiobook_segment_model
    expert_mode = _clean_expert_mode_config(raw.get("expert_mode"))
    if expert_mode:
        config["expert_mode"] = expert_mode
        if _coerce_bool(expert_mode.get("enabled"), default=False):
            config["log_llm_payload"] = True
    return config


def extract_llm_config(config_path: str) -> dict:
    return _normalized_llm_config(load_json(str(Path(config_path).resolve())))


def load_runtime_config(project_path: str) -> dict:
    project = load_json(str(Path(project_path) / "project.json"))
    return {
        **_normalized_llm_config(project.get("llm_config") or {}),
        "project_path": str(Path(project_path).resolve()),
        "planning_mode": normalize_planning_mode(project.get("planning_mode"), default=DEFAULT_PLANNING_MODE),
        "workflow_mode": normalize_workflow_mode(
            project.get("workflow_mode")
            or (project.get("llm_config") or {}).get("workflow_mode")
            or DEFAULT_WORKFLOW_MODE
        ),
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
        "workflow_mode": normalize_workflow_mode(
            runtime_overrides.get("workflow_mode")
            or project.get("workflow_mode")
            or saved.get("workflow_mode")
            or DEFAULT_WORKFLOW_MODE
        ),
        "writing_quality_mode": normalize_writing_quality_mode(
            runtime_overrides.get("writing_quality_mode") or saved.get("writing_quality_mode")
        ),
        "review_mode": normalize_review_mode(
            runtime_overrides.get("review_mode") or saved.get("review_mode")
        ),
        "log_llm_payload": _coerce_bool(
            runtime_overrides.get("log_llm_payload"),
            default=_coerce_bool(saved.get("log_llm_payload")),
        ),
        "project_path": str(Path(project_path).resolve()),
    }
    if provider_requires_api_key(provider) and not runtime["api_key"]:
        raise RuntimeError(f"provider={provider} missing API key, please fill api_keys.sh")
    quality_model = merge_quality_model_configs(saved.get("quality_model"), runtime_overrides.get("quality_model"))
    if quality_model:
        quality_provider = normalize_provider(quality_model.get("model_provider"), default=provider)
        if not str(quality_model.get("api_key", "") or "").strip():
            quality_model["api_key"] = (
                runtime["api_key"]
                if quality_provider == provider
                else api_key_for_provider(quality_provider, api_keys)
            )
        runtime["quality_model"] = quality_model
        quality_runtime, _ = resolve_quality_model_config(runtime)
        quality_provider = normalize_provider(quality_runtime.get("model_provider"), default=provider)
        if provider_requires_api_key(quality_provider) and not str(quality_runtime.get("api_key", "") or "").strip():
            raise RuntimeError(f"quality provider={quality_provider} missing API key, please fill api_keys.sh")
    audiobook_segment_model = merge_quality_model_configs(
        saved.get("audiobook_segment_model"),
        runtime_overrides.get("audiobook_segment_model"),
    )
    if audiobook_segment_model:
        segment_provider = normalize_provider(audiobook_segment_model.get("model_provider"), default=provider)
        if not str(audiobook_segment_model.get("api_key", "") or "").strip():
            audiobook_segment_model["api_key"] = (
                runtime["api_key"]
                if segment_provider == provider
                else api_key_for_provider(segment_provider, api_keys)
            )
        runtime["audiobook_segment_model"] = audiobook_segment_model
        segment_runtime, _ = resolve_audiobook_segment_model_config(runtime)
        segment_provider = normalize_provider(segment_runtime.get("model_provider"), default=provider)
        if provider_requires_api_key(segment_provider) and not str(segment_runtime.get("api_key", "") or "").strip():
            raise RuntimeError(f"audiobook segment provider={segment_provider} missing API key, please fill api_keys.sh")
    expert_mode = merge_expert_mode_configs(saved.get("expert_mode"), runtime_overrides.get("expert_mode"))
    if expert_mode:
        models = []
        for model_config in expert_mode.get("models") or []:
            model_copy = dict(model_config)
            model_provider = normalize_provider(model_copy.get("model_provider"), default=provider)
            if not str(model_copy.get("api_key", "") or "").strip():
                model_copy["api_key"] = (
                    runtime["api_key"]
                    if model_provider == provider
                    else api_key_for_provider(model_provider, api_keys)
                )
            models.append(model_copy)
        if models:
            expert_mode["models"] = models
        runtime["expert_mode"] = expert_mode
        if _coerce_bool(expert_mode.get("enabled"), default=False):
            runtime["log_llm_payload"] = True
            for expert_runtime in resolve_expert_model_configs(runtime):
                expert_provider = normalize_provider(expert_runtime.get("model_provider"), default=provider)
                if provider_requires_api_key(expert_provider) and not str(expert_runtime.get("api_key", "") or "").strip():
                    raise RuntimeError(f"expert provider={expert_provider} missing API key, please fill api_keys.sh")
    return runtime
