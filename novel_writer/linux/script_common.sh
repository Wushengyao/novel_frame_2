#!/usr/bin/env bash

SCRIPT_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DIR="$(cd "${SCRIPT_COMMON_DIR}/.." && pwd)"

script_log() {
  local level="$1"
  shift
  printf '[%s] [%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$level" "$*" >&2
}

log_info() {
  script_log "INFO" "$@"
}

log_success() {
  script_log "SUCCESS" "$@"
}

log_warning() {
  script_log "WARN" "$@"
}

log_error() {
  script_log "ERROR" "$@"
}

prompt_optional_value() {
  local prompt_text="$1"
  local default_value="${2:-}"
  local input_value=""

  if [[ ! -t 0 && ! -t 1 ]]; then
    printf '%s\n' "$default_value"
    return 0
  fi

  if [[ -n "$default_value" ]]; then
    read -r -p "$prompt_text [$default_value] " input_value
    printf '%s\n' "${input_value:-$default_value}"
    return 0
  fi

  read -r -p "$prompt_text " input_value
  printf '%s\n' "$input_value"
}

resolve_python_exe() {
  if [[ -n "${NOVEL_PYTHON_EXE:-}" && -x "${NOVEL_PYTHON_EXE}" ]]; then
    printf '%s\n' "${NOVEL_PYTHON_EXE}"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  echo "未找到 Python，请先安装 Python，或设置 NOVEL_PYTHON_EXE。" >&2
  exit 1
}

load_api_keys() {
  local keys_file="${PROJECT_ROOT_DIR}/api_keys.sh"
  if [[ ! -f "$keys_file" ]]; then
    export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
    export GROK_API_KEY="${GROK_API_KEY:-}"
    export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
    export DOUBAO_API_KEY="${DOUBAO_API_KEY:-}"
    export OLLAMA_API_KEY="${OLLAMA_API_KEY:-}"
    log_warning "未找到 API key 文件: $keys_file，后续将仅依赖环境变量或无需 API key 的 provider。"
    return 0
  fi

  # shellcheck disable=SC1090
  source "$keys_file"
}

normalize_provider() {
  local provider="${1:-}"
  case "$provider" in
    gemini|grok|deepseek|doubao|ollama) printf '%s\n' "$provider" ;;
    *)
      echo "不支持的 provider: $provider（可选: gemini / grok / deepseek / doubao / ollama）" >&2
      exit 1
      ;;
  esac
}

normalize_planning_mode() {
  local mode="${1:-}"
  case "${mode,,}" in
    none|volume|chapter) printf '%s\n' "${mode,,}" ;;
    "")
      printf '%s\n' "chapter"
      ;;
    *)
      echo "Unsupported planning mode: $mode (allowed: none / volume / chapter)" >&2
      exit 1
      ;;
  esac
}

default_model_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    gemini) printf '%s\n' "gemini-3.1-pro-preview" ;;
    grok) printf '%s\n' "grok-4.20-beta-latest-reasoning" ;;
    deepseek) printf '%s\n' "deepseek-reasoner" ;;
    doubao) printf '%s\n' "doubao-seed-2-0-pro-260215" ;;
    ollama) printf '%s\n' "qwen3.5:35b" ;;
  esac
}

default_api_base_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    gemini|grok|deepseek) printf '%s\n' "" ;;
    doubao) printf '%s\n' "https://ark.cn-beijing.volces.com/api/v3" ;;
    ollama) printf '%s\n' "http://127.0.0.1:11434/v1" ;;
  esac
}

default_timeout_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    ollama) printf '%s\n' "900" ;;
    gemini|grok|deepseek|doubao) printf '%s\n' "120" ;;
  esac
}

api_key_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    gemini) printf '%s\n' "${GEMINI_API_KEY:-}" ;;
    grok) printf '%s\n' "${GROK_API_KEY:-}" ;;
    deepseek) printf '%s\n' "${DEEPSEEK_API_KEY:-}" ;;
    doubao) printf '%s\n' "${DOUBAO_API_KEY:-}" ;;
    ollama) printf '%s\n' "${OLLAMA_API_KEY:-}" ;;
  esac
}

ensure_api_key_present() {
  local provider="$1"
  local api_key="$2"
  if [[ "$provider" == "ollama" ]]; then
    return 0
  fi
  if [[ -z "$api_key" ]]; then
    echo "provider=$provider 缺少 API key，请先填写 ${PROJECT_ROOT_DIR}/api_keys.sh" >&2
    exit 1
  fi
}

make_temp_config_path() {
  mktemp "${TMPDIR:-/tmp}/novel_writer_config.XXXXXX.json"
}

get_latest_project_path() {
  local output_root="$1"
  find "$output_root" -maxdepth 1 -mindepth 1 -type d -name 'novel_project_*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

test_illustration_connection_failure() {
  local text="${1:-}"
  [[ "$text" == *"Failed to connect to ComfyUI"* ]] \
    || [[ "$text" == *"actively refused"* ]] \
    || [[ "$text" == *"Connection refused"* ]] \
    || [[ "$text" == *"WinError 10061"* ]] \
    || [[ "$text" == *"No connection could be made"* ]] \
    || [[ "$text" == *"timed out"* ]]
}

write_init_config() {
  local output_path="$1"
  local python_exe
  python_exe="$(resolve_python_exe)"
  "$python_exe" - "$output_path" "$PROJECT_ROOT_DIR" <<'PY'
import json
import os
import pathlib
import sys

path = sys.argv[1]
project_root = pathlib.Path(sys.argv[2]).resolve()
output_root = project_root / "output"
output_root.mkdir(parents=True, exist_ok=True)
provider = os.environ["NOVEL_PROVIDER"]
default_timeouts = {
    "ollama": 900,
    "gemini": 120,
    "grok": 120,
    "deepseek": 120,
    "doubao": 120,
}
data = {
    "project_name": os.environ["NOVEL_PROJECT_NAME"],
    "project_description": os.environ["NOVEL_PROJECT_DESCRIPTION"],
    "project_path": str(output_root / "novel_project_{project_id}"),
    "init_with_llm": True,
    "story_request": os.environ["NOVEL_STORY_REQUEST"],
    "planning_mode": os.environ.get("NOVEL_PLANNING_MODE", "chapter").strip() or "chapter",
    "model_provider": provider,
    "model_name": os.environ["NOVEL_MODEL_NAME"],
    "api_base": os.environ["NOVEL_API_BASE"],
    "api_key": os.environ["NOVEL_API_KEY"],
    "temperature": float(os.environ["NOVEL_TEMPERATURE"]),
    "max_tokens": int(os.environ["NOVEL_MAX_TOKENS"]),
    "timeout": int(os.environ.get("NOVEL_TIMEOUT") or default_timeouts.get(provider, 120)),
}
quality_model = {}
quality_provider = os.environ.get("NOVEL_QUALITY_PROVIDER", "").strip()
quality_model_name = os.environ.get("NOVEL_QUALITY_MODEL_NAME", "").strip()
if quality_provider:
    quality_model["model_provider"] = quality_provider
    if not quality_model_name:
        quality_model.pop("model_name", None)
        quality_model.pop("model", None)
if quality_model_name:
    quality_model["model_name"] = quality_model_name
    quality_model["model"] = quality_model_name
for env_name, key in (
    ("NOVEL_QUALITY_API_BASE", "api_base"),
    ("NOVEL_QUALITY_API_KEY", "api_key"),
    ("NOVEL_QUALITY_TEMPERATURE", "temperature"),
    ("NOVEL_QUALITY_MAX_TOKENS", "max_tokens"),
    ("NOVEL_QUALITY_TIMEOUT", "timeout"),
):
    value = os.environ.get(env_name, "").strip()
    if not value:
        continue
    if key == "temperature":
        quality_model[key] = float(value)
    elif key in {"max_tokens", "timeout"}:
        quality_model[key] = int(value)
    else:
        quality_model[key] = value
if quality_model:
    data["quality_model"] = quality_model

outline_request = os.environ.get("NOVEL_OUTLINE_REQUEST", "").strip()
if outline_request:
    data["outline_request"] = outline_request

with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
PY
}

write_continue_config() {
  local output_path="$1"
  local project_path="$2"
  local python_exe
  python_exe="$(resolve_python_exe)"
  "$python_exe" - "$output_path" "$project_path" <<'PY'
import json
import os
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
project_path = pathlib.Path(sys.argv[2])
project = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
saved = project.get("llm_config", {})
saved_provider = str(saved.get("model_provider", "gemini") or "gemini").strip().lower()
resolved_provider = os.environ.get("NOVEL_PROVIDER_OVERRIDE", "").strip() or saved_provider
saved_planning_mode = str(project.get("planning_mode", "chapter") or "chapter").strip().lower() or "chapter"
resolved_planning_mode = os.environ.get("NOVEL_PLANNING_MODE_OVERRIDE", "").strip() or saved_planning_mode
default_models = {
  "gemini": "gemini-3.1-pro-preview",
  "grok": "grok-4.20-beta-latest-reasoning",
  "deepseek": "deepseek-reasoner",
  "doubao": "doubao-seed-2-0-pro-260215",
  "ollama": "llama3.2",
}
default_api_bases = {
  "doubao": "https://ark.cn-beijing.volces.com/api/v3",
  "ollama": "http://127.0.0.1:11434/v1",
}
default_timeouts = {
  "gemini": 120,
  "grok": 120,
  "deepseek": 120,
  "doubao": 120,
  "ollama": 900,
}

model_name_override = os.environ.get("NOVEL_MODEL_NAME_OVERRIDE", "").strip()
api_base_override = os.environ.get("NOVEL_API_BASE_OVERRIDE", "").strip()

data = {
  "planning_mode": resolved_planning_mode,
  "model_provider": resolved_provider,
  "model_name": model_name_override
  or (
    default_models.get(resolved_provider, "")
    if resolved_provider != saved_provider
    else (saved.get("model_name") or saved.get("model", ""))
  )
  or default_models.get(resolved_provider, ""),
  "api_base": api_base_override
  or (
    default_api_bases.get(resolved_provider, "")
    if resolved_provider != saved_provider
    else (saved.get("api_base", "") or default_api_bases.get(resolved_provider, ""))
  ),
    "api_key": os.environ["NOVEL_API_KEY"],
    "temperature": float(os.environ.get("NOVEL_TEMPERATURE_OVERRIDE", "") or saved.get("temperature", 0.8)),
    "max_tokens": int(os.environ.get("NOVEL_MAX_TOKENS_OVERRIDE", "") or saved.get("max_tokens", 4000)),
    "timeout": int(
      os.environ.get("NOVEL_TIMEOUT_OVERRIDE", "")
      or (
        max(int(saved.get("timeout", 0) or 0), default_timeouts.get("ollama", 900))
        if resolved_provider == "ollama"
        else (saved.get("timeout", default_timeouts.get(resolved_provider, 120)))
      )
    ),
}
saved_quality = saved.get("quality_model") if isinstance(saved.get("quality_model"), dict) else {}
quality_model = dict(saved_quality)
quality_provider = os.environ.get("NOVEL_QUALITY_PROVIDER", "").strip()
quality_model_name = os.environ.get("NOVEL_QUALITY_MODEL_NAME", "").strip()
if quality_provider:
    quality_model["model_provider"] = quality_provider
    if not quality_model_name:
        quality_model.pop("model_name", None)
        quality_model.pop("model", None)
if quality_model_name:
    quality_model["model_name"] = quality_model_name
    quality_model["model"] = quality_model_name
for env_name, key in (
    ("NOVEL_QUALITY_API_BASE", "api_base"),
    ("NOVEL_QUALITY_API_KEY", "api_key"),
    ("NOVEL_QUALITY_TEMPERATURE", "temperature"),
    ("NOVEL_QUALITY_MAX_TOKENS", "max_tokens"),
    ("NOVEL_QUALITY_TIMEOUT", "timeout"),
):
    value = os.environ.get(env_name, "").strip()
    if not value:
        continue
    if key == "temperature":
        quality_model[key] = float(value)
    elif key in {"max_tokens", "timeout"}:
        quality_model[key] = int(value)
    else:
        quality_model[key] = value
if quality_model:
    data["quality_model"] = quality_model

with open(config_path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
PY
}

write_illustrate_config() {
  local output_path="$1"
  local project_path="$2"
  local python_exe
  python_exe="$(resolve_python_exe)"
  "$python_exe" - "$output_path" "$project_path" <<'PY'
import json
import os
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
project_path = pathlib.Path(sys.argv[2])
project = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
saved = project.get("llm_config", {})
resolved_provider = str(saved.get("model_provider", "") or "").strip().lower()
default_timeouts = {
  "gemini": 120,
  "grok": 120,
  "deepseek": 120,
  "doubao": 120,
  "ollama": 900,
}

data = {
  "model_provider": resolved_provider,
  "model_name": os.environ.get("NOVEL_MODEL_NAME_OVERRIDE", "").strip()
  or str(saved.get("model_name") or saved.get("model", "") or ""),
  "api_base": os.environ.get("NOVEL_API_BASE_OVERRIDE", "").strip()
  or str(saved.get("api_base", "") or ""),
  "api_key": os.environ.get("NOVEL_API_KEY", "").strip(),
  "temperature": float(os.environ.get("NOVEL_TEMPERATURE_OVERRIDE", "") or saved.get("temperature", 0.8)),
  "max_tokens": int(os.environ.get("NOVEL_MAX_TOKENS_OVERRIDE", "") or saved.get("max_tokens", 4000)),
  "timeout": int(
    os.environ.get("NOVEL_TIMEOUT_OVERRIDE", "")
    or (
      max(int(saved.get("timeout", 0) or 0), default_timeouts.get("ollama", 900))
      if resolved_provider == "ollama"
      else (saved.get("timeout", default_timeouts.get(resolved_provider, 120)))
    )
  ),
}

with open(config_path, "w", encoding="utf-8") as fh:
  json.dump(data, fh, ensure_ascii=False, indent=2)
PY
}
