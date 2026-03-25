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
    echo "缺少 API key 文件: $keys_file" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$keys_file"
}

normalize_provider() {
  local provider="${1:-}"
  case "$provider" in
    gemini|grok|deepseek|doubao) printf '%s\n' "$provider" ;;
    *)
      echo "不支持的 provider: $provider（可选: gemini / grok / deepseek / doubao）" >&2
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
  esac
}

default_api_base_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    gemini|grok|deepseek) printf '%s\n' "" ;;
    doubao) printf '%s\n' "https://ark.cn-beijing.volces.com/api/v3" ;;
  esac
}

default_thinking_level_for_provider() {
  local provider
  provider="$(normalize_provider "$1")"
  case "$provider" in
    gemini) printf '%s\n' "medium" ;;
    grok|deepseek|doubao) printf '%s\n' "" ;;
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
  esac
}

ensure_api_key_present() {
  local provider="$1"
  local api_key="$2"
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
data = {
    "project_name": os.environ["NOVEL_PROJECT_NAME"],
    "project_description": os.environ["NOVEL_PROJECT_DESCRIPTION"],
    "project_path": str(output_root / "novel_project_{project_id}"),
    "init_with_llm": True,
    "story_request": os.environ["NOVEL_STORY_REQUEST"],
    "model_provider": provider,
    "model_name": os.environ["NOVEL_MODEL_NAME"],
    "api_base": os.environ["NOVEL_API_BASE"],
    "api_key": os.environ["NOVEL_API_KEY"],
    "temperature": float(os.environ["NOVEL_TEMPERATURE"]),
    "max_tokens": int(os.environ["NOVEL_MAX_TOKENS"]),
    "timeout": int(os.environ["NOVEL_TIMEOUT"]),
}
thinking_level = os.environ.get("NOVEL_THINKING_LEVEL", "").strip()
if thinking_level:
    data["thinking_level"] = thinking_level

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
default_models = {
  "gemini": "gemini-3.1-pro-preview",
  "grok": "grok-4.20-beta-latest-reasoning",
  "deepseek": "deepseek-reasoner",
  "doubao": "doubao-seed-2-0-pro-260215",
}
default_api_bases = {
  "doubao": "https://ark.cn-beijing.volces.com/api/v3",
}
default_thinking_levels = {
  "gemini": "medium",
}

model_name_override = os.environ.get("NOVEL_MODEL_NAME_OVERRIDE", "").strip()
api_base_override = os.environ.get("NOVEL_API_BASE_OVERRIDE", "").strip()

data = {
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
    "timeout": int(os.environ.get("NOVEL_TIMEOUT_OVERRIDE", "") or saved.get("timeout", 120)),
}

thinking_level = os.environ.get("NOVEL_THINKING_LEVEL_OVERRIDE", "").strip()
if not thinking_level:
  if resolved_provider == saved_provider:
    thinking_level = str(saved.get("thinking_level", "") or "")
  else:
    thinking_level = default_thinking_levels.get(resolved_provider, "")
if thinking_level:
    data["thinking_level"] = thinking_level

thinking_budget = str(saved.get("thinking_budget", "") or "").strip()
if thinking_budget:
    data["thinking_budget"] = thinking_budget

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

data = {
  "model_provider": resolved_provider,
  "model_name": os.environ.get("NOVEL_MODEL_NAME_OVERRIDE", "").strip()
  or str(saved.get("model_name") or saved.get("model", "") or ""),
  "api_base": os.environ.get("NOVEL_API_BASE_OVERRIDE", "").strip()
  or str(saved.get("api_base", "") or ""),
  "api_key": os.environ.get("NOVEL_API_KEY", "").strip(),
  "temperature": float(os.environ.get("NOVEL_TEMPERATURE_OVERRIDE", "") or saved.get("temperature", 0.8)),
  "max_tokens": int(os.environ.get("NOVEL_MAX_TOKENS_OVERRIDE", "") or saved.get("max_tokens", 4000)),
  "timeout": int(os.environ.get("NOVEL_TIMEOUT_OVERRIDE", "") or saved.get("timeout", 120)),
}

thinking_level = str(saved.get("thinking_level", "") or "").strip()
if thinking_level:
  data["thinking_level"] = thinking_level

thinking_budget = str(saved.get("thinking_budget", "") or "").strip()
if thinking_budget:
  data["thinking_budget"] = thinking_budget

with open(config_path, "w", encoding="utf-8") as fh:
  json.dump(data, fh, ensure_ascii=False, indent=2)
PY
}
