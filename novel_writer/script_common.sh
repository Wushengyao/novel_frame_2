#!/usr/bin/env bash

SCRIPT_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

load_api_keys() {
  local keys_file="${SCRIPT_COMMON_DIR}/api_keys.sh"
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
    gemini) printf '%s\n' "gemini-3.1-flash-lite-preview" ;;
    grok) printf '%s\n' "grok-4.20-beta-latest-non-reasoning" ;;
    deepseek) printf '%s\n' "deepseek-chat" ;;
    doubao) printf '%s\n' "doubao-seed-1-8-251228" ;;
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
    echo "provider=$provider 缺少 API key，请先填写 ${SCRIPT_COMMON_DIR}/api_keys.sh" >&2
    exit 1
  fi
}

make_temp_config_path() {
  mktemp "${TMPDIR:-/tmp}/novel_writer_config.XXXXXX.json"
}

write_init_config() {
  local output_path="$1"
  python3 - "$output_path" "$SCRIPT_COMMON_DIR" <<'PY'
import json
import os
import pathlib
import sys

path = sys.argv[1]
script_dir = pathlib.Path(sys.argv[2]).resolve()
output_root = script_dir / "output"
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

with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
PY
}

write_continue_config() {
  local output_path="$1"
  local project_path="$2"
  python3 - "$output_path" "$project_path" <<'PY'
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
  "gemini": "gemini-3.1-flash-lite-preview",
  "grok": "grok-4.20-beta-latest-non-reasoning",
  "deepseek": "deepseek-chat",
  "doubao": "doubao-seed-1-8-251228",
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
