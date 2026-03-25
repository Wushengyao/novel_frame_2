#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_PROJECT_PATH=""
DEFAULT_CHAPTER="latest"
DEFAULT_USER_REQUEST=""
DEFAULT_FORCE="false"
DEFAULT_CHECKPOINT=""

# Optional runtime overrides
DEFAULT_MODEL_NAME_OVERRIDE=""
DEFAULT_API_BASE_OVERRIDE=""
DEFAULT_TEMPERATURE_OVERRIDE=""
DEFAULT_MAX_TOKENS_OVERRIDE=""
DEFAULT_TIMEOUT_OVERRIDE=""

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
load_api_keys
PYTHON_EXE="$(resolve_python_exe)"
log_info "quick_illustrate: 已加载脚本和 API keys。"

if [[ $# -lt 1 ]]; then
  PROJECT_PATH="$(prompt_optional_value "Project directory" "$DEFAULT_PROJECT_PATH")"
else
  PROJECT_PATH="${1:-$DEFAULT_PROJECT_PATH}"
fi

if [[ $# -lt 2 ]]; then
  CHAPTER="$(prompt_optional_value "Chapter" "$DEFAULT_CHAPTER")"
else
  CHAPTER="${2:-$DEFAULT_CHAPTER}"
fi

if [[ $# -lt 3 ]]; then
  USER_REQUEST="$(prompt_optional_value "User request (optional)" "$DEFAULT_USER_REQUEST")"
else
  USER_REQUEST="${3:-$DEFAULT_USER_REQUEST}"
fi

if [[ $# -lt 4 ]]; then
  FORCE_VALUE="$(prompt_optional_value "Force regenerate? (true/false)" "$DEFAULT_FORCE")"
else
  FORCE_VALUE="${4:-$DEFAULT_FORCE}"
fi

if [[ $# -lt 5 ]]; then
  CHECKPOINT="$(prompt_optional_value "Checkpoint (optional)" "$DEFAULT_CHECKPOINT")"
else
  CHECKPOINT="${5:-$DEFAULT_CHECKPOINT}"
fi

if [[ ! -d "$PROJECT_PATH" ]]; then
  echo "项目目录不存在: $PROJECT_PATH" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_PATH/project.json" ]]; then
  echo "项目目录中缺少 project.json: $PROJECT_PATH" >&2
  exit 1
fi

RESOLVED_PROVIDER="$("$PYTHON_EXE" - "$PROJECT_PATH" <<'PY'
import json
import pathlib
import sys

project_path = pathlib.Path(sys.argv[1])
project = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
saved = project.get("llm_config", {})
print(str(saved.get("model_provider", "") or "").strip().lower())
PY
)"

NOVEL_MODEL_NAME_OVERRIDE="${NOVEL_MODEL_NAME_OVERRIDE:-$DEFAULT_MODEL_NAME_OVERRIDE}"
NOVEL_API_BASE_OVERRIDE="${NOVEL_API_BASE_OVERRIDE:-$DEFAULT_API_BASE_OVERRIDE}"
NOVEL_TEMPERATURE_OVERRIDE="${NOVEL_TEMPERATURE_OVERRIDE:-$DEFAULT_TEMPERATURE_OVERRIDE}"
NOVEL_MAX_TOKENS_OVERRIDE="${NOVEL_MAX_TOKENS_OVERRIDE:-$DEFAULT_MAX_TOKENS_OVERRIDE}"
NOVEL_TIMEOUT_OVERRIDE="${NOVEL_TIMEOUT_OVERRIDE:-$DEFAULT_TIMEOUT_OVERRIDE}"
NOVEL_API_KEY="${NOVEL_API_KEY:-$(api_key_for_provider "$RESOLVED_PROVIDER")}"

ensure_api_key_present "$RESOLVED_PROVIDER" "$NOVEL_API_KEY"
log_info "quick_illustrate: project=$PROJECT_PATH, chapter=$CHAPTER, provider=$RESOLVED_PROVIDER"

export NOVEL_MODEL_NAME_OVERRIDE
export NOVEL_API_BASE_OVERRIDE
export NOVEL_TEMPERATURE_OVERRIDE
export NOVEL_MAX_TOKENS_OVERRIDE
export NOVEL_TIMEOUT_OVERRIDE
export NOVEL_API_KEY

TEMP_CONFIG="$(make_temp_config_path)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
log_info "quick_illustrate: 正在写入临时配置 $TEMP_CONFIG"
write_illustrate_config "$TEMP_CONFIG" "$PROJECT_PATH"

ILLUSTRATE_ARGS=(
  "$PYTHON_EXE" "$PROJECT_ROOT/app.py" illustrate
  --project "$PROJECT_PATH"
  --chapter "$CHAPTER"
  --config "$TEMP_CONFIG"
)

if [[ -n "$USER_REQUEST" ]]; then
  ILLUSTRATE_ARGS+=(--user-request "$USER_REQUEST")
fi

if [[ "${FORCE_VALUE,,}" == "true" ]]; then
  ILLUSTRATE_ARGS+=(--force)
fi

if [[ -n "$CHECKPOINT" ]]; then
  ILLUSTRATE_ARGS+=(--checkpoint "$CHECKPOINT")
fi

log_info "quick_illustrate: 开始执行插图生成。"
"${ILLUSTRATE_ARGS[@]}"
log_success "quick_illustrate: 插图生成流程结束。"
