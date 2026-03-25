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
DEFAULT_STAGE="all"
DEFAULT_USER_REQUEST=""
DEFAULT_VOLUME_NUMBER=""
DEFAULT_PROVIDER_OVERRIDE=""

# Optional runtime overrides
DEFAULT_MODEL_NAME_OVERRIDE=""
DEFAULT_API_BASE_OVERRIDE=""
DEFAULT_TEMPERATURE_OVERRIDE=""
DEFAULT_MAX_TOKENS_OVERRIDE=""
DEFAULT_TIMEOUT_OVERRIDE=""
DEFAULT_THINKING_LEVEL_OVERRIDE=""

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
load_api_keys
PYTHON_EXE="$(resolve_python_exe)"
log_info "quick_outline: 已加载脚本和 API keys。"

if [[ $# -lt 1 ]]; then
  PROJECT_PATH="$(prompt_optional_value "Project directory" "$DEFAULT_PROJECT_PATH")"
else
  PROJECT_PATH="${1:-$DEFAULT_PROJECT_PATH}"
fi

if [[ $# -lt 2 ]]; then
  STAGE="$(prompt_optional_value "Outline stage (volumes/chapters/all)" "$DEFAULT_STAGE")"
else
  STAGE="${2:-$DEFAULT_STAGE}"
fi

if [[ $# -lt 3 ]]; then
  USER_REQUEST="$(prompt_optional_value "Outline request (optional)" "$DEFAULT_USER_REQUEST")"
else
  USER_REQUEST="${3:-$DEFAULT_USER_REQUEST}"
fi

if [[ $# -lt 4 ]]; then
  VOLUME_NUMBER="$(prompt_optional_value "Volume number (optional, only for chapters)" "$DEFAULT_VOLUME_NUMBER")"
else
  VOLUME_NUMBER="${4:-$DEFAULT_VOLUME_NUMBER}"
fi

if [[ $# -lt 5 ]]; then
  PROVIDER_OVERRIDE="$(prompt_optional_value "Provider override (optional: gemini/grok/deepseek/doubao/ollama)" "$DEFAULT_PROVIDER_OVERRIDE")"
else
  PROVIDER_OVERRIDE="${5:-$DEFAULT_PROVIDER_OVERRIDE}"
fi

if [[ -z "$PROJECT_PATH" ]]; then
  echo "用法: ./linux/quick_outline.sh <项目目录> [volumes|chapters|all] [大纲额外要求] [卷号] [provider覆盖]" >&2
  exit 1
fi

if [[ ! -d "$PROJECT_PATH" ]]; then
  echo "项目目录不存在: $PROJECT_PATH" >&2
  exit 1
fi

if [[ ! -f "$PROJECT_PATH/project.json" ]]; then
  echo "项目目录中缺少 project.json: $PROJECT_PATH" >&2
  exit 1
fi

case "${STAGE:-all}" in
  volumes|chapters|all) ;;
  *)
    echo "不支持的 stage: $STAGE（可选: volumes / chapters / all）" >&2
    exit 1
    ;;
esac

if [[ -n "$PROVIDER_OVERRIDE" ]]; then
  PROVIDER_OVERRIDE="$(normalize_provider "$PROVIDER_OVERRIDE")"
fi

RESOLVED_PROVIDER="$("$PYTHON_EXE" - "$PROJECT_PATH" "$PROVIDER_OVERRIDE" <<'PY'
import json
import pathlib
import sys

project_path = pathlib.Path(sys.argv[1])
override = sys.argv[2].strip()
project = json.loads((project_path / "project.json").read_text(encoding="utf-8"))
saved = project.get("llm_config", {})
print(override or saved.get("model_provider", "gemini"))
PY
)"

NOVEL_PROVIDER_OVERRIDE="$RESOLVED_PROVIDER"
NOVEL_MODEL_NAME_OVERRIDE="${NOVEL_MODEL_NAME_OVERRIDE:-$DEFAULT_MODEL_NAME_OVERRIDE}"
NOVEL_API_BASE_OVERRIDE="${NOVEL_API_BASE_OVERRIDE:-$DEFAULT_API_BASE_OVERRIDE}"
NOVEL_TEMPERATURE_OVERRIDE="${NOVEL_TEMPERATURE_OVERRIDE:-$DEFAULT_TEMPERATURE_OVERRIDE}"
NOVEL_MAX_TOKENS_OVERRIDE="${NOVEL_MAX_TOKENS_OVERRIDE:-$DEFAULT_MAX_TOKENS_OVERRIDE}"
NOVEL_TIMEOUT_OVERRIDE="${NOVEL_TIMEOUT_OVERRIDE:-$DEFAULT_TIMEOUT_OVERRIDE}"
NOVEL_THINKING_LEVEL_OVERRIDE="${NOVEL_THINKING_LEVEL_OVERRIDE:-$DEFAULT_THINKING_LEVEL_OVERRIDE}"
NOVEL_API_KEY="${NOVEL_API_KEY:-$(api_key_for_provider "$RESOLVED_PROVIDER")}"

ensure_api_key_present "$RESOLVED_PROVIDER" "$NOVEL_API_KEY"
log_info "quick_outline: project=$PROJECT_PATH, stage=$STAGE, provider=$RESOLVED_PROVIDER"

export NOVEL_PROVIDER_OVERRIDE
export NOVEL_MODEL_NAME_OVERRIDE
export NOVEL_API_BASE_OVERRIDE
export NOVEL_TEMPERATURE_OVERRIDE
export NOVEL_MAX_TOKENS_OVERRIDE
export NOVEL_TIMEOUT_OVERRIDE
export NOVEL_THINKING_LEVEL_OVERRIDE
export NOVEL_API_KEY

TEMP_CONFIG="$(make_temp_config_path)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
log_info "quick_outline: 正在写入临时配置 $TEMP_CONFIG"
write_continue_config "$TEMP_CONFIG" "$PROJECT_PATH"

OUTLINE_ARGS=(
  "$PYTHON_EXE" "$PROJECT_ROOT/app.py" outline
  --project "$PROJECT_PATH"
  --config "$TEMP_CONFIG"
  --stage "$STAGE"
)

if [[ -n "$USER_REQUEST" ]]; then
  OUTLINE_ARGS+=(--user-request "$USER_REQUEST")
fi

if [[ "$STAGE" == "chapters" && -n "$VOLUME_NUMBER" ]]; then
  OUTLINE_ARGS+=(--volume "$VOLUME_NUMBER")
fi

log_info "quick_outline: 开始执行大纲重生成。"
"${OUTLINE_ARGS[@]}"
log_success "quick_outline: 大纲重生成完成。"
log_info "quick_outline: 输出项目状态。"
"$PYTHON_EXE" "$PROJECT_ROOT/app.py" status --project "$PROJECT_PATH"
log_success "quick_outline: 流程结束。"
