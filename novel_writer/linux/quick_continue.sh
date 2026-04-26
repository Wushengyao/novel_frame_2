#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_PROJECT_PATH="/home/wsy/novel_frame_2/novel_writer/output/novel_project_20260318T063939Z_e97b9a09"
DEFAULT_CHAPTER_COUNT="3"
DEFAULT_USER_REQUEST=""
DEFAULT_PROVIDER_OVERRIDE=""
DEFAULT_PLANNING_MODE_OVERRIDE=""
DEFAULT_CONTINUE_MODE="direct"
DEFAULT_GUIDED_OPTION_COUNT="4"
DEFAULT_GUIDED_FEEDBACK=""

# Optional runtime overrides
DEFAULT_MODEL_NAME_OVERRIDE=""
DEFAULT_API_BASE_OVERRIDE=""
DEFAULT_TEMPERATURE_OVERRIDE=""
DEFAULT_MAX_TOKENS_OVERRIDE=""
DEFAULT_TIMEOUT_OVERRIDE=""
DEFAULT_QUALITY_PROVIDER=""
DEFAULT_QUALITY_MODEL_NAME=""
DEFAULT_QUALITY_API_BASE=""
DEFAULT_QUALITY_TEMPERATURE=""
DEFAULT_QUALITY_MAX_TOKENS=""
DEFAULT_QUALITY_TIMEOUT=""
DEFAULT_AUTO_ILLUSTRATE="false"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
load_api_keys
PYTHON_EXE="$(resolve_python_exe)"
log_info "quick_continue: 已加载脚本和 API keys。"

if [[ $# -lt 1 ]]; then
  PROJECT_PATH="$(prompt_optional_value "Project directory" "$DEFAULT_PROJECT_PATH")"
else
  PROJECT_PATH="${1:-$DEFAULT_PROJECT_PATH}"
fi

if [[ $# -lt 2 ]]; then
  CHAPTER_COUNT="$(prompt_optional_value "Chapter count" "$DEFAULT_CHAPTER_COUNT")"
else
  CHAPTER_COUNT="${2:-$DEFAULT_CHAPTER_COUNT}"
fi

if [[ $# -lt 3 ]]; then
  USER_REQUEST="$(prompt_optional_value "User request (optional)" "$DEFAULT_USER_REQUEST")"
else
  USER_REQUEST="${3:-$DEFAULT_USER_REQUEST}"
fi

if [[ $# -lt 4 ]]; then
  PROVIDER_OVERRIDE="$(prompt_optional_value "Provider override (optional: gemini/grok/deepseek/doubao/ollama)" "$DEFAULT_PROVIDER_OVERRIDE")"
else
  PROVIDER_OVERRIDE="${4:-$DEFAULT_PROVIDER_OVERRIDE}"
fi

if [[ $# -lt 5 ]]; then
  PLANNING_MODE_OVERRIDE="$(prompt_optional_value "Planning mode override (optional: none/volume/chapter)" "$DEFAULT_PLANNING_MODE_OVERRIDE")"
else
  PLANNING_MODE_OVERRIDE="${5:-$DEFAULT_PLANNING_MODE_OVERRIDE}"
fi
if [[ -n "$PLANNING_MODE_OVERRIDE" ]]; then
  PLANNING_MODE_OVERRIDE="$(normalize_planning_mode "$PLANNING_MODE_OVERRIDE")"
fi

if [[ $# -lt 6 ]]; then
  CONTINUE_MODE="$(prompt_optional_value "Continue mode (optional: direct/guided)" "$DEFAULT_CONTINUE_MODE")"
else
  CONTINUE_MODE="${6:-$DEFAULT_CONTINUE_MODE}"
fi
CONTINUE_MODE="${CONTINUE_MODE,,}"
if [[ "$CONTINUE_MODE" != "direct" && "$CONTINUE_MODE" != "guided" ]]; then
  echo "Unsupported continue mode: $CONTINUE_MODE (allowed: direct / guided)" >&2
  exit 1
fi

if [[ -z "$PROJECT_PATH" ]]; then
  echo "用法: ./linux/quick_continue.sh <项目目录> [续写章节数] [用户额外要求] [provider覆盖] [planning mode override]" >&2
  echo "也可以直接编辑脚本顶部的 Editable Parameters 区域，然后直接运行 ./linux/quick_continue.sh" >&2
  echo "示例: ./linux/quick_continue.sh ./novel_project_xxx 3 \"想先解决水源问题\"" >&2
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
NOVEL_QUALITY_PROVIDER="${NOVEL_QUALITY_PROVIDER:-$DEFAULT_QUALITY_PROVIDER}"
if [[ -n "$NOVEL_QUALITY_PROVIDER" ]]; then
  NOVEL_QUALITY_PROVIDER="$(normalize_provider "$NOVEL_QUALITY_PROVIDER")"
fi
NOVEL_QUALITY_MODEL_NAME="${NOVEL_QUALITY_MODEL_NAME:-$DEFAULT_QUALITY_MODEL_NAME}"
NOVEL_QUALITY_API_BASE="${NOVEL_QUALITY_API_BASE:-$DEFAULT_QUALITY_API_BASE}"
NOVEL_QUALITY_TEMPERATURE="${NOVEL_QUALITY_TEMPERATURE:-$DEFAULT_QUALITY_TEMPERATURE}"
NOVEL_QUALITY_MAX_TOKENS="${NOVEL_QUALITY_MAX_TOKENS:-$DEFAULT_QUALITY_MAX_TOKENS}"
NOVEL_QUALITY_TIMEOUT="${NOVEL_QUALITY_TIMEOUT:-$DEFAULT_QUALITY_TIMEOUT}"
if [[ -n "${NOVEL_PLANNING_MODE_OVERRIDE:-}" ]]; then
  NOVEL_PLANNING_MODE_OVERRIDE="$(normalize_planning_mode "$NOVEL_PLANNING_MODE_OVERRIDE")"
else
  NOVEL_PLANNING_MODE_OVERRIDE="$PLANNING_MODE_OVERRIDE"
fi
NOVEL_API_KEY="${NOVEL_API_KEY:-$(api_key_for_provider "$RESOLVED_PROVIDER")}"
SAVED_QUALITY_PROVIDER="$("$PYTHON_EXE" - "$PROJECT_PATH" <<'PY'
import json
import pathlib
import sys

project = json.loads((pathlib.Path(sys.argv[1]) / "project.json").read_text(encoding="utf-8"))
quality = project.get("llm_config", {}).get("quality_model", {})
print(str(quality.get("model_provider", "") or "").strip())
PY
)"
EFFECTIVE_QUALITY_PROVIDER="${NOVEL_QUALITY_PROVIDER:-$SAVED_QUALITY_PROVIDER}"
NOVEL_QUALITY_API_KEY="${NOVEL_QUALITY_API_KEY:-}"
if [[ -n "$EFFECTIVE_QUALITY_PROVIDER" ]]; then
  EFFECTIVE_QUALITY_PROVIDER="$(normalize_provider "$EFFECTIVE_QUALITY_PROVIDER")"
  NOVEL_QUALITY_API_KEY="${NOVEL_QUALITY_API_KEY:-$(api_key_for_provider "$EFFECTIVE_QUALITY_PROVIDER")}"
  ensure_api_key_present "$EFFECTIVE_QUALITY_PROVIDER" "$NOVEL_QUALITY_API_KEY"
fi

ensure_api_key_present "$RESOLVED_PROVIDER" "$NOVEL_API_KEY"
log_info "quick_continue: project=$PROJECT_PATH, provider=$RESOLVED_PROVIDER, count=$CHAPTER_COUNT"

if [[ "$CONTINUE_MODE" == "guided" && "$CHAPTER_COUNT" != "1" ]]; then
  log_warning "quick_continue: guided 模式固定只续写 1 章，已自动将 count 调整为 1。"
  CHAPTER_COUNT="1"
fi

export NOVEL_PROVIDER_OVERRIDE
export NOVEL_MODEL_NAME_OVERRIDE
export NOVEL_API_BASE_OVERRIDE
export NOVEL_TEMPERATURE_OVERRIDE
export NOVEL_MAX_TOKENS_OVERRIDE
export NOVEL_TIMEOUT_OVERRIDE
export NOVEL_PLANNING_MODE_OVERRIDE
export NOVEL_API_KEY
export NOVEL_QUALITY_PROVIDER
export NOVEL_QUALITY_MODEL_NAME
export NOVEL_QUALITY_API_BASE
export NOVEL_QUALITY_API_KEY
export NOVEL_QUALITY_TEMPERATURE
export NOVEL_QUALITY_MAX_TOKENS
export NOVEL_QUALITY_TIMEOUT

TEMP_CONFIG="$(make_temp_config_path)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
log_info "quick_continue: 正在写入临时配置 $TEMP_CONFIG"
write_continue_config "$TEMP_CONFIG" "$PROJECT_PATH"

if [[ "$CONTINUE_MODE" == "guided" ]]; then
  OPTIONS_ARGS=(
    "$PYTHON_EXE" "$PROJECT_ROOT/app.py" options
    --project "$PROJECT_PATH"
    --config "$TEMP_CONFIG"
    --option-count "$DEFAULT_GUIDED_OPTION_COUNT"
  )
  GUIDED_OBJECTIVE="$(prompt_optional_value "Objective override (optional)" "")"
  if [[ -n "$GUIDED_OBJECTIVE" ]]; then
    OPTIONS_ARGS+=(--objective "$GUIDED_OBJECTIVE")
  fi
  if [[ -n "$USER_REQUEST" ]]; then
    OPTIONS_ARGS+=(--user-request "$USER_REQUEST")
  fi

  set +e
  log_info "quick_continue: 正在生成下一章推进选项。"
  OPTIONS_OUTPUT="$("${OPTIONS_ARGS[@]}" 2>&1)"
  OPTIONS_EXIT_CODE=$?
  set -e
  printf '%s\n' "$OPTIONS_OUTPUT"

  if [[ $OPTIONS_EXIT_CODE -ne 0 ]]; then
    log_error "quick_continue: 生成推进选项失败，退出码: $OPTIONS_EXIT_CODE"
    exit "$OPTIONS_EXIT_CODE"
  fi

  SESSION_ID="$(printf '%s\n' "$OPTIONS_OUTPUT" | sed -n 's/^Session ID: //p' | head -n 1)"
  RECOMMENDED_OPTION="$(printf '%s\n' "$OPTIONS_OUTPUT" | sed -n 's/^Recommended Option: //p' | head -n 1)"
  if [[ -z "$SESSION_ID" ]]; then
    log_error "quick_continue: 未能从 options 输出中解析出 Session ID。"
    exit 1
  fi

  SELECTED_OPTION="$(prompt_optional_value "Progression option (number or option_id)" "$RECOMMENDED_OPTION")"
  GUIDED_FEEDBACK="$(prompt_optional_value "Guided feedback (optional)" "$DEFAULT_GUIDED_FEEDBACK")"

  NEXT_ARGS=(
    "$PYTHON_EXE" "$PROJECT_ROOT/app.py" next
    --project "$PROJECT_PATH"
    --config "$TEMP_CONFIG"
    --count "1"
    --progression-session "$SESSION_ID"
    --progression-option "$SELECTED_OPTION"
  )
  if [[ -n "$GUIDED_FEEDBACK" ]]; then
    NEXT_ARGS+=(--progression-feedback "$GUIDED_FEEDBACK")
  fi
else
  AUTO_SELECTION_MODE="$(prompt_optional_value "Auto plan selection mode (recommended/random)" "recommended")"
  if [[ "$AUTO_SELECTION_MODE" != "random" ]]; then
    AUTO_SELECTION_MODE="recommended"
  fi
  NEXT_ARGS=(
    "$PYTHON_EXE" "$PROJECT_ROOT/app.py" next
    --project "$PROJECT_PATH"
    --config "$TEMP_CONFIG"
    --count "$CHAPTER_COUNT"
    --selection-mode "$AUTO_SELECTION_MODE"
  )

  if [[ -n "$USER_REQUEST" ]]; then
    NEXT_ARGS+=(--user-request "$USER_REQUEST")
  fi
fi

set +e
log_info "quick_continue: 开始执行正文续写。"
NEXT_OUTPUT="$("${NEXT_ARGS[@]}" 2>&1)"
NEXT_EXIT_CODE=$?
set -e
printf '%s\n' "$NEXT_OUTPUT"

if [[ $NEXT_EXIT_CODE -ne 0 ]]; then
  log_error "quick_continue: 章节生成失败，退出码: $NEXT_EXIT_CODE"
  exit "$NEXT_EXIT_CODE"
fi
log_success "quick_continue: 正文续写完成。"

if [[ "${DEFAULT_AUTO_ILLUSTRATE,,}" == "true" ]]; then
  mapfile -t GENERATED_CHAPTER_PATHS < <(printf '%s\n' "$NEXT_OUTPUT" | sed -n 's/^新章节已保存: //p')

  if [[ ${#GENERATED_CHAPTER_PATHS[@]} -gt 0 ]]; then
    log_info "quick_continue: 正在尝试自动创建插图。"
    for chapter_path in "${GENERATED_CHAPTER_PATHS[@]}"; do
      ILLUSTRATE_ARGS=(
        "$PYTHON_EXE" "$PROJECT_ROOT/app.py" illustrate
        --project "$PROJECT_PATH"
        --chapter "$chapter_path"
        --config "$TEMP_CONFIG"
      )

      set +e
      ILLUSTRATE_OUTPUT="$("${ILLUSTRATE_ARGS[@]}" 2>&1)"
      ILLUSTRATE_EXIT_CODE=$?
      set -e
      printf '%s\n' "$ILLUSTRATE_OUTPUT"

      if [[ $ILLUSTRATE_EXIT_CODE -ne 0 ]]; then
        if test_illustration_connection_failure "$ILLUSTRATE_OUTPUT"; then
          log_warning "quick_continue: ComfyUI 不可连接，已跳过自动插图创建。"
          break
        fi
        log_error "quick_continue: 插图生成失败，退出码: $ILLUSTRATE_EXIT_CODE"
        exit "$ILLUSTRATE_EXIT_CODE"
      fi
    done
    log_success "quick_continue: 自动插图流程结束。"
  else
    log_warning "quick_continue: 未检测到新章节路径，已跳过自动插图创建。"
  fi
fi

log_info "quick_continue: 输出项目状态。"
"$PYTHON_EXE" "$PROJECT_ROOT/app.py" status --project "$PROJECT_PATH"
log_success "quick_continue: 流程结束。"
