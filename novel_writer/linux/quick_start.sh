#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_PROVIDER="ollama"
DEFAULT_STORY_REQUEST="""故事发生在一座高级奢华的校园中，3位主角都是学生。男主是团队力量担当，乐观；女主1号是倾国倾城的美丽少女，身材娇小纤细，团队智力担当，傲娇；女主二号同样美丽动人，善于照顾他人，温柔。小说故事聚焦于他们合作生存的过程上，从初期的保暖，到逐步确保水源和食物来源，然后再逐步提升生活水平。总体风格温馨，并加入情感升温。小说应当详细描写他们协力生存的方方面面，重点描写他们搭建/升级温暖的避难所，并且包括过程中的感情升温与适量的香艳情节。
故事情节方面：
1、故事的开始是放假期间只有主角们在校，突然极寒天气与暴风雪来临，他们被困在学校中。一开始他们认为只是短暂的极端天气很快会有救援，所以在只是团聚在女生宿舍避寒并且做了短期规划。
2、但是显然他们低估了极寒风暴的力量，温度持续下降，救援也不会来。他们必须转战更加保暖的地方御寒（比如桑拿房）、搜集并储备大量物资，并尝试资源再生与可持续利用，不断改善生活条件，由生存转向生活。
3、桑拿房附近的资源也会耗尽，因此它们研究出一套校内探查与物资搜刮的装备与行动方案。可靠安全地收集更多物资，进一步提高生活水平，并逐步实现可持续。
4、新的希望，外部电台发来断续的信号，他们决定去看看。工作内容转向保温载具的设计与改造。
5、...
"""
DEFAULT_PROJECT_NAME="雪封穹顶"
DEFAULT_PROJECT_DESCRIPTION="由模型根据需求自动生成设定的长篇小说项目。"
DEFAULT_OUTLINE_REQUEST=""
DEFAULT_PLANNING_MODE="chapter"
DEFAULT_WORKFLOW_MODE="classic"

# Optional runtime overrides
DEFAULT_MODEL_NAME=""
DEFAULT_API_BASE=""
DEFAULT_TEMPERATURE="1.0"
DEFAULT_MAX_TOKENS="10240"
DEFAULT_TIMEOUT=""
DEFAULT_QUALITY_PROVIDER=""
DEFAULT_QUALITY_MODEL_NAME=""
DEFAULT_QUALITY_API_BASE=""
DEFAULT_QUALITY_TEMPERATURE=""
DEFAULT_QUALITY_MAX_TOKENS=""
DEFAULT_QUALITY_TIMEOUT=""
DEFAULT_AUTO_CREATE_COVER_AND_PORTRAITS="false"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
load_api_keys
PYTHON_EXE="$(resolve_python_exe)"
log_info "quick_start: 已加载脚本和 API keys。"

if [[ $# -lt 1 ]]; then
  PROVIDER="$(normalize_provider "$(prompt_optional_value "Provider (gemini/grok/deepseek/doubao/ollama/llama_cpp)" "$DEFAULT_PROVIDER")")"
else
  PROVIDER="$(normalize_provider "${1:-$DEFAULT_PROVIDER}")"
fi

if [[ $# -lt 2 ]]; then
  STORY_REQUEST="$(prompt_optional_value "Story request" "$DEFAULT_STORY_REQUEST")"
else
  STORY_REQUEST="${2:-$DEFAULT_STORY_REQUEST}"
fi

if [[ $# -lt 3 ]]; then
  PROJECT_NAME="$(prompt_optional_value "Project name" "$DEFAULT_PROJECT_NAME")"
else
  PROJECT_NAME="${3:-$DEFAULT_PROJECT_NAME}"
fi

if [[ $# -lt 4 ]]; then
  PROJECT_DESCRIPTION="$(prompt_optional_value "Project description" "$DEFAULT_PROJECT_DESCRIPTION")"
else
  PROJECT_DESCRIPTION="${4:-$DEFAULT_PROJECT_DESCRIPTION}"
fi

if [[ $# -lt 5 ]]; then
  OUTLINE_REQUEST="$(prompt_optional_value "Outline request (optional)" "$DEFAULT_OUTLINE_REQUEST")"
else
  OUTLINE_REQUEST="${5:-$DEFAULT_OUTLINE_REQUEST}"
fi

if [[ $# -lt 6 ]]; then
  PLANNING_MODE="$(prompt_optional_value "Planning mode (none/volume/chapter)" "$DEFAULT_PLANNING_MODE")"
else
  PLANNING_MODE="${6:-$DEFAULT_PLANNING_MODE}"
fi
PLANNING_MODE="$(normalize_planning_mode "$PLANNING_MODE")"

if [[ $# -lt 7 ]]; then
  WORKFLOW_MODE="$(prompt_optional_value "Workflow mode (classic/agentic)" "$DEFAULT_WORKFLOW_MODE")"
else
  WORKFLOW_MODE="${7:-$DEFAULT_WORKFLOW_MODE}"
fi
WORKFLOW_MODE="$(normalize_workflow_mode "$WORKFLOW_MODE")"

if [[ -z "$STORY_REQUEST" ]]; then
  echo "用法: ./linux/quick_start.sh <provider> <故事需求> [项目名] [项目简介] [大纲额外要求] [planning mode] [workflow mode]" >&2
  echo "也可以直接编辑脚本顶部的 Editable Parameters 区域，然后直接运行 ./linux/quick_start.sh" >&2
  echo "示例: ./linux/quick_start.sh gemini \"现代奢华校园中的极寒生存故事，男女主合作求生。\"" >&2
  exit 1
fi

NOVEL_PROVIDER="$PROVIDER"
NOVEL_PROJECT_NAME="$PROJECT_NAME"
NOVEL_PROJECT_DESCRIPTION="$PROJECT_DESCRIPTION"
NOVEL_STORY_REQUEST="$STORY_REQUEST"
NOVEL_OUTLINE_REQUEST="$OUTLINE_REQUEST"
NOVEL_PLANNING_MODE="$PLANNING_MODE"
NOVEL_WORKFLOW_MODE="${NOVEL_WORKFLOW_MODE:-$WORKFLOW_MODE}"
NOVEL_WORKFLOW_MODE="$(normalize_workflow_mode "$NOVEL_WORKFLOW_MODE")"
NOVEL_MODEL_NAME="${NOVEL_MODEL_NAME:-${DEFAULT_MODEL_NAME:-$(default_model_for_provider "$PROVIDER")}}"
NOVEL_API_BASE="${NOVEL_API_BASE:-${DEFAULT_API_BASE:-$(default_api_base_for_provider "$PROVIDER")}}"
NOVEL_API_KEY="${NOVEL_API_KEY:-$(api_key_for_provider "$PROVIDER")}"
NOVEL_TEMPERATURE="${NOVEL_TEMPERATURE:-$DEFAULT_TEMPERATURE}"
NOVEL_MAX_TOKENS="${NOVEL_MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
NOVEL_TIMEOUT="${NOVEL_TIMEOUT:-${DEFAULT_TIMEOUT:-$(default_timeout_for_provider "$PROVIDER")}}"
NOVEL_QUALITY_PROVIDER="${NOVEL_QUALITY_PROVIDER:-$DEFAULT_QUALITY_PROVIDER}"
if [[ -n "$NOVEL_QUALITY_PROVIDER" ]]; then
  NOVEL_QUALITY_PROVIDER="$(normalize_provider "$NOVEL_QUALITY_PROVIDER")"
fi
NOVEL_QUALITY_MODEL_NAME="${NOVEL_QUALITY_MODEL_NAME:-$DEFAULT_QUALITY_MODEL_NAME}"
NOVEL_QUALITY_API_BASE="${NOVEL_QUALITY_API_BASE:-$DEFAULT_QUALITY_API_BASE}"
NOVEL_QUALITY_TEMPERATURE="${NOVEL_QUALITY_TEMPERATURE:-$DEFAULT_QUALITY_TEMPERATURE}"
NOVEL_QUALITY_MAX_TOKENS="${NOVEL_QUALITY_MAX_TOKENS:-$DEFAULT_QUALITY_MAX_TOKENS}"
NOVEL_QUALITY_TIMEOUT="${NOVEL_QUALITY_TIMEOUT:-$DEFAULT_QUALITY_TIMEOUT}"
NOVEL_QUALITY_API_KEY="${NOVEL_QUALITY_API_KEY:-}"
if [[ -n "$NOVEL_QUALITY_PROVIDER" ]]; then
  NOVEL_QUALITY_API_KEY="${NOVEL_QUALITY_API_KEY:-$(api_key_for_provider "$NOVEL_QUALITY_PROVIDER")}"
  ensure_api_key_present "$NOVEL_QUALITY_PROVIDER" "$NOVEL_QUALITY_API_KEY"
fi

ensure_api_key_present "$PROVIDER" "$NOVEL_API_KEY"
log_info "quick_start: provider=$PROVIDER, project_name=$PROJECT_NAME"

export NOVEL_PROVIDER
export NOVEL_PROJECT_NAME
export NOVEL_PROJECT_DESCRIPTION
export NOVEL_STORY_REQUEST
export NOVEL_OUTLINE_REQUEST
export NOVEL_PLANNING_MODE
export NOVEL_WORKFLOW_MODE
export NOVEL_MODEL_NAME
export NOVEL_API_BASE
export NOVEL_API_KEY
export NOVEL_TEMPERATURE
export NOVEL_MAX_TOKENS
export NOVEL_TIMEOUT
export NOVEL_QUALITY_PROVIDER
export NOVEL_QUALITY_MODEL_NAME
export NOVEL_QUALITY_API_BASE
export NOVEL_QUALITY_API_KEY
export NOVEL_QUALITY_TEMPERATURE
export NOVEL_QUALITY_MAX_TOKENS
export NOVEL_QUALITY_TIMEOUT

TEMP_CONFIG="$(make_temp_config_path)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
log_info "quick_start: 正在写入临时配置 $TEMP_CONFIG"
write_init_config "$TEMP_CONFIG"

log_info "quick_start: 开始执行初始化。"
INIT_OUTPUT="$("$PYTHON_EXE" "$PROJECT_ROOT/app.py" init --config "$TEMP_CONFIG")"
printf '%s\n' "$INIT_OUTPUT"
log_success "quick_start: 初始化命令执行完成。"

PROJECT_PATH="$(printf '%s\n' "$INIT_OUTPUT" | sed -n 's/^项目已初始化: //p' | tail -n 1)"
if [[ -z "$PROJECT_PATH" ]]; then
  PROJECT_PATH="$(get_latest_project_path "$PROJECT_ROOT/output")"
fi
if [[ -z "$PROJECT_PATH" ]]; then
  echo "无法从 init 输出中解析项目路径。" >&2
  exit 1
fi

if [[ "${DEFAULT_AUTO_CREATE_COVER_AND_PORTRAITS,,}" == "true" ]]; then
  log_info "quick_start: 正在尝试自动创建小说封面和人物立绘。"
  set +e
  ASSET_OUTPUT="$("$PYTHON_EXE" "$PROJECT_ROOT/app.py" illustrate-assets --project "$PROJECT_PATH" 2>&1)"
  ASSET_EXIT_CODE=$?
  set -e
  printf '%s\n' "$ASSET_OUTPUT"

  if [[ $ASSET_EXIT_CODE -ne 0 ]]; then
    if test_illustration_connection_failure "$ASSET_OUTPUT"; then
      log_warning "quick_start: ComfyUI 不可连接，已跳过自动创建封面和人物立绘。"
    else
      log_error "quick_start: 封面/人物立绘生成失败，退出码: $ASSET_EXIT_CODE"
      exit "$ASSET_EXIT_CODE"
    fi
  else
    log_success "quick_start: 自动创建封面和人物立绘完成。"
  fi
fi

log_info "quick_start: 输出项目状态。"
"$PYTHON_EXE" "$PROJECT_ROOT/app.py" status --project "$PROJECT_PATH"
log_success "quick_start: 流程结束。"
printf '重生成大纲示例: ./linux/quick_outline.sh "%s" all "想补强的剧情要求"\n' "$PROJECT_PATH"
printf '续写示例: ./linux/quick_continue.sh "%s" 3 "想看的情节"\n' "$PROJECT_PATH"
