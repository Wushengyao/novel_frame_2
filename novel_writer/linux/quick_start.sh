#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_PROVIDER="gemini"
DEFAULT_STORY_REQUEST="""故事发生在高中校园中，男女主都是学生。故事的开始是放假期间只有主角们在校，突然极寒天气与暴风雪来临，他们被困在学校中，他们如何御寒储备物资并生存生活下去，故事由此展开。角色方面，男主力气较大，行为经常有意外的效果，团队力量担当，乐观；女主1号是倾国倾城的美丽少女，身材娇小纤细，团队智力担当，傲娇；女主二号同样美丽动人，善于照顾他人，温柔。小说故事聚焦于他们合作生存的过程上，从初期的保暖，到逐步确保水源和食物来源，然后再逐步提升生活水平。请注意：小说需要具备长篇潜力。"""
DEFAULT_PROJECT_NAME="雪封穹顶"
DEFAULT_PROJECT_DESCRIPTION="由模型根据需求自动生成设定的长篇小说项目。"

# Optional runtime overrides
DEFAULT_MODEL_NAME=""
DEFAULT_API_BASE=""
DEFAULT_TEMPERATURE="1.0"
DEFAULT_MAX_TOKENS="10240"
DEFAULT_TIMEOUT="120"
DEFAULT_THINKING_LEVEL="medium"
DEFAULT_AUTO_CREATE_COVER_AND_PORTRAITS="true"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
load_api_keys
PYTHON_EXE="$(resolve_python_exe)"

if [[ $# -lt 1 ]]; then
  PROVIDER="$(normalize_provider "$(prompt_optional_value "Provider (gemini/grok/deepseek/doubao)" "$DEFAULT_PROVIDER")")"
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

if [[ -z "$STORY_REQUEST" ]]; then
  echo "用法: ./linux/quick_start.sh <provider> <故事需求> [项目名] [项目简介]" >&2
  echo "也可以直接编辑脚本顶部的 Editable Parameters 区域，然后直接运行 ./linux/quick_start.sh" >&2
  echo "示例: ./linux/quick_start.sh gemini \"现代奢华校园中的极寒生存故事，男女主合作求生。\"" >&2
  exit 1
fi

NOVEL_PROVIDER="$PROVIDER"
NOVEL_PROJECT_NAME="$PROJECT_NAME"
NOVEL_PROJECT_DESCRIPTION="$PROJECT_DESCRIPTION"
NOVEL_STORY_REQUEST="$STORY_REQUEST"
NOVEL_MODEL_NAME="${NOVEL_MODEL_NAME:-${DEFAULT_MODEL_NAME:-$(default_model_for_provider "$PROVIDER")}}"
NOVEL_API_BASE="${NOVEL_API_BASE:-${DEFAULT_API_BASE:-$(default_api_base_for_provider "$PROVIDER")}}"
NOVEL_API_KEY="${NOVEL_API_KEY:-$(api_key_for_provider "$PROVIDER")}"
NOVEL_TEMPERATURE="${NOVEL_TEMPERATURE:-$DEFAULT_TEMPERATURE}"
NOVEL_MAX_TOKENS="${NOVEL_MAX_TOKENS:-$DEFAULT_MAX_TOKENS}"
NOVEL_TIMEOUT="${NOVEL_TIMEOUT:-$DEFAULT_TIMEOUT}"
NOVEL_THINKING_LEVEL="${NOVEL_THINKING_LEVEL:-${DEFAULT_THINKING_LEVEL:-$(default_thinking_level_for_provider "$PROVIDER")}}"

ensure_api_key_present "$PROVIDER" "$NOVEL_API_KEY"

export NOVEL_PROVIDER
export NOVEL_PROJECT_NAME
export NOVEL_PROJECT_DESCRIPTION
export NOVEL_STORY_REQUEST
export NOVEL_MODEL_NAME
export NOVEL_API_BASE
export NOVEL_API_KEY
export NOVEL_TEMPERATURE
export NOVEL_MAX_TOKENS
export NOVEL_TIMEOUT
export NOVEL_THINKING_LEVEL

TEMP_CONFIG="$(make_temp_config_path)"
trap 'rm -f "$TEMP_CONFIG"' EXIT
write_init_config "$TEMP_CONFIG"

INIT_OUTPUT="$("$PYTHON_EXE" "$PROJECT_ROOT/app.py" init --config "$TEMP_CONFIG")"
printf '%s\n' "$INIT_OUTPUT"

PROJECT_PATH="$(printf '%s\n' "$INIT_OUTPUT" | sed -n 's/^项目已初始化: //p' | tail -n 1)"
if [[ -z "$PROJECT_PATH" ]]; then
  PROJECT_PATH="$(get_latest_project_path "$PROJECT_ROOT/output")"
fi
if [[ -z "$PROJECT_PATH" ]]; then
  echo "无法从 init 输出中解析项目路径。" >&2
  exit 1
fi

if [[ "${DEFAULT_AUTO_CREATE_COVER_AND_PORTRAITS,,}" == "true" ]]; then
  echo "正在尝试自动创建小说封面和人物立绘..."
  set +e
  ASSET_OUTPUT="$("$PYTHON_EXE" "$PROJECT_ROOT/app.py" illustrate-assets --project "$PROJECT_PATH" 2>&1)"
  ASSET_EXIT_CODE=$?
  set -e
  printf '%s\n' "$ASSET_OUTPUT"

  if [[ $ASSET_EXIT_CODE -ne 0 ]]; then
    if test_illustration_connection_failure "$ASSET_OUTPUT"; then
      echo "ComfyUI 不可连接，已跳过自动创建封面和人物立绘。" >&2
    else
      echo "封面/人物立绘生成失败，退出码: $ASSET_EXIT_CODE" >&2
      exit "$ASSET_EXIT_CODE"
    fi
  fi
fi

"$PYTHON_EXE" "$PROJECT_ROOT/app.py" status --project "$PROJECT_PATH"
printf '续写示例: ./linux/quick_continue.sh "%s" 3 "想看的情节"\n' "$PROJECT_PATH"
