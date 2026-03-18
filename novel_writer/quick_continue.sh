#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_PROJECT_PATH="/home/wsy/novel_frame_2/novel_writer/output/novel_project_20260318T063939Z_e97b9a09"
DEFAULT_CHAPTER_COUNT="3"
DEFAULT_USER_REQUEST=""
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

PROJECT_PATH="${1:-$DEFAULT_PROJECT_PATH}"
CHAPTER_COUNT="${2:-$DEFAULT_CHAPTER_COUNT}"
USER_REQUEST="${3:-$DEFAULT_USER_REQUEST}"
PROVIDER_OVERRIDE="${4:-$DEFAULT_PROVIDER_OVERRIDE}"

if [[ -z "$PROJECT_PATH" ]]; then
  echo "用法: ./quick_continue.sh <项目目录> [续写章节数] [用户额外要求] [provider覆盖]" >&2
  echo "也可以直接编辑脚本顶部的 Editable Parameters 区域，然后直接运行 ./quick_continue.sh" >&2
  echo "示例: ./quick_continue.sh ./novel_project_xxx 3 \"想先解决水源问题\"" >&2
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

RESOLVED_PROVIDER="$(python3 - "$PROJECT_PATH" "$PROVIDER_OVERRIDE" <<'PY'
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
write_continue_config "$TEMP_CONFIG" "$PROJECT_PATH"

NEXT_ARGS=(
  python3 "$SCRIPT_DIR/app.py" next
  --project "$PROJECT_PATH"
  --config "$TEMP_CONFIG"
  --count "$CHAPTER_COUNT"
)

if [[ -n "$USER_REQUEST" ]]; then
  NEXT_ARGS+=(--user-request "$USER_REQUEST")
fi

"${NEXT_ARGS[@]}"
python3 "$SCRIPT_DIR/app.py" status --project "$PROJECT_PATH"
