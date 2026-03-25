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
DEFAULT_TARGET_CHAPTER="0"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
PYTHON_EXE="$(resolve_python_exe)"
log_info "quick_rollback: 已加载脚本。"

if [[ $# -lt 1 ]]; then
  PROJECT_PATH="$(prompt_optional_value "Project directory" "$DEFAULT_PROJECT_PATH")"
else
  PROJECT_PATH="${1:-$DEFAULT_PROJECT_PATH}"
fi

if [[ $# -lt 2 ]]; then
  TARGET_CHAPTER="$(prompt_optional_value "Keep chapters up to" "$DEFAULT_TARGET_CHAPTER")"
else
  TARGET_CHAPTER="${2:-$DEFAULT_TARGET_CHAPTER}"
fi

if [[ -z "$PROJECT_PATH" ]]; then
  echo "用法: ./linux/quick_rollback.sh <项目目录> <保留到第几章>" >&2
  echo "示例: ./linux/quick_rollback.sh ./output/novel_project_xxx 4" >&2
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

if ! [[ "$TARGET_CHAPTER" =~ ^[0-9]+$ ]]; then
  echo "保留章节数必须是非负整数: $TARGET_CHAPTER" >&2
  exit 1
fi

log_info "quick_rollback: project=$PROJECT_PATH, keep_to=$TARGET_CHAPTER"
"$PYTHON_EXE" "$PROJECT_ROOT/app.py" rollback --project "$PROJECT_PATH" --to-chapter "$TARGET_CHAPTER"
log_success "quick_rollback: 回滚流程结束。"
