#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8008"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/script_common.sh"
PYTHON_EXE="$(resolve_python_exe)"

HOST="${1:-$DEFAULT_HOST}"
PORT="${2:-$DEFAULT_PORT}"

cd "$PROJECT_ROOT"
"$PYTHON_EXE" "$PROJECT_ROOT/webui.py" --host "$HOST" --port "$PORT"
