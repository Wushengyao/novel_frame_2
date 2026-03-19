#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================
# Editable Parameters
# 直接修改这里即可，不必每次写命令行参数。
# 如果同时传入命令行参数，则命令行参数优先。
# ==============================
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="8008"

HOST="${1:-$DEFAULT_HOST}"
PORT="${2:-$DEFAULT_PORT}"

cd "$SCRIPT_DIR"
python3 "$SCRIPT_DIR/webui.py" --host "$HOST" --port "$PORT"
