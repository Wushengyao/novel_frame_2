"""Lightweight console logger for CLI workflows."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit(level: str, message: str) -> None:
    print(f"[{_timestamp()}] [{level}] {message}", file=sys.stderr, flush=True)


def log_info(message: str) -> None:
    _emit("INFO", message)


def log_success(message: str) -> None:
    _emit("SUCCESS", message)


def log_warning(message: str) -> None:
    _emit("WARN", message)


def log_error(message: str) -> None:
    _emit("ERROR", message)
