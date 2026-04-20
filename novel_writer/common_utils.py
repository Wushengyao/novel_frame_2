"""Shared helpers for timestamps, progress callbacks, and JSON parsing."""

from __future__ import annotations

import json
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def emit_progress(progress_callback, stage: str, message: str, **extra) -> None:
    if progress_callback is None:
        return
    payload = {
        "stage": stage,
        "message": message,
    }
    payload.update(extra)
    progress_callback(payload)


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_json_object(text: str, error_message: str) -> dict:
    content = (text or "").strip()
    candidates = [content]

    if "```json" in content:
        start = content.find("```json") + len("```json")
        end = content.find("```", start)
        if end != -1:
            candidates.append(content[start:end].strip())
    elif "```" in content:
        start = content.find("```") + len("```")
        end = content.find("```", start)
        if end != -1:
            candidates.append(content[start:end].strip())

    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(content[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    raise ValueError(error_message)
