"""Shared helpers for timestamps, progress callbacks, and JSON parsing."""

from __future__ import annotations

import json
import re
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


def _json_candidates(content: str) -> list[str]:
    candidates = [content]
    for match in re.finditer(r"```[a-zA-Z0-9_-]*\s*(.*?)```", content, flags=re.DOTALL):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)

    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(content[brace_start : brace_end + 1].strip())

    return candidates


def extract_json_object(text: str, error_message: str) -> dict:
    content = (text or "").strip().lstrip("\ufeff")
    decoder = json.JSONDecoder()

    for candidate in _json_candidates(content):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data

        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                data, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data

    raise ValueError(error_message)
