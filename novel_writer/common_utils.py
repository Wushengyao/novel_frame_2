"""Shared helpers for timestamps, progress callbacks, and JSON parsing."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


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


def save_failed_llm_output(
    project_path: str,
    phase: str,
    response_text: str,
    *,
    error: str = "",
    context: dict | None = None,
) -> Path | None:
    """Persist raw model text for malformed responses without storing request secrets."""
    if not str(response_text or "").strip():
        return None
    base = Path(str(project_path or "")).expanduser()
    if not base:
        return None
    timestamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z").replace("+00:00", "Z")
    safe_phase = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(phase or "llm").strip()) or "llm"
    path = base / "failed_llm_outputs" / f"{timestamp}_{safe_phase}.json"
    payload = {
        "created_at": utc_now(),
        "phase": safe_phase,
        "error": str(error or "")[:1200],
        "response_text": str(response_text or ""),
    }
    if context:
        payload["context"] = {
            key: value
            for key, value in context.items()
            if key not in {"api_key", "authorization", "headers", "request"}
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
