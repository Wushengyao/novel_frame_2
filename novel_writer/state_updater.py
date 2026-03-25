"""Update plot state from newly generated chapter text."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from console_logger import log_info, log_success, log_warning
from llm_client import generate_text_with_metadata
from prompt_builder import build_summary_prompt
from project_manager import load_json, save_json, update_project_stats


SUMMARY_KEYS = (
    "recent_events",
    "open_threads",
    "foreshadowing",
    "character_updates",
    "next_chapter_goal",
)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    candidates = [text]

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end != -1:
            candidates.append(text[start:end].strip())
    elif "```" in text:
        start = text.find("```") + len("```")
        end = text.find("```", start)
        if end != -1:
            candidates.append(text[start:end].strip())

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(text[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON from summary response.")


def _normalize_summary(summary: dict) -> dict:
    normalized = {}
    for key in SUMMARY_KEYS:
        value = summary.get(key, [] if key != "next_chapter_goal" else "")
        if key == "next_chapter_goal":
            normalized[key] = value if isinstance(value, str) else str(value)
        elif isinstance(value, list):
            normalized[key] = value
        elif value is None:
            normalized[key] = []
        else:
            normalized[key] = [str(value)]
    return normalized


def _fallback_summary(new_text: str, current_state: dict) -> dict:
    fallback = _normalize_summary(current_state)
    excerpt = new_text.strip().replace("\n", " ")
    fallback["recent_events"] = current_state.get("recent_events", []) + [excerpt[:200]]
    if not fallback["next_chapter_goal"]:
        fallback["next_chapter_goal"] = current_state.get("next_chapter_goal", "")
    return fallback


def update_plot_state(project_path: str, new_text: str, config: dict) -> None:
    log_info(f"剧情状态更新: 开始处理项目 {project_path}")
    base = Path(project_path)
    plot_state_path = base / "plot_state.json"
    characters_path = base / "characters.json"
    summaries_dir = base / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    current_state = load_json(str(plot_state_path))
    characters = load_json(str(characters_path))
    prompt_data = {
        "plot_state": current_state,
        "characters": characters,
    }
    prompt = build_summary_prompt(prompt_data, new_text)

    summary = None
    last_error = None
    for attempt in range(2):
        try:
            log_info(f"剧情状态更新: 第 {attempt + 1} 次请求模型总结本章。")
            response_text, metadata = generate_text_with_metadata(prompt, config)
        except Exception as exc:  # pragma: no cover - intentional resilience path
            update_project_stats(project_path, phase="summary", success=False, usage=None)
            last_error = exc
            log_warning(f"剧情状态更新: 第 {attempt + 1} 次请求失败，原因: {exc}")
            continue

        try:
            update_project_stats(
                project_path,
                phase="summary",
                success=True,
                usage=metadata.get("usage"),
            )
            summary = _normalize_summary(_extract_json_object(response_text))
            log_success("剧情状态更新: 模型总结成功，已解析状态 JSON。")
            break
        except Exception as exc:  # pragma: no cover - intentional resilience path
            last_error = exc
            log_warning(f"剧情状态更新: 返回内容解析失败，原因: {exc}")

    if summary is None:
        summary = _fallback_summary(new_text, current_state)
        log_warning("剧情状态更新: 改用兜底摘要更新 plot_state。")
        if last_error is not None:
            summary["last_summary_error"] = str(last_error)

    updated_state = deepcopy(current_state)
    for key in SUMMARY_KEYS:
        updated_state[key] = summary[key]
    save_json(str(plot_state_path), updated_state)

    chapter_count = load_json(str(base / "project.json")).get("chapter_count", 0)
    summary_path = summaries_dir / f"summary_{chapter_count:04d}.json"
    save_json(str(summary_path), summary)
    log_success(f"剧情状态更新: 已写入 plot_state.json 和 {summary_path.name}")
