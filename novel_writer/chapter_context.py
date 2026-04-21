"""Shared helpers for resolving next-chapter context by planning mode."""

from __future__ import annotations

from copy import deepcopy

from console_logger import log_warning
from common_utils import emit_progress, safe_int
from outline_manager import (
    ensure_project_outlines,
    find_next_chapter_context,
    load_outlines,
    normalize_outlines,
    regenerate_volume_outline,
)
from project_manager import (
    DEFAULT_PLANNING_MODE,
    PLANNING_MODE_CHAPTER,
    PLANNING_MODE_NONE,
    PLANNING_MODE_VOLUME,
    load_project,
    normalize_planning_mode,
)


def resolve_planning_mode(config: dict, project_data: dict | None = None) -> str:
    project = (project_data or {}).get("project") or {}
    return normalize_planning_mode(
        config.get("planning_mode") or project.get("planning_mode"),
        default=DEFAULT_PLANNING_MODE,
    )


def ensure_volume_outlines(project_path: str, config: dict, progress_callback=None) -> dict:
    outlines = load_outlines(project_path)
    if outlines.get("volumes"):
        return normalize_outlines(outlines)
    log_warning("next_chapter: volume outlines missing, regenerating")
    emit_progress(progress_callback, "outline_prepare", "正在补生成分卷大纲")
    return regenerate_volume_outline(
        project_path,
        config,
        user_request="",
        progress_callback=progress_callback,
    )


def collect_upcoming_chapter_contexts(outlines: dict, written_chapter_count: int, count: int) -> list[dict]:
    start_number = written_chapter_count + 1
    end_number = written_chapter_count + count
    contexts = []
    for volume in outlines.get("volumes", []):
        for chapter in volume.get("chapters", []):
            chapter_number = safe_int(chapter.get("chapter_number"), 0)
            if start_number <= chapter_number <= end_number:
                contexts.append(
                    {
                        "volume": deepcopy(volume),
                        "chapter": deepcopy(chapter),
                    }
                )
    return contexts


def collect_upcoming_volume_contexts(outlines: dict, written_chapter_count: int, count: int) -> list[dict]:
    start_number = written_chapter_count + 1
    end_number = written_chapter_count + count
    normalized = normalize_outlines(outlines)
    contexts = []
    chapter_number = 0
    for volume in normalized.get("volumes", []):
        planned_count = max(1, safe_int(volume.get("planned_chapter_count"), 1))
        for chapter_in_volume in range(1, planned_count + 1):
            chapter_number += 1
            if not (start_number <= chapter_number <= end_number):
                continue
            contexts.append(
                {
                    "volume": deepcopy(volume),
                    "chapter": {
                        "chapter_number": chapter_number,
                        "chapter_in_volume": chapter_in_volume,
                        "title": "",
                        "summary": str(volume.get("summary", "") or "").strip(),
                        "goal": str(volume.get("story_goal", "") or "").strip(),
                        "key_events": [],
                        "status": "planned",
                    },
                }
            )
    return contexts


def collect_upcoming_freeform_contexts(project_data: dict, count: int) -> list[dict]:
    current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
    next_goal = str(project_data.get("plot_state", {}).get("next_chapter_goal", "") or "").strip()
    contexts = []
    for offset in range(1, count + 1):
        contexts.append(
            {
                "volume": {},
                "chapter": {
                    "chapter_number": current_chapter_count + offset,
                    "chapter_in_volume": offset,
                    "title": "",
                    "summary": next_goal,
                    "goal": next_goal,
                    "key_events": [],
                    "status": "planned",
                },
            }
        )
    return contexts


def get_next_context_for_mode(
    project_path: str,
    config: dict,
    planning_mode: str,
    progress_callback=None,
) -> tuple[dict, dict]:
    project_data = load_project(project_path)
    current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)

    if planning_mode == PLANNING_MODE_CHAPTER:
        outlines = ensure_project_outlines(
            project_path,
            config,
            sync_progress=False,
            progress_callback=progress_callback,
        )
        project_data["outlines"] = outlines
        next_context = find_next_chapter_context(outlines, current_chapter_count)
        if next_context is None:
            raise ValueError("No usable next chapter outline was found. Regenerate chapter outlines first.")
        return project_data, next_context

    if planning_mode == PLANNING_MODE_VOLUME:
        outlines = ensure_volume_outlines(project_path, config, progress_callback=progress_callback)
        project_data["outlines"] = outlines
        upcoming_contexts = collect_upcoming_volume_contexts(outlines, current_chapter_count, 1)
        if not upcoming_contexts:
            raise ValueError("No usable next volume outline was found. Regenerate volume outlines first.")
        return project_data, upcoming_contexts[0]

    if planning_mode == PLANNING_MODE_NONE:
        return project_data, {"volume": {}, "chapter": {}}

    return project_data, {"volume": {}, "chapter": {}}


def peek_next_context_for_mode(project_data: dict, planning_mode: str) -> dict:
    current_chapter_count = int((project_data.get("project") or {}).get("chapter_count", 0) or 0)
    normalized_mode = normalize_planning_mode(planning_mode, default=DEFAULT_PLANNING_MODE)

    if normalized_mode == PLANNING_MODE_CHAPTER:
        outlines = project_data.get("outlines") or {"meta": {}, "volumes": []}
        next_context = find_next_chapter_context(outlines, current_chapter_count)
        if next_context is not None:
            return next_context
        return {"volume": {}, "chapter": {}}

    if normalized_mode == PLANNING_MODE_VOLUME:
        outlines = normalize_outlines(project_data.get("outlines") or {"meta": {}, "volumes": []})
        upcoming_contexts = collect_upcoming_volume_contexts(outlines, current_chapter_count, 1)
        if upcoming_contexts:
            return upcoming_contexts[0]
        return {"volume": {}, "chapter": {}}

    if normalized_mode == PLANNING_MODE_NONE:
        contexts = collect_upcoming_freeform_contexts(project_data, 1)
        if contexts:
            return contexts[0]
        return {"volume": {}, "chapter": {}}

    return {"volume": {}, "chapter": {}}


def compact_contexts(upcoming_contexts: list[dict]) -> list[dict]:
    compact = []
    for context in upcoming_contexts:
        volume = context.get("volume") or {}
        chapter = context.get("chapter") or {}
        compact.append(
            {
                "volume_number": volume.get("volume_number", 0),
                "volume_title": volume.get("title", ""),
                "volume_summary": volume.get("summary", ""),
                "chapter_number": chapter.get("chapter_number", 0),
                "chapter_in_volume": chapter.get("chapter_in_volume", 0),
                "title": chapter.get("title", ""),
                "summary": chapter.get("summary", ""),
                "goal": chapter.get("goal", ""),
                "key_events": chapter.get("key_events", []),
            }
        )
    return compact
