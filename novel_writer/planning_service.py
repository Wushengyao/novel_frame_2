"""Batch chapter planning helpers shared by CLI and Web flows."""

from __future__ import annotations

from copy import deepcopy

from chapter_context import (
    collect_upcoming_chapter_contexts,
    collect_upcoming_freeform_contexts,
    collect_upcoming_volume_contexts,
    compact_contexts,
    ensure_volume_outlines,
    resolve_planning_mode,
)
from common_utils import emit_progress, extract_json_object, safe_int
from console_logger import log_info, log_success, log_warning
from context_builder import build_batch_plan_context
from llm_client import generate_text_with_metadata
from outline_manager import ensure_project_outlines
from project_manager import (
    PLANNING_MODE_CHAPTER,
    PLANNING_MODE_VOLUME,
    load_project,
    record_context_telemetry,
    update_project_stats,
)
from prompt_builder import build_batch_chapter_plan_prompt, build_system_prompt


def _fallback_batch_plan(
    upcoming_contexts: list[dict],
    user_request: str,
    *,
    allow_outline_override: bool,
) -> dict[int, dict]:
    request = user_request.strip()
    total = len(upcoming_contexts)
    if not request or total == 0:
        return {}

    plan_by_number = {}
    for index, context in enumerate(upcoming_contexts):
        chapter = deepcopy(context["chapter"])
        if total == 1:
            guidance = f'Use "{request}" as the core focus of this chapter and integrate it naturally.'
            role = "direct"
            focus = request
        elif index == 0:
            guidance = f'This chapter should set up "{request}" through preparation, motivation, division of work, or obstacles.'
            role = "setup"
            focus = f"setup for {request}"
        elif index == total - 1:
            guidance = f'This chapter should deliver a meaningful payoff for "{request}" without repeating earlier setup beats.'
            role = "payoff"
            focus = f"payoff for {request}"
        else:
            guidance = f'This chapter should advance "{request}" with new progress, setbacks, adjustments, or character interaction.'
            role = "progress"
            focus = f"progress on {request}"

        chapter["request_focus"] = focus
        chapter["request_role"] = role
        chapter["writer_guidance"] = guidance
        plan_by_number[safe_int(chapter.get("chapter_number"), 0)] = {
            "chapter_outline": chapter if allow_outline_override else None,
            "user_request": guidance,
        }
    return plan_by_number


def _normalize_batch_plan_response(
    data: dict,
    upcoming_contexts: list[dict],
    *,
    allow_outline_override: bool,
) -> dict[int, dict]:
    raw_chapters = data.get("chapters")
    if not isinstance(raw_chapters, list):
        raise ValueError("batch plan response missing chapters list")

    expected_numbers = {
        safe_int(context["chapter"].get("chapter_number"), 0): context
        for context in upcoming_contexts
    }
    if len(raw_chapters) != len(expected_numbers):
        raise ValueError("batch plan chapter count does not match requested count")

    raw_by_number = {}
    for item in raw_chapters:
        if not isinstance(item, dict):
            raise ValueError("batch plan item must be an object")
        chapter_number = safe_int(item.get("chapter_number"), 0)
        if chapter_number not in expected_numbers:
            raise ValueError(f"unexpected chapter_number in batch plan: {chapter_number}")
        raw_by_number[chapter_number] = item

    if len(raw_by_number) != len(expected_numbers):
        raise ValueError("batch plan contains duplicate or missing chapter numbers")

    normalized = {}
    for chapter_number, context in expected_numbers.items():
        raw = raw_by_number[chapter_number]
        merged_outline = deepcopy(context["chapter"])
        for key in ("title", "summary", "goal", "request_focus", "request_role", "writer_guidance"):
            value = str(raw.get(key, "") or "").strip()
            if value:
                merged_outline[key] = value

        key_events = raw.get("key_events") or []
        if not isinstance(key_events, list):
            key_events = [key_events]
        normalized_events = [str(item).strip() for item in key_events if str(item).strip()]
        if normalized_events:
            merged_outline["key_events"] = normalized_events[:5]

        user_request = str(raw.get("writer_guidance") or raw.get("request_focus") or "").strip()
        normalized[chapter_number] = {
            "chapter_outline": merged_outline if allow_outline_override else None,
            "user_request": user_request,
        }
    return normalized


def plan_batch_chapters(
    project_path: str,
    config: dict,
    count: int,
    user_request: str,
    progress_callback=None,
) -> tuple[str, dict[int, dict]]:
    request = user_request.strip()
    project_data = load_project(project_path)
    planning_mode = resolve_planning_mode(config, project_data)
    if count <= 1 or not request:
        return planning_mode, {}

    log_info(
        f"next_chapters: planning batch request for next {count} chapters "
        f"project={project_path} mode={planning_mode}"
    )
    emit_progress(progress_callback, "chapter_batch_plan", "正在规划接下来几章如何分配你想看的情节")

    if planning_mode == PLANNING_MODE_CHAPTER:
        outlines = ensure_project_outlines(
            project_path,
            config,
            sync_progress=False,
            progress_callback=progress_callback,
        )
        project_data["outlines"] = outlines
        current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
        upcoming_contexts = collect_upcoming_chapter_contexts(outlines, current_chapter_count, count)
        allow_outline_override = True
    elif planning_mode == PLANNING_MODE_VOLUME:
        outlines = ensure_volume_outlines(project_path, config, progress_callback=progress_callback)
        project_data["outlines"] = outlines
        current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
        upcoming_contexts = collect_upcoming_volume_contexts(outlines, current_chapter_count, count)
        allow_outline_override = False
    else:
        upcoming_contexts = collect_upcoming_freeform_contexts(project_data, count)
        allow_outline_override = False

    if len(upcoming_contexts) != count:
        log_warning(
            "next_chapters: could not collect enough upcoming chapter contexts, "
            "falling back to heuristic request distribution"
        )
        return planning_mode, _fallback_batch_plan(
            upcoming_contexts,
            request,
            allow_outline_override=allow_outline_override,
        )

    prompt_context = build_batch_plan_context(
        project_path,
        project_data,
        compact_contexts(upcoming_contexts),
        request,
    )
    prompt = build_batch_chapter_plan_prompt(prompt_context, compact_contexts(upcoming_contexts), request)
    record_context_telemetry(
        project_path,
        "outline",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=planning_mode,
        extra={
            "prompt_type": "batch_plan",
            "requested_chapter_count": count,
        },
    )
    try:
        response_text, metadata = generate_text_with_metadata(
            prompt,
            config,
            system_prompt=build_system_prompt("planner"),
            response_format="json",
        )
        update_project_stats(
            project_path,
            phase="outline",
            success=True,
            usage=metadata.get("usage"),
            metadata=metadata,
        )
        plan = _normalize_batch_plan_response(
            extract_json_object(response_text, "Could not parse JSON from batch chapter plan response."),
            upcoming_contexts,
            allow_outline_override=allow_outline_override,
        )
        log_success(f"next_chapters: batch request planned for {len(plan)} chapters")
        return planning_mode, plan
    except Exception as exc:
        update_project_stats(project_path, phase="outline", success=False, usage=None)
        log_warning(f"next_chapters: batch planning failed, fallback to heuristic. reason: {exc}")
        return planning_mode, _fallback_batch_plan(
            upcoming_contexts,
            request,
            allow_outline_override=allow_outline_override,
        )
