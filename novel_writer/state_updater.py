"""Update plot state from newly generated chapter text."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from common_utils import emit_progress, extract_json_object, safe_int
from console_logger import log_info, log_success, log_warning
from context_builder import (
    SUMMARY_LIST_LIMITS,
    build_retrieval_tags,
    build_summary_context,
    normalize_craft_notes,
    normalize_live_plot_state,
    write_arc_summary,
)
from llm_client import generate_text_with_metadata
from prompt_builder import build_summary_prompt, build_system_prompt
from project_manager import (
    load_json,
    load_project,
    record_context_telemetry,
    save_json,
    update_project_stats,
)


def _normalize_list(value: object, *, max_items: int) -> list[str]:
    items = value or []
    if not isinstance(items, list):
        items = [items]
    normalized = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
        if len(normalized) >= max_items:
            break
    return normalized


def _normalize_summary(summary: dict, current_state: dict) -> dict:
    live_state = normalize_live_plot_state({**current_state, **(summary if isinstance(summary, dict) else {})})
    normalized = {
        "chapter_summary": str((summary or {}).get("chapter_summary", "") or "").strip(),
        "current_location": live_state.get("current_location", ""),
        "current_time": live_state.get("current_time", ""),
        "current_arc": live_state.get("current_arc", ""),
        "recent_events": live_state.get("recent_events", []),
        "open_threads": live_state.get("open_threads", []),
        "resolved_threads": live_state.get("resolved_threads", []),
        "foreshadowing": live_state.get("foreshadowing", []),
        "continuity_anchors": live_state.get("continuity_anchors", []),
        "causal_links": live_state.get("causal_links", []),
        "character_updates": live_state.get("character_updates", []),
        "active_characters": live_state.get("active_characters", []),
        "retrieval_tags": _normalize_list((summary or {}).get("retrieval_tags"), max_items=SUMMARY_LIST_LIMITS["retrieval_tags"]),
        "next_chapter_goal": str((summary or {}).get("next_chapter_goal", "") or "").strip(),
        "craft_notes": normalize_craft_notes((summary or {}).get("craft_notes")),
    }
    if not normalized["chapter_summary"]:
        normalized["chapter_summary"] = "；".join(normalized["recent_events"][:2])[:280]
    if not normalized["retrieval_tags"]:
        normalized["retrieval_tags"] = build_retrieval_tags(normalized)
    return normalized


def _fallback_summary(new_text: str, current_state: dict) -> dict:
    fallback = _normalize_summary(current_state, current_state)
    excerpt = new_text.strip().replace("\n", " ")
    fallback["chapter_summary"] = excerpt[:220]
    fallback["recent_events"] = _normalize_list(
        list(current_state.get("recent_events", [])) + [excerpt[:200]],
        max_items=SUMMARY_LIST_LIMITS["recent_events"],
    )
    if not fallback["next_chapter_goal"]:
        fallback["next_chapter_goal"] = current_state.get("next_chapter_goal", "")
    if not fallback["current_arc"]:
        fallback["current_arc"] = current_state.get("current_arc", "")
    fallback["retrieval_tags"] = build_retrieval_tags(fallback)
    fallback["craft_notes"] = normalize_craft_notes({})
    return fallback


def _merge_state(current_state: dict, summary: dict) -> dict:
    updated = normalize_live_plot_state(deepcopy(current_state))

    for key in ("current_location", "current_time", "current_arc", "next_chapter_goal"):
        value = str(summary.get(key, "") or "").strip()
        if value:
            updated[key] = value

    for key in (
        "recent_events",
        "foreshadowing",
        "continuity_anchors",
        "causal_links",
        "character_updates",
        "active_characters",
        "resolved_threads",
    ):
        updated[key] = _normalize_list(summary.get(key), max_items=SUMMARY_LIST_LIMITS[key])

    resolved = set(updated.get("resolved_threads", []))
    open_threads = [
        item
        for item in _normalize_list(summary.get("open_threads"), max_items=SUMMARY_LIST_LIMITS["open_threads"])
        if item not in resolved
    ]
    updated["open_threads"] = open_threads[: SUMMARY_LIST_LIMITS["open_threads"]]
    return updated


def update_plot_state(
    project_path: str,
    new_text: str,
    config: dict,
    progress_callback=None,
    log_context: dict | None = None,
) -> None:
    log_info(f"剧情状态更新: 开始处理项目 {project_path}")
    emit_progress(progress_callback, "summary_prepare", "正在总结新章节并刷新剧情状态")
    base = Path(project_path)
    plot_state_path = base / "plot_state.json"
    summaries_dir = base / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    project_data = load_project(project_path)
    chapter_number = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    current_state = normalize_live_plot_state(load_json(str(plot_state_path)))
    prompt_context = build_summary_context(project_path, project_data, new_text)
    prompt = build_summary_prompt(prompt_context, new_text)
    record_context_telemetry(
        project_path,
        "summary",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number},
    )

    summary = None
    last_error = None
    summary_log_context = {
        "phase": "summary",
        "project_id": str(project_data["project"]["project_id"] or "").strip(),
        "project_path": str(base.resolve()),
        "chapter_count": int(load_json(str(base / "project.json")).get("chapter_count", 0) or 0),
        "section_chars": prompt_context.get("section_chars", {}),
    }
    if log_context:
        summary_log_context.update(log_context)
        summary_log_context["phase"] = "summary"
    for attempt in range(2):
        try:
            log_info(f"剧情状态更新: 第 {attempt + 1} 次请求模型总结本章。")
            emit_progress(progress_callback, "summary_request", f"正在请求剧情状态总结（第 {attempt + 1}/2 次）")
            response_text, metadata = generate_text_with_metadata(
                prompt,
                config,
                log_context=summary_log_context,
                system_prompt=build_system_prompt("summary"),
                response_format="json",
            )
        except Exception as exc:  # pragma: no cover - intentional resilience path
            update_project_stats(project_path, phase="summary", success=False, usage=None, chapter_number=chapter_number)
            last_error = exc
            log_warning(f"剧情状态更新: 第 {attempt + 1} 次请求失败，原因: {exc}")
            continue

        try:
            update_project_stats(
                project_path,
                phase="summary",
                success=True,
                usage=metadata.get("usage"),
                metadata=metadata,
                chapter_number=chapter_number,
            )
            summary = _normalize_summary(
                extract_json_object(response_text, "Could not parse JSON from summary response."),
                current_state,
            )
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

    updated_state = _merge_state(current_state, summary)
    save_json(str(plot_state_path), updated_state)

    chapter_count = load_json(str(base / "project.json")).get("chapter_count", 0)
    summary_path = summaries_dir / f"summary_{chapter_count:04d}.json"
    save_json(str(summary_path), summary)
    write_arc_summary(project_path, safe_int(chapter_count, 0))
    emit_progress(progress_callback, "summary_done", "剧情状态已更新并保存")
    log_success(f"剧情状态更新: 已写入 plot_state.json 和 {summary_path.name}")
