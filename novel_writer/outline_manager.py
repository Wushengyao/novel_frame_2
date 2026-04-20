"""Two-level outline generation and tracking helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from common_utils import emit_progress, extract_json_object, safe_int, utc_now
from console_logger import log_error, log_info, log_success, log_warning
from context_builder import build_chapter_outline_context, build_volume_outline_context
from llm_client import generate_text_with_metadata
from prompt_builder import build_chapter_outline_prompt, build_volume_outline_prompt
from project_manager import (
    load_json,
    load_project,
    record_context_telemetry,
    save_json,
    update_project_stats,
)


EMPTY_OUTLINE_META = {
    "chapter_outline_stale": False,
    "last_volume_outline_request": "",
    "last_chapter_outline_request": "",
    "updated_at": "",
}

EMPTY_VOLUME = {
    "volume_number": 1,
    "title": "",
    "summary": "",
    "story_goal": "",
    "planned_chapter_count": 0,
    "chapters": [],
}

EMPTY_CHAPTER_OUTLINE = {
    "chapter_number": 1,
    "chapter_in_volume": 1,
    "title": "",
    "summary": "",
    "goal": "",
    "key_events": [],
    "status": "planned",
}

OUTLINE_MAX_ATTEMPTS = 3


class OutlineGenerationError(RuntimeError):
    """Raised when outline generation does not produce a usable result."""


def load_outlines(project_path: str) -> dict:
    outlines_path = Path(project_path) / "outlines.json"
    if not outlines_path.exists():
        return {
            "meta": deepcopy(EMPTY_OUTLINE_META),
            "volumes": [],
        }
    data = load_json(str(outlines_path))
    return normalize_outlines(data)


def save_outlines(project_path: str, outlines: dict) -> None:
    normalized = normalize_outlines(outlines)
    normalized["meta"]["updated_at"] = utc_now()
    save_json(str(Path(project_path) / "outlines.json"), normalized)


def normalize_outlines(data: dict | None) -> dict:
    source = data if isinstance(data, dict) else {}
    meta = deepcopy(EMPTY_OUTLINE_META)
    if isinstance(source.get("meta"), dict):
        meta.update(source["meta"])

    volumes = []
    for index, raw_volume in enumerate(source.get("volumes") or [], start=1):
        volume = deepcopy(EMPTY_VOLUME)
        if isinstance(raw_volume, dict):
            volume.update(raw_volume)
        volume["volume_number"] = safe_int(volume.get("volume_number"), index) or index
        volume["title"] = str(volume.get("title", "") or "").strip()
        volume["summary"] = str(volume.get("summary", "") or "").strip()
        volume["story_goal"] = str(volume.get("story_goal", "") or "").strip()
        volume["planned_chapter_count"] = max(1, safe_int(volume.get("planned_chapter_count"), 1))
        chapters = []
        for chapter_index, raw_chapter in enumerate(volume.get("chapters") or [], start=1):
            chapter = deepcopy(EMPTY_CHAPTER_OUTLINE)
            if isinstance(raw_chapter, dict):
                chapter.update(raw_chapter)
            chapter["chapter_in_volume"] = safe_int(chapter.get("chapter_in_volume"), chapter_index) or chapter_index
            chapter["chapter_number"] = safe_int(chapter.get("chapter_number"), 0)
            chapter["title"] = str(chapter.get("title", "") or "").strip()
            chapter["summary"] = str(chapter.get("summary", "") or "").strip()
            chapter["goal"] = str(chapter.get("goal", "") or "").strip()
            key_events = chapter.get("key_events") or []
            if not isinstance(key_events, list):
                key_events = [str(key_events)]
            chapter["key_events"] = [str(item).strip() for item in key_events if str(item).strip()]
            status = str(chapter.get("status", "planned") or "planned").strip().lower()
            chapter["status"] = status if status in {"planned", "completed"} else "planned"
            chapters.append(chapter)
        volume["chapters"] = chapters
        volumes.append(volume)

    normalized = {
        "meta": meta,
        "volumes": volumes,
    }
    return _assign_chapter_numbers(normalized)


def _assign_chapter_numbers(outlines: dict) -> dict:
    normalized = deepcopy(outlines)
    chapter_number = 1
    for volume in normalized.get("volumes", []):
        chapters = volume.get("chapters") or []
        if volume.get("planned_chapter_count", 0) < len(chapters):
            volume["planned_chapter_count"] = len(chapters)
        for chapter_index, chapter in enumerate(chapters, start=1):
            chapter["chapter_in_volume"] = chapter_index
            chapter["chapter_number"] = chapter_number
            chapter_number += 1
    return normalized


def _iter_chapters(outlines: dict) -> list[tuple[dict, dict]]:
    pairs = []
    for volume in outlines.get("volumes", []):
        for chapter in volume.get("chapters", []):
            pairs.append((volume, chapter))
    return pairs


def _build_completed_chapter_context(project_data: dict) -> list[dict]:
    outlines = normalize_outlines(project_data.get("outlines"))
    completed = []
    for volume, chapter in _iter_chapters(outlines):
        if chapter.get("status") != "completed":
            continue
        completed.append(
            {
                "chapter_number": chapter.get("chapter_number", 0),
                "volume_number": volume.get("volume_number", 0),
                "title": chapter.get("title", ""),
                "summary": chapter.get("summary", ""),
                "goal": chapter.get("goal", ""),
            }
        )
    return completed[-8:]


def _ensure_outline_model_config(config: dict) -> None:
    provider = str(config.get("model_provider", "") or "").strip().lower()
    model_name = str(config.get("model") or config.get("model_name") or "").strip()
    api_key = str(config.get("api_key", "") or "").strip()
    api_base = str(config.get("api_base", "") or "").strip()

    if not model_name:
        raise OutlineGenerationError("大纲生成失败：缺少模型名称。")
    if provider == "openai_compatible" and (not api_key or not api_base):
        raise OutlineGenerationError("大纲生成失败：openai_compatible 缺少 api_key 或 api_base。")
    if provider == "ollama" and not api_base:
        raise OutlineGenerationError("大纲生成失败：ollama 缺少 api_base。")
    if provider in {"gemini", "grok", "deepseek", "doubao"} and not api_key:
        raise OutlineGenerationError(f"大纲生成失败：provider={provider} 缺少 api_key。")


def _validate_volume_outline_response(data: dict) -> dict:
    raw_volumes = data.get("volumes")
    if not isinstance(raw_volumes, list) or not raw_volumes:
        raise ValueError("模型返回的分卷结果为空。")
    for index, raw_volume in enumerate(raw_volumes, start=1):
        if not isinstance(raw_volume, dict):
            raise ValueError(f"第 {index} 卷数据格式非法。")
        if safe_int(raw_volume.get("planned_chapter_count"), 0) < 1:
            raise ValueError(f"第 {index} 卷 planned_chapter_count 非法。")

    normalized = normalize_outlines(data)
    for volume in normalized.get("volumes", []):
        if not volume.get("title"):
            raise ValueError(f"第 {volume.get('volume_number', '?')} 卷缺少 title。")
        if not volume.get("summary"):
            raise ValueError(f"第 {volume.get('volume_number', '?')} 卷缺少 summary。")
        if not volume.get("story_goal"):
            raise ValueError(f"第 {volume.get('volume_number', '?')} 卷缺少 story_goal。")
    return normalized


def _validate_chapter_outline_response(data: dict, volume: dict) -> dict:
    planned_count = max(1, safe_int(volume.get("planned_chapter_count"), 1))
    raw_chapters = data.get("chapters")
    if not isinstance(raw_chapters, list):
        raise ValueError("模型返回缺少 chapters 列表。")
    if len(raw_chapters) != planned_count:
        raise ValueError(
            f"模型返回章节数不匹配：期望 {planned_count} 章，实际 {len(raw_chapters)} 章。"
        )

    normalized_volume = normalize_outlines(
        {
            "volumes": [
                {
                    **volume,
                    "chapters": raw_chapters,
                }
            ]
        }
    )["volumes"][0]
    chapters = normalized_volume.get("chapters") or []

    for chapter in chapters:
        chapter_label = f"第 {chapter.get('chapter_in_volume', '?')} 章"
        if not chapter.get("title"):
            raise ValueError(f"{chapter_label} 缺少 title。")
        if not chapter.get("summary"):
            raise ValueError(f"{chapter_label} 缺少 summary。")
        if not chapter.get("goal"):
            raise ValueError(f"{chapter_label} 缺少 goal。")
        key_events = chapter.get("key_events") or []
        if not 2 <= len(key_events) <= 5:
            raise ValueError(f"{chapter_label} 的 key_events 数量必须在 2 到 5 之间。")

    return {
        "volume_number": normalized_volume.get("volume_number", volume.get("volume_number", 1)),
        "chapters": chapters,
    }


def _generate_outline_json(
    prompt: str,
    config: dict,
    project_path: str,
    *,
    validator,
    context: str,
    max_attempts: int = OUTLINE_MAX_ATTEMPTS,
) -> dict:
    _ensure_outline_model_config(config)
    last_error = None

    for attempt in range(max_attempts):
        try:
            log_info(f"{context}: 第 {attempt + 1}/{max_attempts} 次请求模型。")
            response_text, metadata = generate_text_with_metadata(prompt, config)
        except Exception as exc:  # pragma: no cover - resilience path
            update_project_stats(project_path, phase="outline", success=False, usage=None)
            last_error = exc
            log_warning(f"{context}: 第 {attempt + 1}/{max_attempts} 次请求失败，原因: {exc}")
            continue

        try:
            parsed = extract_json_object(response_text, "Could not parse JSON from outline response.")
            validated = validator(parsed)
        except Exception as exc:  # pragma: no cover - resilience path
            update_project_stats(
                project_path,
                phase="outline",
                success=False,
                usage=metadata.get("usage"),
            )
            last_error = exc
            log_warning(f"{context}: 第 {attempt + 1}/{max_attempts} 次返回内容无效，原因: {exc}")
            continue

        update_project_stats(
            project_path,
            phase="outline",
            success=True,
            usage=metadata.get("usage"),
        )
        log_success(f"{context}: 第 {attempt + 1}/{max_attempts} 次请求成功。")
        return validated

    raise OutlineGenerationError(
        f"{context} 在 {max_attempts} 次尝试后仍未生成可用结果。最后错误: {last_error}"
    )


def regenerate_volume_outline(
    project_path: str,
    config: dict,
    user_request: str = "",
    progress_callback=None,
) -> dict:
    log_info(f"分卷大纲: 开始生成，项目={project_path}")
    emit_progress(progress_callback, "volume_outline", "正在生成分卷大纲")
    project_data = load_project(project_path)
    prompt_context = build_volume_outline_context(
        project_path,
        project_data,
        _build_completed_chapter_context(project_data),
        user_request,
    )
    prompt = build_volume_outline_prompt(
        prompt_context,
        user_request=user_request,
    )
    record_context_telemetry(
        project_path,
        "outline",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"prompt_type": "volume_outline"},
    )
    outlines = _generate_outline_json(
        prompt,
        config,
        project_path,
        validator=_validate_volume_outline_response,
        context="分卷大纲",
    )
    log_success(f"分卷大纲: 模型生成完成，共 {len(outlines.get('volumes', []))} 卷。")
    outlines["meta"]["chapter_outline_stale"] = True
    outlines["meta"]["last_volume_outline_request"] = user_request.strip()
    save_outlines(project_path, outlines)
    emit_progress(progress_callback, "volume_outline_done", "分卷大纲已写入")
    log_success("分卷大纲: 已写入 outlines.json，并标记分章大纲需要同步。")
    return outlines


def _find_volume(outlines: dict, volume_number: int) -> dict | None:
    for volume in outlines.get("volumes", []):
        if safe_int(volume.get("volume_number"), 0) == volume_number:
            return deepcopy(volume)
    return None


def _all_volumes_have_chapters(outlines: dict) -> bool:
    volumes = outlines.get("volumes", [])
    if not volumes:
        return False
    for volume in volumes:
        planned_count = max(1, safe_int(volume.get("planned_chapter_count"), 1))
        if len(volume.get("chapters") or []) < planned_count:
            return False
    return True


def _save_partial_chapter_outlines(
    project_path: str,
    existing_outlines: dict,
    new_volumes: list[dict],
    user_request: str,
) -> None:
    partial_outlines = deepcopy(existing_outlines)
    remaining_volumes = [
        deepcopy(volume)
        for volume in existing_outlines.get("volumes", [])[len(new_volumes) :]
    ]
    partial_outlines["volumes"] = new_volumes + remaining_volumes
    partial_outlines = normalize_outlines(partial_outlines)
    partial_outlines["meta"]["last_chapter_outline_request"] = user_request.strip()
    partial_outlines["meta"]["chapter_outline_stale"] = True
    save_outlines(project_path, partial_outlines)


def _build_placeholder_completed_chapters(count: int) -> list[dict]:
    chapters = []
    for chapter_index in range(1, count + 1):
        chapters.append(
            {
                "chapter_in_volume": chapter_index,
                "title": f"已完成章节{chapter_index}",
                "summary": "该章正文已经存在，请以已写内容和剧情状态为准。",
                "goal": "与已写正文保持一致。",
                "key_events": [],
                "status": "completed",
            }
        )
    return chapters


def regenerate_chapter_outline(
    project_path: str,
    config: dict,
    volume_number: int | None = None,
    user_request: str = "",
    progress_callback=None,
) -> dict:
    if volume_number is None:
        log_info(f"分章大纲: 开始为全部卷生成，项目={project_path}")
    else:
        log_info(f"分章大纲: 开始为第 {volume_number} 卷生成，项目={project_path}")

    existing_outlines = load_outlines(project_path)
    if not existing_outlines.get("volumes"):
        log_warning("分章大纲: 当前没有分卷大纲，先自动生成分卷大纲。")
        emit_progress(progress_callback, "chapter_outline_prepare", "缺少分卷大纲，正在补生成")
        existing_outlines = regenerate_volume_outline(
            project_path,
            config,
            user_request="",
            progress_callback=progress_callback,
        )

    project_data = load_project(project_path)
    written_remaining = int(project_data.get("project", {}).get("chapter_count", 0) or 0)
    updated_outlines = deepcopy(existing_outlines)
    new_volumes = []

    for volume in existing_outlines.get("volumes", []):
        planned_count = max(1, safe_int(volume.get("planned_chapter_count"), 1))
        existing_volume = _find_volume(existing_outlines, safe_int(volume.get("volume_number"), 0))
        existing_completed = []
        if existing_volume and existing_volume.get("chapters"):
            for chapter in existing_volume.get("chapters", []):
                if chapter.get("status") == "completed":
                    existing_completed.append(deepcopy(chapter))

        completed_count = min(written_remaining, len(existing_completed) or planned_count)
        written_remaining = max(0, written_remaining - completed_count)
        preserved_completed = existing_completed[:completed_count] or _build_placeholder_completed_chapters(completed_count)

        should_regenerate = volume_number is None or safe_int(volume.get("volume_number"), 0) == volume_number
        if should_regenerate:
            emit_progress(
                progress_callback,
                "chapter_outline_volume",
                f"正在生成第 {volume.get('volume_number', '?')} 卷分章大纲",
            )
            log_info(
                f"分章大纲: 正在生成第 {volume.get('volume_number', '?')} 卷，"
                f"计划 {planned_count} 章，已完成 {completed_count} 章。"
            )
            prompt_context = build_chapter_outline_context(
                project_path,
                project_data,
                volume,
                new_volumes,
                preserved_completed,
                user_request,
            )
            prompt = build_chapter_outline_prompt(
                prompt_context,
                volume=volume,
                previous_volumes=new_volumes,
                completed_chapters=preserved_completed,
                user_request=user_request,
            )
            record_context_telemetry(
                project_path,
                "outline",
                prompt_chars=len(prompt),
                section_chars=prompt_context.get("section_chars"),
                planning_mode=config.get("planning_mode", ""),
                extra={
                    "prompt_type": "chapter_outline",
                    "target_volume_number": safe_int(volume.get("volume_number"), 0),
                },
            )
            try:
                generated = _generate_outline_json(
                    prompt,
                    config,
                    project_path,
                    validator=lambda data, current_volume=volume: _validate_chapter_outline_response(data, current_volume),
                    context=f"分章大纲: 第 {volume.get('volume_number', '?')} 卷",
                )
            except Exception as exc:
                log_error(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷生成失败，原因: {exc}")
                raise

            generated_chapters = generated.get("chapters") or []
            volume_chapters = preserved_completed + generated_chapters[completed_count:planned_count]
            log_success(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷生成完成。")
        else:
            volume_chapters = deepcopy(existing_volume.get("chapters") or []) if existing_volume else []
            log_info(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷沿用现有章纲。")

        volume_copy = deepcopy(volume)
        volume_copy["chapters"] = volume_chapters[:planned_count]
        new_volumes.append(volume_copy)
        _save_partial_chapter_outlines(project_path, existing_outlines, new_volumes, user_request)
        emit_progress(
            progress_callback,
            "chapter_outline_saved",
            f"第 {volume.get('volume_number', '?')} 卷分章大纲已保存",
        )
        log_info(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷进度已保存到 outlines.json。")

    updated_outlines["volumes"] = new_volumes
    updated_outlines = normalize_outlines(updated_outlines)
    updated_outlines["meta"]["last_chapter_outline_request"] = user_request.strip()
    updated_outlines["meta"]["chapter_outline_stale"] = not _all_volumes_have_chapters(updated_outlines)
    save_outlines(project_path, updated_outlines)
    sync_outline_progress(project_path, updated_outlines)
    if updated_outlines["meta"]["chapter_outline_stale"]:
        log_warning("分章大纲: 已保存，但仍有卷缺少完整章纲。")
    else:
        log_success("分章大纲: 全部章纲已保存并同步到剧情状态。")
    emit_progress(progress_callback, "chapter_outline_done", "分章大纲生成完成")
    return updated_outlines


def sync_outline_progress(project_path: str, outlines: dict | None = None) -> dict:
    log_info(f"章纲进度同步: 开始同步项目 {project_path}")
    outline_data = normalize_outlines(outlines) if outlines is not None else load_outlines(project_path)
    written_count = int(load_json(str(Path(project_path) / "project.json")).get("chapter_count", 0) or 0)
    chapter_index = 0
    for volume in outline_data.get("volumes", []):
        for chapter in volume.get("chapters", []):
            chapter_index += 1
            chapter["status"] = "completed" if chapter_index <= written_count else "planned"
    outline_data = normalize_outlines(outline_data)
    save_outlines(project_path, outline_data)
    _sync_plot_state_next_goal(project_path, outline_data)
    log_success(f"章纲进度同步: 已按已写章节数 {written_count} 更新章纲状态。")
    return outline_data


def _sync_plot_state_next_goal(project_path: str, outlines: dict) -> None:
    next_context = find_next_chapter_context(
        outlines,
        int(load_json(str(Path(project_path) / "project.json")).get("chapter_count", 0) or 0),
    )
    if next_context is None:
        return
    plot_state_path = Path(project_path) / "plot_state.json"
    plot_state = load_json(str(plot_state_path))
    chapter = next_context["chapter"]
    plot_state["next_chapter_goal"] = chapter.get("goal") or chapter.get("summary") or plot_state.get("next_chapter_goal", "")
    save_json(str(plot_state_path), plot_state)


def apply_chapter_outline_override(project_path: str, chapter_number: int, chapter_outline: dict) -> dict:
    outlines = load_outlines(project_path)
    target_number = max(1, safe_int(chapter_number, 0))
    raw_key_events = chapter_outline.get("key_events") or []
    if not isinstance(raw_key_events, list):
        raw_key_events = [raw_key_events]
    updated = False
    normalized_key_events = [
        str(item).strip()
        for item in raw_key_events
        if str(item).strip()
    ][:5]
    for volume in outlines.get("volumes", []):
        for chapter in volume.get("chapters", []):
            if safe_int(chapter.get("chapter_number"), 0) != target_number:
                continue
            for key in ("title", "summary", "goal"):
                value = str(chapter_outline.get(key, "") or "").strip()
                if value:
                    chapter[key] = value
            if normalized_key_events:
                chapter["key_events"] = normalized_key_events
            updated = True
            break
        if updated:
            break

    if not updated:
        raise ValueError(f"未找到第 {target_number} 章章纲，无法应用推进方案。")

    outlines = normalize_outlines(outlines)
    save_outlines(project_path, outlines)
    _sync_plot_state_next_goal(project_path, outlines)
    return outlines


def find_next_chapter_context(outlines: dict, written_chapter_count: int) -> dict | None:
    normalized = normalize_outlines(outlines)
    for volume in normalized.get("volumes", []):
        for chapter in volume.get("chapters", []):
            if safe_int(chapter.get("chapter_number"), 0) == written_chapter_count + 1:
                return {
                    "volume": deepcopy(volume),
                    "chapter": deepcopy(chapter),
                }
    return None


def ensure_project_outlines(
    project_path: str,
    config: dict,
    *,
    sync_progress: bool = True,
    progress_callback=None,
) -> dict:
    outlines = load_outlines(project_path)
    if not outlines.get("volumes"):
        log_warning("章纲检查: 未发现 outlines.json，准备自动补全分卷和分章大纲。")
        emit_progress(progress_callback, "outline_prepare", "未发现大纲，正在自动补生成")
        regenerate_volume_outline(project_path, config, user_request="", progress_callback=progress_callback)
        outlines = regenerate_chapter_outline(
            project_path,
            config,
            volume_number=None,
            user_request="",
            progress_callback=progress_callback,
        )
    elif outlines.get("meta", {}).get("chapter_outline_stale"):
        log_error("章纲检查: 分卷大纲已经更新，但分章大纲尚未同步。")
        raise ValueError("分卷大纲已经更新，但分章大纲尚未同步，请先重新生成分章大纲。")
    elif not _all_volumes_have_chapters(outlines):
        log_warning("章纲检查: 检测到章纲不完整，准备自动补全。")
        emit_progress(progress_callback, "outline_repair", "检测到分章大纲不完整，正在补全")
        outlines = regenerate_chapter_outline(
            project_path,
            config,
            volume_number=None,
            user_request="",
            progress_callback=progress_callback,
        )
    else:
        outlines = normalize_outlines(outlines)
        if sync_progress:
            outlines = sync_outline_progress(project_path, outlines)
        log_success("章纲检查: 分卷和分章大纲均可用。")
    emit_progress(progress_callback, "outline_ready", "大纲检查完成")
    return outlines


def get_outline_status(project_path: str) -> dict:
    outlines = load_outlines(project_path)
    project = load_json(str(Path(project_path) / "project.json"))
    next_context = find_next_chapter_context(outlines, int(project.get("chapter_count", 0) or 0))
    return {
        "has_outlines": bool(outlines.get("volumes")),
        "volume_count": len(outlines.get("volumes", [])),
        "chapter_outline_stale": bool(outlines.get("meta", {}).get("chapter_outline_stale")),
        "next_context": next_context,
    }
