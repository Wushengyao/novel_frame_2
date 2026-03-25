"""Two-level outline generation and tracking helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from console_logger import log_error, log_info, log_success, log_warning
from llm_client import generate_text_with_metadata
from prompt_builder import build_chapter_outline_prompt, build_volume_outline_prompt
from project_manager import load_json, load_project, save_json, update_project_stats


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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
    raise ValueError("Could not parse JSON from outline response.")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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
    normalized["meta"]["updated_at"] = _utc_now()
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
        volume["volume_number"] = _safe_int(volume.get("volume_number"), index) or index
        volume["title"] = str(volume.get("title", "") or "").strip()
        volume["summary"] = str(volume.get("summary", "") or "").strip()
        volume["story_goal"] = str(volume.get("story_goal", "") or "").strip()
        volume["planned_chapter_count"] = max(1, _safe_int(volume.get("planned_chapter_count"), 1))
        chapters = []
        for chapter_index, raw_chapter in enumerate(volume.get("chapters") or [], start=1):
            chapter = deepcopy(EMPTY_CHAPTER_OUTLINE)
            if isinstance(raw_chapter, dict):
                chapter.update(raw_chapter)
            chapter["chapter_in_volume"] = _safe_int(chapter.get("chapter_in_volume"), chapter_index) or chapter_index
            chapter["chapter_number"] = _safe_int(chapter.get("chapter_number"), 0)
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


def _fallback_volume_outline(project_data: dict, user_request: str = "") -> dict:
    project = project_data.get("project", {})
    story_request = str(project.get("story_request", "") or "").strip()
    prompt_text = user_request.strip() or story_request or str(project.get("description", "") or "").strip()
    base_summary = prompt_text[:180] or "围绕主角群展开长期故事，逐步推进主线与人物关系。"
    return normalize_outlines(
        {
            "meta": deepcopy(EMPTY_OUTLINE_META),
            "volumes": [
                {
                    "volume_number": 1,
                    "title": "第一卷 困局初成",
                    "summary": base_summary or "建立核心困局、人物关系与主线目标。",
                    "story_goal": "完成开篇铺陈，确立核心矛盾与初步生存/行动逻辑。",
                    "planned_chapter_count": 6,
                },
                {
                    "volume_number": 2,
                    "title": "第二卷 局势扩展",
                    "summary": "在既有困局基础上扩大行动范围，引入更多资源、矛盾和人物关系变化。",
                    "story_goal": "推动主线升级，让人物关系和外部挑战同步加深。",
                    "planned_chapter_count": 6,
                },
                {
                    "volume_number": 3,
                    "title": "第三卷 危机升级",
                    "summary": "让前期伏笔集中兑现，使故事进入更复杂、更难回避的阶段。",
                    "story_goal": "形成阶段性高潮，并为下一阶段发展预留空间。",
                    "planned_chapter_count": 6,
                },
            ],
        }
    )


def _fallback_chapters_for_volume(volume: dict) -> list[dict]:
    planned_count = max(1, _safe_int(volume.get("planned_chapter_count"), 1))
    default_titles = [
        "局面建立",
        "第一次应对",
        "局势试探",
        "问题加深",
        "阶段突破",
        "余波扩散",
        "新线索出现",
        "冲突转向",
        "阶段收束",
        "新的悬念",
    ]
    chapters = []
    for chapter_index in range(1, planned_count + 1):
        title = default_titles[chapter_index - 1] if chapter_index <= len(default_titles) else f"推进节点{chapter_index}"
        chapters.append(
            {
                "chapter_in_volume": chapter_index,
                "title": title,
                "summary": f"围绕“{volume.get('title', '当前卷')}”的阶段目标，推进到第{chapter_index}个关键节点。",
                "goal": f"完成第{chapter_index}章应承担的推进任务，并自然承接上下章。",
                "key_events": [
                    "推动当前阶段的核心事件",
                    "深化人物互动或矛盾",
                    "为下一章留下明确延续点",
                ],
                "status": "planned",
            }
        )
    return chapters


def _generate_outline_json(prompt: str, config: dict, project_path: str) -> dict | None:
    provider = str(config.get("model_provider", "") or "").strip().lower()
    model_name = str(config.get("model") or config.get("model_name") or "").strip()
    api_key = str(config.get("api_key", "") or "").strip()
    api_base = str(config.get("api_base", "") or "").strip()
    if not model_name:
        log_warning("大纲生成: 未提供模型名称，直接使用兜底大纲。")
        return None
    if provider == "openai_compatible" and (not api_key or not api_base):
        log_warning("大纲生成: openai_compatible 缺少 api_key 或 api_base，直接使用兜底大纲。")
        return None
    if provider == "ollama" and not api_base:
        log_warning("大纲生成: ollama 缺少 api_base，直接使用兜底大纲。")
        return None
    if provider in {"gemini", "grok", "deepseek", "doubao"} and not api_key:
        log_warning(f"大纲生成: provider={provider} 缺少 api_key，直接使用兜底大纲。")
        return None

    last_error = None
    for attempt in range(2):
        try:
            log_info(f"大纲生成: 第 {attempt + 1} 次请求模型。")
            response_text, metadata = generate_text_with_metadata(prompt, config)
        except Exception as exc:  # pragma: no cover - resilience path
            update_project_stats(project_path, phase="outline", success=False, usage=None)
            last_error = exc
            log_warning(f"大纲生成: 第 {attempt + 1} 次请求失败，原因: {exc}")
            continue

        try:
            update_project_stats(
                project_path,
                phase="outline",
                success=True,
                usage=metadata.get("usage"),
            )
            log_success("大纲生成: 模型返回成功，已解析 JSON。")
            return _extract_json_object(response_text)
        except Exception as exc:  # pragma: no cover - resilience path
            last_error = exc
            log_warning(f"大纲生成: 返回内容解析失败，原因: {exc}")
    if last_error is not None:
        log_warning(f"大纲生成: 改用兜底大纲。最后错误: {last_error}")
        return None
    return None


def regenerate_volume_outline(project_path: str, config: dict, user_request: str = "") -> dict:
    log_info(f"分卷大纲: 开始生成，项目={project_path}")
    project_data = load_project(project_path)
    prompt = build_volume_outline_prompt(
        {
            **project_data,
            "completed_chapters": _build_completed_chapter_context(project_data),
        },
        user_request=user_request,
    )
    generated = _generate_outline_json(prompt, config, project_path)
    outlines = normalize_outlines(generated) if generated else _fallback_volume_outline(project_data, user_request=user_request)
    if generated:
        log_success(f"分卷大纲: 模型生成完成，共 {len(outlines.get('volumes', []))} 卷。")
    else:
        log_warning(f"分卷大纲: 使用兜底结果，共 {len(outlines.get('volumes', []))} 卷。")
    outlines["meta"]["chapter_outline_stale"] = True
    outlines["meta"]["last_volume_outline_request"] = user_request.strip()
    save_outlines(project_path, outlines)
    log_success("分卷大纲: 已写入 outlines.json，并标记分章大纲需要同步。")
    return outlines


def _find_volume(outlines: dict, volume_number: int) -> dict | None:
    for volume in outlines.get("volumes", []):
        if _safe_int(volume.get("volume_number"), 0) == volume_number:
            return deepcopy(volume)
    return None


def _all_volumes_have_chapters(outlines: dict) -> bool:
    volumes = outlines.get("volumes", [])
    if not volumes:
        return False
    for volume in volumes:
        planned_count = max(1, _safe_int(volume.get("planned_chapter_count"), 1))
        if len(volume.get("chapters") or []) < planned_count:
            return False
    return True


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
) -> dict:
    if volume_number is None:
        log_info(f"分章大纲: 开始为全部卷生成，项目={project_path}")
    else:
        log_info(f"分章大纲: 开始为第 {volume_number} 卷生成，项目={project_path}")
    existing_outlines = load_outlines(project_path)
    if not existing_outlines.get("volumes"):
        log_warning("分章大纲: 当前没有分卷大纲，先自动生成分卷大纲。")
        existing_outlines = regenerate_volume_outline(project_path, config, user_request="")

    project_data = load_project(project_path)
    written_remaining = int(project_data.get("project", {}).get("chapter_count", 0) or 0)
    updated_outlines = deepcopy(existing_outlines)
    new_volumes = []

    for volume in existing_outlines.get("volumes", []):
        planned_count = max(1, _safe_int(volume.get("planned_chapter_count"), 1))
        existing_volume = _find_volume(existing_outlines, _safe_int(volume.get("volume_number"), 0))
        existing_completed = []
        if existing_volume and existing_volume.get("chapters"):
            for chapter in existing_volume.get("chapters", []):
                if chapter.get("status") == "completed":
                    existing_completed.append(deepcopy(chapter))

        completed_count = min(written_remaining, len(existing_completed) or planned_count)
        written_remaining = max(0, written_remaining - completed_count)
        preserved_completed = existing_completed[:completed_count] or _build_placeholder_completed_chapters(completed_count)

        should_regenerate = volume_number is None or _safe_int(volume.get("volume_number"), 0) == volume_number
        if should_regenerate:
            log_info(
                f"分章大纲: 正在生成第 {volume.get('volume_number', '?')} 卷，"
                f"计划 {planned_count} 章，已完成 {completed_count} 章。"
            )
            prompt = build_chapter_outline_prompt(
                project_data,
                volume=volume,
                previous_volumes=new_volumes,
                completed_chapters=preserved_completed,
                user_request=user_request,
            )
            generated = _generate_outline_json(prompt, config, project_path)
            generated_chapters = []
            if generated and isinstance(generated.get("chapters"), list):
                generated_chapters = generated.get("chapters") or []

            normalized_generated = normalize_outlines(
                {
                    "volumes": [
                        {
                            **volume,
                            "chapters": generated_chapters or _fallback_chapters_for_volume(volume),
                        }
                    ]
                }
            )["volumes"][0]["chapters"]
            volume_chapters = preserved_completed + normalized_generated[completed_count:planned_count]
            log_success(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷生成完成。")
        else:
            volume_chapters = deepcopy(existing_volume.get("chapters") or [])
            log_info(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷沿用现有章纲。")

        if len(volume_chapters) < planned_count:
            volume_chapters.extend(_fallback_chapters_for_volume(volume)[len(volume_chapters) : planned_count])
            log_warning(f"分章大纲: 第 {volume.get('volume_number', '?')} 卷章纲不足，已用兜底章纲补齐。")
        volume_copy = deepcopy(volume)
        volume_copy["chapters"] = volume_chapters[:planned_count]
        new_volumes.append(volume_copy)

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
    next_context = find_next_chapter_context(outlines, int(load_json(str(Path(project_path) / "project.json")).get("chapter_count", 0) or 0))
    if next_context is None:
        return
    plot_state_path = Path(project_path) / "plot_state.json"
    plot_state = load_json(str(plot_state_path))
    chapter = next_context["chapter"]
    plot_state["next_chapter_goal"] = chapter.get("goal") or chapter.get("summary") or plot_state.get("next_chapter_goal", "")
    save_json(str(plot_state_path), plot_state)


def find_next_chapter_context(outlines: dict, written_chapter_count: int) -> dict | None:
    normalized = normalize_outlines(outlines)
    for volume in normalized.get("volumes", []):
        for chapter in volume.get("chapters", []):
            if _safe_int(chapter.get("chapter_number"), 0) == written_chapter_count + 1:
                return {
                    "volume": deepcopy(volume),
                    "chapter": deepcopy(chapter),
                }
    return None


def ensure_project_outlines(project_path: str, config: dict) -> dict:
    outlines = load_outlines(project_path)
    if not outlines.get("volumes"):
        log_warning("章纲检查: 未发现 outlines.json，准备自动补全分卷和分章大纲。")
        regenerate_volume_outline(project_path, config, user_request="")
        outlines = regenerate_chapter_outline(project_path, config, volume_number=None, user_request="")
    elif outlines.get("meta", {}).get("chapter_outline_stale"):
        log_error("章纲检查: 分卷大纲已更新，但分章大纲尚未同步。")
        raise ValueError("分卷大纲已经更新，但分章大纲尚未同步，请先重新生成分章大纲。")
    elif not _all_volumes_have_chapters(outlines):
        log_warning("章纲检查: 检测到章纲不完整，准备自动补齐。")
        outlines = regenerate_chapter_outline(project_path, config, volume_number=None, user_request="")
    else:
        outlines = sync_outline_progress(project_path, outlines)
        log_success("章纲检查: 分卷和分章大纲均可用。")
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
