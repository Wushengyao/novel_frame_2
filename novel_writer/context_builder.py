"""Context assembly, memory retrieval, and prompt budget helpers."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from common_utils import safe_int
from project_manager import (
    EMPTY_AUTHOR_INTENT,
    EMPTY_PLOT_STATE,
    ensure_author_intent,
    load_json,
    normalize_planning_mode,
    save_json,
)


WRITER_SECTION_LIMITS = {
    "author_intent": 600,
    "chapter_task": 500,
    "live_state": 900,
    "retrieved_memory": 700,
    "recent_scene": 2200,
    "style_contract": 400,
    "static_world": 500,
    "static_characters": 700,
}
WRITER_SOFT_TOTAL_CHARS = 7000
WRITER_HARD_TOTAL_CHARS = 8000
WRITER_TOTAL_REDUCTION_ORDER = (
    "retrieved_memory",
    "recent_scene",
    "live_state",
    "static_characters",
    "static_world",
)

RECENT_SUMMARY_COUNT = 2
RETRIEVED_MEMORY_LIMIT = 3
ARC_SUMMARY_SPAN = 5

SUMMARY_LIST_LIMITS = {
    "recent_events": 6,
    "open_threads": 8,
    "resolved_threads": 8,
    "foreshadowing": 6,
    "character_updates": 6,
    "active_characters": 6,
    "retrieval_tags": 12,
}

STOP_WORDS = {
    "的",
    "了",
    "和",
    "是",
    "在",
    "与",
    "及",
    "并",
    "把",
    "将",
    "让",
    "为",
    "对",
    "中",
    "后",
    "前",
    "以及",
    "一个",
    "他们",
    "她们",
    "我们",
    "进行",
    "需要",
    "当前",
    "本章",
    "下一章",
    "chapter",
    "story",
    "goal",
}


def _trim_text(text: str, max_chars: int) -> str:
    content = str(text or "").strip()
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    if max_chars <= 1:
        return content[:max_chars]
    return content[: max_chars - 1].rstrip() + "…"


def _json_block(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_string_list(value: object, *, max_items: int) -> list[str]:
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


def normalize_live_plot_state(plot_state: dict | None) -> dict:
    normalized = deepcopy(EMPTY_PLOT_STATE)
    if isinstance(plot_state, dict):
        for key, value in plot_state.items():
            normalized[key] = value

    for key in (
        "current_arc",
        "current_location",
        "current_time",
        "main_plot",
        "next_chapter_goal",
    ):
        normalized[key] = str(normalized.get(key, "") or "").strip()

    for key, limit in SUMMARY_LIST_LIMITS.items():
        normalized[key] = _normalize_string_list(normalized.get(key), max_items=limit)
    return normalized


def normalize_author_intent(author_intent: dict | None) -> dict:
    normalized = deepcopy(EMPTY_AUTHOR_INTENT)
    if isinstance(author_intent, dict):
        for key, value in author_intent.items():
            normalized[key] = value
    normalized["premise"] = str(normalized.get("premise", "") or "").strip()
    normalized["long_arc"] = str(normalized.get("long_arc", "") or "").strip()
    normalized["tone_contract"] = str(normalized.get("tone_contract", "") or "").strip()
    normalized["creativity_guidance"] = str(normalized.get("creativity_guidance", "") or "").strip()
    normalized["must_haves"] = _normalize_string_list(normalized.get("must_haves"), max_items=6)
    normalized["must_not_break"] = _normalize_string_list(normalized.get("must_not_break"), max_items=6)
    return normalized


def _format_bullets(title: str, items: list[str], *, max_chars: int) -> str:
    if not items or max_chars <= 0:
        return ""
    lines = [title]
    for item in items:
        candidate = "\n".join(lines + [f"- {item}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"- {item}")
    return "\n".join(lines)


def _extract_keywords(text: str) -> set[str]:
    content = str(text or "").strip().lower()
    if not content:
        return set()

    tokens = set()
    for word in re.findall(r"[a-z0-9_]{2,}", content):
        if word not in STOP_WORDS:
            tokens.add(word)

    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", content):
        if chunk not in STOP_WORDS:
            tokens.add(chunk)
        for index in range(len(chunk) - 1):
            token = chunk[index : index + 2]
            if token not in STOP_WORDS:
                tokens.add(token)
        for index in range(len(chunk) - 2):
            token = chunk[index : index + 3]
            if token not in STOP_WORDS:
                tokens.add(token)
    return tokens


def _overlap_score(query_keywords: set[str], candidate_keywords: set[str]) -> int:
    if not query_keywords or not candidate_keywords:
        return 0
    return len(query_keywords & candidate_keywords)


def select_recent_scene_window(chapter_text: str, *, min_chars: int = 1600, max_chars: int = 2200) -> str:
    text = str(chapter_text or "").strip()
    if not text:
        return ""

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    selected: list[str] = []
    current_length = 0
    for paragraph in reversed(paragraphs):
        extra = len(paragraph) + (2 if selected else 0)
        if current_length >= min_chars and current_length + extra > max_chars:
            break
        if current_length + extra > max_chars and selected:
            break
        selected.append(paragraph)
        current_length += extra
        if current_length >= min_chars:
            break

    if not selected:
        return _trim_text(paragraphs[-1], max_chars)
    return "\n\n".join(reversed(selected))


def _summary_path(project_path: str, chapter_number: int) -> Path:
    return Path(project_path) / "summaries" / f"summary_{chapter_number:04d}.json"


def _task_card_path(project_path: str, chapter_number: int) -> Path:
    return Path(project_path) / "task_cards" / f"chapter_{chapter_number:04d}.json"


def _arc_summary_path(project_path: str, arc_index: int) -> Path:
    return Path(project_path) / "arc_summaries" / f"arc_{arc_index:04d}.json"


def _normalize_summary_payload(summary: dict | None, *, chapter_number: int) -> dict:
    source = summary if isinstance(summary, dict) else {}
    recent_events = _normalize_string_list(source.get("recent_events"), max_items=SUMMARY_LIST_LIMITS["recent_events"])
    chapter_summary = str(source.get("chapter_summary", "") or "").strip()
    if not chapter_summary:
        chapter_summary = "；".join(recent_events[:2]) or str(source.get("next_chapter_goal", "") or "").strip()

    payload = {
        "chapter_number": chapter_number,
        "chapter_summary": chapter_summary[:280],
        "current_location": str(source.get("current_location", "") or "").strip(),
        "current_time": str(source.get("current_time", "") or "").strip(),
        "current_arc": str(source.get("current_arc", "") or "").strip(),
        "recent_events": recent_events,
        "open_threads": _normalize_string_list(source.get("open_threads"), max_items=SUMMARY_LIST_LIMITS["open_threads"]),
        "resolved_threads": _normalize_string_list(source.get("resolved_threads"), max_items=SUMMARY_LIST_LIMITS["resolved_threads"]),
        "foreshadowing": _normalize_string_list(source.get("foreshadowing"), max_items=SUMMARY_LIST_LIMITS["foreshadowing"]),
        "character_updates": _normalize_string_list(source.get("character_updates"), max_items=SUMMARY_LIST_LIMITS["character_updates"]),
        "active_characters": _normalize_string_list(source.get("active_characters"), max_items=SUMMARY_LIST_LIMITS["active_characters"]),
        "retrieval_tags": _normalize_string_list(source.get("retrieval_tags"), max_items=SUMMARY_LIST_LIMITS["retrieval_tags"]),
        "next_chapter_goal": str(source.get("next_chapter_goal", "") or "").strip(),
    }
    if not payload["retrieval_tags"]:
        payload["retrieval_tags"] = build_retrieval_tags(payload)
    return payload


def load_recent_summary_payloads(project_path: str, chapter_count: int, *, limit: int = RECENT_SUMMARY_COUNT) -> list[dict]:
    payloads = []
    for chapter_number in range(max(1, chapter_count - limit + 1), chapter_count + 1):
        path = _summary_path(project_path, chapter_number)
        if not path.exists():
            continue
        payloads.append(_normalize_summary_payload(load_json(str(path)), chapter_number=chapter_number))
    return payloads


def build_retrieval_tags(summary_payload: dict) -> list[str]:
    text_parts = [
        str(summary_payload.get("chapter_summary", "") or "").strip(),
        str(summary_payload.get("current_location", "") or "").strip(),
        str(summary_payload.get("current_time", "") or "").strip(),
        str(summary_payload.get("current_arc", "") or "").strip(),
        " ".join(summary_payload.get("open_threads") or []),
        " ".join(summary_payload.get("resolved_threads") or []),
        " ".join(summary_payload.get("active_characters") or []),
    ]
    keywords = sorted(_extract_keywords(" ".join(part for part in text_parts if part)))
    tags = []
    for keyword in keywords:
        if keyword in STOP_WORDS:
            continue
        tags.append(keyword)
        if len(tags) >= SUMMARY_LIST_LIMITS["retrieval_tags"]:
            break
    return tags


def build_arc_summary_payload(arc_index: int, chapter_payloads: list[dict]) -> dict:
    chapter_payloads = [payload for payload in chapter_payloads if isinstance(payload, dict)]
    if not chapter_payloads:
        return {
            "arc_index": arc_index,
            "chapter_range": [],
            "summary": "",
            "current_arc": "",
            "open_threads": [],
            "resolved_threads": [],
            "active_characters": [],
            "key_locations": [],
            "retrieval_tags": [],
        }

    start = min(safe_int(payload.get("chapter_number"), 0) for payload in chapter_payloads)
    end = max(safe_int(payload.get("chapter_number"), 0) for payload in chapter_payloads)
    summary_lines = []
    for payload in chapter_payloads[-3:]:
        text = str(payload.get("chapter_summary", "") or "").strip()
        if text:
            summary_lines.append(text)

    key_locations = []
    for payload in chapter_payloads:
        location = str(payload.get("current_location", "") or "").strip()
        if location and location not in key_locations:
            key_locations.append(location)
        if len(key_locations) >= 4:
            break

    result = {
        "arc_index": arc_index,
        "chapter_range": [start, end],
        "summary": _trim_text("；".join(summary_lines), 320),
        "current_arc": str(chapter_payloads[-1].get("current_arc", "") or "").strip(),
        "open_threads": _normalize_string_list(
            [item for payload in chapter_payloads for item in payload.get("open_threads") or []],
            max_items=SUMMARY_LIST_LIMITS["open_threads"],
        ),
        "resolved_threads": _normalize_string_list(
            [item for payload in chapter_payloads for item in payload.get("resolved_threads") or []],
            max_items=SUMMARY_LIST_LIMITS["resolved_threads"],
        ),
        "active_characters": _normalize_string_list(
            [item for payload in chapter_payloads for item in payload.get("active_characters") or []],
            max_items=SUMMARY_LIST_LIMITS["active_characters"],
        ),
        "key_locations": key_locations,
    }
    result["retrieval_tags"] = build_retrieval_tags(
        {
            "chapter_summary": result["summary"],
            "current_location": " ".join(result["key_locations"]),
            "current_time": "",
            "current_arc": result["current_arc"],
            "open_threads": result["open_threads"],
            "resolved_threads": result["resolved_threads"],
            "active_characters": result["active_characters"],
        }
    )
    return result


def write_arc_summary(project_path: str, chapter_count: int) -> dict | None:
    if chapter_count < ARC_SUMMARY_SPAN or chapter_count % ARC_SUMMARY_SPAN != 0:
        return None

    payloads = []
    start = chapter_count - ARC_SUMMARY_SPAN + 1
    for chapter_number in range(start, chapter_count + 1):
        path = _summary_path(project_path, chapter_number)
        if not path.exists():
            return None
        payloads.append(_normalize_summary_payload(load_json(str(path)), chapter_number=chapter_number))

    arc_index = chapter_count // ARC_SUMMARY_SPAN
    payload = build_arc_summary_payload(arc_index, payloads)
    path = _arc_summary_path(project_path, arc_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(path), payload)
    return payload


def _compact_world_block(world: dict, task_text: str, *, max_chars: int) -> str:
    background = _normalize_string_list(world.get("background"), max_items=4)
    rules = _normalize_string_list(world.get("rules"), max_items=4)
    query = _extract_keywords(task_text)
    if query:
        background.sort(key=lambda item: _overlap_score(query, _extract_keywords(item)), reverse=True)
        rules.sort(key=lambda item: _overlap_score(query, _extract_keywords(item)), reverse=True)

    lines = []
    title = str(world.get("title", "") or "").strip()
    setting = str(world.get("setting", "") or "").strip()
    genre = str(world.get("genre", "") or "").strip()
    header = " / ".join(part for part in (title, genre, setting) if part)
    if header:
        lines.append(header)
    for item in background[:2]:
        candidate = "\n".join(lines + [f"背景: {item}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"背景: {item}")
    for item in rules[:2]:
        candidate = "\n".join(lines + [f"规则: {item}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"规则: {item}")
    return _trim_text("\n".join(lines), max_chars)


def _compact_character_description(text: str, *, max_chars: int) -> str:
    content = str(text or "").strip()
    if len(content) <= max_chars:
        return content
    sentences = [item.strip() for item in re.split(r"[。！？!?；;]\s*", content) if item.strip()]
    lines = []
    for sentence in sentences:
        candidate = "；".join(lines + [sentence])
        if len(candidate) > max_chars:
            break
        lines.append(sentence)
    return _trim_text("；".join(lines) or content, max_chars)


def _active_character_names(plot_state: dict, characters: dict) -> list[str]:
    active = _normalize_string_list(plot_state.get("active_characters"), max_items=SUMMARY_LIST_LIMITS["active_characters"])
    if active:
        return active
    protagonists = characters.get("protagonists") or []
    return [str(item.get("name", "") or "").strip() for item in protagonists if str(item.get("name", "") or "").strip()]


def _compact_characters_block(characters: dict, plot_state: dict, *, max_chars: int) -> str:
    active_names = _active_character_names(plot_state, characters)
    all_characters = []
    for group in ("protagonists", "supporting"):
        for character in characters.get(group) or []:
            name = str(character.get("name", "") or "").strip()
            if not name:
                continue
            priority = 0 if name in active_names else 1
            all_characters.append((priority, character))
    all_characters.sort(key=lambda item: (item[0], str(item[1].get("name", "") or "")))

    lines = []
    for _, character in all_characters[:6]:
        name = str(character.get("name", "") or "").strip()
        role = str(character.get("role", "") or "").strip()
        description = _compact_character_description(character.get("description", ""), max_chars=110)
        line = f"{name}｜{role}｜{description}".strip("｜")
        candidate = "\n".join(lines + [line])
        if len(candidate) > max_chars:
            break
        lines.append(line)
    return _trim_text("\n".join(lines), max_chars)


def _build_author_intent_block(author_intent: dict, *, max_chars: int) -> str:
    intent = normalize_author_intent(author_intent)
    lines = []
    if intent["premise"]:
        lines.append(f"核心前提: {intent['premise']}")
    if intent["long_arc"]:
        lines.append(f"长期主线: {intent['long_arc']}")
    if intent["tone_contract"]:
        lines.append(f"语气约束: {intent['tone_contract']}")
    if intent["must_haves"]:
        lines.append("必须保留:")
        for item in intent["must_haves"]:
            candidate = "\n".join(lines + [f"- {item}"])
            if len(candidate) > max_chars:
                break
            lines.append(f"- {item}")
    if intent["must_not_break"]:
        header = "\n".join(lines + ["不能破坏:"]) if lines else "不能破坏:"
        if len(header) <= max_chars:
            lines.append("不能破坏:")
            for item in intent["must_not_break"]:
                candidate = "\n".join(lines + [f"- {item}"])
                if len(candidate) > max_chars:
                    break
                lines.append(f"- {item}")
    return _trim_text("\n".join(lines), max_chars)


def _build_style_contract_block(style: dict, author_intent: dict, *, max_chars: int) -> str:
    tone = str(style.get("tone", "") or "").strip()
    pov = str(style.get("pov", "") or "").strip()
    requirements = _normalize_string_list(style.get("requirements"), max_items=5)
    creativity = str(author_intent.get("creativity_guidance", "") or "").strip()
    lines = []
    if tone:
        lines.append(f"基调: {tone}")
    if pov:
        lines.append(f"视角: {pov}")
    for requirement in requirements:
        candidate = "\n".join(lines + [f"要求: {requirement}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"要求: {requirement}")
    if creativity:
        candidate = "\n".join(lines + [f"创作弹性: {creativity}"])
        if len(candidate) <= max_chars:
            lines.append(f"创作弹性: {creativity}")
    return _trim_text("\n".join(lines), max_chars)


def build_chapter_task_card(
    project_path: str,
    project_data: dict,
    next_context: dict,
    *,
    planning_mode: str,
    user_request: str = "",
    persist: bool = True,
) -> dict:
    chapter = deepcopy(next_context.get("chapter") or {})
    volume = deepcopy(next_context.get("volume") or {})
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    chapter_number = max(1, safe_int(chapter.get("chapter_number"), safe_int(project_data.get("project", {}).get("chapter_count"), 0) + 1))
    normalized_mode = normalize_planning_mode(planning_mode)

    if normalized_mode == "chapter" and chapter:
        task_card = {
            "chapter_number": chapter_number,
            "planning_mode": normalized_mode,
            "source": "chapter_outline",
            "title": str(chapter.get("title", "") or "").strip() or f"第 {chapter_number} 章任务",
            "summary": str(chapter.get("summary", "") or "").strip(),
            "goal": str(chapter.get("goal", "") or "").strip(),
            "key_events": _normalize_string_list(chapter.get("key_events"), max_items=5),
            "volume_title": str(volume.get("title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or "").strip(),
            "writer_guidance": str(user_request or "").strip(),
        }
    else:
        focus = (
            str(user_request or "").strip()
            or str(chapter.get("summary", "") or "").strip()
            or str(plot_state.get("next_chapter_goal", "") or "").strip()
            or str(volume.get("summary", "") or "").strip()
            or str(plot_state.get("main_plot", "") or "").strip()
        )
        goal = (
            str(plot_state.get("next_chapter_goal", "") or "").strip()
            or str(chapter.get("goal", "") or "").strip()
            or str(volume.get("story_goal", "") or "").strip()
            or focus
        )
        task_card = {
            "chapter_number": chapter_number,
            "planning_mode": normalized_mode,
            "source": "volume_outline" if normalized_mode == "volume" else "freeform",
            "title": str(chapter.get("title", "") or "").strip() or f"第 {chapter_number} 章任务",
            "summary": focus,
            "goal": goal,
            "key_events": [
                "承接上一章留下的局势、情绪与未解问题。",
                _trim_text(f"围绕“{focus}”形成当前章的主要推进。", 80),
                "让人物关系、资源状态或外部局势出现至少一项可见变化。",
                "结尾保留下一章可自然承接的悬念或推进点。",
            ],
            "volume_title": str(volume.get("title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or "").strip(),
            "writer_guidance": str(user_request or "").strip() or f"以“{goal}”为本章核心任务，允许自由选择更有活力的推进方式。",
        }

    task_card["summary"] = _trim_text(task_card.get("summary", ""), 220)
    task_card["goal"] = _trim_text(task_card.get("goal", ""), 180)
    task_card["writer_guidance"] = _trim_text(task_card.get("writer_guidance", ""), 220)
    task_card["key_events"] = _normalize_string_list(task_card.get("key_events"), max_items=5)

    if persist:
        path = _task_card_path(project_path, chapter_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(str(path), task_card)
    return task_card


def load_task_card(project_path: str, chapter_number: int) -> dict | None:
    path = _task_card_path(project_path, chapter_number)
    if not path.exists():
        return None
    return load_json(str(path))


def _build_chapter_task_block(task_card: dict, *, max_chars: int) -> str:
    lines = []
    if task_card.get("title"):
        lines.append(f"标题: {task_card['title']}")
    if task_card.get("summary"):
        lines.append(f"本章会发生: {task_card['summary']}")
    if task_card.get("goal"):
        lines.append(f"叙事目标: {task_card['goal']}")
    if task_card.get("volume_goal"):
        candidate = "\n".join(lines + [f"阶段目标: {task_card['volume_goal']}"])
        if len(candidate) <= max_chars:
            lines.append(f"阶段目标: {task_card['volume_goal']}")
    for item in task_card.get("key_events") or []:
        candidate = "\n".join(lines + [f"- {item}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"- {item}")
    if task_card.get("writer_guidance"):
        candidate = "\n".join(lines + [f"补充偏好: {task_card['writer_guidance']}"])
        if len(candidate) <= max_chars:
            lines.append(f"补充偏好: {task_card['writer_guidance']}")
    return _trim_text("\n".join(lines), max_chars)


def _build_live_state_block(plot_state: dict, *, max_chars: int) -> str:
    state = normalize_live_plot_state(plot_state)
    lines = []
    for label, key in (
        ("主线", "main_plot"),
        ("当前弧线", "current_arc"),
        ("当前位置", "current_location"),
        ("当前时间", "current_time"),
        ("下一目标", "next_chapter_goal"),
    ):
        value = str(state.get(key, "") or "").strip()
        if not value:
            continue
        candidate = "\n".join(lines + [f"{label}: {value}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"{label}: {value}")

    for title, key in (
        ("最近事件", "recent_events"),
        ("未解线程", "open_threads"),
        ("已解线程", "resolved_threads"),
        ("伏笔", "foreshadowing"),
    ):
        block = _format_bullets(title, state.get(key) or [], max_chars=max_chars - len("\n".join(lines)))
        if not block:
            continue
        candidate = "\n".join(lines + [block]) if lines else block
        if len(candidate) > max_chars:
            continue
        lines.append(block)

    return _trim_text("\n".join(lines), max_chars)


def _build_recent_scene_block(project_path: str, chapter_count: int, recent_text: str, *, max_chars: int) -> str:
    excerpt = select_recent_scene_window(recent_text, min_chars=1600, max_chars=max(900, max_chars - 420))
    recent_summaries = load_recent_summary_payloads(project_path, chapter_count, limit=RECENT_SUMMARY_COUNT)
    summary_lines = []
    for payload in recent_summaries:
        location = str(payload.get("current_location", "") or "").strip()
        piece = f"第{payload['chapter_number']}章: {payload.get('chapter_summary', '')}"
        if location:
            piece += f" @ {location}"
        summary_lines.append(piece)

    lines = []
    if summary_lines:
        summary_block = "最近两章摘要:\n" + "\n".join(f"- {item}" for item in summary_lines)
        if len(summary_block) < max_chars:
            lines.append(summary_block)
    if excerpt:
        lines.append("最近场景窗口:\n" + excerpt)
    return _trim_text("\n\n".join(lines), max_chars)


def _collect_summary_memory_candidates(project_path: str, chapter_count: int) -> list[dict]:
    candidates = []
    for path in sorted((Path(project_path) / "summaries").glob("summary_*.json")):
        chapter_number = safe_int(path.stem.split("_")[-1], 0)
        if chapter_number <= 0 or chapter_number > chapter_count - RECENT_SUMMARY_COUNT:
            continue
        payload = _normalize_summary_payload(load_json(str(path)), chapter_number=chapter_number)
        candidates.append(
            {
                "kind": "chapter",
                "chapter_number": chapter_number,
                "text": "；".join(
                    part
                    for part in (
                        payload.get("chapter_summary", ""),
                        " ".join(payload.get("open_threads") or []),
                        " ".join(payload.get("resolved_threads") or []),
                        " ".join(payload.get("character_updates") or []),
                    )
                    if part
                ),
                "tags": set(payload.get("retrieval_tags") or []),
            }
        )

    for path in sorted((Path(project_path) / "arc_summaries").glob("arc_*.json")):
        data = load_json(str(path))
        candidates.append(
            {
                "kind": "arc",
                "arc_index": safe_int(data.get("arc_index"), 0),
                "text": "；".join(
                    part
                    for part in (
                        str(data.get("summary", "") or "").strip(),
                        " ".join(_normalize_string_list(data.get("open_threads"), max_items=SUMMARY_LIST_LIMITS["open_threads"])),
                        " ".join(_normalize_string_list(data.get("resolved_threads"), max_items=SUMMARY_LIST_LIMITS["resolved_threads"])),
                    )
                    if part
                ),
                "tags": set(_normalize_string_list(data.get("retrieval_tags"), max_items=SUMMARY_LIST_LIMITS["retrieval_tags"])),
            }
        )
    return candidates


def _build_retrieved_memory_block(
    project_path: str,
    chapter_count: int,
    task_card: dict,
    plot_state: dict,
    recent_scene: str,
    *,
    max_chars: int,
) -> str:
    query_text = " ".join(
        part
        for part in (
            str(task_card.get("summary", "") or "").strip(),
            str(task_card.get("goal", "") or "").strip(),
            str(task_card.get("writer_guidance", "") or "").strip(),
            str(plot_state.get("current_arc", "") or "").strip(),
            " ".join(plot_state.get("open_threads") or []),
            recent_scene,
        )
        if part
    )
    query_keywords = _extract_keywords(query_text)
    ranked = []
    for candidate in _collect_summary_memory_candidates(project_path, chapter_count):
        score = _overlap_score(query_keywords, candidate.get("tags") or _extract_keywords(candidate.get("text", "")))
        if score <= 0:
            continue
        ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)

    lines = []
    for _, candidate in ranked[:RETRIEVED_MEMORY_LIMIT]:
        if candidate["kind"] == "chapter":
            label = f"第{candidate['chapter_number']}章记忆"
        else:
            label = f"阶段记忆 #{candidate['arc_index']}"
        text = _trim_text(candidate.get("text", ""), 220)
        candidate_line = f"- {label}: {text}"
        next_block = "\n".join(lines + [candidate_line])
        if len(next_block) > max_chars:
            break
        lines.append(candidate_line)
    return _trim_text("\n".join(lines), max_chars)


def _reduce_sections_to_target(sections: dict[str, str], target_chars: int) -> dict[str, str]:
    trimmed = dict(sections)
    total = sum(len(value) for value in trimmed.values() if value)
    if total <= target_chars:
        return trimmed

    overflow = total - target_chars
    for section_name in WRITER_TOTAL_REDUCTION_ORDER:
        content = trimmed.get(section_name, "")
        if not content or overflow <= 0:
            continue
        target = max(160, len(content) - overflow)
        updated = _trim_text(content, target)
        overflow -= max(0, len(content) - len(updated))
        trimmed[section_name] = updated

    return trimmed


def _apply_writer_total_budget(sections: dict[str, str]) -> dict[str, str]:
    trimmed = _reduce_sections_to_target(sections, WRITER_SOFT_TOTAL_CHARS)
    return _reduce_sections_to_target(trimmed, WRITER_HARD_TOTAL_CHARS)


def build_writer_context(
    project_path: str,
    project_data: dict,
    next_context: dict,
    recent_text: str,
    *,
    user_request: str = "",
    planning_mode: str,
) -> dict:
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    task_card = build_chapter_task_card(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        user_request=user_request,
    )
    task_text = " ".join(
        part
        for part in (
            task_card.get("summary", ""),
            task_card.get("goal", ""),
            task_card.get("writer_guidance", ""),
        )
        if part
    )

    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=WRITER_SECTION_LIMITS["author_intent"]),
        "chapter_task": _build_chapter_task_block(task_card, max_chars=WRITER_SECTION_LIMITS["chapter_task"]),
        "live_state": _build_live_state_block(plot_state, max_chars=WRITER_SECTION_LIMITS["live_state"]),
        "style_contract": _build_style_contract_block(
            project_data.get("style") or {},
            author_intent,
            max_chars=WRITER_SECTION_LIMITS["style_contract"],
        ),
        "static_world": _compact_world_block(
            project_data.get("world") or {},
            task_text,
            max_chars=WRITER_SECTION_LIMITS["static_world"],
        ),
        "static_characters": _compact_characters_block(
            project_data.get("characters") or {},
            plot_state,
            max_chars=WRITER_SECTION_LIMITS["static_characters"],
        ),
    }
    sections["recent_scene"] = _build_recent_scene_block(
        project_path,
        chapter_count,
        recent_text,
        max_chars=WRITER_SECTION_LIMITS["recent_scene"],
    )
    sections["retrieved_memory"] = _build_retrieved_memory_block(
        project_path,
        chapter_count,
        task_card,
        plot_state,
        sections["recent_scene"],
        max_chars=WRITER_SECTION_LIMITS["retrieved_memory"],
    )
    sections = _apply_writer_total_budget(sections)

    return {
        "planning_mode": normalize_planning_mode(planning_mode),
        "chapter_count": chapter_count,
        "task_card": task_card,
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if value},
    }


def build_summary_context(project_path: str, project_data: dict, new_text: str) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=520),
        "live_state": _build_live_state_block(plot_state, max_chars=900),
        "static_characters": _compact_characters_block(
            project_data.get("characters") or {},
            plot_state,
            max_chars=700,
        ),
        "chapter_text": str(new_text or "").strip(),
    }
    return {
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if value},
    }


def build_batch_plan_context(project_path: str, project_data: dict, upcoming_chapters: list[dict], user_request: str) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    last_chapter_path = Path(project_path) / "chapters" / f"chapter_{chapter_count:04d}.md"
    recent_text = last_chapter_path.read_text(encoding="utf-8") if last_chapter_path.exists() else ""
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=700),
        "live_state": _build_live_state_block(plot_state, max_chars=1200),
        "static_world": _compact_world_block(project_data.get("world") or {}, user_request, max_chars=500),
        "static_characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=700),
        "recent_scene": _build_recent_scene_block(project_path, chapter_count, recent_text, max_chars=1800),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=400),
        "upcoming_chapters": _json_block(upcoming_chapters),
        "user_request": str(user_request or "").strip(),
    }
    return {
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if value},
    }


def build_progression_context(
    project_path: str,
    project_data: dict,
    next_context: dict,
    recent_text: str,
    *,
    user_request: str,
    option_count: int,
    planning_mode: str,
) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    task_card = build_chapter_task_card(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        user_request=user_request,
        persist=False,
    )
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=700),
        "chapter_task": _build_chapter_task_block(task_card, max_chars=560),
        "live_state": _build_live_state_block(plot_state, max_chars=1000),
        "static_world": _compact_world_block(project_data.get("world") or {}, user_request, max_chars=500),
        "static_characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=700),
        "recent_scene": _build_recent_scene_block(project_path, chapter_count, recent_text, max_chars=2200),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=420),
        "planning_mode": normalize_planning_mode(planning_mode),
        "user_request": str(user_request or "").strip() or "无额外要求。请仅基于当前状态给出下一章推进选项。",
        "option_count": max(1, int(option_count or 4)),
    }
    return {
        "task_card": task_card,
        "sections": sections,
        "section_chars": {key: len(str(value)) for key, value in sections.items() if value not in (None, "") and not isinstance(value, int)},
    }


def _compact_completed_chapter_list(completed_chapters: list[dict], *, max_items: int = 8) -> list[dict]:
    compact = []
    for chapter in completed_chapters[:max_items]:
        compact.append(
            {
                "chapter_number": safe_int(chapter.get("chapter_number"), 0),
                "title": str(chapter.get("title", "") or "").strip(),
                "summary": _trim_text(str(chapter.get("summary", "") or "").strip(), 160),
                "goal": _trim_text(str(chapter.get("goal", "") or "").strip(), 120),
            }
        )
    return compact


def _compact_previous_volumes(previous_volumes: list[dict], *, max_items: int = 6) -> list[dict]:
    compact = []
    for volume in previous_volumes[:max_items]:
        compact.append(
            {
                "volume_number": safe_int(volume.get("volume_number"), 0),
                "title": str(volume.get("title", "") or "").strip(),
                "summary": _trim_text(str(volume.get("summary", "") or "").strip(), 180),
                "story_goal": _trim_text(str(volume.get("story_goal", "") or "").strip(), 140),
                "planned_chapter_count": safe_int(volume.get("planned_chapter_count"), 0),
            }
        )
    return compact


def build_volume_outline_context(project_path: str, project_data: dict, completed_chapters: list[dict], user_request: str) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=900),
        "world": _compact_world_block(project_data.get("world") or {}, user_request, max_chars=700),
        "characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=900),
        "live_state": _build_live_state_block(plot_state, max_chars=1100),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=420),
        "completed_chapters": _json_block(_compact_completed_chapter_list(completed_chapters, max_items=8)),
        "user_request": str(user_request or "").strip() or "无额外要求。请基于现有设定给出合理的长篇分卷规划。",
    }
    return {
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if isinstance(value, str) and value},
    }


def build_chapter_outline_context(
    project_path: str,
    project_data: dict,
    volume: dict,
    previous_volumes: list[dict] | None,
    completed_chapters: list[dict] | None,
    user_request: str,
) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    compact_volume = {
        "volume_number": safe_int(volume.get("volume_number"), 0),
        "title": str(volume.get("title", "") or "").strip(),
        "summary": _trim_text(str(volume.get("summary", "") or "").strip(), 220),
        "story_goal": _trim_text(str(volume.get("story_goal", "") or "").strip(), 180),
        "planned_chapter_count": safe_int(volume.get("planned_chapter_count"), 0),
    }
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=900),
        "world": _compact_world_block(project_data.get("world") or {}, compact_volume["summary"], max_chars=600),
        "characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=900),
        "live_state": _build_live_state_block(plot_state, max_chars=1100),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=420),
        "previous_volumes": _json_block(_compact_previous_volumes(previous_volumes or [], max_items=6)),
        "current_volume": _json_block(compact_volume),
        "completed_chapters": _json_block(_compact_completed_chapter_list(completed_chapters or [], max_items=8)),
        "user_request": str(user_request or "").strip() or "无额外要求。请基于当前卷纲细化出稳定的分章推进。",
    }
    return {
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if isinstance(value, str) and value},
    }
