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
    ensure_reader_setup,
    format_chapter_heading,
    is_generic_chapter_title,
    load_json,
    normalize_planning_mode,
    save_json,
    sanitize_chapter_title,
)


WRITER_SECTION_LIMITS = {
    "author_intent": 1400,
    "creative_contract": 1400,
    "opening_contract": 1200,
    "continuation_contract": 1000,
    "reader_setup": 1200,
    "chapter_task": 900,
    "live_state": 1800,
    "retrieved_memory": 1100,
    "recent_craft_memory": 1200,
    "recent_scene": 3600,
    "craft_brief": 1400,
    "style_contract": 700,
    "static_world": 900,
    "static_characters": 1600,
}
WRITER_SOFT_TOTAL_CHARS = 15000
WRITER_HARD_TOTAL_CHARS = 18000
WRITER_TOTAL_REDUCTION_ORDER = (
    "retrieved_memory",
    "recent_craft_memory",
    "recent_scene",
    "craft_brief",
    "live_state",
    "static_characters",
    "static_world",
)

RECENT_SUMMARY_COUNT = 2
RECENT_CRAFT_MEMORY_COUNT = 3
RETRIEVED_MEMORY_LIMIT = 3
ARC_SUMMARY_SPAN = 5

SUMMARY_LIST_LIMITS = {
    "recent_events": 6,
    "open_threads": 8,
    "resolved_threads": 8,
    "foreshadowing": 6,
    "continuity_anchors": 8,
    "causal_links": 8,
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

TASK_CARD_PRIORITY = {
    "progression_selected": 4,
    "high_auto_plan": 3,
    "chapter_outline": 3,
    "volume_outline": 2,
    "plot_state": 1,
    "freeform": 0,
}

FIRST_CHAPTER_READER_ENTRY_STEP = (
    "先建立读者入口：用具体场景交代地点/时间/危机、核心人物身份与同场原因、"
    "当前行动理由，再推进本章任务事件。"
)


def _trim_text(text: str, max_chars: int) -> str:
    content = str(text or "").strip()
    if max_chars <= 0:
        return ""
    if len(content) <= max_chars:
        return content
    if max_chars <= 1:
        return content[:max_chars]
    return content[: max_chars - 1].rstrip() + "…"


def _normalized_compare_text(text: object) -> str:
    return re.sub(r"[\s\u3000,，.。:：;；!！?？\"“”'‘’()（）\-_/]+", "", str(text or "").strip())


def _is_duplicateish(left: object, right: object) -> bool:
    left_text = _normalized_compare_text(left)
    right_text = _normalized_compare_text(right)
    if not left_text or not right_text:
        return False
    return left_text in right_text or right_text in left_text


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


def _title_from_candidate(value: object, *, chapter_number: int) -> str:
    text = sanitize_chapter_title(value, chapter_number=chapter_number)
    if not text:
        return ""
    quoted = re.search(r"[“「『](.{2,32}?)[”」』]", text)
    if quoted:
        text = quoted.group(1).strip()
    pieces = [part.strip() for part in re.split(r"[\n，,。；;！!？?]", text) if part.strip()]
    if pieces:
        text = pieces[0]
    text = re.sub(r"^(?:本章(?:会|将|要|需要)?|围绕|沿着|落实|执行)\s*", "", text).strip(" “”。；;，,")
    text = sanitize_chapter_title(text, chapter_number=chapter_number)
    if not text or is_generic_chapter_title(text, chapter_number=chapter_number):
        return ""
    return _trim_text(text, 24)


def _derive_chapter_title(task_card: dict, *, chapter_number: int) -> str:
    for candidate in (
        task_card.get("chapter_title"),
        task_card.get("title"),
        task_card.get("objective"),
        task_card.get("goal"),
        task_card.get("plan_summary"),
        task_card.get("summary"),
    ):
        title = _title_from_candidate(candidate, chapter_number=chapter_number)
        if title:
            return title

    for item in _normalize_string_list(task_card.get("plan_steps") or task_card.get("key_events"), max_items=5):
        if "读者入口" in item or item.startswith("承接上一章"):
            continue
        title = _title_from_candidate(item, chapter_number=chapter_number)
        if title:
            return title
    return ""


def _is_first_chapter_target(project_data: dict, chapter_number: int) -> bool:
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    return chapter_count == 0 and max(1, safe_int(chapter_number, 1)) == 1


def _with_first_chapter_reader_entry(
    task_card: dict,
    project_data: dict,
    *,
    chapter_number: int,
) -> dict:
    if not _is_first_chapter_target(project_data, chapter_number):
        return task_card

    updated = deepcopy(task_card)
    steps = _normalize_string_list(updated.get("plan_steps") or updated.get("key_events"), max_items=5)
    has_reader_entry = any(
        "读者入口" in item
        or ("读者" in item and ("设定" in item or "开篇" in item or "交代" in item))
        for item in steps
    )
    if not has_reader_entry:
        steps = [FIRST_CHAPTER_READER_ENTRY_STEP] + steps
    updated["plan_steps"] = steps[:5]
    updated["key_events"] = updated["plan_steps"]
    return updated


def _synthesize_task_summary(
    *,
    title: str,
    goal: str,
    key_events: list[str],
    source: str,
) -> str:
    events = [
        event
        for event in _normalize_string_list(key_events, max_items=5)
        if "读者入口" not in event and not event.startswith("承接上一章")
    ][:3]
    if events:
        candidate = "；".join(events[:2]).strip()
        if candidate and candidate != goal:
            return _trim_text(candidate, 220)

    focus = str(goal or title or "").strip()
    if not focus:
        return ""

    prefix_map = {
        "chapter_outline": "本章会具体展开",
        "progression_selected": "本章会沿着已选推进方案落实",
        "high_auto_plan": "本章会沿着高质量细化方案落实",
        "volume_outline": "本章会围绕阶段任务推进",
        "plot_state": "本章会围绕当前待办推进",
        "freeform": "本章会围绕当前局势推进",
    }
    prefix = prefix_map.get(source, "本章会围绕以下重点推进")
    return _trim_text(f"{prefix}“{focus}”，并让局势出现新的变化。", 220)


def _join_semicolon(items: list[str], *, max_items: int = 2) -> str:
    return "；".join(_normalize_string_list(items, max_items=max_items))


def _build_live_state_plan_summary(plot_state: dict, *, goal: str, focus: str) -> str:
    state = normalize_live_plot_state(plot_state)
    target = str(goal or focus or "").strip()
    lines = []
    location_bits = [
        str(state.get("current_time", "") or "").strip(),
        str(state.get("current_location", "") or "").strip(),
    ]
    location = "，".join(part for part in location_bits if part)
    if location:
        lines.append(f"从{location}开场")

    recent = _normalize_string_list(state.get("recent_events"), max_items=5)[-2:]
    if recent:
        lines.append(f"承接上一章结果：{_join_semicolon(recent)}")

    if target:
        lines.append(f"本章要落实：{target.rstrip('。！？!?；;，, ')}")

    open_threads = _normalize_string_list(state.get("open_threads"), max_items=3)
    if open_threads:
        lines.append(f"优先处理未解压力：{_join_semicolon(open_threads)}")

    if not lines:
        return _synthesize_task_summary(title="", goal=target, key_events=[], source="plot_state")
    text = "。".join(part.rstrip("。！？!?；;，, ") for part in lines if part).strip()
    return _trim_text(f"{text}。", 360)


def _build_live_state_plan_steps(plot_state: dict, *, focus: str) -> list[str]:
    state = normalize_live_plot_state(plot_state)
    steps = []

    recent = _normalize_string_list(state.get("recent_events"), max_items=5)[-2:]
    location_bits = [
        str(state.get("current_time", "") or "").strip(),
        str(state.get("current_location", "") or "").strip(),
    ]
    location = "，".join(part for part in location_bits if part)
    if location or recent:
        pieces = []
        if location:
            pieces.append(f"从{location}的具体场面开章")
        if recent:
            pieces.append(f"接住上一章结果：{_join_semicolon(recent)}")
        steps.append(_trim_text("；".join(pieces), 120))

    open_threads = _normalize_string_list(state.get("open_threads"), max_items=3)
    if open_threads:
        steps.append(_trim_text(f"优先把未解压力推到台前：{_join_semicolon(open_threads, max_items=3)}", 120))

    target = str(focus or state.get("next_chapter_goal", "") or "").strip()
    if target:
        steps.append(_trim_text(f"围绕“{target}”安排可见行动，让人物做出选择并承担结果。", 120))

    anchors = _normalize_string_list(state.get("continuity_anchors"), max_items=3)
    if anchors:
        steps.append(_trim_text(f"写作时保留连续性锚点：{_join_semicolon(anchors, max_items=2)}", 120))

    foreshadowing = _normalize_string_list(state.get("foreshadowing"), max_items=3)
    if foreshadowing:
        steps.append(_trim_text(f"利用既有伏笔或隐患制造新变化：{_join_semicolon(foreshadowing, max_items=2)}", 120))

    if len(steps) < 4:
        steps.append("让人物关系、资源状态、外部局势或环境认知至少出现一项可验证变化。")
    if len(steps) < 5:
        steps.append("结尾保留下一章可自然承接的悬念、决定或行动入口。")
    return _normalize_string_list(steps, max_items=5)


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
    normalized["narrative_engine"] = str(normalized.get("narrative_engine", "") or "").strip()
    normalized["relationship_engine"] = str(normalized.get("relationship_engine", "") or "").strip()
    normalized["creativity_guidance"] = str(normalized.get("creativity_guidance", "") or "").strip()
    normalized["voice_rules"] = _normalize_string_list(normalized.get("voice_rules"), max_items=6)
    normalized["scene_promises"] = _normalize_string_list(normalized.get("scene_promises"), max_items=8)
    normalized["anti_flat_rules"] = _normalize_string_list(normalized.get("anti_flat_rules"), max_items=8)
    normalized["must_haves"] = _normalize_string_list(normalized.get("must_haves"), max_items=8)
    normalized["must_not_break"] = _normalize_string_list(normalized.get("must_not_break"), max_items=6)
    return normalized


def normalize_craft_notes(craft_notes: dict | None) -> dict:
    source = craft_notes if isinstance(craft_notes, dict) else {}
    return {
        "repeated_actions": _normalize_string_list(source.get("repeated_actions"), max_items=6),
        "recurring_gestures": _normalize_string_list(source.get("recurring_gestures"), max_items=6),
        "scene_type": str(source.get("scene_type", "") or "").strip(),
        "emotional_beat": str(source.get("emotional_beat", "") or "").strip(),
        "ending_pattern": str(source.get("ending_pattern", "") or "").strip(),
        "notable_phrasing": _normalize_string_list(source.get("notable_phrasing"), max_items=6),
    }


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
        "continuity_anchors": _normalize_string_list(source.get("continuity_anchors"), max_items=SUMMARY_LIST_LIMITS["continuity_anchors"]),
        "causal_links": _normalize_string_list(source.get("causal_links"), max_items=SUMMARY_LIST_LIMITS["causal_links"]),
        "character_updates": _normalize_string_list(source.get("character_updates"), max_items=SUMMARY_LIST_LIMITS["character_updates"]),
        "active_characters": _normalize_string_list(source.get("active_characters"), max_items=SUMMARY_LIST_LIMITS["active_characters"]),
        "retrieval_tags": _normalize_string_list(source.get("retrieval_tags"), max_items=SUMMARY_LIST_LIMITS["retrieval_tags"]),
        "next_chapter_goal": str(source.get("next_chapter_goal", "") or "").strip(),
        "craft_notes": normalize_craft_notes(source.get("craft_notes")),
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
        " ".join(summary_payload.get("continuity_anchors") or []),
        " ".join(summary_payload.get("causal_links") or []),
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
            "continuity_anchors": [],
            "causal_links": [],
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
        "continuity_anchors": _normalize_string_list(
            [item for payload in chapter_payloads for item in payload.get("continuity_anchors") or []],
            max_items=SUMMARY_LIST_LIMITS["continuity_anchors"],
        ),
        "causal_links": _normalize_string_list(
            [item for payload in chapter_payloads for item in payload.get("causal_links") or []],
            max_items=SUMMARY_LIST_LIMITS["causal_links"],
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
            "continuity_anchors": result["continuity_anchors"],
            "causal_links": result["causal_links"],
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


def _split_compact_fragments(text: str) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []
    fragments = []
    for part in re.split(r"[\n\r]+|[。！？!?；;]+", source):
        cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", part).strip("，,、；;：: ")
        if cleaned:
            fragments.append(cleaned)
    return fragments


def _is_low_signal_author_premise(text: str) -> bool:
    normalized = _normalized_compare_text(text)
    if not normalized:
        return True
    generic_markers = (
        "由模型根据需求自动生成设定",
        "长篇小说项目",
        "structuredmemorynovelwritingproject",
        "testproject",
        "测试项目",
        "用于分析链路",
        "用于分析",
        "提示词链路",
        "workflowprobe",
    )
    return any(marker in normalized for marker in generic_markers)


def _summarize_author_premise(author_intent: dict, *, max_chars: int) -> str:
    intent = normalize_author_intent(author_intent)
    premise_source = str(intent.get("premise", "") or "").strip()
    if _is_low_signal_author_premise(premise_source):
        fallback = str(intent.get("long_arc", "") or "").strip()
        if fallback:
            premise_source = fallback

    fragments = _split_compact_fragments(premise_source)
    if not fragments:
        return ""

    selected: list[str] = []
    for fragment in fragments:
        compact = _trim_text(fragment, 80)
        if any(_is_duplicateish(compact, existing) for existing in selected):
            continue
        candidate = "；".join(selected + [compact])
        if len(candidate) > max_chars and selected:
            break
        selected.append(compact)
        if len(selected) >= 2:
            break

    summary = "；".join(selected) if selected else _trim_text(fragments[0], max_chars)
    if not summary:
        return ""

    normalized = _normalized_compare_text(summary)
    if normalized.startswith("故事发生在") or normalized.startswith("小说故事聚焦于"):
        prefix = "围绕"
    else:
        prefix = "本书围绕"
    return _trim_text(f"{prefix}{summary}展开。", max_chars)


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
    premise_summary = _summarize_author_premise(intent, max_chars=min(240, max_chars))
    if premise_summary:
        lines.append(f"写作核心: {premise_summary}")
    if intent.get("narrative_engine") and not _is_duplicateish(intent["narrative_engine"], premise_summary):
        lines.append(f"叙事引擎: {_trim_text(intent['narrative_engine'], 150)}")

    emphasis_items = []
    for item in intent["must_haves"]:
        compact = _trim_text(item, 72)
        if any(_is_duplicateish(compact, existing) for existing in emphasis_items):
            continue
        if premise_summary and _is_duplicateish(compact, premise_summary):
            continue
        emphasis_items.append(compact)
        if len(emphasis_items) >= 4:
            break
    if emphasis_items:
        lines.append(f"优先强调: {'；'.join(emphasis_items)}")
    return _trim_text("\n".join(lines), max_chars)


def _build_creative_contract_block(author_intent: dict, *, max_chars: int) -> str:
    intent = normalize_author_intent(author_intent)
    lines: list[str] = []
    if intent.get("relationship_engine"):
        lines.append(f"关系引擎: {_trim_text(intent['relationship_engine'], 180)}")
    if intent["voice_rules"]:
        lines.append(f"叙述声音: {'；'.join(_trim_text(item, 64) for item in intent['voice_rules'][:4])}")
    if intent["scene_promises"]:
        lines.append(f"场景承诺: {'；'.join(_trim_text(item, 64) for item in intent['scene_promises'][:5])}")
    if intent["anti_flat_rules"]:
        lines.append(f"平淡规避: {'；'.join(_trim_text(item, 64) for item in intent['anti_flat_rules'][:4])}")
    adult_boundary_source = " ".join(
        [intent.get("tone_contract", ""), *intent["voice_rules"], *intent["must_not_break"]]
    )
    if any(marker in adult_boundary_source for marker in ("成人", "暧昧", "黄段子", "露骨")):
        boundary = "成人暧昧只写成年人之间的张力"
        if any(marker in adult_boundary_source for marker in ("露骨", "性行为")):
            boundary = f"{boundary}，不写露骨性行为"
        if not any(_is_duplicateish(boundary, line) for line in lines):
            lines.append(f"边界: {boundary}")
    if intent.get("creativity_guidance"):
        lines.append(f"创作弹性: {_trim_text(intent['creativity_guidance'], 100)}")
    return _trim_text("\n".join(lines), max_chars)


def _build_style_contract_block(style: dict, author_intent: dict, *, max_chars: int) -> str:
    intent = normalize_author_intent(author_intent)
    tone = str(style.get("tone", "") or "").strip()
    pov = str(style.get("pov", "") or "").strip()
    requirements = _normalize_string_list(style.get("requirements"), max_items=5)
    creativity = str(intent.get("creativity_guidance", "") or "").strip()
    lines = []
    if tone and not _is_duplicateish(tone, intent.get("tone_contract", "")):
        lines.append(f"基调: {tone}")
    if pov:
        lines.append(f"视角: {pov}")
    added_requirements = 0
    for requirement in requirements:
        if _is_duplicateish(requirement, intent.get("tone_contract", "")):
            continue
        if any(_is_duplicateish(requirement, item) for item in intent.get("must_haves", [])):
            continue
        candidate = "\n".join(lines + [f"要求: {requirement}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"要求: {requirement}")
        added_requirements += 1
        if added_requirements >= 2:
            break
    if creativity and not any(_is_duplicateish(creativity, line) for line in lines):
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
    task_card = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=persist,
    )
    if user_request:
        return merge_task_card_guidance(task_card, user_request)
    return task_card


def _normalize_task_card_payload(
    task_card: dict,
    *,
    chapter_number: int,
    planning_mode: str,
    volume: dict | None = None,
) -> dict:
    volume = volume if isinstance(volume, dict) else {}
    chapter_number = max(1, chapter_number)
    normalized_mode = normalize_planning_mode(task_card.get("planning_mode") or planning_mode)
    source = str(task_card.get("source", "") or "").strip() or "freeform"
    if source not in TASK_CARD_PRIORITY:
        source = "freeform"
    chapter_title = _derive_chapter_title(task_card, chapter_number=chapter_number)

    normalized = {
        "chapter_number": chapter_number,
        "planning_mode": normalized_mode,
        "source": source,
        "title": chapter_title or f"第 {chapter_number} 章",
        "chapter_heading": format_chapter_heading(chapter_number, chapter_title),
        "objective": str(task_card.get("objective", "") or task_card.get("goal", "") or "").strip(),
        "plan_summary": str(task_card.get("plan_summary", "") or task_card.get("summary", "") or "").strip(),
        "plan_steps": _normalize_string_list(task_card.get("plan_steps") or task_card.get("key_events"), max_items=5),
        "plan_guidance": str(task_card.get("plan_guidance", "") or task_card.get("writer_guidance", "") or "").strip(),
        "volume_title": str(task_card.get("volume_title", "") or volume.get("title", "") or "").strip(),
        "volume_goal": str(task_card.get("volume_goal", "") or volume.get("story_goal", "") or "").strip(),
    }
    if not normalized["plan_summary"]:
        normalized["plan_summary"] = normalized["objective"]
    if not normalized["objective"]:
        normalized["objective"] = normalized["plan_summary"]
    if normalized["plan_summary"] and normalized["plan_summary"] == normalized["objective"]:
        normalized["plan_summary"] = _synthesize_task_summary(
            title=normalized.get("title", ""),
            goal=normalized.get("objective", ""),
            key_events=normalized.get("plan_steps") or [],
            source=normalized.get("source", ""),
        )
    normalized["objective"] = _trim_text(normalized.get("objective", ""), 180)
    normalized["plan_summary"] = _trim_text(normalized.get("plan_summary", ""), 360)
    normalized["plan_guidance"] = _trim_text(normalized.get("plan_guidance", ""), 420)
    normalized["plan_steps"] = _normalize_string_list(normalized.get("plan_steps"), max_items=5)
    normalized["goal"] = normalized["objective"]
    normalized["summary"] = normalized["plan_summary"]
    normalized["writer_guidance"] = normalized["plan_guidance"]
    normalized["key_events"] = normalized["plan_steps"]
    normalized["chapter_title"] = normalized["title"]

    if normalized["source"] in {"progression_selected", "high_auto_plan"}:
        derived = task_card.get("derived_from") or {}
        normalized["derived_from"] = {
            "session_id": str(derived.get("session_id", "") or "").strip(),
            "option_id": str(derived.get("option_id", "") or "").strip(),
            "base_planning_mode": normalize_planning_mode(derived.get("base_planning_mode") or normalized_mode),
            "baseline_source": str(derived.get("baseline_source", "") or "").strip(),
            "workflow_id": str(derived.get("workflow_id", "") or "").strip(),
        }
    return normalized


def override_task_card_objective(task_card: dict, objective: str) -> dict:
    text = str(objective or "").strip()
    if not text:
        return deepcopy(task_card)
    updated = deepcopy(task_card)
    updated["objective"] = text
    updated["goal"] = text
    return _normalize_task_card_payload(
        updated,
        chapter_number=max(1, safe_int(updated.get("chapter_number"), 1)),
        planning_mode=str(updated.get("planning_mode", "") or ""),
    )


def _build_baseline_task_card(
    project_data: dict,
    next_context: dict,
    *,
    planning_mode: str,
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
            "objective": str(chapter.get("goal", "") or "").strip(),
            "plan_summary": str(chapter.get("summary", "") or "").strip(),
            "plan_steps": _normalize_string_list(chapter.get("key_events"), max_items=5),
            "volume_title": str(volume.get("title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or "").strip(),
            "plan_guidance": "",
        }
    else:
        chapter_summary = str(chapter.get("summary", "") or "").strip()
        chapter_goal = str(chapter.get("goal", "") or "").strip()
        next_goal = str(plot_state.get("next_chapter_goal", "") or "").strip()
        volume_summary = str(volume.get("summary", "") or "").strip()
        volume_goal = str(volume.get("story_goal", "") or "").strip()
        focus = (
            chapter_summary
            or next_goal
            or volume_summary
            or str(plot_state.get("main_plot", "") or "").strip()
        )
        goal = (
            chapter_goal
            or next_goal
            or volume_goal
            or focus
        )
        if normalized_mode == "volume" and (volume_summary or volume_goal):
            source = "volume_outline"
        elif next_goal:
            source = "plot_state"
        else:
            source = "freeform"
        if source in {"plot_state", "freeform"}:
            plan_summary = _build_live_state_plan_summary(plot_state, goal=goal, focus=focus)
            plan_steps = _build_live_state_plan_steps(plot_state, focus=focus or goal)
        else:
            plan_summary = focus
            plan_steps = [
                "承接上一章留下的局势、情绪与未解问题。",
                _trim_text(f"围绕“{focus}”形成当前章的主要推进。", 80),
                "让人物关系、资源状态或外部局势出现至少一项可见变化。",
                "结尾保留下一章可自然承接的悬念或推进点。",
            ]
        task_card = {
            "chapter_number": chapter_number,
            "planning_mode": normalized_mode,
            "source": source,
            "title": str(chapter.get("title", "") or "").strip() or f"第 {chapter_number} 章任务",
            "objective": goal,
            "plan_summary": plan_summary,
            "plan_steps": plan_steps,
            "volume_title": str(volume.get("title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or "").strip(),
            "plan_guidance": f"以“{goal}”为本章核心任务，允许自由选择更有活力的推进方式。" if goal else "",
        }

    task_card = _with_first_chapter_reader_entry(
        task_card,
        project_data,
        chapter_number=chapter_number,
    )
    return _normalize_task_card_payload(
        task_card,
        chapter_number=chapter_number,
        planning_mode=normalized_mode,
        volume=volume,
    )


def save_task_card(project_path: str, task_card: dict) -> dict:
    chapter_number = max(1, safe_int(task_card.get("chapter_number"), 0))
    normalized = _normalize_task_card_payload(
        task_card,
        chapter_number=chapter_number,
        planning_mode=str(task_card.get("planning_mode", "") or ""),
    )
    path = _task_card_path(project_path, chapter_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(path), normalized)
    return normalized


def resolve_effective_chapter_task(
    project_path: str,
    project_data: dict,
    next_context: dict,
    *,
    planning_mode: str,
    persist: bool = True,
) -> dict:
    chapter = deepcopy(next_context.get("chapter") or {})
    volume = deepcopy(next_context.get("volume") or {})
    chapter_number = max(1, safe_int(chapter.get("chapter_number"), safe_int(project_data.get("project", {}).get("chapter_count"), 0) + 1))
    existing = load_task_card(project_path, chapter_number)
    if existing and str(existing.get("source", "") or "").strip() in {"progression_selected", "high_auto_plan"}:
        return _normalize_task_card_payload(
            existing,
            chapter_number=chapter_number,
            planning_mode=planning_mode,
            volume=volume,
        )

    task_card = _build_baseline_task_card(
        project_data,
        next_context,
        planning_mode=planning_mode,
    )
    if persist:
        return save_task_card(project_path, task_card)
    return task_card


def build_high_auto_plan_task_card(
    project_path: str,
    project_data: dict,
    next_context: dict,
    plan_payload: dict,
    baseline_task: dict,
    *,
    workflow_id: str,
    planning_mode: str,
    persist: bool = True,
) -> dict:
    chapter = deepcopy(next_context.get("chapter") or {})
    volume = deepcopy(next_context.get("volume") or {})
    baseline = deepcopy(baseline_task or {})
    payload = plan_payload if isinstance(plan_payload, dict) else {}
    chapter_number = max(1, safe_int(chapter.get("chapter_number"), safe_int(project_data.get("project", {}).get("chapter_count"), 0) + 1))
    objective = (
        str(baseline.get("objective", "") or baseline.get("goal", "") or "").strip()
        or str(payload.get("objective", "") or payload.get("goal", "") or "").strip()
    )
    plan_summary = (
        str(payload.get("plan_summary", "") or payload.get("summary", "") or "").strip()
        or str(baseline.get("plan_summary", "") or baseline.get("summary", "") or "").strip()
    )
    plan_steps = (
        payload.get("plan_steps")
        or payload.get("key_events")
        or baseline.get("plan_steps")
        or baseline.get("key_events")
        or []
    )
    plan_guidance = str(payload.get("plan_guidance", "") or payload.get("writer_guidance", "") or "").strip()
    guidance_parts = [
        "高质量模式自动细化 plan。请把它作为本章执行蓝图，不要退回只完成 objective 的清单式写法。",
        plan_guidance,
    ]
    if objective and not _is_duplicateish(plan_summary, objective):
        guidance_parts.append(f"本章仍需完成 objective：{objective}")

    raw_task_card = _with_first_chapter_reader_entry(
        {
            "chapter_number": chapter_number,
            "planning_mode": planning_mode,
            "source": "high_auto_plan",
            "title": (
                str(payload.get("title", "") or "").strip()
                or str(baseline.get("title", "") or "").strip()
                or str(chapter.get("title", "") or "").strip()
            ),
            "objective": objective or plan_summary,
            "plan_summary": plan_summary,
            "plan_steps": plan_steps,
            "volume_title": str(volume.get("title", "") or baseline.get("volume_title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or baseline.get("volume_goal", "") or "").strip(),
            "plan_guidance": "\n".join(part for part in guidance_parts if part),
            "derived_from": {
                "option_id": "high_auto_plan",
                "workflow_id": str(workflow_id or "").strip(),
                "base_planning_mode": normalize_planning_mode(planning_mode),
                "baseline_source": str(baseline.get("source", "") or "").strip(),
            },
        },
        project_data,
        chapter_number=chapter_number,
    )
    task_card = _normalize_task_card_payload(
        raw_task_card,
        chapter_number=chapter_number,
        planning_mode=planning_mode,
        volume=volume,
    )
    if persist:
        return save_task_card(project_path, task_card)
    return task_card


def merge_task_card_guidance(task_card: dict, guidance: str, *, prefix: str = "用户当前补充：") -> dict:
    extra = str(guidance or "").strip()
    if not extra:
        return deepcopy(task_card)
    merged = deepcopy(task_card)
    extra_line = f"{prefix}{extra}" if prefix else extra
    base = str(merged.get("plan_guidance", "") or merged.get("writer_guidance", "") or "").strip()
    if extra_line in base or extra in base:
        return merged
    merged["plan_guidance"] = _trim_text("\n".join(part for part in (base, extra_line) if part), 220)
    merged["writer_guidance"] = merged["plan_guidance"]
    return merged


def build_progression_selected_task_card(
    project_path: str,
    project_data: dict,
    next_context: dict,
    selected_option: dict,
    baseline_task: dict | None = None,
    *,
    session_id: str,
    option_id: str,
    planning_mode: str,
    baseline_source: str,
    selection_feedback: str = "",
    persist: bool = True,
) -> dict:
    chapter = deepcopy(next_context.get("chapter") or {})
    volume = deepcopy(next_context.get("volume") or {})
    baseline = deepcopy(baseline_task or {})
    chapter_number = max(1, safe_int(chapter.get("chapter_number"), safe_int(project_data.get("project", {}).get("chapter_count"), 0) + 1))
    chapter_outline = deepcopy(selected_option.get("chapter_outline") or {})
    feedback = str(selection_feedback or "").strip()
    objective = (
        str(baseline.get("objective", "") or baseline.get("goal", "") or "").strip()
        or str(chapter_outline.get("goal", "") or "").strip()
    )
    plan_summary = (
        str(selected_option.get("plan_summary", "") or selected_option.get("summary", "") or "").strip()
        or str(baseline.get("plan_summary", "") or baseline.get("summary", "") or "").strip()
        or str(chapter_outline.get("summary", "") or "").strip()
    )
    plan_steps = (
        selected_option.get("plan_steps")
        or selected_option.get("key_events")
        or baseline.get("plan_steps")
        or baseline.get("key_events")
        or chapter_outline.get("key_events")
        or []
    )
    plan_guidance = str(
        selected_option.get("plan_guidance", "")
        or selected_option.get("writer_guidance", "")
        or ""
    ).strip()

    guidance_parts = [plan_guidance]
    if objective and not _is_duplicateish(plan_summary, objective):
        guidance_parts.append(f"本章仍需完成 objective：{objective}")
    if feedback:
        guidance_parts.append(f"用户补充细化：{feedback}")
        guidance_parts.append("这些补充只能作为已选 plan 的细化与微调，不能推翻本章 objective。")

    raw_task_card = _with_first_chapter_reader_entry(
        {
            "chapter_number": chapter_number,
            "planning_mode": planning_mode,
            "source": "progression_selected",
            "title": (
                str(selected_option.get("title", "") or "").strip()
                or str(baseline.get("title", "") or "").strip()
                or str(chapter_outline.get("title", "") or "").strip()
            ),
            "objective": objective or plan_summary,
            "plan_summary": plan_summary,
            "plan_steps": plan_steps,
            "volume_title": str(volume.get("title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or "").strip(),
            "plan_guidance": "\n".join(part for part in guidance_parts if part),
            "derived_from": {
                "session_id": session_id,
                "option_id": option_id,
                "base_planning_mode": normalize_planning_mode(planning_mode),
                "baseline_source": str(baseline_source or "").strip(),
            },
        },
        project_data,
        chapter_number=chapter_number,
    )
    task_card = _normalize_task_card_payload(
        raw_task_card,
        chapter_number=chapter_number,
        planning_mode=planning_mode,
        volume=volume,
    )
    if persist:
        return save_task_card(project_path, task_card)
    return task_card


def _normalize_custom_progression_lines(custom_idea: str, *, max_items: int = 4) -> list[str]:
    lines = []
    for raw_line in re.split(r"[\r\n]+", str(custom_idea or "").strip()):
        cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", raw_line).strip()
        if cleaned:
            lines.append(cleaned)
    if len(lines) >= 2:
        return _normalize_string_list(lines, max_items=max_items)

    fragments = []
    for piece in re.split(r"[。！？!?；;]+", str(custom_idea or "").strip()):
        cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", piece).strip()
        if cleaned:
            fragments.append(cleaned)
    return _normalize_string_list(fragments, max_items=max_items)


def build_custom_progression_task_card(
    project_path: str,
    project_data: dict,
    next_context: dict,
    baseline_task: dict,
    custom_idea: str,
    *,
    session_id: str,
    option_id: str,
    planning_mode: str,
    baseline_source: str,
    persist: bool = True,
) -> dict:
    custom_text = str(custom_idea or "").strip()
    if not custom_text:
        raise ValueError("选择空白自定义项后，必须填写你自己的创意与想看的情节。")

    chapter = deepcopy(next_context.get("chapter") or {})
    volume = deepcopy(next_context.get("volume") or {})
    chapter_number = max(1, safe_int(chapter.get("chapter_number"), safe_int(project_data.get("project", {}).get("chapter_count"), 0) + 1))

    custom_lines = _normalize_custom_progression_lines(custom_text)
    title_seed = custom_lines[0] if custom_lines else custom_text
    title = _trim_text(title_seed, 28) or str(baseline_task.get("title", "") or "").strip() or f"第 {chapter_number} 章自定义推进"
    objective = _trim_text(
        str(baseline_task.get("objective", "") or baseline_task.get("goal", "") or "").strip(),
        180,
    ) or _trim_text(custom_lines[0] if custom_lines else custom_text, 180)
    plan_summary = _trim_text(custom_text, 220)

    if len(custom_lines) >= 2:
        plan_steps = [_trim_text(item, 70) for item in custom_lines[:4]]
    else:
        focus = _trim_text(objective or plan_summary, 48)
        plan_steps = [
            _trim_text(f"围绕“{focus}”推进本章主要情节。", 70),
            "承接当前状态、人物关系与未解问题，保持前后连贯。",
            "让人物关系、局势或线索至少出现一项清晰变化。",
            "结尾保留下一章可自然承接的变化或悬念。",
        ]

    raw_task_card = _with_first_chapter_reader_entry(
        {
            "chapter_number": chapter_number,
            "planning_mode": planning_mode,
            "source": "progression_selected",
            "title": title,
            "objective": objective,
            "plan_summary": plan_summary,
            "plan_steps": plan_steps,
            "volume_title": str(volume.get("title", "") or baseline_task.get("volume_title", "") or "").strip(),
            "volume_goal": str(volume.get("story_goal", "") or baseline_task.get("volume_goal", "") or "").strip(),
            "plan_guidance": "\n".join(
                [
                    "本章采用用户自定义 plan。",
                    f"用户自定义 plan：{plan_summary}",
                    "请优先落实这段 plan，同时保持与当前 objective、人物关系和卷目标一致。",
                ]
            ),
            "derived_from": {
                "session_id": session_id,
                "option_id": option_id,
                "base_planning_mode": normalize_planning_mode(planning_mode),
                "baseline_source": str(baseline_source or "").strip(),
            },
        },
        project_data,
        chapter_number=chapter_number,
    )
    task_card = _normalize_task_card_payload(
        raw_task_card,
        chapter_number=chapter_number,
        planning_mode=planning_mode,
        volume=volume,
    )
    if persist:
        return save_task_card(project_path, task_card)
    return task_card


def load_task_card(project_path: str, chapter_number: int) -> dict | None:
    path = _task_card_path(project_path, chapter_number)
    if not path.exists():
        return None
    payload = load_json(str(path))
    return _normalize_task_card_payload(
        payload,
        chapter_number=max(1, chapter_number),
        planning_mode=str(payload.get("planning_mode", "") or ""),
    )


def _build_chapter_task_block(task_card: dict, *, max_chars: int) -> str:
    objective = str(task_card.get("objective", "") or task_card.get("goal", "") or "").strip()
    plan_summary = str(task_card.get("plan_summary", "") or task_card.get("summary", "") or "").strip()
    volume_goal = str(task_card.get("volume_goal", "") or "").strip()
    plan_guidance = str(task_card.get("plan_guidance", "") or task_card.get("writer_guidance", "") or "").strip()
    lines = []
    chapter_number = max(1, safe_int(task_card.get("chapter_number"), 1))
    chapter_heading = (
        str(task_card.get("chapter_heading", "") or "").strip()
        or format_chapter_heading(chapter_number, task_card.get("title", ""))
    )
    if chapter_heading:
        lines.append(f"章节标题: {chapter_heading}")
    if objective:
        lines.append(f"本章 objective: {objective}")
    if plan_summary and not _is_duplicateish(plan_summary, objective):
        lines.append(f"执行计划: {plan_summary}")
    if volume_goal and not _is_duplicateish(plan_summary, volume_goal) and not _is_duplicateish(objective, volume_goal):
        candidate = "\n".join(lines + [f"阶段目标: {volume_goal}"])
        if len(candidate) <= max_chars:
            lines.append(f"阶段目标: {volume_goal}")
    for item in (task_card.get("plan_steps") or task_card.get("key_events") or [])[:5]:
        candidate = "\n".join(lines + [f"- {item}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"- {item}")
    formulaic_guidance = objective and plan_guidance.startswith(f"以“{objective}”为本章核心任务")
    if plan_guidance and not formulaic_guidance and not _is_duplicateish(plan_guidance, plan_summary) and not _is_duplicateish(plan_guidance, objective):
        candidate = "\n".join(lines + [f"计划补充: {plan_guidance}"])
        if len(candidate) <= max_chars:
            lines.append(f"计划补充: {plan_guidance}")
    return _trim_text("\n".join(lines), max_chars)


def _build_live_state_block(plot_state: dict, *, max_chars: int, include_next_goal: bool = True) -> str:
    state = normalize_live_plot_state(plot_state)
    lines = []
    entries = [
        ("主线", "main_plot"),
        ("当前弧线", "current_arc"),
        ("当前位置", "current_location"),
        ("当前时间", "current_time"),
    ]
    if include_next_goal:
        entries.append(("下一目标", "next_chapter_goal"))
    for label, key in entries:
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
        ("连续性锚点", "continuity_anchors"),
        ("因果/动机线", "causal_links"),
    ):
        block = _format_bullets(title, state.get(key) or [], max_chars=max_chars - len("\n".join(lines)))
        if not block:
            continue
        candidate = "\n".join(lines + [block]) if lines else block
        if len(candidate) > max_chars:
            continue
        lines.append(block)

    return _trim_text("\n".join(lines), max_chars)


def _build_opening_contract_block(project_data: dict, task_card: dict, *, max_chars: int) -> str:
    chapter_number = max(1, safe_int(task_card.get("chapter_number"), 1))
    if not _is_first_chapter_target(project_data, chapter_number):
        return ""

    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    anchors = []
    for label, key in (
        ("地点", "current_location"),
        ("时间", "current_time"),
        ("弧线", "current_arc"),
    ):
        value = str(plot_state.get(key, "") or "").strip()
        if value:
            anchors.append(f"{label}={_trim_text(value, 72)}")

    active_names = _normalize_string_list(plot_state.get("active_characters"), max_items=4)
    if not active_names:
        active_names = _active_character_names(plot_state, project_data.get("characters") or {})[:4]
    if active_names:
        anchors.append("出场人物=" + "、".join(active_names))

    lines = [
        "这是正文第一章，读者还不知道设定文件里的前情；任务卡里的事件不能按续写模式直接起跳。",
        "开章可以有强钩子，但必须先把设定转化为读者可感知的场景：当下处境、核心人物身份、彼此为何同场、当前压力、行动理由。",
        "不要默认读者知道人物关系、世界冲突或上一段未写出的前情；避免一开场就喊名、执行计划清单或跳到结果。",
    ]
    if anchors:
        lines.append("可用首章锚点：" + "；".join(anchors[:5]))
    return _trim_text("\n".join(lines), max_chars)


def _build_continuation_contract_block(project_data: dict, task_card: dict, *, max_chars: int) -> str:
    chapter_number = max(1, safe_int(task_card.get("chapter_number"), 1))
    if _is_first_chapter_target(project_data, chapter_number):
        return ""

    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    anchors = []
    for label, key in (
        ("地点", "current_location"),
        ("时间", "current_time"),
        ("弧线", "current_arc"),
    ):
        value = str(plot_state.get(key, "") or "").strip()
        if value:
            anchors.append(f"{label}={_trim_text(value, 72)}")
    active_names = _normalize_string_list(plot_state.get("active_characters"), max_items=4)
    if not active_names:
        active_names = _active_character_names(plot_state, project_data.get("characters") or {})[:4]
    if active_names:
        anchors.append("出场人物=" + "、".join(active_names))

    lines = [
        "这是续文章节，不要把正文写成“承接上一章状态 + 按任务清单执行 + 结尾铺垫”的模板。",
        "开章必须落在一个可感知的具体场面：地点、时间、人物身体/情绪状态、眼前压力和行动理由都要通过动作、感官、心理或对白呈现。",
        "前 800 字内要建立本章戏剧问题和人物互动火花；核心行动必须包含选择、阻力、结果，以及关系、资源、局势或认知的可验证变化。",
        "可以承接最近正文的余波，但不要机械复刻上一章的动作姿态、情绪节拍或结尾方式。",
    ]
    if anchors:
        lines.append("可用续写锚点：" + "；".join(anchors[:5]))
    return _trim_text("\n".join(lines), max_chars)


def _build_reader_setup_block(
    project_path: str,
    project_data: dict,
    task_card: dict,
    *,
    max_chars: int,
) -> str:
    chapter_number = max(1, safe_int(task_card.get("chapter_number"), 1))
    if not _is_first_chapter_target(project_data, chapter_number):
        return ""

    text = str(project_data.get("reader_setup") or "").strip()
    if not text:
        text = ensure_reader_setup(project_path, project_data).strip()
        project_data["reader_setup"] = text
    if not text:
        return ""

    return _trim_text(
        "以下是项目提供给读者的非剧透开卷导语。它可以帮助读者先看到设定入口，"
        "但第一章正文仍必须自洽，不能假设读者一定读过，也不能把它原样复述成设定说明书。\n"
        + text,
        max_chars,
    )


def _build_recent_scene_block(project_path: str, chapter_count: int, recent_text: str, *, max_chars: int) -> str:
    excerpt = select_recent_scene_window(recent_text, min_chars=1200, max_chars=max(700, max_chars))
    if excerpt:
        return _trim_text(excerpt, max_chars)

    recent_summaries = load_recent_summary_payloads(project_path, chapter_count, limit=1)
    if recent_summaries:
        payload = recent_summaries[-1]
        location = str(payload.get("current_location", "") or "").strip()
        piece = f"第{payload['chapter_number']}章: {payload.get('chapter_summary', '')}"
        if location:
            piece += f" @ {location}"
        return _trim_text("- " + piece, max_chars)
    return ""


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
                        " ".join(payload.get("continuity_anchors") or []),
                        " ".join(payload.get("causal_links") or []),
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
                        " ".join(_normalize_string_list(data.get("continuity_anchors"), max_items=SUMMARY_LIST_LIMITS["continuity_anchors"])),
                        " ".join(_normalize_string_list(data.get("causal_links"), max_items=SUMMARY_LIST_LIMITS["causal_links"])),
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
    lines = []
    query_text = " ".join(
        part
        for part in (
            str(plot_state.get("main_plot", "") or "").strip(),
            str(task_card.get("summary", "") or "").strip(),
            str(task_card.get("goal", "") or "").strip(),
            str(task_card.get("writer_guidance", "") or "").strip(),
            str(plot_state.get("current_arc", "") or "").strip(),
            " ".join(plot_state.get("open_threads") or []),
            " ".join(plot_state.get("foreshadowing") or []),
            " ".join(plot_state.get("continuity_anchors") or []),
            " ".join(plot_state.get("causal_links") or []),
            " ".join(plot_state.get("recent_events") or []),
            " ".join(plot_state.get("character_updates") or []),
            recent_scene,
        )
        if part
    )
    query_keywords = _extract_keywords(query_text)
    current_open_keywords = _extract_keywords(" ".join(plot_state.get("open_threads") or []))
    ranked = []
    for candidate in _collect_summary_memory_candidates(project_path, chapter_count):
        candidate_keywords = candidate.get("tags") or _extract_keywords(candidate.get("text", ""))
        score = _overlap_score(query_keywords, candidate_keywords)
        if current_open_keywords:
            score += _overlap_score(current_open_keywords, candidate_keywords) * 2
        if candidate.get("kind") == "arc":
            score += 1
        if score <= 0:
            continue
        recency = safe_int(candidate.get("chapter_number") or candidate.get("arc_index"), 0)
        ranked.append((score, recency, candidate))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    for _, _, candidate in ranked[:RETRIEVED_MEMORY_LIMIT]:
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


def _build_recent_craft_memory_block(project_path: str, chapter_count: int, *, max_chars: int) -> str:
    lines = []
    for payload in load_recent_summary_payloads(project_path, chapter_count, limit=RECENT_CRAFT_MEMORY_COUNT):
        notes = normalize_craft_notes(payload.get("craft_notes"))
        fragments = []
        scene_type = notes.get("scene_type", "")
        emotional_beat = notes.get("emotional_beat", "")
        ending_pattern = notes.get("ending_pattern", "")
        if scene_type:
            fragments.append(f"场景类型={scene_type}")
        if emotional_beat:
            fragments.append(f"情绪节拍={emotional_beat}")
        if ending_pattern:
            fragments.append(f"结尾方式={ending_pattern}")
        repeated = (notes.get("repeated_actions") or []) + (notes.get("recurring_gestures") or [])
        if repeated:
            fragments.append(f"动作/姿态={'; '.join(repeated[:4])}")
        phrasing = notes.get("notable_phrasing") or []
        if phrasing:
            fragments.append(f"常用措辞={'; '.join(phrasing[:3])}")
        if not fragments:
            continue
        candidate = f"- 第{payload['chapter_number']}章写法记忆: " + "；".join(fragments)
        next_block = "\n".join(lines + [candidate])
        if len(next_block) > max_chars:
            break
        lines.append(candidate)
    if not lines:
        return ""
    return _trim_text(
        "\n".join(lines)
        + "\n使用方式: 下一章要保留连续性，但避免机械复用这些动作、姿态、情绪节拍和结尾方式；如必须复用，需赋予新的功能、代价或关系变化。",
        max_chars,
    )


def _build_craft_brief_block(craft_brief: dict | None, *, max_chars: int) -> str:
    if not isinstance(craft_brief, dict):
        return ""
    lines = []
    mapping = (
        ("钩子", "chapter_hook"),
        ("读者入口/连续性桥", "context_bridge"),
        ("戏剧问题", "dramatic_question"),
        ("冲突压力", "conflict_pressure"),
        ("行动理由", "action_reasoning"),
        ("情绪转折", "emotional_turn"),
        ("场景调度", "scene_movement"),
        ("感官调色", "sensory_palette"),
        ("新鲜互动", "fresh_interaction_patterns"),
        ("禁用重复", "forbidden_repeats"),
        ("验收标准", "success_criteria"),
        ("补充", "focus_notes"),
    )
    for label, key in mapping:
        value = craft_brief.get(key)
        if isinstance(value, list):
            text = "；".join(str(item).strip() for item in value if str(item).strip())
        else:
            text = str(value or "").strip()
        if not text:
            continue
        candidate = "\n".join(lines + [f"{label}: {text}"])
        if len(candidate) > max_chars:
            break
        lines.append(f"{label}: {text}")
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
    craft_brief: dict | None = None,
    include_continuation_contract: bool = False,
) -> dict:
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    effective_task_card = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
    )
    task_card = merge_task_card_guidance(effective_task_card, user_request) if user_request else effective_task_card
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
        "creative_contract": _build_creative_contract_block(
            author_intent,
            max_chars=WRITER_SECTION_LIMITS["creative_contract"],
        ),
        "opening_contract": _build_opening_contract_block(
            project_data,
            task_card,
            max_chars=WRITER_SECTION_LIMITS["opening_contract"],
        ),
        "continuation_contract": (
            _build_continuation_contract_block(
                project_data,
                task_card,
                max_chars=WRITER_SECTION_LIMITS["continuation_contract"],
            )
            if include_continuation_contract
            else ""
        ),
        "reader_setup": _build_reader_setup_block(
            project_path,
            project_data,
            task_card,
            max_chars=WRITER_SECTION_LIMITS["reader_setup"],
        ),
        "chapter_task": _build_chapter_task_block(task_card, max_chars=WRITER_SECTION_LIMITS["chapter_task"]),
        "live_state": _build_live_state_block(
            plot_state,
            max_chars=WRITER_SECTION_LIMITS["live_state"],
            include_next_goal=False,
        ),
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
        "recent_craft_memory": _build_recent_craft_memory_block(
            project_path,
            chapter_count,
            max_chars=WRITER_SECTION_LIMITS["recent_craft_memory"],
        ),
        "craft_brief": _build_craft_brief_block(
            craft_brief,
            max_chars=WRITER_SECTION_LIMITS["craft_brief"],
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
        "effective_task_card": effective_task_card,
        "sections": sections,
        "section_chars": {key: len(value) for key, value in sections.items() if value},
    }


def build_summary_context(project_path: str, project_data: dict, new_text: str) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    completed_task = load_task_card(project_path, chapter_count) if chapter_count > 0 else None
    sections = {
        "live_state": _build_live_state_block(plot_state, max_chars=900),
        "static_characters": _compact_characters_block(
            project_data.get("characters") or {},
            plot_state,
            max_chars=320,
        ),
        "chapter_text": str(new_text or "").strip(),
    }
    if completed_task:
        sections["completed_task"] = _build_chapter_task_block(completed_task, max_chars=480)
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
        "author_intent": _build_author_intent_block(author_intent, max_chars=780),
        "creative_contract": _build_creative_contract_block(author_intent, max_chars=760),
        "live_state": _build_live_state_block(plot_state, max_chars=1200),
        "static_world": _compact_world_block(project_data.get("world") or {}, user_request, max_chars=500),
        "static_characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=700),
        "recent_scene": _build_recent_scene_block(project_path, chapter_count, recent_text, max_chars=1800),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=460),
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
    task_card: dict | None = None,
    option_count: int,
    planning_mode: str,
) -> dict:
    plot_state = normalize_live_plot_state(project_data.get("plot_state"))
    author_intent = normalize_author_intent(project_data.get("author_intent") or ensure_author_intent(project_path))
    chapter_count = safe_int(project_data.get("project", {}).get("chapter_count"), 0)
    effective_task = task_card or resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=False,
    )
    task_text = " ".join(
        part
        for part in (
            effective_task.get("summary", ""),
            effective_task.get("goal", ""),
            effective_task.get("writer_guidance", ""),
            user_request,
        )
        if part
    )
    sections = {
        "author_intent": _build_author_intent_block(author_intent, max_chars=780),
        "creative_contract": _build_creative_contract_block(author_intent, max_chars=760),
        "opening_contract": _build_opening_contract_block(project_data, effective_task, max_chars=760),
        "reader_setup": _build_reader_setup_block(project_path, project_data, effective_task, max_chars=760),
        "chapter_task": _build_chapter_task_block(effective_task, max_chars=480),
        "live_state": _build_live_state_block(plot_state, max_chars=860, include_next_goal=False),
        "static_world": _compact_world_block(project_data.get("world") or {}, task_text, max_chars=500),
        "static_characters": _compact_characters_block(project_data.get("characters") or {}, plot_state, max_chars=700),
        "recent_scene": _build_recent_scene_block(project_path, chapter_count, recent_text, max_chars=1700),
        "style_contract": _build_style_contract_block(project_data.get("style") or {}, author_intent, max_chars=360),
        "planning_mode": normalize_planning_mode(planning_mode),
        "user_request": str(user_request or "").strip() or "无额外要求。请仅基于当前状态给出下一章推进选项。",
        "option_count": max(1, int(option_count or 4)),
    }
    sections["retrieved_memory"] = _build_retrieved_memory_block(
        project_path,
        chapter_count,
        effective_task,
        plot_state,
        sections["recent_scene"],
        max_chars=420,
    )
    return {
        "task_card": effective_task,
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
