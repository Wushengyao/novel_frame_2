"""Creative quality stages for chapter generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common_utils import emit_progress, extract_json_object
from console_logger import log_info, log_success, log_warning
from llm_client import generate_text_with_metadata
from project_manager import record_context_telemetry, save_json, update_project_stats
from prompt_builder import (
    build_craft_brief_prompt,
    build_quality_review_prompt,
    build_rewrite_prompt,
)
from runtime_config import (
    REVIEW_MODE_AUTO,
    REVIEW_MODE_MANUAL,
    WRITING_QUALITY_BALANCED,
    WRITING_QUALITY_HIGH,
    WRITING_QUALITY_LIGHT,
    normalize_review_mode,
    normalize_writing_quality_mode,
)


CRAFT_BRIEF_DIR_NAME = "craft_briefs"
QUALITY_REVIEW_DIR_NAME = "quality_reviews"
REVIEW_SCORE_KEYS = (
    "task_completion",
    "reader_hook",
    "scene_freshness",
    "character_specificity",
    "motivation_causality",
    "repetition_risk",
    "continuity",
)
REVIEW_PASS_AVERAGE = 7.0
REVIEW_PASS_MINIMUM = 5.0


def _coerce_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, score))


def _coerce_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on", "通过", "pass", "passed"}:
        return True
    if normalized in {"false", "0", "no", "n", "off", "不通过", "fail", "failed"}:
        return False
    return default


def _normalize_string_list(value: object, *, max_items: int = 8) -> list[str]:
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    result = []
    seen = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
        if len(result) >= max_items:
            break
    return result


def normalize_craft_brief(payload: dict | None, fallback: dict | None = None) -> dict:
    source = payload if isinstance(payload, dict) else {}
    fallback_source = fallback if isinstance(fallback, dict) else {}
    return {
        "chapter_hook": str(source.get("chapter_hook") or fallback_source.get("chapter_hook") or "").strip(),
        "context_bridge": str(source.get("context_bridge") or fallback_source.get("context_bridge") or "").strip(),
        "dramatic_question": str(source.get("dramatic_question") or fallback_source.get("dramatic_question") or "").strip(),
        "conflict_pressure": str(source.get("conflict_pressure") or fallback_source.get("conflict_pressure") or "").strip(),
        "action_reasoning": str(source.get("action_reasoning") or fallback_source.get("action_reasoning") or "").strip(),
        "emotional_turn": str(source.get("emotional_turn") or fallback_source.get("emotional_turn") or "").strip(),
        "scene_movement": _normalize_string_list(source.get("scene_movement") or fallback_source.get("scene_movement"), max_items=6),
        "sensory_palette": _normalize_string_list(source.get("sensory_palette") or fallback_source.get("sensory_palette"), max_items=6),
        "fresh_interaction_patterns": _normalize_string_list(
            source.get("fresh_interaction_patterns") or fallback_source.get("fresh_interaction_patterns"),
            max_items=6,
        ),
        "forbidden_repeats": _normalize_string_list(source.get("forbidden_repeats") or fallback_source.get("forbidden_repeats"), max_items=8),
        "focus_notes": str(source.get("focus_notes") or fallback_source.get("focus_notes") or "").strip(),
    }


def fallback_craft_brief(prompt_context: dict) -> dict:
    task_card = prompt_context.get("task_card") or {}
    sections = prompt_context.get("sections") or {}
    objective = str(task_card.get("objective") or task_card.get("goal") or "").strip()
    recent_craft = str(sections.get("recent_craft_memory") or "").strip()
    forbidden_repeats = []
    if recent_craft:
        forbidden_repeats.append("避免机械复用近期章节里的动作、姿态、情绪节拍和结尾方式。")
    return normalize_craft_brief(
        {
            "chapter_hook": objective or "用一个具体压力或异常变化开章。",
            "context_bridge": "开场补足读者需要理解的处境、人物关系和连续性锚点。",
            "dramatic_question": objective or "本章能否完成当前任务并付出新的代价？",
            "conflict_pressure": "让外部压力、资源限制或人物选择推动场景变化。",
            "action_reasoning": "让关键行动来自当前压力、人物目标和可见限制，而不是无缘无故发生。",
            "emotional_turn": "至少让一名核心人物的判断、信任或欲望出现可见变化。",
            "scene_movement": ["开章给出具体压力", "中段让人物做出选择", "结尾兑现结果或留下新代价"],
            "sensory_palette": ["选择与本章地点相关的两到三种感官细节"],
            "fresh_interaction_patterns": ["用新的动作逻辑和互动方式表现关系，不重复上一章姿态"],
            "forbidden_repeats": forbidden_repeats,
            "focus_notes": "这是本地兜底蓝图；优先保证任务完成、场景推进和写法变化。",
        }
    )


def normalize_quality_review(payload: dict | None, *, fallback_passed: bool = True) -> dict:
    source = payload if isinstance(payload, dict) else {}
    raw_scores = source.get("scores") if isinstance(source.get("scores"), dict) else {}
    if not source:
        default_score = 10.0 if fallback_passed else 0.0
        scores = {key: default_score for key in REVIEW_SCORE_KEYS}
    else:
        motivation_fallback = min(
            (
                _coerce_score(raw_scores.get(key))
                for key in ("task_completion", "character_specificity", "continuity")
                if key in raw_scores
            ),
            default=0.0,
        )
        scores = {
            key: _coerce_score(
                raw_scores.get(key, motivation_fallback if key == "motivation_causality" else None)
            )
            for key in REVIEW_SCORE_KEYS
        }
    average = sum(scores.values()) / len(REVIEW_SCORE_KEYS)
    computed_passed = average >= REVIEW_PASS_AVERAGE and min(scores.values()) >= REVIEW_PASS_MINIMUM
    explicit_passed = _coerce_bool(source.get("passed"), default=computed_passed) if "passed" in source else computed_passed
    passed = (explicit_passed and computed_passed) if source else fallback_passed
    return {
        "scores": scores,
        "average_score": round(average, 2),
        "passed": passed,
        "strengths": _normalize_string_list(source.get("strengths"), max_items=8),
        "issues": _normalize_string_list(source.get("issues"), max_items=8),
        "revision_guidance": str(source.get("revision_guidance", "") or "").strip(),
        "repeat_examples": _normalize_string_list(source.get("repeat_examples"), max_items=8),
    }


def quality_review_passed(review: dict) -> bool:
    if not isinstance(review, dict):
        return True
    if not bool(review.get("passed", True)):
        return False
    scores = review.get("scores") if isinstance(review.get("scores"), dict) else {}
    if not scores:
        return True
    values = [_coerce_score(scores.get(key)) for key in REVIEW_SCORE_KEYS]
    return sum(values) / len(values) >= REVIEW_PASS_AVERAGE and min(values) >= REVIEW_PASS_MINIMUM


def craft_brief_path(project_path: str, chapter_number: int) -> Path:
    return Path(project_path) / CRAFT_BRIEF_DIR_NAME / f"chapter_{chapter_number:04d}.json"


def quality_review_path(project_path: str, chapter_number: int, attempt: int) -> Path:
    return Path(project_path) / QUALITY_REVIEW_DIR_NAME / f"chapter_{chapter_number:04d}_attempt_{attempt}.json"


def generate_craft_brief(
    project_path: str,
    prompt_context: dict,
    config: dict,
    *,
    log_context: dict[str, Any],
    progress_callback=None,
) -> dict:
    chapter_number = int((prompt_context.get("task_card") or {}).get("chapter_number", 0) or 0)
    fallback = fallback_craft_brief(prompt_context)
    prompt = build_craft_brief_prompt(prompt_context)
    record_context_telemetry(
        project_path,
        "craft_brief",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number},
    )
    request_log_context = {**log_context, "phase": "craft_brief"}
    try:
        log_info("craft_brief: requesting model brief")
        emit_progress(progress_callback, "craft_brief", "Generating chapter craft brief")
        response_text, metadata = generate_text_with_metadata(prompt, config, log_context=request_log_context)
        update_project_stats(project_path, phase="craft_brief", success=True, usage=metadata.get("usage"))
    except Exception as exc:  # pragma: no cover - resilience path
        update_project_stats(project_path, phase="craft_brief", success=False, usage=None)
        log_warning(f"craft_brief: using fallback brief, reason: {exc}")
        brief = deepcopy(fallback)
        brief["fallback_reason"] = str(exc)
    else:
        try:
            brief = normalize_craft_brief(extract_json_object(response_text, "Could not parse JSON from craft brief response."), fallback)
        except Exception as exc:  # pragma: no cover - malformed model output fallback
            log_warning(f"craft_brief: using fallback brief, reason: {exc}")
            brief = deepcopy(fallback)
            brief["fallback_reason"] = str(exc)

    save_json(str(craft_brief_path(project_path, chapter_number)), brief)
    log_success(f"craft_brief: saved chapter_{chapter_number:04d}.json")
    return brief


def review_chapter_draft(
    project_path: str,
    prompt_context: dict,
    draft_text: str,
    config: dict,
    *,
    attempt: int,
    strict: bool,
    log_context: dict[str, Any],
    progress_callback=None,
) -> dict:
    chapter_number = int((prompt_context.get("task_card") or {}).get("chapter_number", 0) or 0)
    prompt = build_quality_review_prompt(prompt_context, draft_text, strict=strict)
    record_context_telemetry(
        project_path,
        "quality_review",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number, "attempt": attempt, "strict": strict},
    )
    request_log_context = {**log_context, "phase": "quality_review", "attempt": attempt}
    try:
        log_info(f"quality_review: requesting review attempt={attempt}")
        emit_progress(progress_callback, "quality_review", f"Reviewing chapter draft (attempt {attempt})")
        response_text, metadata = generate_text_with_metadata(prompt, config, log_context=request_log_context)
        update_project_stats(project_path, phase="quality_review", success=True, usage=metadata.get("usage"))
    except Exception as exc:  # pragma: no cover - resilience path
        update_project_stats(project_path, phase="quality_review", success=False, usage=None)
        log_warning(f"quality_review: using passing fallback report, reason: {exc}")
        review = normalize_quality_review(None, fallback_passed=True)
        review["fallback_reason"] = str(exc)
    else:
        try:
            review = normalize_quality_review(extract_json_object(response_text, "Could not parse JSON from quality review response."))
        except Exception as exc:  # pragma: no cover - malformed model output fallback
            log_warning(f"quality_review: using passing fallback report, reason: {exc}")
            review = normalize_quality_review(None, fallback_passed=True)
            review["fallback_reason"] = str(exc)

    save_json(str(quality_review_path(project_path, chapter_number, attempt)), review)
    log_success(f"quality_review: saved chapter_{chapter_number:04d}_attempt_{attempt}.json")
    return review


def rewrite_chapter_draft(
    project_path: str,
    prompt_context: dict,
    draft_text: str,
    review_report: dict,
    config: dict,
    *,
    log_context: dict[str, Any],
    progress_callback=None,
) -> str:
    chapter_number = int((prompt_context.get("task_card") or {}).get("chapter_number", 0) or 0)
    prompt = build_rewrite_prompt(prompt_context, draft_text, review_report)
    record_context_telemetry(
        project_path,
        "rewrite",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number},
    )
    request_log_context = {**log_context, "phase": "rewrite"}
    try:
        log_info("rewrite: requesting improved chapter draft")
        emit_progress(progress_callback, "rewrite", "Rewriting chapter draft after quality review")
        response_text, metadata = generate_text_with_metadata(prompt, config, log_context=request_log_context)
        update_project_stats(project_path, phase="rewrite", success=True, usage=metadata.get("usage"))
        return response_text
    except Exception:
        update_project_stats(project_path, phase="rewrite", success=False, usage=None)
        raise


def quality_mode_uses_craft_brief(mode: object) -> bool:
    return normalize_writing_quality_mode(mode) in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}


def quality_mode_uses_review(mode: object) -> bool:
    return normalize_writing_quality_mode(mode) in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}


def quality_mode_allows_rewrite(mode: object, review_mode: object) -> bool:
    return (
        normalize_writing_quality_mode(mode) == WRITING_QUALITY_HIGH
        and normalize_review_mode(review_mode) == REVIEW_MODE_AUTO
    )


def normalize_quality_config(config: dict) -> tuple[str, str]:
    return (
        normalize_writing_quality_mode(config.get("writing_quality_mode")),
        normalize_review_mode(config.get("review_mode")),
    )


__all__ = [
    "REVIEW_MODE_AUTO",
    "REVIEW_MODE_MANUAL",
    "WRITING_QUALITY_BALANCED",
    "WRITING_QUALITY_HIGH",
    "WRITING_QUALITY_LIGHT",
    "craft_brief_path",
    "generate_craft_brief",
    "normalize_craft_brief",
    "normalize_quality_config",
    "quality_mode_allows_rewrite",
    "quality_mode_uses_craft_brief",
    "quality_mode_uses_review",
    "quality_review_passed",
    "quality_review_path",
    "review_chapter_draft",
    "rewrite_chapter_draft",
]
