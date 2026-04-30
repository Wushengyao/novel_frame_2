"""Creative quality stages for chapter generation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from common_utils import emit_progress, extract_json_object, save_failed_llm_output
from console_logger import log_info, log_success, log_warning
from llm_client import generate_text_with_metadata, raise_if_llm_response_truncated
from project_manager import load_json, record_context_telemetry, save_json, update_project_stats
from prompt_builder import (
    build_craft_brief_prompt,
    build_quality_review_prompt,
    build_rewrite_prompt,
    build_system_prompt,
)
from runtime_config import (
    REVIEW_MODE_AUTO,
    REVIEW_MODE_MANUAL,
    WRITING_QUALITY_BALANCED,
    WRITING_QUALITY_HIGH,
    WRITING_QUALITY_LIGHT,
    normalize_review_mode,
    normalize_writing_quality_mode,
    resolve_quality_model_config,
)


CRAFT_BRIEF_DIR_NAME = "craft_briefs"
QUALITY_REVIEW_DIR_NAME = "quality_reviews"
QUALITY_DRAFT_DIR_NAME = "quality_drafts"
REWRITE_REQUEST_ATTEMPTS = 2
REWRITE_TEXT_KEYS = (
    "rewritten_text",
    "chapter_text",
    "content",
    "text",
    "body",
    "正文",
    "重写正文",
)
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
REVIEW_FLATNESS_REWRITE_MINIMUM = 7.0
FLATNESS_SCORE_KEYS = (
    "reader_hook",
    "scene_freshness",
    "character_specificity",
    "repetition_risk",
)
QUALITY_REVIEW_SCHEMA_VERSION = 2


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


def _normalize_score_reasons(value: object) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {
        key: str(source.get(key) or "").strip()
        for key in REVIEW_SCORE_KEYS
        if str(source.get(key) or "").strip()
    }


def _normalize_issue_objects(value: object, *, max_items: int = 8, default_severity: str = "major") -> list[dict[str, str]]:
    items = value if isinstance(value, list) else [value] if value not in (None, "") else []
    result: list[dict[str, str]] = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("issue") or item.get("description") or item.get("summary") or "").strip()
            evidence = str(item.get("evidence") or item.get("example") or "").strip()
            fix = str(item.get("fix") or item.get("guidance") or item.get("recommendation") or "").strip()
            category = str(item.get("category") or "").strip()
            severity = str(item.get("severity") or default_severity).strip().lower() or default_severity
        else:
            text = str(item or "").strip()
            evidence = ""
            fix = ""
            category = ""
            severity = default_severity
        if not text:
            continue
        dedupe_key = (category, severity, text, evidence, fix)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        result.append(
            {
                "category": category,
                "severity": severity,
                "issue": text,
                "evidence": evidence,
                "fix": fix,
            }
        )
        if len(result) >= max_items:
            break
    return result


def _has_blocker(review: dict) -> bool:
    issues = review.get("blocking_issues") if isinstance(review, dict) else []
    if not isinstance(issues, list):
        return False
    for issue in issues:
        if isinstance(issue, dict):
            severity = str(issue.get("severity") or "").strip().lower()
            if severity == "blocker":
                return True
    return False


def _flatness_low_scores(scores: dict[str, float]) -> list[str]:
    return [
        key
        for key in FLATNESS_SCORE_KEYS
        if _coerce_score(scores.get(key)) < REVIEW_FLATNESS_REWRITE_MINIMUM
    ]


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
        "success_criteria": _normalize_string_list(source.get("success_criteria") or fallback_source.get("success_criteria"), max_items=5),
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
            "success_criteria": [
                f"完成或明确推进本章任务：{objective}" if objective else "完成或明确推进本章任务卡的核心目标。",
                "关键行动有清楚的当前压力、人物选择和可见结果。",
                "结尾形成新的信息、代价、决定或悬念，避免原地空转。",
            ],
            "focus_notes": "这是本地兜底蓝图；优先保证任务完成、场景推进和写法变化。",
        }
    )


def normalize_quality_review(payload: dict | None, *, fallback_passed: bool = False, review_unavailable: bool = False) -> dict:
    source = payload if isinstance(payload, dict) else {}
    raw_scores = source.get("scores") if isinstance(source.get("scores"), dict) else {}
    unavailable = review_unavailable or _coerce_bool(source.get("review_unavailable"), default=False)
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
    blocking_issues = _normalize_issue_objects(source.get("blocking_issues"), max_items=8, default_severity="blocker")
    flatness_issues = _flatness_low_scores(scores)
    computed_passed = (
        average >= REVIEW_PASS_AVERAGE
        and min(scores.values()) >= REVIEW_PASS_MINIMUM
        and not any(issue.get("severity") == "blocker" for issue in blocking_issues)
        and not unavailable
    )
    explicit_passed = _coerce_bool(source.get("passed"), default=computed_passed) if "passed" in source else computed_passed
    passed = (explicit_passed and computed_passed) if source else (fallback_passed and not unavailable)
    issues = _normalize_string_list(source.get("issues"), max_items=8)
    nice_to_have = _normalize_string_list(source.get("nice_to_have"), max_items=8)
    rewrite_plan = _normalize_string_list(source.get("rewrite_plan"), max_items=8)
    revision_guidance = str(source.get("revision_guidance", "") or "").strip()
    if flatness_issues and source:
        labels = "、".join(flatness_issues)
        flatness_note = f"平淡度风险：{labels} 低于 {REVIEW_FLATNESS_REWRITE_MINIMUM:g}，需要增强钩子、角色声音、互动火花或场景新变化。"
        if flatness_note not in issues:
            issues.append(flatness_note)
        if not rewrite_plan:
            rewrite_plan = [
                "重写开章钩子，让场景压力更早出现。",
                "补强角色专属口吻、动作反应和互动火花。",
                "把概括性推进改成可见的动作、感官、心理和对话交替。",
                "在结尾兑现新的信息、代价、关系变化或生存成果。",
            ]
        if not revision_guidance:
            revision_guidance = "围绕低分平淡项轻量改写，保留已完成的剧情目标，重点增强钩子、角色声音、场景新鲜度和互动火花。"
    explicit_needs_rewrite = _coerce_bool(source.get("needs_rewrite"), default=False) if source else False
    needs_rewrite = bool(flatness_issues and source and not unavailable) or explicit_needs_rewrite
    return {
        "schema_version": QUALITY_REVIEW_SCHEMA_VERSION,
        "scores": scores,
        "average_score": round(average, 2),
        "passed": passed,
        "review_unavailable": unavailable,
        "needs_rewrite": needs_rewrite,
        "flatness_issues": flatness_issues,
        "score_reasons": _normalize_score_reasons(source.get("score_reasons")),
        "strengths": _normalize_string_list(source.get("strengths"), max_items=8),
        "issues": issues[:8],
        "blocking_issues": blocking_issues,
        "nice_to_have": nice_to_have[:8],
        "revision_guidance": revision_guidance,
        "rewrite_plan": rewrite_plan[:8],
        "repeat_examples": _normalize_string_list(source.get("repeat_examples"), max_items=8),
    }


def _extract_quality_review_payload(response_text: str) -> dict:
    payload = extract_json_object(response_text, "Could not parse JSON from quality review response.")
    if not isinstance(payload.get("scores"), dict):
        raise ValueError("quality review response missing scores object")
    return payload


def quality_review_passed(review: dict) -> bool:
    if not isinstance(review, dict):
        return False
    if bool(review.get("review_unavailable", False)):
        return False
    if _has_blocker(review):
        return False
    if not bool(review.get("passed", True)):
        return False
    scores = review.get("scores") if isinstance(review.get("scores"), dict) else {}
    if not scores:
        return True
    values = [_coerce_score(scores.get(key)) for key in REVIEW_SCORE_KEYS]
    return sum(values) / len(values) >= REVIEW_PASS_AVERAGE and min(values) >= REVIEW_PASS_MINIMUM


def quality_review_available(review: dict) -> bool:
    return isinstance(review, dict) and not bool(review.get("review_unavailable", False))


def quality_review_needs_rewrite(review: dict, mode: object) -> bool:
    if not quality_review_available(review):
        return False
    normalized_mode = normalize_writing_quality_mode(mode)
    if normalized_mode not in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}:
        return False
    if not quality_review_passed(review):
        return True
    return bool(review.get("needs_rewrite"))


def _strip_single_fenced_block(text: str) -> str:
    content = str(text or "").strip()
    if not content.startswith("```"):
        return content
    first_newline = content.find("\n")
    closing = content.rfind("```")
    if first_newline == -1 or closing <= first_newline:
        return content
    return content[first_newline + 1 : closing].strip()


def _looks_like_json_response(text: str) -> bool:
    content = str(text or "").strip().lstrip("\ufeff")
    if content.startswith("{"):
        return True
    if content.startswith("```"):
        return _strip_single_fenced_block(content).lstrip("\ufeff").lstrip().startswith("{")
    return False


def _rewrite_text_from_payload(payload: dict) -> str:
    for key in REWRITE_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("result", "data", "chapter"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            text = _rewrite_text_from_payload(nested)
            if text:
                return text
    return ""


def normalize_rewrite_response_text(response_text: str) -> str:
    content = _strip_single_fenced_block(response_text)
    if not content:
        raise ValueError("Rewrite response is empty.")

    if _looks_like_json_response(response_text):
        payload = extract_json_object(response_text, "Could not parse JSON from rewrite response.")
        rewritten_text = _rewrite_text_from_payload(payload)
        if not rewritten_text:
            raise ValueError("Rewrite response JSON did not contain rewritten chapter text.")
        return _strip_single_fenced_block(rewritten_text)

    return content


def craft_brief_path(project_path: str, chapter_number: int) -> Path:
    return Path(project_path) / CRAFT_BRIEF_DIR_NAME / f"chapter_{chapter_number:04d}.json"


def quality_review_path(project_path: str, chapter_number: int, attempt: int) -> Path:
    return Path(project_path) / QUALITY_REVIEW_DIR_NAME / f"chapter_{chapter_number:04d}_attempt_{attempt}.json"


def pre_rewrite_draft_path(project_path: str, chapter_number: int, rewrite_attempt: int) -> Path:
    return Path(project_path) / QUALITY_DRAFT_DIR_NAME / f"chapter_{chapter_number:04d}_before_rewrite_{rewrite_attempt}.md"


def save_pre_rewrite_draft(project_path: str, chapter_number: int, rewrite_attempt: int, draft_text: str) -> Path:
    path = pre_rewrite_draft_path(project_path, chapter_number, rewrite_attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(draft_text, encoding="utf-8")
    return path


def list_quality_artifacts(project_path: str, chapter_number: int) -> dict:
    base = Path(project_path)
    review_pattern = f"chapter_{chapter_number:04d}_attempt_*.json"
    draft_pattern = f"chapter_{chapter_number:04d}_before_rewrite_*.md"
    reports = []
    for path in sorted((base / QUALITY_REVIEW_DIR_NAME).glob(review_pattern)):
        match = path.name.rsplit("_attempt_", 1)
        attempt = 0
        if len(match) == 2:
            try:
                attempt_text = match[1][:-5] if match[1].endswith(".json") else match[1]
                attempt = int(attempt_text)
            except ValueError:
                attempt = 0
        report = {}
        error = ""
        try:
            report = load_json(str(path))
        except Exception as exc:  # pragma: no cover - damaged local artifact
            error = str(exc)
        reports.append(
            {
                "attempt": attempt,
                "file_name": path.name,
                "path": str(path),
                "report": report,
                "error": error,
            }
        )

    drafts = []
    for path in sorted((base / QUALITY_DRAFT_DIR_NAME).glob(draft_pattern)):
        match = path.name.rsplit("_before_rewrite_", 1)
        rewrite_attempt = 0
        if len(match) == 2:
            try:
                attempt_text = match[1][:-3] if match[1].endswith(".md") else match[1]
                rewrite_attempt = int(attempt_text)
            except ValueError:
                rewrite_attempt = 0
        drafts.append(
            {
                "rewrite_attempt": rewrite_attempt,
                "file_name": path.name,
                "path": str(path),
            }
        )

    reports.sort(key=lambda item: item.get("attempt", 0))
    drafts.sort(key=lambda item: item.get("rewrite_attempt", 0))
    return {
        "chapter_number": chapter_number,
        "reports": reports,
        "pre_rewrite_drafts": drafts,
        "rewrite_count": len(drafts),
    }


def _quality_request_config(config: dict, log_context: dict[str, Any], phase: str) -> tuple[dict, dict[str, Any], dict[str, object]]:
    request_config, uses_quality_model = resolve_quality_model_config(config)
    request_log_context = {
        **log_context,
        "phase": phase,
        "uses_quality_model": uses_quality_model,
    }
    telemetry_extra: dict[str, object] = {"uses_quality_model": uses_quality_model}
    if uses_quality_model:
        provider = str(request_config.get("model_provider", "") or "")
        model = str(request_config.get("model_name") or request_config.get("model") or "")
        request_log_context["quality_model_provider"] = provider
        request_log_context["quality_model"] = model
        telemetry_extra["quality_model_provider"] = provider
        telemetry_extra["quality_model"] = model
    return request_config, request_log_context, telemetry_extra


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
    request_config, request_log_context, quality_extra = _quality_request_config(config, log_context, "craft_brief")
    record_context_telemetry(
        project_path,
        "craft_brief",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number, **quality_extra},
    )
    try:
        log_info("craft_brief: requesting model brief")
        emit_progress(progress_callback, "craft_brief", "Generating chapter craft brief")
        response_text, metadata = generate_text_with_metadata(
            prompt,
            request_config,
            log_context=request_log_context,
            system_prompt=build_system_prompt("craft_brief"),
            response_format="json",
        )
        update_project_stats(project_path, phase="craft_brief", success=True, usage=metadata.get("usage"), metadata=metadata)
    except Exception as exc:  # pragma: no cover - resilience path
        update_project_stats(project_path, phase="craft_brief", success=False, usage=None)
        log_warning(f"craft_brief: using fallback brief, reason: {exc}")
        brief = deepcopy(fallback)
        brief["fallback_reason"] = str(exc)
    else:
        try:
            brief = normalize_craft_brief(extract_json_object(response_text, "Could not parse JSON from craft brief response."), fallback)
        except Exception as exc:  # pragma: no cover - malformed model output fallback
            save_failed_llm_output(
                project_path,
                "craft_brief",
                response_text,
                error=str(exc),
                context=request_log_context,
            )
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
    request_config, request_log_context, quality_extra = _quality_request_config(config, log_context, "quality_review")
    record_context_telemetry(
        project_path,
        "quality_review",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number, "attempt": attempt, "strict": strict, **quality_extra},
    )
    request_log_context["attempt"] = attempt
    review = None
    last_error = ""
    for request_attempt in (1, 2):
        request_log_context["quality_request_attempt"] = request_attempt
        try:
            log_info(f"quality_review: requesting review attempt={attempt} request_attempt={request_attempt}")
            message = f"Reviewing chapter draft (attempt {attempt})"
            if request_attempt > 1:
                message += " retry"
            emit_progress(progress_callback, "quality_review", message)
            response_text, metadata = generate_text_with_metadata(
                prompt,
                request_config,
                log_context=request_log_context,
                system_prompt=build_system_prompt("quality_review"),
                response_format="json",
            )
            update_project_stats(project_path, phase="quality_review", success=True, usage=metadata.get("usage"), metadata=metadata)
        except Exception as exc:  # pragma: no cover - resilience path
            update_project_stats(project_path, phase="quality_review", success=False, usage=None)
            last_error = str(exc)
            log_warning(f"quality_review: request failed attempt={attempt} request_attempt={request_attempt}, reason: {exc}")
            continue

        try:
            review = normalize_quality_review(_extract_quality_review_payload(response_text))
            break
        except Exception as exc:  # pragma: no cover - malformed model output fallback
            save_failed_llm_output(
                project_path,
                "quality_review",
                response_text,
                error=str(exc),
                context=request_log_context,
            )
            last_error = str(exc)
            log_warning(f"quality_review: response parse failed attempt={attempt} request_attempt={request_attempt}, reason: {exc}")

    if review is None:
        log_warning(f"quality_review: unavailable after retry; saving non-passing fallback report, reason: {last_error}")
        review = normalize_quality_review(None, fallback_passed=False, review_unavailable=True)
        review["fallback_reason"] = last_error

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
    request_config, request_log_context, quality_extra = _quality_request_config(config, log_context, "rewrite")
    record_context_telemetry(
        project_path,
        "rewrite",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=config.get("planning_mode", ""),
        extra={"target_chapter_number": chapter_number, **quality_extra},
    )
    last_error = ""
    for request_attempt in range(1, REWRITE_REQUEST_ATTEMPTS + 1):
        request_log_context["rewrite_request_attempt"] = request_attempt
        try:
            log_info(f"rewrite: requesting improved chapter draft request_attempt={request_attempt}")
            message = "Rewriting chapter draft after quality review"
            if request_attempt > 1:
                message += " retry"
            emit_progress(progress_callback, "rewrite", message)
            response_text, metadata = generate_text_with_metadata(
                prompt,
                request_config,
                log_context=request_log_context,
                system_prompt=build_system_prompt("rewrite"),
            )
            raise_if_llm_response_truncated(metadata, phase="rewrite")
        except Exception as exc:
            update_project_stats(project_path, phase="rewrite", success=False, usage=None)
            last_error = str(exc)
            log_warning(f"rewrite: request failed request_attempt={request_attempt}, reason: {exc}")
            continue

        try:
            rewritten_text = normalize_rewrite_response_text(response_text)
        except Exception as exc:
            update_project_stats(project_path, phase="rewrite", success=False, usage=metadata.get("usage"), metadata=metadata)
            last_error = str(exc)
            log_warning(f"rewrite: response parse failed request_attempt={request_attempt}, reason: {exc}")
            continue

        update_project_stats(project_path, phase="rewrite", success=True, usage=metadata.get("usage"), metadata=metadata)
        return rewritten_text

    raise RuntimeError(f"rewrite failed after retry: {last_error}")


def quality_mode_uses_craft_brief(mode: object) -> bool:
    return normalize_writing_quality_mode(mode) in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}


def quality_mode_uses_review(mode: object) -> bool:
    return normalize_writing_quality_mode(mode) in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}


def quality_mode_allows_rewrite(mode: object, review_mode: object) -> bool:
    return (
        normalize_writing_quality_mode(mode) in {WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH}
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
    "normalize_rewrite_response_text",
    "quality_mode_allows_rewrite",
    "quality_mode_uses_craft_brief",
    "quality_mode_uses_review",
    "quality_review_available",
    "quality_review_needs_rewrite",
    "quality_review_passed",
    "quality_review_path",
    "list_quality_artifacts",
    "pre_rewrite_draft_path",
    "review_chapter_draft",
    "rewrite_chapter_draft",
    "save_pre_rewrite_draft",
]
