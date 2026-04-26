"""Guided next-chapter progression option generation and session storage."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from chapter_context import get_next_context_for_mode, peek_next_context_for_mode, resolve_planning_mode
from common_utils import emit_progress, extract_json_object, safe_int, utc_now
from console_logger import log_success
from context_builder import (
    build_custom_progression_task_card,
    build_progression_context,
    build_progression_selected_task_card,
    override_task_card_objective,
    resolve_effective_chapter_task,
)
from llm_client import generate_text_with_metadata
from project_manager import (
    PLANNING_MODE_NONE,
    get_last_chapter_text,
    load_json,
    load_project,
    record_context_telemetry,
    save_json,
    update_project_stats,
)
from prompt_builder import build_auto_objective_prompt, build_progression_options_prompt, build_system_prompt
from runtime_config import sanitize_runtime_overrides


DEFAULT_OPTION_COUNT = 4
ALLOWED_OPTION_COUNTS = {3, 4, 5}
SESSION_DIR_NAME = "progression_sessions"
CUSTOM_PROGRESSION_OPTION_ID = "custom_user_option"
SELECTION_MODE_MANUAL = "manual"
SELECTION_MODE_RECOMMENDED = "recommended"
SELECTION_MODE_RANDOM = "random"
ALLOWED_SELECTION_MODES = {
    SELECTION_MODE_MANUAL,
    SELECTION_MODE_RECOMMENDED,
    SELECTION_MODE_RANDOM,
}
SELECTION_ORIGIN_USER = "user"
SELECTION_ORIGIN_AUTO = "auto"


def validate_option_count(option_count: object) -> int:
    count = safe_int(option_count, DEFAULT_OPTION_COUNT)
    if count not in ALLOWED_OPTION_COUNTS:
        raise ValueError("option_count must be one of: 3, 4, 5.")
    return count


def validate_selection_mode(selection_mode: object, *, allow_manual: bool = True) -> str:
    mode = str(selection_mode or "").strip().lower()
    if not mode:
        return SELECTION_MODE_MANUAL if allow_manual else SELECTION_MODE_RECOMMENDED
    if mode == SELECTION_MODE_MANUAL and not allow_manual:
        raise ValueError("selection_mode must be one of: recommended, random.")
    if mode not in ALLOWED_SELECTION_MODES:
        if allow_manual:
            raise ValueError("selection_mode must be one of: manual, recommended, random.")
        raise ValueError("selection_mode must be one of: recommended, random.")
    return mode


def _session_dir(project_path: str) -> Path:
    return Path(project_path) / SESSION_DIR_NAME


def _session_path(project_path: str, session_id: str) -> Path:
    return _session_dir(project_path) / f"progression_{session_id}.json"


def _normalize_key_events(raw_key_events: object) -> list[str]:
    key_events = raw_key_events or []
    if not isinstance(key_events, list):
        key_events = [key_events]
    return [str(item).strip() for item in key_events if str(item).strip()][:5]


def _normalize_option(option: dict, fallback_id: str) -> dict:
    option_id = str(option.get("option_id") or fallback_id).strip() or fallback_id
    plan_summary = str(option.get("plan_summary", "") or option.get("summary", "") or "").strip()
    plan_steps = _normalize_key_events(option.get("plan_steps") or option.get("key_events", []))
    plan_guidance = str(option.get("plan_guidance", "") or option.get("writer_guidance", "") or "").strip()
    return {
        "option_id": option_id,
        "title": str(option.get("title", "") or "").strip(),
        "plan_summary": plan_summary,
        "plan_steps": plan_steps,
        "plan_guidance": plan_guidance,
        "summary": plan_summary,
        "key_events": plan_steps,
        "writer_guidance": plan_guidance,
        "recommended": bool(option.get("recommended")),
        "custom": bool(option.get("custom")),
    }


def _non_custom_options(options: list[dict]) -> list[dict]:
    return [option for option in options if not option.get("custom")]


def normalize_progression_options_response(data: dict, option_count: int) -> dict:
    count = validate_option_count(option_count)
    raw_options = data.get("options")
    if not isinstance(raw_options, list):
        raise ValueError("progression response missing options list")
    if len(raw_options) != count:
        raise ValueError("progression option count does not match requested count")

    normalized = []
    option_ids = set()
    for index, raw_option in enumerate(raw_options, start=1):
        if not isinstance(raw_option, dict):
            raise ValueError("progression option item must be an object")
        option = _normalize_option(raw_option, fallback_id=f"option_{index}")
        if not option["title"] or not option["plan_summary"] or not option["plan_guidance"]:
            raise ValueError(f"progression option {option['option_id']} is missing required fields")
        if len(option["plan_steps"]) < 2:
            raise ValueError(f"progression option {option['option_id']} must include at least 2 key_events")
        if option["option_id"] in option_ids:
            raise ValueError("progression options contain duplicate option_id values")
        option_ids.add(option["option_id"])
        normalized.append(option)

    recommended_option_id = str(data.get("recommended_option_id", "") or "").strip()
    if recommended_option_id:
        found = False
        for option in normalized:
            option["recommended"] = option["option_id"] == recommended_option_id
            found = found or option["recommended"]
        if not found:
            raise ValueError("recommended_option_id does not match any option_id")

    recommended = [option for option in normalized if option.get("recommended")]
    if len(recommended) != 1:
        raise ValueError("progression options must contain exactly one recommended option")

    return {
        "recommended_option_id": recommended[0]["option_id"],
        "options": normalized,
    }


def build_custom_progression_option() -> dict:
    return {
        "option_id": CUSTOM_PROGRESSION_OPTION_ID,
        "title": "空白自定义项",
        "plan_summary": "不采用上面的候选方案，改由你自己定义这一章想看的创意和情节。",
        "plan_steps": [
            "由你在下方填写这一章真正想发生的内容。",
            "系统会保留当前 objective、剧情状态和卷目标作为上位约束。",
        ],
        "plan_guidance": "请把用户随后填写的自定义创意作为本章执行 plan，不要改写 objective。",
        "recommended": False,
        "custom": True,
    }


def save_progression_session(project_path: str, session: dict) -> str:
    session_id = str(session.get("session_id", "") or "").strip()
    if not session_id:
        raise ValueError("progression session missing session_id")
    path = _session_path(project_path, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(path), session)
    return str(path)


def load_progression_session(project_path: str, session_id: str) -> dict:
    return load_json(str(_session_path(project_path, session_id)))


def _project_chapter_count(project_path: str) -> int:
    project = load_json(str(Path(project_path) / "project.json"))
    return max(0, int(project.get("chapter_count", 0) or 0))


def _update_session_status(project_path: str, session: dict, *, status: str) -> dict:
    updated = deepcopy(session)
    updated["status"] = status
    save_progression_session(project_path, updated)
    return updated


def is_progression_session_stale(project_path: str, session: dict) -> bool:
    current_chapter_count = _project_chapter_count(project_path)
    session_chapter_count = safe_int(session.get("project_chapter_count"), -1)
    target_chapter_number = safe_int(session.get("target_chapter_number"), 0)
    return (
        current_chapter_count != session_chapter_count
        or target_chapter_number != current_chapter_count + 1
    )


def ensure_fresh_progression_session(project_path: str, session: dict) -> dict:
    if is_progression_session_stale(project_path, session):
        return _update_session_status(project_path, session, status="stale")
    return session


def list_progression_sessions(project_path: str, *, include_stale: bool = False) -> list[dict]:
    sessions_dir = _session_dir(project_path)
    if not sessions_dir.exists():
        return []
    sessions = []
    for path in sorted(sessions_dir.glob("progression_*.json"), reverse=True):
        try:
            session = load_json(str(path))
        except Exception:
            continue
        session = ensure_fresh_progression_session(project_path, session)
        if not include_stale and session.get("status") == "stale":
            continue
        sessions.append(session)
    sessions.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return sessions


def get_latest_active_progression_session(project_path: str) -> dict | None:
    for session in list_progression_sessions(project_path):
        if session.get("status") in {"pending", "selected"}:
            return session
    return None


def mark_active_progression_sessions_stale(project_path: str) -> int:
    sessions_dir = _session_dir(project_path)
    if not sessions_dir.exists():
        return 0

    stale_count = 0
    for path in sorted(sessions_dir.glob("progression_*.json")):
        try:
            session = load_json(str(path))
        except Exception:
            continue
        if session.get("status") not in {"pending", "selected"}:
            continue
        session["status"] = "stale"
        save_progression_session(project_path, session)
        stale_count += 1
    return stale_count


def auto_select_progression_option(session: dict, selection_mode: object) -> str:
    mode = validate_selection_mode(selection_mode, allow_manual=False)
    options = _non_custom_options(session.get("options") or [])
    if not options:
        raise ValueError("当前没有可用的模型推进选项可供自动选择。")
    if mode == SELECTION_MODE_RECOMMENDED:
        recommended_option_id = str(session.get("recommended_option_id", "") or "").strip()
        if recommended_option_id:
            return recommended_option_id
        return str(options[0].get("option_id", "") or "").strip()
    return str(random.choice(options).get("option_id", "") or "").strip()


def generate_auto_chapter_objective(
    project_path: str,
    config: dict,
    *,
    user_request: str = "",
    progress_callback=None,
) -> str:
    project_data = load_project(project_path)
    planning_mode = resolve_planning_mode(config, project_data)
    if planning_mode != PLANNING_MODE_NONE:
        raise ValueError("auto objective generation only supports planning_mode=none")

    target_chapter_number = safe_int(project_data["project"].get("chapter_count"), 0) + 1
    emit_progress(progress_callback, "auto_objective_prepare", "正在为下一章提炼 objective")
    project_data, next_context = get_next_context_for_mode(
        project_path,
        config,
        planning_mode,
        progress_callback=progress_callback,
    )
    recent_text = get_last_chapter_text(project_path)
    if not recent_text:
        recent_text = "这是开篇前状态。请围绕第一章如何自然开场来提炼 objective。"
    base_task = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=False,
    )
    prompt_context = build_progression_context(
        project_path,
        project_data,
        next_context,
        recent_text,
        user_request=user_request,
        task_card=base_task,
        option_count=DEFAULT_OPTION_COUNT,
        planning_mode=planning_mode,
    )
    prompt = build_auto_objective_prompt(
        prompt_context,
        recent_text,
        next_context,
        user_request=user_request,
        planning_mode=planning_mode,
    )
    record_context_telemetry(
        project_path,
        "outline",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=planning_mode,
        extra={
            "prompt_type": "auto_objective",
            "target_chapter_number": target_chapter_number,
        },
    )

    log_context = {
        "phase": "outline",
        "project_id": str(project_data["project"].get("project_id") or "").strip(),
        "project_path": str(Path(project_path).resolve()),
        "planning_mode": planning_mode,
        "target_chapter_number": target_chapter_number,
        "prompt_type": "auto_objective",
    }
    request_context = user_request.strip()
    if request_context:
        log_context["user_request"] = request_context[:280]

    try:
        response_text, metadata = generate_text_with_metadata(
            prompt,
            config,
            log_context=log_context,
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
        objective_payload = extract_json_object(
            response_text,
            "Could not parse JSON from auto objective response.",
        )
        objective = str(objective_payload.get("objective", "") or "").strip()
        if not objective:
            raise ValueError("auto objective response missing objective")
    except Exception:
        update_project_stats(project_path, phase="outline", success=False, usage=None)
        raise

    return override_task_card_objective(base_task, objective).get("objective", "")


def generate_progression_options(
    project_path: str,
    config: dict,
    *,
    user_request: str = "",
    objective_override: str = "",
    option_count: int = DEFAULT_OPTION_COUNT,
    runtime_overrides: dict | None = None,
    progress_callback=None,
) -> dict:
    count = validate_option_count(option_count)
    project_data = load_project(project_path)
    planning_mode = resolve_planning_mode(config, project_data)
    target_chapter_number = safe_int(project_data["project"].get("chapter_count"), 0) + 1
    emit_progress(progress_callback, "progression_options_prepare", "正在生成下一章剧情推进选项")
    project_data, next_context = get_next_context_for_mode(
        project_path,
        config,
        planning_mode,
        progress_callback=progress_callback,
    )
    recent_text = get_last_chapter_text(project_path)
    if not recent_text:
        recent_text = "这是开篇前状态。请围绕第一章如何自然开场来给出推进选项。"
    base_task = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=False,
    )
    task_card = override_task_card_objective(base_task, objective_override)
    prompt_context = build_progression_context(
        project_path,
        project_data,
        next_context,
        recent_text,
        user_request=user_request,
        task_card=task_card,
        option_count=count,
        planning_mode=planning_mode,
    )
    prompt = build_progression_options_prompt(
        prompt_context,
        recent_text,
        next_context,
        user_request=user_request,
        option_count=count,
        planning_mode=planning_mode,
    )
    record_context_telemetry(
        project_path,
        "outline",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=planning_mode,
        extra={
            "prompt_type": "progression_options",
            "target_chapter_number": target_chapter_number,
        },
    )

    log_context = {
        "phase": "outline",
        "project_id": str(project_data["project"].get("project_id") or "").strip(),
        "project_path": str(Path(project_path).resolve()),
        "planning_mode": planning_mode,
        "target_chapter_number": target_chapter_number,
        "option_count": count,
    }
    request_context = user_request.strip()
    if request_context:
        log_context["user_request"] = request_context[:280]
    objective_context = str(task_card.get("objective", "") or task_card.get("goal", "") or "").strip()
    if objective_context:
        log_context["objective"] = objective_context[:180]

    try:
        response_text, metadata = generate_text_with_metadata(
            prompt,
            config,
            log_context=log_context,
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
        normalized = normalize_progression_options_response(
            extract_json_object(response_text, "Could not parse JSON from progression options response."),
            count,
        )
    except Exception:
        update_project_stats(project_path, phase="outline", success=False, usage=None)
        raise

    current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
    session_id = f"{utc_now().replace('-', '').replace(':', '').replace('+00:00', 'Z')}_{uuid4().hex[:8]}"
    session = {
        "session_id": session_id,
        "created_at": utc_now(),
        "project_chapter_count": current_chapter_count,
        "target_chapter_number": current_chapter_count + 1,
        "planning_mode": planning_mode,
        "source_user_request": user_request.strip(),
        "objective": objective_context,
        "runtime_overrides": sanitize_runtime_overrides(runtime_overrides),
        "recommended_option_id": normalized["recommended_option_id"],
        "options": normalized["options"] + [build_custom_progression_option()],
        "status": "pending",
        "selection_mode": SELECTION_MODE_MANUAL,
        "selection_origin": SELECTION_ORIGIN_USER,
        "selected_option_id": "",
        "selection_feedback": "",
        "selected_at": "",
        "auto_batch_request": "",
    }
    save_progression_session(project_path, session)
    log_success(
        "progression_options: generated "
        f"{count} model options plus 1 custom blank option for chapter {session['target_chapter_number']}"
    )
    return session


def resolve_progression_selection(
    project_path: str,
    session_id: str,
    option_ref: str,
    *,
    selection_feedback: str = "",
    selection_mode: str = SELECTION_MODE_MANUAL,
    selection_origin: str = SELECTION_ORIGIN_USER,
    auto_batch_request: str = "",
) -> dict:
    session = ensure_fresh_progression_session(project_path, load_progression_session(project_path, session_id))
    if session.get("status") == "stale":
        raise ValueError("当前推进选项已过期，请重新生成推进选项。")

    option_ref = str(option_ref or "").strip()
    if not option_ref:
        raise ValueError("请选择一个推进选项。")

    options = session.get("options") or []
    selected = None
    if option_ref.isdigit():
        index = int(option_ref) - 1
        if 0 <= index < len(options):
            selected = options[index]
    if selected is None:
        for option in options:
            if str(option.get("option_id", "")).strip() == option_ref:
                selected = option
                break
    if selected is None:
        raise ValueError("未找到对应的推进选项，请重新生成推进选项。")
    if selected.get("custom") and not str(selection_feedback or "").strip():
        raise ValueError("选择空白自定义项后，必须填写你自己的创意与想看的情节。")

    normalized_selection_mode = validate_selection_mode(selection_mode)
    normalized_selection_origin = (
        SELECTION_ORIGIN_AUTO if str(selection_origin or "").strip().lower() == SELECTION_ORIGIN_AUTO else SELECTION_ORIGIN_USER
    )
    session["selected_option_id"] = selected["option_id"]
    session["selection_feedback"] = selection_feedback.strip()
    session["status"] = "selected"
    session["selection_mode"] = normalized_selection_mode
    session["selection_origin"] = normalized_selection_origin
    session["selected_at"] = utc_now()
    session["auto_batch_request"] = str(auto_batch_request or session.get("auto_batch_request", "") or "").strip()
    save_progression_session(project_path, session)

    project_data = load_project(project_path)
    planning_mode = str(session.get("planning_mode", "") or "").strip()
    next_context = peek_next_context_for_mode(project_data, planning_mode)
    baseline_task = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=False,
    )
    session_objective = str(session.get("objective", "") or "").strip()
    if session_objective:
        baseline_task = override_task_card_objective(baseline_task, session_objective)
    if selected.get("custom"):
        selected_task_card = build_custom_progression_task_card(
            project_path,
            project_data,
            next_context,
            baseline_task,
            selection_feedback,
            session_id=str(session.get("session_id", "") or "").strip(),
            option_id=selected["option_id"],
            planning_mode=planning_mode,
            baseline_source=str(baseline_task.get("source", "") or "").strip(),
            persist=True,
        )
    else:
        selected_task_card = build_progression_selected_task_card(
            project_path,
            project_data,
            next_context,
            selected,
            baseline_task,
            session_id=str(session.get("session_id", "") or "").strip(),
            option_id=selected["option_id"],
            planning_mode=planning_mode,
            baseline_source=str(baseline_task.get("source", "") or "").strip(),
            selection_feedback=selection_feedback,
            persist=True,
        )
    log_success("progression_options: persisted selected progression task card")

    return {
        "session": session,
        "planning_mode": planning_mode,
        "user_request": "",
        "chapter_outline_override": None,
        "selected_option": selected,
        "task_card": selected_task_card,
    }
