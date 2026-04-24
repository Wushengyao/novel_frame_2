"""CLI entry point for the structured-memory novel writer MVP."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chapter_context import get_next_context_for_mode, peek_next_context_for_mode
from common_utils import emit_progress, utc_now
from console_logger import log_error, log_info, log_success, log_warning
from context_builder import build_writer_context, resolve_effective_chapter_task
from audiobook_manager import chapter_refs_for_all, generate_audiobook_chapters
from illustration_manager import illustrate_chapters, illustrate_project_assets
from llm_client import generate_text_with_metadata
from outline_manager import (
    find_next_chapter_context,
    get_outline_status,
    regenerate_chapter_outline,
    regenerate_volume_outline,
    sync_outline_progress,
)
from progression_manager import (
    CUSTOM_PROGRESSION_OPTION_ID,
    SELECTION_MODE_RECOMMENDED,
    auto_select_progression_option,
    generate_auto_chapter_objective,
    generate_progression_options,
    resolve_progression_selection,
    validate_selection_mode,
)
from prompt_builder import build_writer_prompt
from quality_manager import (
    generate_craft_brief,
    normalize_quality_config,
    quality_mode_allows_rewrite,
    quality_mode_uses_craft_brief,
    quality_mode_uses_review,
    quality_review_passed,
    review_chapter_draft,
    rewrite_chapter_draft,
)
from project_manager import (
    PLANNING_MODE_CHAPTER,
    PLANNING_MODE_NONE,
    PLANNING_MODE_VOLUME,
    create_state_snapshot,
    ensure_state_snapshot,
    get_latest_state_snapshot_chapter,
    get_last_chapter_text,
    init_project,
    load_json,
    load_project,
    normalize_chapter_text,
    normalize_planning_mode,
    record_context_telemetry,
    rollback_project,
    save_chapter,
    update_project_stats,
)
from runtime_config import (
    REVIEW_MODES,
    WRITING_QUALITY_HIGH,
    WRITING_QUALITY_MODES,
    extract_llm_config,
    load_runtime_config,
)
from state_updater import update_plot_state
from version import APP_NAME, DISPLAY_VERSION


def _add_illustration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, help="Max concurrent illustration jobs")
    parser.add_argument("--comfyui-api-base", help="ComfyUI API base URL")
    parser.add_argument("--comfyui-root", help="Optional ComfyUI root directory")
    parser.add_argument("--checkpoint", help="Checkpoint name for CheckpointLoaderSimple")
    parser.add_argument("--workflow-template", help="Optional ComfyUI workflow template JSON path")
    parser.add_argument("--width", type=int, help="Illustration width")
    parser.add_argument("--height", type=int, help="Illustration height")
    parser.add_argument("--steps", type=int, help="Sampling steps")
    parser.add_argument("--cfg", type=float, help="CFG scale")
    parser.add_argument("--sampler-name", help="Sampler name")
    parser.add_argument("--scheduler", help="Scheduler name")
    parser.add_argument("--seed", type=int, help="Optional fixed illustration seed")


def _extract_illustration_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "comfyui_api_base": getattr(args, "comfyui_api_base", None),
        "comfyui_root": getattr(args, "comfyui_root", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "workflow_template": getattr(args, "workflow_template", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "steps": getattr(args, "steps", None),
        "cfg": getattr(args, "cfg", None),
        "sampler_name": getattr(args, "sampler_name", None),
        "scheduler": getattr(args, "scheduler", None),
        "seed": getattr(args, "seed", None),
    }
    return {key: value for key, value in mapping.items() if value not in (None, "")}


def _extract_illustration_workers(args: argparse.Namespace) -> int | None:
    workers = getattr(args, "workers", None)
    if workers is None:
        return None
    return max(1, int(workers))


def _launch_background_illustration_job(
    project_path: str,
    *,
    chapter_refs: list[str],
    config_path: str | None = None,
    user_request: str = "",
    force: bool = False,
    overrides: dict | None = None,
    max_workers: int | None = None,
) -> dict:
    jobs_dir = Path(project_path) / "illustrations" / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    log_path = jobs_dir / f"illustrate_{utc_now().replace(':', '').replace('-', '').replace('+00:00', 'Z')}.log"

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "illustrate",
        "--project",
        str(project_path),
    ]
    for chapter_ref in chapter_refs:
        command.extend(["--chapter-ref", str(chapter_ref)])
    if config_path:
        command.extend(["--config", str(config_path)])
    if user_request:
        command.extend(["--user-request", user_request])
    if force:
        command.append("--force")
    if max_workers:
        command.extend(["--workers", str(max_workers)])

    flag_map = {
        "comfyui_api_base": "--comfyui-api-base",
        "comfyui_root": "--comfyui-root",
        "checkpoint": "--checkpoint",
        "workflow_template": "--workflow-template",
        "width": "--width",
        "height": "--height",
        "steps": "--steps",
        "cfg": "--cfg",
        "sampler_name": "--sampler-name",
        "scheduler": "--scheduler",
        "seed": "--seed",
    }
    for key, value in (overrides or {}).items():
        flag = flag_map.get(key)
        if flag:
            command.extend([flag, str(value)])

    with log_path.open("w", encoding="utf-8") as log_file:
        popen_kwargs = {
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "cwd": str(Path(project_path).resolve()),
            "close_fds": True,
        }
        if os.name == "nt":
            creationflags = 0
            creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            process = subprocess.Popen(command, creationflags=creationflags, **popen_kwargs)
        else:
            process = subprocess.Popen(command, start_new_session=True, **popen_kwargs)

    return {
        "pid": process.pid,
        "log_path": str(log_path),
    }


def run_next_chapter(
    project_path: str,
    config: dict,
    user_request: str = "",
    *,
    chapter_outline_override: dict | None = None,
    planning_mode: str | None = None,
    log_context: dict[str, object] | None = None,
    progress_callback=None,
) -> str:
    log_info(f"next_chapter: prepare project={project_path}")
    effective_mode = normalize_planning_mode(planning_mode or config.get("planning_mode"))
    emit_progress(progress_callback, "chapter_prepare", f"Preparing next chapter with planning mode: {effective_mode}")
    project_data, next_context = get_next_context_for_mode(
        project_path,
        config,
        effective_mode,
        progress_callback=progress_callback,
    )
    current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
    emit_progress(progress_callback, "chapter_snapshot_prepare", "Saving pre-write snapshot")
    ensure_state_snapshot(
        project_path,
        chapter_count=current_chapter_count,
        note="pre-write checkpoint",
    )

    if effective_mode == PLANNING_MODE_CHAPTER and chapter_outline_override:
        merged_chapter = deepcopy(next_context["chapter"])
        merged_chapter.update(chapter_outline_override)
        next_context["chapter"] = merged_chapter

    target_chapter_number = next_context["chapter"].get("chapter_number", current_chapter_count + 1)
    resolved_project_id = str(project_data["project"].get("project_id") or "").strip()
    log_context_payload = {
        "phase": "writer",
        "project_id": resolved_project_id,
        "project_path": str(Path(project_path).resolve()),
        "planning_mode": effective_mode,
        "target_chapter_number": target_chapter_number,
        "source": "run_next_chapter",
    }
    if user_request:
        log_context_payload["user_request"] = user_request[:280]
    if chapter_outline_override:
        log_context_payload["has_chapter_outline_override"] = True
    if log_context:
        log_context_payload.update(log_context)

    log_info(
        "next_chapter: writing "
        f"mode={effective_mode} "
        f"volume={next_context['volume'].get('volume_number', '?')} "
        f"chapter={target_chapter_number} "
        f"title={next_context['chapter'].get('title', '')}"
    )

    last_chapter = get_last_chapter_text(project_path)
    writing_quality_mode, review_mode = normalize_quality_config(config)
    prompt_context = build_writer_context(
        project_path,
        project_data,
        next_context,
        last_chapter,
        user_request=user_request,
        planning_mode=effective_mode,
    )
    if quality_mode_uses_craft_brief(writing_quality_mode):
        craft_brief = generate_craft_brief(
            project_path,
            prompt_context,
            config,
            log_context=log_context_payload,
            progress_callback=progress_callback,
        )
        prompt_context = build_writer_context(
            project_path,
            project_data,
            next_context,
            last_chapter,
            user_request=user_request,
            planning_mode=effective_mode,
            craft_brief=craft_brief,
        )
    prompt = build_writer_prompt(prompt_context)
    record_context_telemetry(
        project_path,
        "writer",
        prompt_chars=len(prompt),
        section_chars=prompt_context.get("section_chars"),
        planning_mode=effective_mode,
        extra={
            "target_chapter_number": prompt_context.get("task_card", {}).get("chapter_number"),
            "prompt_soft_budget": 7000,
            "prompt_hard_budget": 8000,
            "writing_quality_mode": writing_quality_mode,
            "review_mode": review_mode,
        },
    )

    try:
        log_info("next_chapter: requesting model output")
        emit_progress(progress_callback, "chapter_write", "Generating chapter text")
        response_text, metadata = generate_text_with_metadata(
            prompt,
            config,
            log_context=log_context_payload,
        )
    except Exception:
        update_project_stats(project_path, phase="writer", success=False, usage=None)
        log_error("next_chapter: writer request failed")
        raise

    update_project_stats(
        project_path,
        phase="writer",
        success=True,
        usage=metadata.get("usage"),
    )
    chapter_text = normalize_chapter_text(response_text)
    if quality_mode_uses_review(writing_quality_mode):
        review = review_chapter_draft(
            project_path,
            prompt_context,
            chapter_text,
            config,
            attempt=1,
            strict=writing_quality_mode == WRITING_QUALITY_HIGH,
            log_context=log_context_payload,
            progress_callback=progress_callback,
        )
        if quality_mode_allows_rewrite(writing_quality_mode, review_mode) and not quality_review_passed(review):
            try:
                rewritten_text = rewrite_chapter_draft(
                    project_path,
                    prompt_context,
                    chapter_text,
                    review,
                    config,
                    log_context=log_context_payload,
                    progress_callback=progress_callback,
                )
                chapter_text = normalize_chapter_text(rewritten_text)
            except Exception as exc:  # pragma: no cover - keep original draft if rewrite fails
                log_warning(f"rewrite: failed; keeping original draft. reason={exc}")
    emit_progress(progress_callback, "chapter_save", "Saving chapter file")
    chapter_path = save_chapter(project_path, chapter_text)
    log_success(f"next_chapter: saved to {chapter_path}")

    log_info("next_chapter: updating plot_state")
    emit_progress(progress_callback, "chapter_summary", "Updating plot state")
    update_plot_state(project_path, chapter_text, config, progress_callback=progress_callback)
    if effective_mode != PLANNING_MODE_NONE:
        log_info("next_chapter: syncing outline progress")
        emit_progress(progress_callback, "chapter_outline_sync", "Syncing outline progress")
        sync_outline_progress(project_path)

    emit_progress(progress_callback, "chapter_snapshot", "Saving post-write snapshot")
    snapshot_path = create_state_snapshot(project_path, note="post-write checkpoint")
    log_success(f"next_chapter: snapshot saved to {snapshot_path}")
    emit_progress(progress_callback, "chapter_done", f"Chapter completed: {Path(chapter_path).name}")
    return chapter_path


def run_next_chapter_from_progression(
    project_path: str,
    config: dict,
    *,
    progression_session: str,
    progression_option: str,
    progression_feedback: str = "",
    progress_callback=None,
) -> str:
    selection = resolve_progression_selection(
        project_path,
        progression_session,
        progression_option,
        selection_feedback=progression_feedback,
    )
    return run_next_chapter(
        project_path,
        config,
        user_request=selection["user_request"],
        chapter_outline_override=None,
        planning_mode=selection.get("planning_mode"),
        log_context={
            "phase": "writer",
            "source": "run_next_chapter_from_progression",
            "progression_session": progression_session,
            "progression_option": progression_option,
            "has_progression_feedback": bool(str(progression_feedback or "").strip()),
            "selection_feedback": (progression_feedback or "").strip()[:120],
        },
        progress_callback=progress_callback,
    )


def run_next_chapters(
    project_path: str,
    config: dict,
    count: int,
    user_request: str = "",
    *,
    selection_mode: str = SELECTION_MODE_RECOMMENDED,
    runtime_overrides: dict | None = None,
    progress_callback=None,
) -> list[str]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    normalized_selection_mode = validate_selection_mode(selection_mode, allow_manual=False)
    chapter_paths = []
    for index in range(count):
        emit_progress(
            progress_callback,
            "chapter_batch",
            f"Writing chapter {index + 1}/{count}",
            current=index,
            total=count,
        )
        planning_mode = normalize_planning_mode(config.get("planning_mode"))
        objective_override = ""
        if planning_mode == PLANNING_MODE_NONE:
            objective_override = generate_auto_chapter_objective(
                project_path,
                config,
                user_request=user_request,
                progress_callback=progress_callback,
            )
        session = generate_progression_options(
            project_path,
            config,
            user_request=user_request,
            objective_override=objective_override,
            runtime_overrides=runtime_overrides,
            progress_callback=progress_callback,
        )
        option_ref = auto_select_progression_option(session, normalized_selection_mode)
        selection = resolve_progression_selection(
            project_path,
            str(session.get("session_id", "") or "").strip(),
            option_ref,
            selection_mode=normalized_selection_mode,
            selection_origin="auto",
            auto_batch_request=user_request,
        )
        chapter_paths.append(
            run_next_chapter(
                project_path,
                config,
                user_request="",
                chapter_outline_override=None,
                planning_mode=selection.get("planning_mode"),
                log_context={
                    "phase": "writer",
                    "source": "run_next_chapters_auto",
                    "auto_selection_mode": normalized_selection_mode,
                    "progression_session": str(session.get("session_id", "") or "").strip(),
                    "progression_option": option_ref,
                    "auto_batch_request": user_request[:120],
                },
                progress_callback=progress_callback,
            )
        )
        emit_progress(
            progress_callback,
            "chapter_batch_done",
            f"Chapter {index + 1}/{count} completed",
            current=index + 1,
            total=count,
        )
    return chapter_paths


def _print_status(project_path: str) -> None:
    project_data = load_project(project_path)
    project = project_data["project"]
    plot_state = project_data["plot_state"]
    llm_config = project.get("llm_config", {})
    stats = project.get("stats") or {}
    total_stats = stats.get("total") or {}

    print(f"Project ID: {project.get('project_id', 'legacy_project')}")
    print(f"Project Name: {project.get('name', '')}")
    print(f"Chapter Count: {project.get('chapter_count', 0)}")
    print(f"Updated At: {project.get('updated_at', '')}")

    latest_snapshot = get_latest_state_snapshot_chapter(project_path)
    print(f"Latest Snapshot: {latest_snapshot if latest_snapshot is not None else 'none'}")
    print(f"Provider: {llm_config.get('model_provider', '')}")
    print(f"Model: {llm_config.get('model_name') or llm_config.get('model', '')}")
    print(f"Planning Mode: {normalize_planning_mode(project.get('planning_mode'))}")
    print(f"Writing Quality Mode: {llm_config.get('writing_quality_mode', 'balanced')}")
    print(f"Review Mode: {llm_config.get('review_mode', 'auto')}")
    print(
        "Requests: "
        f"{total_stats.get('requests', 0)} "
        f"(success={total_stats.get('successes', 0)}, failure={total_stats.get('failures', 0)})"
    )
    print(
        "Tokens: "
        f"prompt={total_stats.get('prompt_tokens', 0)}, "
        f"completion={total_stats.get('completion_tokens', 0)}, "
        f"total={total_stats.get('total_tokens', 0)}"
    )

    extras = []
    if total_stats.get("cached_tokens", 0):
        extras.append(f"cached={total_stats.get('cached_tokens', 0)}")
    if total_stats.get("reasoning_tokens", 0):
        extras.append(f"reasoning={total_stats.get('reasoning_tokens', 0)}")
    if total_stats.get("thought_tokens", 0):
        extras.append(f"thoughts={total_stats.get('thought_tokens', 0)}")
    if extras:
        print("Extra Tokens: " + ", ".join(extras))

    print(f"Current Location: {plot_state.get('current_location', '')}")
    print(f"Current Time: {plot_state.get('current_time', '')}")
    planning_mode = normalize_planning_mode(project.get("planning_mode"))
    next_context = peek_next_context_for_mode(project_data, planning_mode)
    effective_task = resolve_effective_chapter_task(
        project_path,
        project_data,
        next_context,
        planning_mode=planning_mode,
        persist=False,
    )
    print(f"Current Chapter Objective: {effective_task.get('objective', '') or effective_task.get('goal', '')}")
    print(f"Current Chapter Plan: {effective_task.get('plan_summary', '') or effective_task.get('summary', '')}")
    if effective_task.get("source"):
        print(f"Current Task Source: {effective_task.get('source', '')}")
    if effective_task.get("volume_goal"):
        print(f"Volume Goal: {effective_task.get('volume_goal', '')}")
    print(f"Live Next Goal: {plot_state.get('next_chapter_goal', '')}")

    outline_status = get_outline_status(project_path)
    if outline_status.get("has_outlines"):
        print(f"Volumes: {outline_status.get('volume_count', 0)}")
        if (
            planning_mode == PLANNING_MODE_CHAPTER
            and outline_status.get("chapter_outline_stale")
        ):
            print("Chapter outlines are stale.")
        next_context = outline_status.get("next_context")
        if next_context:
            volume = next_context.get("volume") or {}
            chapter = next_context.get("chapter") or {}
            print(
                "Next Outline: "
                f"Vol.{volume.get('volume_number', '?')} {volume.get('title', '')} / "
                f"Ch.{chapter.get('chapter_number', '?')} {chapter.get('title', '')}"
            )
            print(f"Next Summary: {chapter.get('summary', '')}")

    open_threads = plot_state.get("open_threads", [])
    if open_threads:
        print("Open Threads:")
        for item in open_threads:
            print(f"- {item}")
def main() -> None:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} CLI")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {DISPLAY_VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a novel project")
    init_parser.add_argument("--config", required=True, help="Path to config.json")

    next_parser = subparsers.add_parser("next", help="Generate the next chapter")
    next_parser.add_argument("--project", required=True, help="Path to novel_project")
    next_parser.add_argument("--config", help="Optional config.json to override saved LLM settings")
    next_parser.add_argument("--count", type=int, default=1, help="Generate multiple chapters sequentially")
    next_parser.add_argument("--user-request", default="", help="Optional user preference for this batch")
    next_parser.add_argument(
        "--selection-mode",
        default=SELECTION_MODE_RECOMMENDED,
        choices=("recommended", "random"),
        help="How automatic continuation chooses a plan for each chapter",
    )
    next_parser.add_argument("--progression-session", default="", help="Guided progression session id")
    next_parser.add_argument("--progression-option", default="", help="Guided progression option number or option_id")
    next_parser.add_argument("--progression-feedback", default="", help="Optional refinement for the selected guided option")
    next_parser.add_argument(
        "--planning-mode",
        choices=(PLANNING_MODE_NONE, PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER),
        help="Override planning mode for this run",
    )
    next_parser.add_argument(
        "--writing-quality-mode",
        choices=tuple(sorted(WRITING_QUALITY_MODES)),
        help="Override writing quality pipeline for this run",
    )
    next_parser.add_argument(
        "--review-mode",
        choices=tuple(sorted(REVIEW_MODES)),
        help="Override quality review behavior for this run",
    )
    next_parser.add_argument("--illustrate", action="store_true", help="Generate chapter illustrations")
    next_parser.add_argument("--illustration-request", default="", help="Optional extra art direction")
    next_parser.add_argument(
        "--force-illustration",
        action="store_true",
        help="Regenerate illustrations even if metadata already exists",
    )
    next_parser.add_argument(
        "--illustrate-blocking",
        action="store_true",
        help="Wait for illustrations instead of launching them in the background",
    )
    _add_illustration_arguments(next_parser)

    illustrate_parser = subparsers.add_parser("illustrate", help="Generate illustrations for chapters")
    illustrate_parser.add_argument("--project", required=True, help="Path to novel_project")
    illustrate_parser.add_argument(
        "--chapter",
        default="latest",
        help="Chapter slug/path to illustrate, default latest. Ignored when --all or --chapter-ref is used",
    )
    illustrate_parser.add_argument(
        "--chapter-ref",
        action="append",
        help="Explicit chapter slug/path. Can be passed multiple times",
    )
    illustrate_parser.add_argument("--all", action="store_true", help="Generate illustrations for all chapters")
    illustrate_parser.add_argument("--config", help="Optional config.json for LLM prompt refinement")
    illustrate_parser.add_argument("--user-request", default="", help="Optional extra art direction")
    illustrate_parser.add_argument("--force", action="store_true", help="Regenerate existing illustrations")
    _add_illustration_arguments(illustrate_parser)

    assets_parser = subparsers.add_parser("illustrate-assets", help="Generate cover and character portraits")
    assets_parser.add_argument("--project", required=True, help="Path to novel_project")
    assets_parser.add_argument("--user-request", default="", help="Optional extra art direction")
    assets_parser.add_argument("--force", action="store_true", help="Regenerate existing assets")
    _add_illustration_arguments(assets_parser)

    audiobook_parser = subparsers.add_parser("audiobook", help="Generate audiobook WAV files for chapters")
    audiobook_parser.add_argument("--project", required=True, help="Path to novel_project")
    audiobook_parser.add_argument("--chapter", default="latest", help="Chapter slug/path to synthesize")
    audiobook_parser.add_argument(
        "--chapter-ref",
        action="append",
        help="Explicit chapter slug/path. Can be passed multiple times",
    )
    audiobook_parser.add_argument("--all", action="store_true", help="Generate audiobook WAV files for all chapters")
    audiobook_parser.add_argument("--force", action="store_true", help="Regenerate existing audiobook files")
    audiobook_parser.add_argument("--narrator-preset", default="", help="Narrator preset id to use")

    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project", required=True, help="Path to novel_project")

    rollback_parser = subparsers.add_parser("rollback", help="Rollback project state to a previous chapter")
    rollback_parser.add_argument("--project", required=True, help="Path to novel_project")
    rollback_parser.add_argument("--to-chapter", required=True, type=int, help="Keep chapters up to this number")

    outline_parser = subparsers.add_parser("outline", help="Generate or regenerate outlines")
    outline_parser.add_argument("--project", required=True, help="Path to novel_project")
    outline_parser.add_argument("--config", help="Optional config.json to override saved LLM settings")
    outline_parser.add_argument(
        "--stage",
        choices=("volumes", "chapters", "all"),
        default="all",
        help="Which outline stage to regenerate",
    )
    outline_parser.add_argument("--volume", type=int, help="Optional volume number for chapter outline generation")
    outline_parser.add_argument("--user-request", default="", help="Optional extra plot requirements")

    options_parser = subparsers.add_parser("options", help="Generate guided next-chapter progression options")
    options_parser.add_argument("--project", required=True, help="Path to novel_project")
    options_parser.add_argument("--config", help="Optional config.json to override saved LLM settings")
    options_parser.add_argument("--objective", default="", help="Optional override for the next chapter objective before generating plans")
    options_parser.add_argument("--user-request", default="", help="Optional preference for guided options")
    options_parser.add_argument("--option-count", type=int, default=4, choices=(3, 4, 5), help="How many model-generated options to generate; a blank custom option is always added")
    options_parser.add_argument(
        "--planning-mode",
        choices=(PLANNING_MODE_NONE, PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER),
        help="Override planning mode for this run",
    )

    args = parser.parse_args()

    if args.command == "init":
        log_info("cli: init")
        project_path = init_project(args.config)
        print(f"Project initialized: {project_path}")
        return

    if args.command == "next":
        log_info("cli: next")
        if args.count < 1:
            parser.error("--count must be at least 1")

        has_progression = bool(
            args.progression_session or args.progression_option or args.progression_feedback
        )
        if has_progression:
            if args.count != 1:
                parser.error("guided progression only supports --count 1; use direct next for batch writing")
            if not args.progression_session or not args.progression_option:
                parser.error("guided progression requires both --progression-session and --progression-option")
            if args.user_request:
                parser.error("--user-request cannot be combined with guided progression selection")

        config = extract_llm_config(args.config) if args.config else load_runtime_config(args.project)
        if args.planning_mode:
            config["planning_mode"] = args.planning_mode
        if args.writing_quality_mode:
            config["writing_quality_mode"] = args.writing_quality_mode
        if args.review_mode:
            config["review_mode"] = args.review_mode
        runtime_overrides = {
            key: value
            for key, value in {
                "planning_mode": args.planning_mode,
                "writing_quality_mode": args.writing_quality_mode,
                "review_mode": args.review_mode,
            }.items()
            if value
        }
        if has_progression:
            chapter_paths = [
                run_next_chapter_from_progression(
                    args.project,
                    config,
                    progression_session=args.progression_session,
                    progression_option=args.progression_option,
                    progression_feedback=args.progression_feedback,
                )
            ]
        else:
            chapter_paths = run_next_chapters(
                args.project,
                config,
                args.count,
                user_request=args.user_request,
                selection_mode=args.selection_mode,
                runtime_overrides=runtime_overrides or None,
            )
        print(f"Generated chapters: {len(chapter_paths)}")
        for chapter_path in chapter_paths:
            print(f"- {chapter_path}")

        if args.illustrate:
            illustration_workers = _extract_illustration_workers(args)
            illustration_overrides = _extract_illustration_overrides(args)
            if args.illustrate_blocking:
                results = illustrate_chapters(
                    args.project,
                    chapter_refs=chapter_paths,
                    llm_config=config,
                    user_request=args.illustration_request,
                    force=args.force_illustration,
                    overrides=illustration_overrides,
                    max_workers=illustration_workers,
                )
                for result in results:
                    state = "reused" if result.get("reused") else "generated"
                    print(f"{state}: {result.get('chapter_slug', '')}")
                    for image in result.get("images", []):
                        print(f"- {image.get('relative_path', '')}")
            else:
                job = _launch_background_illustration_job(
                    args.project,
                    chapter_refs=chapter_paths,
                    config_path=args.config,
                    user_request=args.illustration_request,
                    force=args.force_illustration,
                    overrides=illustration_overrides,
                    max_workers=illustration_workers,
                )
                print(f"Illustration job started in background. pid={job.get('pid', '')}")
                print(f"Illustration log: {job.get('log_path', '')}")
        return

    if args.command == "illustrate":
        log_info("cli: illustrate")
        config = extract_llm_config(args.config) if args.config else None
        chapter_refs = list(args.chapter_ref or [])
        if args.all:
            chapters_dir = Path(args.project) / "chapters"
            chapter_refs = [str(path) for path in sorted(chapters_dir.glob("chapter_*.md"))]
        elif not chapter_refs:
            chapter_refs = [args.chapter]

        results = illustrate_chapters(
            args.project,
            chapter_refs=chapter_refs,
            llm_config=config,
            user_request=args.user_request,
            force=args.force,
            overrides=_extract_illustration_overrides(args),
            max_workers=_extract_illustration_workers(args),
        )
        print(f"Processed illustration chapters: {len(results)}")
        for result in results:
            state = "reused" if result.get("reused") else "generated"
            print(f"{state}: {result.get('chapter_slug', '')}")
            for image in result.get("images", []):
                print(f"- {image.get('relative_path', '')}")
        return

    if args.command == "illustrate-assets":
        log_info("cli: illustrate-assets")
        result = illustrate_project_assets(
            args.project,
            user_request=args.user_request,
            force=args.force,
            overrides=_extract_illustration_overrides(args),
        )
        cover = result.get("cover") or {}
        print(f"Cover: {'reused' if cover.get('reused') else 'generated'}")
        for image in cover.get("images", []):
            print(f"- {image.get('relative_path', '')}")
        portraits = result.get("portraits") or []
        print(f"Character portraits: {len(portraits)}")
        for portrait in portraits:
            state = "reused" if portrait.get("reused") else "generated"
            print(f"{state}: {portrait.get('character_name', portrait.get('asset_slug', ''))}")
            for image in portrait.get("images", []):
                print(f"- {image.get('relative_path', '')}")
        return

    if args.command == "audiobook":
        log_info("cli: audiobook")
        chapter_refs = list(args.chapter_ref or [])
        if args.all:
            chapter_refs = chapter_refs_for_all(args.project)
        elif not chapter_refs:
            chapter_refs = [args.chapter]
        results = generate_audiobook_chapters(
            args.project,
            chapter_refs=chapter_refs,
            force=args.force,
            narrator_preset=args.narrator_preset,
        )
        print(f"Processed audiobook chapters: {len(results)}")
        for result in results:
            state = "reused" if result.get("reused") else "generated"
            print(f"{state}: {result.get('chapter_slug', '')}")
            if result.get("combined_audio"):
                print(f"- {result.get('combined_audio', '')}")
        return

    if args.command == "status":
        log_info("cli: status")
        _print_status(args.project)
        return

    if args.command == "rollback":
        log_info(f"cli: rollback target={args.to_chapter}")
        if args.to_chapter < 0:
            parser.error("--to-chapter must be at least 0")
        result = rollback_project(args.project, args.to_chapter)
        print(
            f"Rollback complete: {result.get('current_chapter_count', 0)} -> "
            f"{result.get('target_chapter_count', 0)}"
        )
        print(f"Restore source: {result.get('restore_source', '')}")
        print(f"Snapshot: {result.get('snapshot_path', '')}")
        removed = result.get("removed") or {}
        print(
            "Removed: "
            f"chapters={len(removed.get('chapters', []))}, "
            f"summaries={len(removed.get('summaries', []))}, "
            f"illustrations={len(removed.get('illustrations', []))}, "
            f"audiobook={len(removed.get('audiobook', []))}, "
            f"snapshots={len(removed.get('snapshots', []))}"
        )
        _print_status(args.project)
        return

    if args.command == "outline":
        log_info(f"cli: outline stage={args.stage}")
        config = extract_llm_config(args.config) if args.config else load_runtime_config(args.project)
        if args.stage in {"volumes", "all"}:
            outlines = regenerate_volume_outline(
                args.project,
                config,
                user_request=args.user_request,
            )
            print(f"Volume outlines generated: {len(outlines.get('volumes', []))}")
        if args.stage in {"chapters", "all"}:
            outlines = regenerate_chapter_outline(
                args.project,
                config,
                volume_number=args.volume if args.stage == "chapters" else None,
                user_request=args.user_request,
            )
            target = f"volume {args.volume}" if args.volume else "all volumes"
            print(f"Chapter outlines generated for {target}")
            next_context = find_next_chapter_context(
                outlines,
                int(load_project(args.project)["project"].get("chapter_count", 0) or 0),
            )
            if next_context:
                volume = next_context["volume"]
                chapter = next_context["chapter"]
                print(
                    "Next chapter aligned to "
                    f"Vol.{volume.get('volume_number', '?')} {volume.get('title', '')} / "
                    f"Ch.{chapter.get('chapter_number', '?')} {chapter.get('title', '')}"
                )
        return

    if args.command == "options":
        log_info("cli: options")
        config = extract_llm_config(args.config) if args.config else load_runtime_config(args.project)
        if args.planning_mode:
            config["planning_mode"] = args.planning_mode
        session = generate_progression_options(
            args.project,
            config,
            objective_override=args.objective,
            user_request=args.user_request,
            option_count=args.option_count,
            runtime_overrides={"planning_mode": args.planning_mode} if args.planning_mode else None,
        )
        print(f"Session ID: {session.get('session_id', '')}")
        print(f"Target Chapter: {session.get('target_chapter_number', '')}")
        print(f"Objective: {session.get('objective', '')}")
        print(f"Recommended Option: {session.get('recommended_option_id', '')}")
        for index, option in enumerate(session.get("options", []), start=1):
            if option.get("custom"):
                marker = " [custom]"
            else:
                marker = " [recommended]" if option.get("recommended") else ""
            print(f"Option {index} [{option.get('option_id', '')}]{marker}: {option.get('title', '')}")
            print(f"  Plan Summary: {option.get('plan_summary', '') or option.get('summary', '')}")
            print(f"  Plan Steps: {'; '.join(option.get('plan_steps', []) or option.get('key_events', []))}")
            print(f"  Plan Guidance: {option.get('plan_guidance', '') or option.get('writer_guidance', '')}")
        print(
            "Tip: choose "
            f"`{CUSTOM_PROGRESSION_OPTION_ID}`"
            " (or its option number) and pass `--progression-feedback` to write a fully custom chapter plan."
        )
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
