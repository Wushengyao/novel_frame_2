"""CLI entry point for the structured-memory novel writer MVP."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from console_logger import log_error, log_info, log_success
from illustration_manager import illustrate_chapters, illustrate_project_assets
from llm_client import generate_text_with_metadata
from outline_manager import (
    ensure_project_outlines,
    find_next_chapter_context,
    get_outline_status,
    regenerate_chapter_outline,
    regenerate_volume_outline,
    sync_outline_progress,
)
from prompt_builder import build_writer_prompt
from project_manager import (
    create_state_snapshot,
    ensure_state_snapshot,
    get_latest_state_snapshot_chapter,
    get_last_chapter_text,
    init_project,
    load_json,
    load_project,
    normalize_chapter_text,
    rollback_project,
    save_chapter,
    update_project_stats,
)
from state_updater import update_plot_state


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    log_path = jobs_dir / f"illustrate_{_utc_now()}.log"

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


def run_next_chapter(project_path: str, config: dict, user_request: str = "") -> str:
    log_info(f"next_chapter: prepare project={project_path}")
    outlines = ensure_project_outlines(project_path, config, sync_progress=False)
    project_data = load_project(project_path)
    current_chapter_count = int(project_data["project"].get("chapter_count", 0) or 0)
    ensure_state_snapshot(
        project_path,
        chapter_count=current_chapter_count,
        note="pre-write checkpoint",
    )

    next_context = find_next_chapter_context(outlines, current_chapter_count)
    if next_context is None:
        log_error("next_chapter: no outline found for the next chapter")
        raise ValueError("No usable next chapter outline was found. Regenerate chapter outlines first.")

    log_info(
        "next_chapter: writing "
        f"volume={next_context['volume'].get('volume_number', '?')} "
        f"chapter={next_context['chapter'].get('chapter_number', '?')} "
        f"title={next_context['chapter'].get('title', '')}"
    )

    last_chapter = get_last_chapter_text(project_path)
    recent_text = (
        last_chapter[-3000:]
        if last_chapter
        else "This is the opening chapter. Please begin the story naturally."
    )
    prompt = build_writer_prompt(
        project_data,
        recent_text,
        user_request=user_request,
        current_volume_outline=next_context["volume"],
        chapter_outline=next_context["chapter"],
    )

    try:
        log_info("next_chapter: requesting model output")
        response_text, metadata = generate_text_with_metadata(prompt, config)
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
    chapter_path = save_chapter(project_path, chapter_text)
    log_success(f"next_chapter: saved to {chapter_path}")

    log_info("next_chapter: updating plot_state")
    update_plot_state(project_path, chapter_text, config)
    log_info("next_chapter: syncing outline progress")
    sync_outline_progress(project_path)

    snapshot_path = create_state_snapshot(project_path, note="post-write checkpoint")
    log_success(f"next_chapter: snapshot saved to {snapshot_path}")
    return chapter_path


def run_next_chapters(project_path: str, config: dict, count: int, user_request: str = "") -> list[str]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    return [run_next_chapter(project_path, config, user_request=user_request) for _ in range(count)]


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
    print(f"Next Goal: {plot_state.get('next_chapter_goal', '')}")

    outline_status = get_outline_status(project_path)
    if outline_status.get("has_outlines"):
        print(f"Volumes: {outline_status.get('volume_count', 0)}")
        if outline_status.get("chapter_outline_stale"):
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


def _load_runtime_config(project_path: str) -> dict:
    project = load_project(project_path)["project"]
    return project.get("llm_config", {})


def _extract_llm_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    raw = load_json(str(config_file))
    return {
        "model_provider": raw.get("model_provider", "openai_compatible"),
        "model": raw.get("model") or raw.get("model_name", ""),
        "model_name": raw.get("model_name") or raw.get("model", ""),
        "api_base": raw.get("api_base", ""),
        "api_key": raw.get("api_key", ""),
        "temperature": raw.get("temperature", 0.8),
        "max_tokens": raw.get("max_tokens", 4000),
        "timeout": raw.get("timeout", 120),
        "thinking_level": raw.get("thinking_level"),
        "thinking_budget": raw.get("thinking_budget"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel writer MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a novel project")
    init_parser.add_argument("--config", required=True, help="Path to config.json")

    next_parser = subparsers.add_parser("next", help="Generate the next chapter")
    next_parser.add_argument("--project", required=True, help="Path to novel_project")
    next_parser.add_argument("--config", help="Optional config.json to override saved LLM settings")
    next_parser.add_argument("--count", type=int, default=1, help="Generate multiple chapters sequentially")
    next_parser.add_argument("--user-request", default="", help="Optional user preference for this batch")
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

        config = _extract_llm_config(args.config) if args.config else _load_runtime_config(args.project)
        chapter_paths = run_next_chapters(
            args.project,
            config,
            args.count,
            user_request=args.user_request,
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
        config = _extract_llm_config(args.config) if args.config else None
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
            f"snapshots={len(removed.get('snapshots', []))}"
        )
        _print_status(args.project)
        return

    if args.command == "outline":
        log_info(f"cli: outline stage={args.stage}")
        config = _extract_llm_config(args.config) if args.config else _load_runtime_config(args.project)
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

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
