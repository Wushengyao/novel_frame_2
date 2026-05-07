"""Agentic workflow entry points built on the existing deterministic workflows."""

from __future__ import annotations

from pathlib import Path

from agent_runtime import SkillResult, WorkflowAgent
from common_utils import emit_progress, utc_now
from outline_manager import regenerate_chapter_outline, regenerate_volume_outline
from progression_manager import generate_progression_options
from project_manager import (
    PLANNING_MODE_CHAPTER,
    PLANNING_MODE_VOLUME,
    STORY_SETUP_FILENAME,
    _build_author_intent_from_project,
    _build_llm_config,
    _build_persisted_llm_config,
    _build_project_id,
    _build_project_stats,
    _ensure_project_subdirs,
    _expert_mode_enabled,
    _generate_initial_story_data,
    _normalize_initial_plot_state,
    _resolve_project_path,
    create_state_snapshot,
    ensure_reader_setup,
    load_json,
    normalize_planning_mode,
    regenerate_initial_project,
    save_json,
)
from workflow_modes import normalize_workflow_mode


def init_project_agentic(config_path: str, progress_callback=None) -> str:
    config_file = Path(config_path).resolve()
    config = load_json(str(config_file))
    config["workflow_mode"] = normalize_workflow_mode(config.get("workflow_mode"))
    project_id = config.get("project_id") or _build_project_id()
    project_path = _resolve_project_path(config_file, config, project_id)
    config["project_path"] = str(project_path.resolve())
    if _expert_mode_enabled(config):
        config["log_llm_payload"] = True

    emit_progress(progress_callback, "init_dirs", "Creating project directories")
    project_path.mkdir(parents=True, exist_ok=True)
    _ensure_project_subdirs(project_path)
    agent = WorkflowAgent(project_path, "init_project", progress_callback=progress_callback)

    try:
        generated_result = agent.run_skill(
            "init.generate_story_data",
            lambda: SkillResult.ok(
                value=_generate_initial_story_data(config, progress_callback=progress_callback),
                artifacts={
                    "story_setup": str(project_path / STORY_SETUP_FILENAME),
                    "world": str(project_path / "world.json"),
                    "characters": str(project_path / "characters.json"),
                    "plot_state": str(project_path / "plot_state.json"),
                    "style": str(project_path / "style.json"),
                },
                message="Initial story data generated",
            ),
            inputs={"project_id": project_id, "planning_mode": normalize_planning_mode(config.get("planning_mode"))},
        )
        generated_data, init_meta = generated_result.value
        story_setup = generated_data.get("story_setup") or {}
        world = generated_data["world"]
        characters = generated_data["characters"]
        plot_state = _normalize_initial_plot_state(generated_data["plot_state"])
        style = generated_data["style"]
        author_intent = _build_author_intent_from_project(
            {
                "story_request": config.get("story_request", ""),
                "description": config.get("project_description", "Structured-memory novel writing project."),
            },
            world,
            style,
            plot_state,
        )
        project_data = {
            "project_id": project_id,
            "name": config.get("project_name", "Novel Project"),
            "description": config.get("project_description", "Structured-memory novel writing project."),
            "project_path": str(project_path),
            "story_request": config.get("story_request", ""),
            "planning_mode": normalize_planning_mode(config.get("planning_mode")),
            "workflow_mode": normalize_workflow_mode(config.get("workflow_mode")),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "chapter_count": 0,
            "init": init_meta,
            "stats": init_meta.get("stats") or _build_project_stats(),
            "llm_config": _build_persisted_llm_config(config),
        }

        def write_project_files() -> SkillResult:
            save_json(str(project_path / "project.json"), project_data)
            save_json(str(project_path / STORY_SETUP_FILENAME), story_setup or {"world": world, "characters": characters})
            save_json(str(project_path / "world.json"), world)
            save_json(str(project_path / "characters.json"), characters)
            save_json(str(project_path / "plot_state.json"), plot_state)
            save_json(str(project_path / "style.json"), style)
            save_json(str(project_path / "author_intent.json"), author_intent)
            ensure_reader_setup(
                str(project_path),
                {
                    "project": project_data,
                    "world": world,
                    "characters": characters,
                    "plot_state": plot_state,
                    "style": style,
                    "author_intent": author_intent,
                },
            )
            return SkillResult.ok(
                artifacts={
                    "project": str(project_path / "project.json"),
                    "reader_setup": str(project_path / "reader_setup.md"),
                    "author_intent": str(project_path / "author_intent.json"),
                },
                message="Project files written",
            )

        agent.run_skill(
            "init.write_project_files",
            write_project_files,
            inputs={"project_id": project_id, "project_path": str(project_path)},
        )

        llm_config = _build_llm_config(config)
        outline_request = str(config.get("outline_request", "") or "").strip()
        planning_mode = normalize_planning_mode(project_data.get("planning_mode"))
        if planning_mode in {PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER}:
            agent.run_skill(
                "outline.regenerate_volume",
                lambda: SkillResult.ok(
                    value=regenerate_volume_outline(
                        str(project_path),
                        llm_config,
                        user_request=outline_request,
                        progress_callback=progress_callback,
                    ),
                    artifacts={"outlines": str(project_path / "outlines.json")},
                    message="Volume outline regenerated",
                ),
                inputs={"project_path": str(project_path), "user_request": outline_request},
            )
        if planning_mode == PLANNING_MODE_CHAPTER:
            agent.run_skill(
                "outline.regenerate_chapter",
                lambda: SkillResult.ok(
                    value=regenerate_chapter_outline(
                        str(project_path),
                        llm_config,
                        volume_number=None,
                        user_request=outline_request,
                        progress_callback=progress_callback,
                    ),
                    artifacts={"outlines": str(project_path / "outlines.json")},
                    message="Chapter outline regenerated",
                ),
                inputs={"project_path": str(project_path), "user_request": outline_request},
            )

        snapshot_result = agent.run_skill(
            "snapshot.create_post",
            lambda: SkillResult.ok(
                value=create_state_snapshot(str(project_path), chapter_count=0, note="post-init checkpoint"),
                message="Initial snapshot saved",
            ),
            inputs={"project_path": str(project_path), "chapter_count": 0},
        )
        snapshot_path = str(snapshot_result.value or "")
        if snapshot_path:
            agent.run.add_artifacts({"snapshot": snapshot_path})
        agent.finish_success(
            message="Project initialization completed",
            artifacts={"project_path": str(project_path), "project": str(project_path / "project.json")},
        )
        emit_progress(progress_callback, "init_done", "Project initialization completed")
        return str(project_path)
    except Exception as exc:
        agent.finish_failure(exc)
        raise


def regenerate_initial_project_agentic(project_path: str, config: dict | None = None, progress_callback=None) -> dict:
    agent = WorkflowAgent(project_path, "regenerate_initial_project", progress_callback=progress_callback)
    try:
        result = agent.run_skill(
            "init.regenerate_project",
            lambda: SkillResult.ok(
                value=regenerate_initial_project(project_path, config, progress_callback=progress_callback),
                artifacts={
                    "project": str(Path(project_path).resolve() / "project.json"),
                    "outlines": str(Path(project_path).resolve() / "outlines.json"),
                },
                message="Initial project settings regenerated",
            ),
            inputs={"project_path": str(Path(project_path).resolve())},
        ).value
        agent.finish_success(
            message="Initial project settings regenerated",
            artifacts={"project_path": str(Path(project_path).resolve())},
        )
        return result
    except Exception as exc:
        agent.finish_failure(exc)
        raise


def regenerate_outline_agentic(
    project_path: str,
    config: dict,
    *,
    stage: str,
    volume_number: int | None = None,
    user_request: str = "",
    progress_callback=None,
) -> dict:
    agent = WorkflowAgent(project_path, "outline_regeneration", progress_callback=progress_callback)
    outlines = {}
    try:
        if stage in {"volumes", "all"}:
            outlines = agent.run_skill(
                "outline.regenerate_volume",
                lambda: SkillResult.ok(
                    value=regenerate_volume_outline(
                        project_path,
                        config,
                        user_request=user_request,
                        progress_callback=progress_callback,
                    ),
                    artifacts={"outlines": str(Path(project_path).resolve() / "outlines.json")},
                    message="Volume outline regenerated",
                ),
                inputs={"project_path": project_path, "user_request": user_request},
            ).value
        if stage in {"chapters", "all"}:
            outlines = agent.run_skill(
                "outline.regenerate_chapter",
                lambda: SkillResult.ok(
                    value=regenerate_chapter_outline(
                        project_path,
                        config,
                        volume_number=volume_number if stage == "chapters" else None,
                        user_request=user_request,
                        progress_callback=progress_callback,
                    ),
                    artifacts={"outlines": str(Path(project_path).resolve() / "outlines.json")},
                    message="Chapter outline regenerated",
                ),
                inputs={"project_path": project_path, "volume_number": volume_number, "user_request": user_request},
            ).value
        agent.finish_success(
            message="Outline regeneration completed",
            artifacts={"outlines": str(Path(project_path).resolve() / "outlines.json")},
        )
        return outlines
    except Exception as exc:
        agent.finish_failure(exc)
        raise


def generate_progression_options_agentic(
    project_path: str,
    config: dict,
    *,
    user_request: str = "",
    objective_override: str = "",
    option_count: int = 4,
    runtime_overrides: dict | None = None,
    progress_callback=None,
) -> dict:
    agent = WorkflowAgent(project_path, "progression_options", progress_callback=progress_callback)
    try:
        session = agent.run_skill(
            "progression.generate_options",
            lambda: SkillResult.ok(
                value=generate_progression_options(
                    project_path,
                    config,
                    user_request=user_request,
                    objective_override=objective_override,
                    option_count=option_count,
                    runtime_overrides=runtime_overrides,
                    progress_callback=progress_callback,
                ),
                artifacts={"progression_sessions": str(Path(project_path).resolve() / "progression_sessions")},
                message="Progression options generated",
            ),
            inputs={
                "project_path": project_path,
                "user_request": user_request[:160],
                "objective_override": objective_override[:160],
                "option_count": option_count,
            },
        ).value
        agent.finish_success(
            message="Progression options generated",
            artifacts={"progression_sessions": str(Path(project_path).resolve() / "progression_sessions")},
        )
        return session
    except Exception as exc:
        agent.finish_failure(exc)
        raise
