"""Project storage helpers for the novel writer MVP."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from console_logger import log_error, log_info, log_success, log_warning
from llm_client import generate_text_with_metadata
from prompt_builder import build_init_prompt


EMPTY_WORLD = {
    "title": "",
    "genre": "",
    "setting": "",
    "background": [],
    "rules": [],
}

EMPTY_CHARACTERS = {
    "protagonists": [],
    "supporting": [],
}

EMPTY_CHARACTER_PROFILE = {
    "name": "",
    "role": "",
    "description": "",
    "appearance": "",
}

EMPTY_PLOT_STATE = {
    "main_plot": "",
    "recent_events": [],
    "open_threads": [],
    "foreshadowing": [],
    "character_updates": [],
    "current_location": "",
    "current_time": "",
    "next_chapter_goal": "",
}

EMPTY_STYLE = {
    "tone": "",
    "pov": "",
    "requirements": [],
}

DEFAULT_WORLD = deepcopy(EMPTY_WORLD)
DEFAULT_CHARACTERS = deepcopy(EMPTY_CHARACTERS)
DEFAULT_PLOT_STATE = deepcopy(EMPTY_PLOT_STATE)
DEFAULT_STYLE = deepcopy(EMPTY_STYLE)
PLANNING_MODE_NONE = "none"
PLANNING_MODE_VOLUME = "volume"
PLANNING_MODE_CHAPTER = "chapter"
DEFAULT_PLANNING_MODE = PLANNING_MODE_VOLUME
PLANNING_MODES = {
    PLANNING_MODE_NONE,
    PLANNING_MODE_VOLUME,
    PLANNING_MODE_CHAPTER,
}

INIT_SECTION_KEYS = ("world", "characters", "plot_state", "style")
CHAPTER_TITLE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?第[0-9零一二三四五六七八九十百千万两〇]+[章节卷回部篇]\s*[：:.-]?\s*.+$"
)
STATS_PHASES = ("init", "outline", "writer", "summary")
SNAPSHOT_DIR_NAME = "snapshots"
SNAPSHOT_STATE_FILES = (
    "project.json",
    "world.json",
    "characters.json",
    "plot_state.json",
    "style.json",
    "outlines.json",
)
ROLLBACK_SUMMARY_KEYS = (
    "recent_events",
    "open_threads",
    "foreshadowing",
    "character_updates",
    "next_chapter_goal",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _emit_progress(progress_callback, stage: str, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "stage": stage,
            "message": message,
        }
    )


def normalize_planning_mode(mode: object, default: str = DEFAULT_PLANNING_MODE) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized in PLANNING_MODES:
        return normalized
    return default


def normalize_chapter_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and CHAPTER_TITLE_PATTERN.match(lines[0].strip()):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)

    return "\n".join(lines).strip()


def _empty_usage_stats() -> dict:
    return {
        "requests": 0,
        "successes": 0,
        "failures": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "thought_tokens": 0,
    }


def _build_project_stats() -> dict:
    return {
        "total": _empty_usage_stats(),
        "by_phase": {phase: _empty_usage_stats() for phase in STATS_PHASES},
    }


def _merge_usage_stats(target: dict, success: bool, usage: dict | None) -> None:
    target["requests"] = int(target.get("requests", 0)) + 1
    if success:
        target["successes"] = int(target.get("successes", 0)) + 1
    else:
        target["failures"] = int(target.get("failures", 0)) + 1

    if not usage:
        return

    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "thought_tokens",
    ):
        target[key] = int(target.get(key, 0)) + int(usage.get(key, 0) or 0)


def update_project_stats(project_path: str, phase: str, success: bool, usage: dict | None = None) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    stats = project_data.get("stats") or _build_project_stats()
    stats.setdefault("total", _empty_usage_stats())
    stats.setdefault("by_phase", {})
    stats["by_phase"].setdefault(phase, _empty_usage_stats())

    _merge_usage_stats(stats["total"], success=success, usage=usage)
    _merge_usage_stats(stats["by_phase"][phase], success=success, usage=usage)

    project_data["stats"] = stats
    project_data["updated_at"] = _utc_now()
    save_json(str(project_file), project_data)


def _build_project_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def _resolve_project_path(config_file: Path, config: dict, project_id: str) -> Path:
    raw_project_path = str(config.get("project_path", "./novel_project"))
    formatted_path = raw_project_path.format(project_id=project_id)
    project_path = Path(formatted_path)
    if not project_path.is_absolute():
        project_path = (config_file.parent / project_path).resolve()

    # If the configured path already contains another project, create a unique sibling.
    if project_path.exists() and (project_path / "project.json").exists():
        project_path = project_path.parent / f"{project_path.name}_{project_id}"
    return project_path


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
    raise ValueError("Could not parse JSON from init response.")


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_llm_config(config: dict) -> dict:
    return {
        "model_provider": config.get("model_provider", "openai_compatible"),
        "model": config.get("model") or config.get("model_name", ""),
        "model_name": config.get("model_name") or config.get("model", ""),
        "api_base": config.get("api_base", ""),
        "api_key": config.get("api_key", ""),
        "temperature": config.get("temperature", 0.8),
        "max_tokens": config.get("max_tokens", 4000),
        "timeout": config.get("timeout", 120),
        "thinking_level": config.get("thinking_level"),
        "thinking_budget": config.get("thinking_budget"),
        "planning_mode": normalize_planning_mode(config.get("planning_mode")),
    }


def _build_persisted_llm_config(config: dict) -> dict:
    persisted = _build_llm_config(config)
    persisted["api_key"] = ""
    return persisted


def _normalize_init_result(data: dict) -> dict:
    normalized = {}
    for key in INIT_SECTION_KEYS:
        value = data.get(key, {})
        normalized[key] = value if isinstance(value, dict) else {}
    normalized["characters"] = _normalize_characters(normalized.get("characters", {}))
    return normalized


def _normalize_character_entry(entry: dict) -> dict:
    normalized = deepcopy(EMPTY_CHARACTER_PROFILE)
    if isinstance(entry, dict):
        normalized = _deep_merge(normalized, entry)
    return {
        key: str(normalized.get(key, "") or "").strip()
        for key in EMPTY_CHARACTER_PROFILE
    }


def _normalize_characters(characters: dict) -> dict:
    normalized = deepcopy(EMPTY_CHARACTERS)
    if isinstance(characters, dict):
        normalized = _deep_merge(normalized, characters)

    result = deepcopy(EMPTY_CHARACTERS)
    for group in ("protagonists", "supporting"):
        items = normalized.get(group)
        if not isinstance(items, list):
            items = []
        result[group] = [_normalize_character_entry(item) for item in items if isinstance(item, dict)]
    return result


def _build_seed_story_data(config: dict) -> dict:
    use_empty_defaults = config.get("init_with_llm", False)
    world_default = EMPTY_WORLD if use_empty_defaults else DEFAULT_WORLD
    characters_default = EMPTY_CHARACTERS if use_empty_defaults else DEFAULT_CHARACTERS
    plot_state_default = EMPTY_PLOT_STATE if use_empty_defaults else DEFAULT_PLOT_STATE
    style_default = EMPTY_STYLE if use_empty_defaults else DEFAULT_STYLE
    return {
        "world": deepcopy(config.get("world", world_default)),
        "characters": _normalize_characters(deepcopy(config.get("characters", characters_default))),
        "plot_state": deepcopy(config.get("plot_state", plot_state_default)),
        "style": deepcopy(config.get("style", style_default)),
    }


def _build_fallback_story_data(config: dict, seed_data: dict) -> dict:
    fallback = deepcopy(seed_data)
    project_name = str(config.get("project_name", "")).strip()
    project_description = str(config.get("project_description", "")).strip()
    story_request = str(config.get("story_request", "")).strip()

    if not any(fallback["world"].values()):
        fallback["world"] = {
            "title": project_name,
            "genre": "",
            "setting": project_description or story_request[:200],
            "background": [story_request] if story_request else [],
            "rules": [],
        }

    if not fallback["characters"].get("protagonists") and not fallback["characters"].get("supporting"):
        fallback["characters"] = deepcopy(EMPTY_CHARACTERS)
    fallback["characters"] = _normalize_characters(fallback["characters"])

    if not any(fallback["plot_state"].values()):
        fallback["plot_state"] = {
            "main_plot": project_description or story_request,
            "recent_events": [],
            "open_threads": [],
            "foreshadowing": [],
            "character_updates": [],
            "current_location": "",
            "current_time": "",
            "next_chapter_goal": "根据用户需求自然展开故事开篇。",
        }

    if not any(fallback["style"].values()):
        fallback["style"] = {
            "tone": "自然、连贯、适合长篇连载",
            "pov": "第三人称",
            "requirements": [
                "人物保持一致",
                "剧情持续推进",
            ],
        }

    return fallback


def _normalize_initial_plot_state(plot_state: dict) -> dict:
    normalized = deepcopy(EMPTY_PLOT_STATE)
    if isinstance(plot_state, dict):
        normalized = _deep_merge(normalized, plot_state)

    # Before chapter 1, these fields should not contain already-happened events
    # or future dangling threads generated during initialization.
    normalized["recent_events"] = []
    normalized["open_threads"] = []
    normalized["foreshadowing"] = []
    normalized["character_updates"] = []

    if not str(normalized.get("next_chapter_goal", "")).strip():
        normalized["next_chapter_goal"] = "作为第一章自然展开故事开篇，建立人物、环境与核心矛盾。"
    return normalized


def _generate_initial_story_data(config: dict) -> tuple[dict, dict]:
    seed_data = _build_seed_story_data(config)
    fallback_data = _build_fallback_story_data(config, seed_data)
    init_stats = _build_project_stats()

    if not config.get("init_with_llm", False):
        log_info("初始化设定: 已关闭 LLM 初始化，使用本地兜底设定。")
        return fallback_data, {
            "used_llm": False,
            "llm_init_error": "",
            "stats": init_stats,
        }

    prompt = build_init_prompt(
        {
            "project_name": config.get("project_name", "Novel Project"),
            "project_description": config.get("project_description", ""),
            "story_request": config.get("story_request", ""),
            "world_seed": seed_data["world"],
            "characters_seed": seed_data["characters"],
            "plot_state_seed": seed_data["plot_state"],
            "style_seed": seed_data["style"],
        }
    )

    llm_config = _build_llm_config(config)
    llm_init_error = ""
    generated_data = None

    log_info("初始化设定: 开始请求模型生成世界观、人物、剧情状态和文风。")
    for attempt in range(2):
        try:
            log_info(f"初始化设定: 第 {attempt + 1} 次请求模型。")
            response_text, metadata = generate_text_with_metadata(prompt, llm_config)
        except Exception as exc:  # pragma: no cover - resilience path
            _merge_usage_stats(init_stats["total"], success=False, usage=None)
            _merge_usage_stats(init_stats["by_phase"]["init"], success=False, usage=None)
            llm_init_error = str(exc)
            log_warning(f"初始化设定: 第 {attempt + 1} 次请求失败，原因: {llm_init_error}")
            continue

        try:
            _merge_usage_stats(init_stats["total"], success=True, usage=metadata.get("usage"))
            _merge_usage_stats(init_stats["by_phase"]["init"], success=True, usage=metadata.get("usage"))
            generated_data = _normalize_init_result(_extract_json_object(response_text))
            log_success("初始化设定: 模型返回成功，已解析初始化设定。")
            break
        except Exception as exc:  # pragma: no cover - resilience path
            llm_init_error = str(exc)
            log_warning(f"初始化设定: 返回内容解析失败，原因: {llm_init_error}")

    if generated_data is None:
        if llm_init_error:
            log_warning(f"初始化设定: 改用本地兜底设定。最后错误: {llm_init_error}")
        else:
            log_warning("初始化设定: 未拿到可用结果，改用本地兜底设定。")
        return fallback_data, {
            "used_llm": False,
            "llm_init_error": llm_init_error,
            "stats": init_stats,
        }

    final_data = {
        key: _deep_merge(fallback_data[key], generated_data.get(key, {}))
        for key in INIT_SECTION_KEYS
    }
    final_data["plot_state"] = _normalize_initial_plot_state(final_data["plot_state"])
    return final_data, {
        "used_llm": True,
        "llm_init_error": llm_init_error,
        "stats": init_stats,
    }


def save_json(path: str, data: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{file_path.stem}_",
        suffix=file_path.suffix or ".json",
        dir=str(file_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, file_path)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise


def load_json(path: str) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def init_project(config_path: str, progress_callback=None) -> str:
    log_info(f"init_project: loading config {config_path}")
    _emit_progress(progress_callback, "init_config", "Loading project config")
    config_file = Path(config_path).resolve()
    config = load_json(str(config_file))
    project_id = config.get("project_id") or _build_project_id()
    project_path = _resolve_project_path(config_file, config, project_id)
    log_info(f"init_project: creating project directory {project_path}")
    _emit_progress(progress_callback, "init_dirs", "Creating project directories")
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "chapters").mkdir(exist_ok=True)
    (project_path / "summaries").mkdir(exist_ok=True)
    (project_path / "illustrations").mkdir(exist_ok=True)
    log_success("init_project: base directories ready")

    _emit_progress(progress_callback, "init_story", "Generating initial story data")
    generated_data, init_meta = _generate_initial_story_data(config)
    world = generated_data["world"]
    characters = generated_data["characters"]
    plot_state = _normalize_initial_plot_state(generated_data["plot_state"])
    style = generated_data["style"]
    log_info("init_project: writing project json files")

    project_data = {
        "project_id": project_id,
        "name": config.get("project_name", "Novel Project"),
        "description": config.get("project_description", "Structured-memory novel writing project."),
        "project_path": str(project_path),
        "story_request": config.get("story_request", ""),
        "planning_mode": normalize_planning_mode(config.get("planning_mode")),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "chapter_count": 0,
        "init": init_meta,
        "stats": init_meta.get("stats") or _build_project_stats(),
        "llm_config": _build_persisted_llm_config(config),
    }

    _emit_progress(progress_callback, "init_files", "Writing project files")
    save_json(str(project_path / "project.json"), project_data)
    save_json(str(project_path / "world.json"), world)
    save_json(str(project_path / "characters.json"), characters)
    save_json(str(project_path / "plot_state.json"), plot_state)
    save_json(str(project_path / "style.json"), style)
    log_success("init_project: project files written")

    from outline_manager import regenerate_chapter_outline, regenerate_volume_outline

    llm_config = _build_llm_config(config)
    outline_request = str(config.get("outline_request", "") or "").strip()
    planning_mode = normalize_planning_mode(project_data.get("planning_mode"))
    if planning_mode in {PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER}:
        log_info("init_project: generating volume outlines")
        _emit_progress(progress_callback, "init_volume_outline", "Generating volume outlines")
        regenerate_volume_outline(
            str(project_path),
            llm_config,
            user_request=outline_request,
            progress_callback=progress_callback,
        )
        log_success("init_project: volume outlines ready")
    if planning_mode == PLANNING_MODE_CHAPTER:
        log_info("init_project: generating chapter outlines")
        _emit_progress(progress_callback, "init_chapter_outline", "Generating chapter outlines")
        regenerate_chapter_outline(
            str(project_path),
            llm_config,
            volume_number=None,
            user_request=outline_request,
            progress_callback=progress_callback,
        )
        log_success("init_project: chapter outlines ready")

    _emit_progress(progress_callback, "init_snapshot", "Saving initial snapshot")
    snapshot_path = create_state_snapshot(str(project_path), chapter_count=0, note="post-init checkpoint")
    log_success(f"init_project: snapshot saved to {snapshot_path}")
    log_success(f"init_project: project initialized at {project_path}")
    _emit_progress(progress_callback, "init_done", "Project initialization completed")
    return str(project_path)


def load_project(project_path: str) -> dict:
    base = Path(project_path)
    outlines_path = base / "outlines.json"
    return {
        "project": load_json(str(base / "project.json")),
        "world": load_json(str(base / "world.json")),
        "characters": _normalize_characters(load_json(str(base / "characters.json"))),
        "plot_state": load_json(str(base / "plot_state.json")),
        "style": load_json(str(base / "style.json")),
        "outlines": load_json(str(outlines_path)) if outlines_path.exists() else {"meta": {}, "volumes": []},
        "chapters_path": str(base / "chapters"),
        "summaries_path": str(base / "summaries"),
        "illustrations_path": str(base / "illustrations"),
    }


def _parse_numbered_name(name: str, prefix: str, suffix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    if suffix and not name.endswith(suffix):
        return None
    raw = name[len(prefix) :]
    if suffix:
        raw = raw[: -len(suffix)]
    if not raw.isdigit():
        return None
    return int(raw)


def _snapshot_dir(project_path: str, chapter_count: int) -> Path:
    return Path(project_path) / SNAPSHOT_DIR_NAME / f"chapter_{chapter_count:04d}"


def _normalize_chapter_count(project_path: str, chapter_count: int | None = None) -> int:
    if chapter_count is not None:
        return max(0, int(chapter_count))
    project_data = load_json(str(Path(project_path) / "project.json"))
    return max(0, int(project_data.get("chapter_count", 0) or 0))


def list_state_snapshots(project_path: str) -> list[int]:
    snapshots_dir = Path(project_path) / SNAPSHOT_DIR_NAME
    if not snapshots_dir.exists():
        return []
    numbers = []
    for path in snapshots_dir.iterdir():
        if not path.is_dir():
            continue
        chapter_number = _parse_numbered_name(path.name, "chapter_", "")
        if chapter_number is not None:
            numbers.append(chapter_number)
    return sorted(numbers)


def get_latest_state_snapshot_chapter(project_path: str) -> int | None:
    snapshots = list_state_snapshots(project_path)
    return snapshots[-1] if snapshots else None


def create_state_snapshot(project_path: str, chapter_count: int | None = None, note: str = "") -> str:
    normalized_count = _normalize_chapter_count(project_path, chapter_count)
    base = Path(project_path)
    snapshot_dir = _snapshot_dir(project_path, normalized_count)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for filename in SNAPSHOT_STATE_FILES:
        source_path = base / filename
        if not source_path.exists():
            continue
        shutil.copy2(source_path, snapshot_dir / filename)
        copied_files.append(filename)

    save_json(
        str(snapshot_dir / "snapshot.json"),
        {
            "chapter_count": normalized_count,
            "created_at": _utc_now(),
            "note": note.strip(),
            "files": copied_files,
        },
    )
    return str(snapshot_dir)


def ensure_state_snapshot(project_path: str, chapter_count: int | None = None, note: str = "") -> str:
    normalized_count = _normalize_chapter_count(project_path, chapter_count)
    snapshot_dir = _snapshot_dir(project_path, normalized_count)
    if snapshot_dir.exists():
        return str(snapshot_dir)
    return create_state_snapshot(project_path, chapter_count=normalized_count, note=note)


def _restore_state_from_snapshot(project_path: str, chapter_count: int) -> bool:
    base = Path(project_path)
    snapshot_dir = _snapshot_dir(project_path, chapter_count)
    if not snapshot_dir.exists():
        return False

    for filename in SNAPSHOT_STATE_FILES:
        if filename == "project.json":
            continue
        source_path = snapshot_dir / filename
        target_path = base / filename
        if source_path.exists():
            shutil.copy2(source_path, target_path)
        elif target_path.exists():
            target_path.unlink()
    return True


def _restore_plot_state_from_summary(project_path: str, chapter_count: int) -> str:
    base = Path(project_path)
    plot_state_path = base / "plot_state.json"
    plot_state = load_json(str(plot_state_path))

    if chapter_count == 0:
        plot_state["recent_events"] = []
        plot_state["open_threads"] = []
        plot_state["foreshadowing"] = []
        plot_state["character_updates"] = []
        plot_state["next_chapter_goal"] = ""
        save_json(str(plot_state_path), plot_state)
        return "best_effort_reset"

    summary_path = base / "summaries" / f"summary_{chapter_count:04d}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"找不到第 {chapter_count} 章对应的快照或 summary 文件，无法回滚到该状态: {summary_path}"
        )

    summary = load_json(str(summary_path))
    for key in ROLLBACK_SUMMARY_KEYS:
        default_value = "" if key == "next_chapter_goal" else []
        value = summary.get(key, default_value)
        if key == "next_chapter_goal":
            plot_state[key] = value if isinstance(value, str) else str(value or "")
        elif isinstance(value, list):
            plot_state[key] = value
        elif value is None:
            plot_state[key] = []
        else:
            plot_state[key] = [str(value)]
    save_json(str(plot_state_path), plot_state)
    return "summary_fallback"


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _delete_future_artifacts(project_path: str, keep_chapter_count: int) -> dict:
    base = Path(project_path)
    removed = {
        "chapters": [],
        "summaries": [],
        "illustrations": [],
        "snapshots": [],
    }

    for chapter_path in sorted((base / "chapters").glob("chapter_*.md")):
        chapter_number = _parse_numbered_name(chapter_path.name, "chapter_", ".md")
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(chapter_path)
            removed["chapters"].append(str(chapter_path.relative_to(base)).replace("\\", "/"))

    for summary_path in sorted((base / "summaries").glob("summary_*.json")):
        chapter_number = _parse_numbered_name(summary_path.name, "summary_", ".json")
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(summary_path)
            removed["summaries"].append(str(summary_path.relative_to(base)).replace("\\", "/"))

    illustrations_dir = base / "illustrations"
    if illustrations_dir.exists():
        for record_dir in sorted(illustrations_dir.glob("chapter_*")):
            chapter_number = _parse_numbered_name(record_dir.name, "chapter_", "")
            if chapter_number is not None and chapter_number > keep_chapter_count:
                _remove_path(record_dir)
                removed["illustrations"].append(str(record_dir.relative_to(base)).replace("\\", "/"))

    snapshots_dir = base / SNAPSHOT_DIR_NAME
    if snapshots_dir.exists():
        for snapshot_dir in sorted(snapshots_dir.glob("chapter_*")):
            chapter_number = _parse_numbered_name(snapshot_dir.name, "chapter_", "")
            if chapter_number is not None and chapter_number > keep_chapter_count:
                _remove_path(snapshot_dir)
                removed["snapshots"].append(str(snapshot_dir.relative_to(base)).replace("\\", "/"))

    return removed


def _update_project_chapter_count(project_path: str, chapter_count: int) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    project_data["chapter_count"] = max(0, int(chapter_count))
    project_data["updated_at"] = _utc_now()
    save_json(str(project_file), project_data)


def rollback_project(project_path: str, to_chapter: int) -> dict:
    base = Path(project_path)
    project_file = base / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"项目目录中缺少 project.json: {project_path}")

    current_project = load_json(str(project_file))
    current_count = max(0, int(current_project.get("chapter_count", 0) or 0))
    target_count = max(0, int(to_chapter))

    if target_count > current_count:
        raise ValueError(f"目标章节数不能大于当前章节数: current={current_count}, target={target_count}")

    if target_count == current_count:
        snapshot_path = create_state_snapshot(project_path, chapter_count=target_count, note="refreshed current state")
        return {
            "current_chapter_count": current_count,
            "target_chapter_count": target_count,
            "restore_source": "noop",
            "snapshot_path": snapshot_path,
            "removed": {
                "chapters": [],
                "summaries": [],
                "illustrations": [],
                "snapshots": [],
            },
        }

    restored_from = "snapshot"
    if not _restore_state_from_snapshot(project_path, target_count):
        restored_from = _restore_plot_state_from_summary(project_path, target_count)

    removed = _delete_future_artifacts(project_path, keep_chapter_count=target_count)
    _update_project_chapter_count(project_path, target_count)

    from outline_manager import sync_outline_progress

    sync_outline_progress(project_path)
    snapshot_path = create_state_snapshot(project_path, chapter_count=target_count, note="post-rollback state")
    return {
        "current_chapter_count": current_count,
        "target_chapter_count": target_count,
        "restore_source": restored_from,
        "snapshot_path": snapshot_path,
        "removed": removed,
    }


def save_chapter(project_path: str, text: str) -> str:
    base = Path(project_path)
    chapters_dir = base / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(chapters_dir.glob("chapter_*.md"))
    next_index = len(existing) + 1
    chapter_name = f"chapter_{next_index:04d}.md"
    chapter_path = chapters_dir / chapter_name
    chapter_path.write_text(normalize_chapter_text(text) + "\n", encoding="utf-8")

    project_file = base / "project.json"
    project_data = load_json(str(project_file))
    project_data["chapter_count"] = next_index
    project_data["updated_at"] = _utc_now()
    save_json(str(project_file), project_data)
    return str(chapter_path)


def get_last_chapter_text(project_path: str) -> str:
    chapters_dir = Path(project_path) / "chapters"
    if not chapters_dir.exists():
        return ""
    chapters = sorted(chapters_dir.glob("chapter_*.md"))
    if not chapters:
        return ""
    return normalize_chapter_text(chapters[-1].read_text(encoding="utf-8"))
