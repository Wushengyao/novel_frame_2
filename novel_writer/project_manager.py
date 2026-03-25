"""Project storage helpers for the novel writer MVP."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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

INIT_SECTION_KEYS = ("world", "characters", "plot_state", "style")
CHAPTER_TITLE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?第[0-9零一二三四五六七八九十百千万两〇]+[章节卷回部篇]\s*[：:.-]?\s*.+$"
)
STATS_PHASES = ("init", "outline", "writer", "summary")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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

    for _ in range(2):
        try:
            response_text, metadata = generate_text_with_metadata(prompt, llm_config)
        except Exception as exc:  # pragma: no cover - resilience path
            _merge_usage_stats(init_stats["total"], success=False, usage=None)
            _merge_usage_stats(init_stats["by_phase"]["init"], success=False, usage=None)
            llm_init_error = str(exc)
            continue

        try:
            _merge_usage_stats(init_stats["total"], success=True, usage=metadata.get("usage"))
            _merge_usage_stats(init_stats["by_phase"]["init"], success=True, usage=metadata.get("usage"))
            generated_data = _normalize_init_result(_extract_json_object(response_text))
            break
        except Exception as exc:  # pragma: no cover - resilience path
            llm_init_error = str(exc)

    if generated_data is None:
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
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_json(path: str) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def init_project(config_path: str) -> str:
    config_file = Path(config_path).resolve()
    config = load_json(str(config_file))
    project_id = config.get("project_id") or _build_project_id()
    project_path = _resolve_project_path(config_file, config, project_id)
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "chapters").mkdir(exist_ok=True)
    (project_path / "summaries").mkdir(exist_ok=True)
    (project_path / "illustrations").mkdir(exist_ok=True)

    generated_data, init_meta = _generate_initial_story_data(config)
    world = generated_data["world"]
    characters = generated_data["characters"]
    plot_state = _normalize_initial_plot_state(generated_data["plot_state"])
    style = generated_data["style"]

    project_data = {
        "project_id": project_id,
        "name": config.get("project_name", "Novel Project"),
        "description": config.get("project_description", "Structured-memory novel writing project."),
        "project_path": str(project_path),
        "story_request": config.get("story_request", ""),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "chapter_count": 0,
        "init": init_meta,
        "stats": init_meta.get("stats") or _build_project_stats(),
        "llm_config": _build_persisted_llm_config(config),
    }

    save_json(str(project_path / "project.json"), project_data)
    save_json(str(project_path / "world.json"), world)
    save_json(str(project_path / "characters.json"), characters)
    save_json(str(project_path / "plot_state.json"), plot_state)
    save_json(str(project_path / "style.json"), style)
    try:
        from outline_manager import regenerate_chapter_outline, regenerate_volume_outline

        llm_config = _build_llm_config(config)
        outline_request = str(config.get("outline_request", "") or "").strip()
        regenerate_volume_outline(str(project_path), llm_config, user_request=outline_request)
        regenerate_chapter_outline(str(project_path), llm_config, volume_number=None, user_request=outline_request)
    except Exception:  # pragma: no cover - outline generation should not block init
        pass
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
