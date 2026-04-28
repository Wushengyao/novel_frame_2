"""Project storage helpers for the novel writer MVP."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from uuid import uuid4

from common_utils import emit_progress, extract_json_object, utc_now
from console_logger import log_error, log_info, log_success, log_warning
from llm_client import generate_text_with_metadata
from pricing import estimate_llm_cost
from prompt_builder import build_init_prompt, build_story_setup_prompt, build_system_prompt


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
    "current_arc": "",
    "recent_events": [],
    "open_threads": [],
    "resolved_threads": [],
    "foreshadowing": [],
    "continuity_anchors": [],
    "causal_links": [],
    "character_updates": [],
    "active_characters": [],
    "current_location": "",
    "current_time": "",
    "next_chapter_goal": "",
}

EMPTY_STYLE = {
    "tone": "",
    "pov": "",
    "requirements": [],
}

EMPTY_AUTHOR_INTENT = {
    "premise": "",
    "long_arc": "",
    "tone_contract": "",
    "must_haves": [],
    "must_not_break": [],
    "creativity_guidance": "",
}

DEFAULT_WORLD = deepcopy(EMPTY_WORLD)
DEFAULT_CHARACTERS = deepcopy(EMPTY_CHARACTERS)
DEFAULT_PLOT_STATE = deepcopy(EMPTY_PLOT_STATE)
DEFAULT_STYLE = deepcopy(EMPTY_STYLE)
PLANNING_MODE_NONE = "none"
PLANNING_MODE_VOLUME = "volume"
PLANNING_MODE_CHAPTER = "chapter"
DEFAULT_PLANNING_MODE = PLANNING_MODE_CHAPTER
PLANNING_MODES = {
    PLANNING_MODE_NONE,
    PLANNING_MODE_VOLUME,
    PLANNING_MODE_CHAPTER,
}

INIT_SECTION_KEYS = ("world", "characters", "plot_state", "style")
STORY_SETUP_SECTION_KEYS = ("world", "characters")
STORY_SETUP_FILENAME = "story_setup.json"
CHAPTER_TITLE_PATTERN = re.compile(
    r"^\s*(?:#{1,6}\s*)?第[0-9零一二三四五六七八九十百千万两〇]+[章节卷回部篇]\s*[：:.-]?\s*.+$"
)
STATS_PHASES = (
    "init",
    "outline",
    "craft_brief",
    "writer",
    "quality_review",
    "rewrite",
    "summary",
    "expert_review",
    "polish",
    "audiobook",
)
SNAPSHOT_DIR_NAME = "snapshots"
SNAPSHOT_STATE_FILES = (
    "project.json",
    STORY_SETUP_FILENAME,
    "world.json",
    "characters.json",
    "plot_state.json",
    "style.json",
    "author_intent.json",
    "outlines.json",
)
ROLLBACK_SUMMARY_KEYS = (
    "current_arc",
    "current_location",
    "current_time",
    "recent_events",
    "open_threads",
    "resolved_threads",
    "foreshadowing",
    "continuity_anchors",
    "causal_links",
    "character_updates",
    "active_characters",
    "next_chapter_goal",
)
PROJECT_WRITE_LOCK_FILENAME = ".project_write.lock"
PROJECT_AUDIO_LOCK_FILENAME = ".project_audio.lock"
PROJECT_LOCK_FILENAMES = (PROJECT_WRITE_LOCK_FILENAME, PROJECT_AUDIO_LOCK_FILENAME)
PROJECT_DIR_PATTERN = re.compile(r"^novel_project_")
PROJECT_WRITE_LOCK_PROCESS_MARKERS = (
    "webui.py",
    "app.py",
    "python",
    "uv",
)
PROJECT_EXPORT_MANIFEST_FILENAME = ".novel_writer_project_export.json"
PROJECT_EXPORT_FORMAT_VERSION = 1
PROJECT_IMPORT_REQUIRED_FILES = (
    "project.json",
    "world.json",
    "characters.json",
    "plot_state.json",
    "style.json",
)
PROJECT_STANDARD_DIRS = (
    "chapters",
    "summaries",
    "arc_summaries",
    "task_cards",
    "craft_briefs",
    "quality_reviews",
    "quality_drafts",
    "expert_reviews",
    "illustrations",
    "audiobook",
    "snapshots",
    "progression_sessions",
    "polish_backups",
    "llm_logs",
)


class ProjectWriteLockError(RuntimeError):
    """Raised when a mutating project workflow cannot acquire the project lock."""


class ProjectWriteLock:
    def __init__(
        self,
        project_path: str,
        *,
        owner: str = "",
        timeout: float = 0,
        poll_interval: float = 0.2,
        lock_filename: str = PROJECT_WRITE_LOCK_FILENAME,
        busy_message: str = "当前项目已有写作任务正在运行，请稍后再试。",
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.lock_path = self.project_path / lock_filename
        self.owner = str(owner or "").strip()
        self.timeout = max(0.0, float(timeout or 0))
        self.poll_interval = max(0.05, float(poll_interval or 0.2))
        self.busy_message = str(busy_message or "").strip()
        self.token = uuid4().hex
        self._acquired = False

    def __enter__(self) -> "ProjectWriteLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> "ProjectWriteLock":
        if self._acquired:
            return self
        if not self.project_path.exists():
            raise FileNotFoundError(f"project path does not exist: {self.project_path}")

        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    payload = {
                        "pid": os.getpid(),
                        "owner": self.owner,
                        "created_at": utc_now(),
                        "project_path": str(self.project_path),
                        "token": self.token,
                    }
                    os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
                    os.write(fd, b"\n")
                except Exception:
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                    raise
                finally:
                    os.close(fd)
                self._acquired = True
                return self
            except FileExistsError as exc:
                if self._discard_stale_lock():
                    continue
                if self.timeout <= 0 or time.monotonic() >= deadline:
                    raise ProjectWriteLockError(self._busy_message()) from exc
                time.sleep(self.poll_interval)

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            if self.lock_path.exists():
                try:
                    lock_data = load_json(str(self.lock_path))
                except Exception:
                    lock_data = {}
                if lock_data.get("token") == self.token:
                    self.lock_path.unlink()
        finally:
            self._acquired = False

    def _busy_message(self) -> str:
        return (
            (self.busy_message or "当前项目已有任务正在运行，请稍后再试。")
            + " "
            f"锁文件: {self.lock_path}。"
            "如果确认没有任务运行，可手动删除该锁文件。"
        )


    def _discard_stale_lock(self) -> bool:
        try:
            lock_text = self.lock_path.read_text(encoding="utf-8")
            lock_data = json.loads(lock_text)
        except (OSError, ValueError, TypeError):
            return False

        if _lock_owner_process_still_active(lock_data):
            return False

        try:
            if self.lock_path.read_text(encoding="utf-8") != lock_text:
                return False
            self.lock_path.unlink()
            return True
        except OSError:
            return False


def acquire_project_write_lock(project_path: str, *, owner: str = "", timeout: float = 0) -> ProjectWriteLock:
    return ProjectWriteLock(project_path, owner=owner, timeout=timeout)


def acquire_project_audio_lock(project_path: str, *, owner: str = "", timeout: float = 0) -> ProjectWriteLock:
    return ProjectWriteLock(
        project_path,
        owner=owner,
        timeout=timeout,
        lock_filename=PROJECT_AUDIO_LOCK_FILENAME,
        busy_message="当前项目已有有声章节生成任务正在运行，请稍后再试。",
    )


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_process_cmdline(pid: int) -> str | None:
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = proc_cmdline.read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip().lower()


def _lock_owner_process_still_active(lock_data: object) -> bool:
    if not isinstance(lock_data, dict):
        return True
    try:
        pid = int(lock_data.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return True
    if not _process_exists(pid):
        return False

    cmdline = _read_process_cmdline(pid)
    if cmdline is None:
        return True
    return any(marker in cmdline for marker in PROJECT_WRITE_LOCK_PROCESS_MARKERS)


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
        "cost": _empty_cost_stats(),
        "context_telemetry": {
            "recent_runs": [],
            "last_by_phase": {},
        },
    }


def _empty_cost_stats() -> dict:
    return {
        "currency": "USD",
        "estimated_total_usd": 0.0,
        "priced_tokens": 0,
        "unpriced_tokens": 0,
        "by_phase": {},
        "by_model": {},
        "started_at": "",
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


def _cost_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _cost_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _normalize_cost_stats(stats: dict) -> dict:
    cost = stats.get("cost") if isinstance(stats.get("cost"), dict) else {}
    normalized = _empty_cost_stats()
    normalized.update(cost)
    normalized["currency"] = "USD"
    normalized["estimated_total_usd"] = _cost_float(normalized.get("estimated_total_usd"))
    normalized["priced_tokens"] = _cost_int(normalized.get("priced_tokens"))
    normalized["unpriced_tokens"] = _cost_int(normalized.get("unpriced_tokens"))
    normalized["by_phase"] = normalized.get("by_phase") if isinstance(normalized.get("by_phase"), dict) else {}
    normalized["by_model"] = normalized.get("by_model") if isinstance(normalized.get("by_model"), dict) else {}
    return normalized


def _merge_cost_entry(target: dict, usage: dict | None, estimate: dict) -> None:
    target["requests"] = _cost_int(target.get("requests")) + 1
    target["estimated_usd"] = _cost_float(target.get("estimated_usd")) + _cost_float(estimate.get("estimated_cost_usd"))
    target["priced_tokens"] = _cost_int(target.get("priced_tokens")) + _cost_int(estimate.get("priced_tokens"))
    target["unpriced_tokens"] = _cost_int(target.get("unpriced_tokens")) + _cost_int(estimate.get("unpriced_tokens"))
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
        target[key] = _cost_int(target.get(key)) + _cost_int(usage.get(key))


def _merge_model_cost_stats(stats: dict, phase: str, success: bool, metadata: dict | None = None) -> None:
    if not success or not isinstance(metadata, dict):
        return
    usage = metadata.get("usage")
    if not isinstance(usage, dict):
        return
    provider = str(metadata.get("provider") or "openai_compatible").strip() or "openai_compatible"
    model = str(metadata.get("model") or "").strip()
    estimate = estimate_llm_cost(provider, model, usage)

    cost = _normalize_cost_stats(stats)
    if not cost.get("started_at"):
        cost["started_at"] = utc_now()
    cost["estimated_total_usd"] = _cost_float(cost.get("estimated_total_usd")) + _cost_float(
        estimate.get("estimated_cost_usd")
    )
    cost["priced_tokens"] = _cost_int(cost.get("priced_tokens")) + _cost_int(estimate.get("priced_tokens"))
    cost["unpriced_tokens"] = _cost_int(cost.get("unpriced_tokens")) + _cost_int(estimate.get("unpriced_tokens"))

    phase_entry = cost["by_phase"].setdefault(phase, {})
    _merge_cost_entry(phase_entry, usage, estimate)

    model_key = f"{estimate.get('provider') or provider}:{estimate.get('model') or model}"
    model_entry = cost["by_model"].setdefault(
        model_key,
        {
            "provider": estimate.get("provider") or provider,
            "model": estimate.get("model") or model,
            "pricing_status": estimate.get("pricing_status", ""),
            "source": estimate.get("source") or {},
            "reason": estimate.get("reason", ""),
        },
    )
    model_entry["provider"] = estimate.get("provider") or provider
    model_entry["model"] = estimate.get("model") or model
    model_entry["pricing_status"] = estimate.get("pricing_status", "")
    model_entry["source"] = estimate.get("source") or {}
    model_entry["reason"] = estimate.get("reason", "")
    _merge_cost_entry(model_entry, usage, estimate)

    stats["cost"] = cost


def _merge_project_stats(stats: dict, phase: str, success: bool, usage: dict | None = None, metadata: dict | None = None) -> None:
    stats.setdefault("total", _empty_usage_stats())
    stats.setdefault("by_phase", {})
    stats["by_phase"].setdefault(phase, _empty_usage_stats())
    stats["cost"] = _normalize_cost_stats(stats)
    _merge_usage_stats(stats["total"], success=success, usage=usage)
    _merge_usage_stats(stats["by_phase"][phase], success=success, usage=usage)
    _merge_model_cost_stats(stats, phase=phase, success=success, metadata=metadata)


def update_project_stats(
    project_path: str,
    phase: str,
    success: bool,
    usage: dict | None = None,
    metadata: dict | None = None,
) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    stats = project_data.get("stats") or _build_project_stats()
    stats.setdefault(
        "context_telemetry",
        {
            "recent_runs": [],
            "last_by_phase": {},
        },
    )

    _merge_project_stats(stats, phase=phase, success=success, usage=usage, metadata=metadata)

    project_data["stats"] = stats
    project_data["updated_at"] = utc_now()
    save_json(str(project_file), project_data)


def record_context_telemetry(
    project_path: str,
    phase: str,
    *,
    prompt_chars: int,
    section_chars: dict[str, int] | None = None,
    planning_mode: str = "",
    extra: dict | None = None,
) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    stats = project_data.get("stats") or _build_project_stats()
    telemetry = stats.setdefault(
        "context_telemetry",
        {
            "recent_runs": [],
            "last_by_phase": {},
        },
    )

    entry = {
        "created_at": utc_now(),
        "phase": str(phase or "").strip(),
        "planning_mode": normalize_planning_mode(planning_mode, default="") if planning_mode else "",
        "prompt_chars": max(0, int(prompt_chars or 0)),
        "section_chars": {
            str(key): max(0, int(value or 0))
            for key, value in (section_chars or {}).items()
        },
    }
    if isinstance(extra, dict):
        for key, value in extra.items():
            if value in (None, "", []):
                continue
            entry[str(key)] = value

    recent_runs = telemetry.setdefault("recent_runs", [])
    recent_runs.append(entry)
    telemetry["recent_runs"] = recent_runs[-120:]
    telemetry.setdefault("last_by_phase", {})[entry["phase"]] = entry

    stats["context_telemetry"] = telemetry
    project_data["stats"] = stats
    project_data["updated_at"] = utc_now()
    save_json(str(project_file), project_data)


def _build_project_id() -> str:
    timestamp = utc_now().replace("-", "").replace(":", "").replace("+00:00", "Z")
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


def _build_quality_model_config(config: dict, *, include_api_key: bool) -> dict:
    raw = config.get("quality_model") if isinstance(config.get("quality_model"), dict) else {}
    quality_model: dict[str, object] = {}
    provider = str(raw.get("model_provider") or raw.get("provider") or "").strip().lower()
    if provider:
        quality_model["model_provider"] = provider
    model = str(raw.get("model_name") or raw.get("model") or "").strip()
    if model:
        quality_model["model_name"] = model
        quality_model["model"] = model
    api_base = str(raw.get("api_base", "") or "").strip()
    if api_base:
        quality_model["api_base"] = api_base
    if include_api_key:
        api_key = str(raw.get("api_key", "") or "").strip()
        if api_key:
            quality_model["api_key"] = api_key
    for key in ("temperature", "max_tokens", "timeout"):
        value = raw.get(key)
        if value not in (None, ""):
            quality_model[key] = value
    return quality_model


def _coerce_config_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _build_expert_mode_config(config: dict, *, include_api_key: bool) -> dict:
    raw = config.get("expert_mode") if isinstance(config.get("expert_mode"), dict) else {}
    expert_mode: dict[str, object] = {}
    if "enabled" in raw:
        expert_mode["enabled"] = _coerce_config_bool(raw.get("enabled"))

    models = []
    raw_models = raw.get("models") if isinstance(raw.get("models"), list) else []
    for item in raw_models[:3]:
        model_config = _build_quality_model_config({"quality_model": item}, include_api_key=include_api_key)
        if model_config:
            models.append(model_config)
    if models:
        expert_mode["models"] = models
    return expert_mode


def _expert_mode_enabled(config: dict) -> bool:
    raw = config.get("expert_mode") if isinstance(config.get("expert_mode"), dict) else {}
    return _coerce_config_bool(raw.get("enabled"), default=False)


def _build_llm_config(config: dict) -> dict:
    expert_enabled = _expert_mode_enabled(config)
    llm_config = {
        "model_provider": config.get("model_provider", "openai_compatible"),
        "model": config.get("model") or config.get("model_name", ""),
        "model_name": config.get("model_name") or config.get("model", ""),
        "api_base": config.get("api_base", ""),
        "api_key": config.get("api_key", ""),
        "temperature": config.get("temperature", 0.8),
        "max_tokens": config.get("max_tokens", 4000),
        "timeout": config.get("timeout", 120),
        "planning_mode": normalize_planning_mode(config.get("planning_mode")),
        "writing_quality_mode": config.get("writing_quality_mode", "balanced"),
        "review_mode": config.get("review_mode", "auto"),
        "log_llm_payload": _coerce_config_bool(config.get("log_llm_payload")) or expert_enabled,
    }
    if config.get("project_path"):
        llm_config["project_path"] = str(config.get("project_path"))
    quality_model = _build_quality_model_config(config, include_api_key=True)
    if quality_model:
        llm_config["quality_model"] = quality_model
    expert_mode = _build_expert_mode_config(config, include_api_key=True)
    if expert_mode:
        llm_config["expert_mode"] = expert_mode
    return llm_config


def _build_persisted_llm_config(config: dict) -> dict:
    persisted = _build_llm_config(config)
    persisted["api_key"] = ""
    persisted.pop("project_path", None)
    if isinstance(persisted.get("quality_model"), dict):
        persisted["quality_model"]["api_key"] = ""
    expert_mode = persisted.get("expert_mode")
    if isinstance(expert_mode, dict):
        for model in expert_mode.get("models") or []:
            if isinstance(model, dict):
                model["api_key"] = ""
    return persisted


def _normalize_world(world: dict) -> dict:
    normalized = deepcopy(EMPTY_WORLD)
    if isinstance(world, dict):
        normalized = _deep_merge(normalized, world)

    result = {}
    for key, default_value in EMPTY_WORLD.items():
        value = normalized.get(key, default_value)
        if isinstance(default_value, list):
            if not isinstance(value, list):
                value = [value]
            result[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            result[key] = str(value or "").strip()
    return result


def _normalize_story_setup_result(data: dict) -> dict:
    normalized = {}
    for key in STORY_SETUP_SECTION_KEYS:
        value = data.get(key, {})
        normalized[key] = value if isinstance(value, dict) else {}
    normalized["world"] = _normalize_world(normalized.get("world", {}))
    normalized["characters"] = _normalize_characters(normalized.get("characters", {}))
    return normalized


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


def _has_character_profiles(characters: dict) -> bool:
    if not isinstance(characters, dict):
        return False
    for group in ("protagonists", "supporting"):
        items = characters.get(group)
        if isinstance(items, list) and any(isinstance(item, dict) for item in items):
            return True
    return False


def _normalize_name_list(value: object, *, max_items: int = 8) -> list[str]:
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


def _prune_initial_supporting_characters(
    characters: dict,
    plot_state: dict,
    *,
    seeded_characters: dict | None = None,
) -> dict:
    normalized = _normalize_characters(characters)
    seeded = _normalize_characters(seeded_characters or {})
    opening_names = set(_normalize_name_list((plot_state or {}).get("active_characters"), max_items=6))
    seeded_supporting_names = {
        str(item.get("name", "") or "").strip()
        for item in seeded.get("supporting") or []
        if str(item.get("name", "") or "").strip()
    }
    keep_names = opening_names | seeded_supporting_names
    normalized["supporting"] = [
        item
        for item in normalized.get("supporting") or []
        if str(item.get("name", "") or "").strip() in keep_names
    ]
    return normalized


def _normalize_author_intent(author_intent: dict | None) -> dict:
    normalized = deepcopy(EMPTY_AUTHOR_INTENT)
    if isinstance(author_intent, dict):
        normalized = _deep_merge(normalized, author_intent)

    result = {
        "premise": str(normalized.get("premise", "") or "").strip(),
        "long_arc": str(normalized.get("long_arc", "") or "").strip(),
        "tone_contract": str(normalized.get("tone_contract", "") or "").strip(),
        "creativity_guidance": str(normalized.get("creativity_guidance", "") or "").strip(),
    }
    for key in ("must_haves", "must_not_break"):
        value = normalized.get(key) or []
        if not isinstance(value, list):
            value = [value]
        result[key] = [str(item).strip() for item in value if str(item).strip()]
    return result


def _extract_story_constraints(text: str, max_items: int = 5) -> list[str]:
    source = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    fragments = []
    for line in source.split("\n"):
        cleaned = line.strip().lstrip("-").strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[0-9]+[、.．:：)]\s*", "", cleaned).strip()
        if cleaned:
            fragments.append(cleaned)

    if not fragments and source.strip():
        fragments = [
            item.strip()
            for item in re.split(r"[。！？!?；;]\s*", source)
            if item.strip()
        ]

    unique = []
    seen = set()
    for fragment in fragments:
        compact = fragment[:80].strip()
        if not compact or compact in seen:
            continue
        unique.append(compact)
        seen.add(compact)
        if len(unique) >= max_items:
            break
    return unique


def _build_author_intent_from_project(project: dict, world: dict, style: dict, plot_state: dict) -> dict:
    story_request = str(project.get("story_request", "") or "").strip()
    project_description = str(project.get("description", "") or "").strip()
    tone = str(style.get("tone", "") or "").strip()
    pov = str(style.get("pov", "") or "").strip()
    premise = project_description or story_request[:240] or str(world.get("setting", "") or "").strip()
    long_arc = (
        str(plot_state.get("main_plot", "") or "").strip()
        or story_request[:240]
        or project_description
    )
    tone_contract = " / ".join(part for part in (tone, pov) if part).strip()
    style_requirements = style.get("requirements") or []
    if not isinstance(style_requirements, list):
        style_requirements = [style_requirements]
    must_haves = [
        str(item).strip()
        for item in style_requirements
        if str(item).strip()
    ]
    must_haves.extend(_extract_story_constraints(story_request, max_items=6))
    must_not_break = [
        "人物行为与既有关系不能失真或跳变。",
        "重大设定、地点状态与时间推进必须与已有正文一致。",
        "不能提前写完后续章节的核心事件。",
    ]
    return _normalize_author_intent(
        {
            "premise": premise[:240],
            "long_arc": long_arc[:240],
            "tone_contract": tone_contract[:180],
            "must_haves": must_haves[:6],
            "must_not_break": must_not_break,
            "creativity_guidance": "在不破坏设定与记忆的前提下，优先写出有新鲜感的场景调度、互动细节与推进方式。",
        }
    )


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


def _story_setup_enabled(config: dict) -> bool:
    if not config.get("init_with_llm", False):
        return False
    if "init_with_story_setup" in config:
        return _coerce_config_bool(config.get("init_with_story_setup"), default=True)
    if "init_story_setup" in config:
        return _coerce_config_bool(config.get("init_story_setup"), default=True)
    return True


def _merge_story_setup_seed(seed_data: dict, story_setup: dict | None) -> dict:
    if not story_setup:
        return deepcopy(seed_data)

    merged = deepcopy(seed_data)
    setup_world = story_setup.get("world") if isinstance(story_setup.get("world"), dict) else {}
    if setup_world:
        merged["world"] = _deep_merge(merged.get("world", {}), setup_world)

    setup_characters = story_setup.get("characters") if isinstance(story_setup.get("characters"), dict) else {}
    if setup_characters:
        merged["characters"] = _normalize_characters(_deep_merge(merged.get("characters", {}), setup_characters))

    return merged


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
            "current_arc": "开篇阶段",
            "recent_events": [],
            "open_threads": [],
            "resolved_threads": [],
            "foreshadowing": [],
            "character_updates": [],
            "active_characters": [],
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
    normalized["resolved_threads"] = []
    normalized["foreshadowing"] = []
    normalized["continuity_anchors"] = []
    normalized["causal_links"] = []
    normalized["character_updates"] = []
    normalized["active_characters"] = []

    if not str(normalized.get("current_arc", "")).strip():
        normalized["current_arc"] = "开篇阶段"

    if not str(normalized.get("next_chapter_goal", "")).strip():
        normalized["next_chapter_goal"] = "作为第一章自然展开故事开篇，建立人物、环境与核心矛盾。"
    return normalized


def _generate_story_setup_data(
    config: dict,
    seed_data: dict,
    init_stats: dict,
    progress_callback=None,
) -> tuple[dict | None, dict]:
    meta = {
        "used_story_setup_llm": False,
        "llm_story_setup_error": "",
    }
    if not _story_setup_enabled(config):
        return None, meta

    prompt = build_story_setup_prompt(
        {
            "project_name": config.get("project_name", "Novel Project"),
            "project_description": config.get("project_description", ""),
            "story_request": config.get("story_request", ""),
            "world_seed": seed_data["world"],
            "characters_seed": seed_data["characters"],
        }
    )
    llm_config = _build_llm_config(config)

    emit_progress(progress_callback, "init_story_setup", "正在根据故事需求生成人物和背景设定")
    log_info("初始化设定: 先请求模型生成人物和背景设定。")
    for attempt in range(2):
        try:
            log_info(f"初始化设定: 人物和背景设定第 {attempt + 1} 次请求模型。")
            response_text, metadata = generate_text_with_metadata(
                prompt,
                llm_config,
                system_prompt=build_system_prompt("planner"),
                response_format="json",
            )
        except Exception as exc:  # pragma: no cover - resilience path
            _merge_project_stats(init_stats, phase="init", success=False, usage=None, metadata=None)
            meta["llm_story_setup_error"] = str(exc)
            log_warning(
                "初始化设定: 人物和背景设定请求失败，"
                f"原因: {meta['llm_story_setup_error']}"
            )
            continue

        try:
            _merge_project_stats(
                init_stats,
                phase="init",
                success=True,
                usage=metadata.get("usage"),
                metadata=metadata,
            )
            setup_data = _normalize_story_setup_result(
                extract_json_object(response_text, "Could not parse JSON from story setup response.")
            )
            meta["used_story_setup_llm"] = True
            log_success("初始化设定: 人物和背景设定已生成。")
            return setup_data, meta
        except Exception as exc:  # pragma: no cover - resilience path
            meta["llm_story_setup_error"] = str(exc)
            log_warning(
                "初始化设定: 人物和背景设定解析失败，"
                f"原因: {meta['llm_story_setup_error']}"
            )

    return None, meta


def _generate_initial_story_data(config: dict, progress_callback=None) -> tuple[dict, dict]:
    seed_data = _build_seed_story_data(config)
    init_stats = _build_project_stats()
    story_setup, story_setup_meta = _generate_story_setup_data(
        config,
        seed_data,
        init_stats,
        progress_callback=progress_callback,
    )
    effective_seed_data = _merge_story_setup_seed(seed_data, story_setup)
    fallback_data = _build_fallback_story_data(config, effective_seed_data)

    if not config.get("init_with_llm", False):
        log_info("初始化设定: 已关闭 LLM 初始化，使用本地兜底设定。")
        fallback_data["story_setup"] = {
            "world": _normalize_world(fallback_data["world"]),
            "characters": _normalize_characters(fallback_data["characters"]),
        }
        return fallback_data, {
            **story_setup_meta,
            "used_llm": False,
            "llm_init_error": "",
            "stats": init_stats,
        }

    prompt = build_init_prompt(
        {
            "project_name": config.get("project_name", "Novel Project"),
            "project_description": config.get("project_description", ""),
            "story_request": config.get("story_request", ""),
            "world_seed": effective_seed_data["world"],
            "characters_seed": effective_seed_data["characters"],
            "plot_state_seed": effective_seed_data["plot_state"],
            "style_seed": effective_seed_data["style"],
        }
    )

    llm_config = _build_llm_config(config)
    llm_init_error = ""
    generated_data = None

    log_info("初始化设定: 开始请求模型生成世界观、人物、剧情状态和文风。")
    for attempt in range(2):
        try:
            log_info(f"初始化设定: 第 {attempt + 1} 次请求模型。")
            response_text, metadata = generate_text_with_metadata(
                prompt,
                llm_config,
                system_prompt=build_system_prompt("planner"),
                response_format="json",
            )
        except Exception as exc:  # pragma: no cover - resilience path
            _merge_project_stats(init_stats, phase="init", success=False, usage=None, metadata=None)
            llm_init_error = str(exc)
            log_warning(f"初始化设定: 第 {attempt + 1} 次请求失败，原因: {llm_init_error}")
            continue

        try:
            _merge_project_stats(
                init_stats,
                phase="init",
                success=True,
                usage=metadata.get("usage"),
                metadata=metadata,
            )
            generated_data = _normalize_init_result(
                extract_json_object(response_text, "Could not parse JSON from init response.")
            )
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
        fallback_data["story_setup"] = story_setup or {
            "world": _normalize_world(fallback_data["world"]),
            "characters": _normalize_characters(fallback_data["characters"]),
        }
        return fallback_data, {
            **story_setup_meta,
            "used_llm": False,
            "llm_init_error": llm_init_error,
            "stats": init_stats,
        }

    final_data = {}
    for key in INIT_SECTION_KEYS:
        generated_section = generated_data.get(key, {})
        if key == "characters" and not _has_character_profiles(generated_section):
            final_data[key] = deepcopy(fallback_data[key])
        else:
            final_data[key] = _deep_merge(fallback_data[key], generated_section)
    final_data["characters"] = _prune_initial_supporting_characters(
        final_data["characters"],
        final_data["plot_state"],
        seeded_characters=seed_data["characters"],
    )
    final_data["plot_state"] = _normalize_initial_plot_state(final_data["plot_state"])
    final_data["story_setup"] = story_setup or {
        "world": _normalize_world(final_data["world"]),
        "characters": _normalize_characters(final_data["characters"]),
    }
    return final_data, {
        **story_setup_meta,
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
    emit_progress(progress_callback, "init_config", "Loading project config")
    config_file = Path(config_path).resolve()
    config = load_json(str(config_file))
    project_id = config.get("project_id") or _build_project_id()
    project_path = _resolve_project_path(config_file, config, project_id)
    config["project_path"] = str(project_path.resolve())
    if _expert_mode_enabled(config):
        config["log_llm_payload"] = True
    log_info(f"init_project: creating project directory {project_path}")
    emit_progress(progress_callback, "init_dirs", "Creating project directories")
    project_path.mkdir(parents=True, exist_ok=True)
    _ensure_project_subdirs(project_path)
    log_success("init_project: base directories ready")

    emit_progress(progress_callback, "init_story", "Generating initial story data")
    generated_data, init_meta = _generate_initial_story_data(config, progress_callback=progress_callback)
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
    log_info("init_project: writing project json files")

    project_data = {
        "project_id": project_id,
        "name": config.get("project_name", "Novel Project"),
        "description": config.get("project_description", "Structured-memory novel writing project."),
        "project_path": str(project_path),
        "story_request": config.get("story_request", ""),
        "planning_mode": normalize_planning_mode(config.get("planning_mode")),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "chapter_count": 0,
        "init": init_meta,
        "stats": init_meta.get("stats") or _build_project_stats(),
        "llm_config": _build_persisted_llm_config(config),
    }

    emit_progress(progress_callback, "init_files", "Writing project files")
    save_json(str(project_path / "project.json"), project_data)
    save_json(str(project_path / STORY_SETUP_FILENAME), story_setup or {"world": world, "characters": characters})
    save_json(str(project_path / "world.json"), world)
    save_json(str(project_path / "characters.json"), characters)
    save_json(str(project_path / "plot_state.json"), plot_state)
    save_json(str(project_path / "style.json"), style)
    save_json(str(project_path / "author_intent.json"), author_intent)
    log_success("init_project: project files written")

    from outline_manager import regenerate_chapter_outline, regenerate_volume_outline

    llm_config = _build_llm_config(config)
    outline_request = str(config.get("outline_request", "") or "").strip()
    planning_mode = normalize_planning_mode(project_data.get("planning_mode"))
    if planning_mode in {PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER}:
        log_info("init_project: generating volume outlines")
        emit_progress(progress_callback, "init_volume_outline", "Generating volume outlines")
        regenerate_volume_outline(
            str(project_path),
            llm_config,
            user_request=outline_request,
            progress_callback=progress_callback,
        )
        log_success("init_project: volume outlines ready")
    if planning_mode == PLANNING_MODE_CHAPTER:
        log_info("init_project: generating chapter outlines")
        emit_progress(progress_callback, "init_chapter_outline", "Generating chapter outlines")
        regenerate_chapter_outline(
            str(project_path),
            llm_config,
            volume_number=None,
            user_request=outline_request,
            progress_callback=progress_callback,
        )
        log_success("init_project: chapter outlines ready")

    emit_progress(progress_callback, "init_snapshot", "Saving initial snapshot")
    snapshot_path = create_state_snapshot(str(project_path), chapter_count=0, note="post-init checkpoint")
    log_success(f"init_project: snapshot saved to {snapshot_path}")
    log_success(f"init_project: project initialized at {project_path}")
    emit_progress(progress_callback, "init_done", "Project initialization completed")
    return str(project_path)


def load_project(project_path: str) -> dict:
    base = Path(project_path)
    outlines_path = base / "outlines.json"
    project = load_json(str(base / "project.json"))
    world = load_json(str(base / "world.json"))
    characters = _normalize_characters(load_json(str(base / "characters.json")))
    plot_state = load_json(str(base / "plot_state.json"))
    style = load_json(str(base / "style.json"))
    author_intent_path = base / "author_intent.json"
    author_intent = _normalize_author_intent(
        load_json(str(author_intent_path))
        if author_intent_path.exists()
        else _build_author_intent_from_project(project, world, style, plot_state)
    )
    return {
        "project": project,
        "world": world,
        "characters": characters,
        "plot_state": plot_state,
        "style": style,
        "author_intent": author_intent,
        "outlines": load_json(str(outlines_path)) if outlines_path.exists() else {"meta": {}, "volumes": []},
        "chapters_path": str(base / "chapters"),
        "summaries_path": str(base / "summaries"),
        "arc_summaries_path": str(base / "arc_summaries"),
        "task_cards_path": str(base / "task_cards"),
        "craft_briefs_path": str(base / "craft_briefs"),
        "quality_reviews_path": str(base / "quality_reviews"),
        "quality_drafts_path": str(base / "quality_drafts"),
        "illustrations_path": str(base / "illustrations"),
        "audiobook_path": str(base / "audiobook"),
    }


def _safe_filename_part(value: object, default: str = "project") -> str:
    text = str(value or "").strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return safe or default


def _safe_project_id(value: object, default: str = "imported") -> str:
    return _safe_filename_part(value, default=default)


def _project_file_lock_is_active(project_path: Path, lock_filename: str) -> bool:
    lock_path = project_path / lock_filename
    if not lock_path.exists():
        return False
    try:
        lock_data = load_json(str(lock_path))
    except Exception:
        return True
    return _lock_owner_process_still_active(lock_data)


def _project_lock_is_active(project_path: Path) -> bool:
    return _project_file_lock_is_active(project_path, PROJECT_WRITE_LOCK_FILENAME)


def project_audio_lock_is_active(project_path: str | Path) -> bool:
    return _project_file_lock_is_active(Path(project_path), PROJECT_AUDIO_LOCK_FILENAME)


def ensure_no_project_audio_lock(project_path: str | Path, action: str) -> None:
    if project_audio_lock_is_active(project_path):
        raise ProjectWriteLockError(f"有声章节正在生成中，可以继续续写，但暂时不能{action}。请等音频任务完成后再试。")


def _project_archive_default_path(project_path: Path) -> Path:
    return project_path.parent / f"{project_path.name}.zip"


def _should_export_project_file(path: Path, project_path: Path, archive_path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        if path.resolve() == archive_path.resolve():
            return False
    except OSError:
        pass
    relative = path.relative_to(project_path)
    if any(lock_name in relative.parts for lock_name in PROJECT_LOCK_FILENAMES):
        return False
    if "__pycache__" in relative.parts:
        return False
    name = path.name
    if name.startswith("."):
        return False
    if name.endswith((".tmp", ".part", ".bak")):
        return False
    return True


def _collect_project_export_files(project_path: Path, archive_path: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in project_path.rglob("*")
            if _should_export_project_file(path, project_path, archive_path)
        ],
        key=lambda item: item.relative_to(project_path).as_posix(),
    )


def export_project_archive(project_path: str, archive_path: str | None = None) -> dict:
    base = Path(project_path).resolve()
    project_file = base / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"项目目录中缺少 project.json: {project_path}")
    if _project_lock_is_active(base):
        raise ProjectWriteLockError("当前项目已有写作任务正在运行，暂时不能导出项目包。")

    archive_file = Path(archive_path).expanduser() if archive_path else _project_archive_default_path(base)
    if archive_file.exists() and archive_file.is_dir():
        archive_file = archive_file / f"{base.name}.zip"
    if not archive_file.is_absolute():
        archive_file = archive_file.resolve()
    archive_file.parent.mkdir(parents=True, exist_ok=True)

    project = load_json(str(project_file))
    project_id = str(project.get("project_id") or base.name).strip() or base.name
    root_dir_name = _safe_filename_part(base.name, default=f"novel_project_{_safe_filename_part(project_id)}")
    files = _collect_project_export_files(base, archive_file)
    total_bytes = sum(path.stat().st_size for path in files)
    manifest = {
        "format_version": PROJECT_EXPORT_FORMAT_VERSION,
        "app": "novel_writer",
        "exported_at": utc_now(),
        "project_id": project_id,
        "project_name": project.get("name", ""),
        "project_dir_name": root_dir_name,
        "chapter_count": project.get("chapter_count", 0),
        "complete_project": True,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }

    fd, temp_archive = tempfile.mkstemp(
        prefix=f".{archive_file.stem}_",
        suffix=".zip",
        dir=str(archive_file.parent),
    )
    os.close(fd)
    temp_archive_path = Path(temp_archive)
    try:
        with zipfile.ZipFile(temp_archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                PROJECT_EXPORT_MANIFEST_FILENAME,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )
            for path in files:
                relative = path.relative_to(base).as_posix()
                archive.write(path, f"{root_dir_name}/{relative}")
        os.replace(temp_archive_path, archive_file)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise

    return {
        "archive_path": str(archive_file),
        "project_id": project_id,
        "project_dir_name": root_dir_name,
        "format_version": PROJECT_EXPORT_FORMAT_VERSION,
        "file_count": len(files),
        "size_bytes": archive_file.stat().st_size,
        "total_bytes": total_bytes,
    }


def _validate_zip_member_name(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        raise ValueError(f"项目包中包含非法路径: {name!r}")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise ValueError(f"项目包中包含绝对路径: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"项目包中包含不安全路径: {name!r}")
    return PurePosixPath(name)


def _read_export_manifest(archive: zipfile.ZipFile) -> dict:
    try:
        with archive.open(PROJECT_EXPORT_MANIFEST_FILENAME) as fh:
            manifest = json.loads(fh.read().decode("utf-8"))
    except KeyError as exc:
        raise ValueError("项目包缺少导出清单。") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("项目包导出清单无法解析。") from exc
    if not isinstance(manifest, dict):
        raise ValueError("项目包导出清单格式无效。")
    if int(manifest.get("format_version", 0) or 0) != PROJECT_EXPORT_FORMAT_VERSION:
        raise ValueError(f"不支持的项目包格式版本: {manifest.get('format_version')!r}")
    return manifest


def _validate_project_archive_members(archive: zipfile.ZipFile) -> tuple[dict, str, dict[str, zipfile.ZipInfo]]:
    manifest = _read_export_manifest(archive)
    members: dict[str, zipfile.ZipInfo] = {}
    top_levels: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if info.is_dir():
            continue
        if name == PROJECT_EXPORT_MANIFEST_FILENAME:
            continue
        path = _validate_zip_member_name(name)
        normalized = path.as_posix()
        if normalized in members:
            raise ValueError(f"项目包中包含重复文件: {normalized}")
        parts = path.parts
        if len(parts) < 2:
            raise ValueError("项目包中的项目文件必须位于单个顶层目录下。")
        top_levels.add(parts[0])
        members[normalized] = info

    if len(top_levels) != 1:
        raise ValueError("项目包必须且只能包含一个项目目录。")
    project_root = next(iter(top_levels))
    missing_required = [
        filename
        for filename in PROJECT_IMPORT_REQUIRED_FILES
        if f"{project_root}/{filename}" not in members
    ]
    if missing_required:
        raise ValueError("项目包中的项目目录缺少必需文件: " + ", ".join(missing_required))
    return manifest, project_root, members


def _existing_project_ids(output_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not output_dir.exists():
        return ids
    for path in output_dir.iterdir():
        project_file = path / "project.json"
        if not path.is_dir() or not project_file.exists():
            continue
        try:
            project = load_json(str(project_file))
        except Exception:
            continue
        project_id = str(project.get("project_id") or "").strip()
        if project_id:
            ids.add(project_id)
    return ids


def _import_project_dir_name(project_root: str, project_id: str) -> str:
    if PROJECT_DIR_PATTERN.match(project_root):
        return project_root
    return f"novel_project_{_safe_filename_part(project_id)}"


def _unique_import_project_id(project_id: str, output_dir: Path, existing_ids: set[str]) -> str:
    base_id = _safe_filename_part(project_id, default="imported")
    for _ in range(100):
        candidate = f"{base_id}_import_{uuid4().hex[:8]}"
        candidate_dir = output_dir / f"novel_project_{candidate}"
        if candidate not in existing_ids and not candidate_dir.exists():
            return candidate
    raise RuntimeError("无法为导入项目生成唯一 project_id。")


def _extract_project_archive(
    archive: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    project_root: str,
    target_dir: Path,
) -> int:
    extracted = 0
    target_root = target_dir.resolve()
    prefix = f"{project_root}/"
    for member_name, info in members.items():
        if not member_name.startswith(prefix):
            raise ValueError("项目包包含不属于项目目录的文件。")
        relative_name = member_name[len(prefix) :]
        relative_path = PurePosixPath(relative_name)
        if relative_path.name in PROJECT_LOCK_FILENAMES:
            continue
        target_path = target_dir.joinpath(*relative_path.parts)
        resolved_target = target_path.resolve()
        if not resolved_target.is_relative_to(target_root):
            raise ValueError(f"项目包中包含不安全路径: {member_name!r}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target_path.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        extracted += 1
    return extracted


def _ensure_project_subdirs(project_path: Path) -> None:
    for dirname in PROJECT_STANDARD_DIRS:
        (project_path / dirname).mkdir(parents=True, exist_ok=True)


def import_project_archive(archive_path: str, output_dir: str, conflict_strategy: str = "rename") -> dict:
    if conflict_strategy != "rename":
        raise ValueError("当前仅支持 conflict_strategy='rename'。")

    archive_file = Path(archive_path).expanduser().resolve()
    if not archive_file.exists() or not archive_file.is_file():
        raise FileNotFoundError(f"项目包不存在: {archive_path}")
    if not zipfile.is_zipfile(archive_file):
        raise ValueError("导入文件不是有效的 ZIP 项目包。")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_project_ids(output_path)

    with zipfile.ZipFile(archive_file, "r") as archive:
        manifest, project_root, members = _validate_project_archive_members(archive)
        with tempfile.TemporaryDirectory(prefix=".project_import_", dir=str(output_path)) as temp_dir:
            temp_root = Path(temp_dir)
            temp_project_path = temp_root / project_root
            temp_project_path.mkdir(parents=True, exist_ok=True)
            extracted_count = _extract_project_archive(archive, members, project_root, temp_project_path)

            project_file = temp_project_path / "project.json"
            project = load_json(str(project_file))
            if not isinstance(project, dict):
                raise ValueError("项目包中的 project.json 格式无效。")
            source_project_id = str(project.get("project_id") or manifest.get("project_id") or project_root).strip()
            if not source_project_id:
                source_project_id = project_root
            final_project_id = _safe_project_id(source_project_id)
            candidate_dir_name = _import_project_dir_name(project_root, final_project_id)
            final_dir = output_path / candidate_dir_name
            project_id_normalized = final_project_id != source_project_id
            project_conflicts = final_project_id in existing_ids or final_dir.exists()
            renamed = project_id_normalized or project_conflicts
            if project_conflicts:
                final_project_id = _unique_import_project_id(source_project_id, output_path, existing_ids)
                final_dir = output_path / f"novel_project_{final_project_id}"

            if final_dir.exists():
                raise FileExistsError(f"导入目标目录已存在: {final_dir}")

            project["project_id"] = final_project_id
            project["project_path"] = str(final_dir)
            save_json(str(project_file), project)
            _ensure_project_subdirs(temp_project_path)
            load_project(str(temp_project_path))
            temp_project_path.replace(final_dir)

    return {
        "project_path": str(final_dir),
        "project_id": final_project_id,
        "source_project_id": source_project_id,
        "renamed": renamed,
        "archive_path": str(archive_file),
        "project_dir_name": final_dir.name,
        "source_project_dir_name": project_root,
        "format_version": PROJECT_EXPORT_FORMAT_VERSION,
        "file_count": extracted_count,
    }


def ensure_author_intent(project_path: str) -> dict:
    base = Path(project_path)
    path = base / "author_intent.json"
    if path.exists():
        return _normalize_author_intent(load_json(str(path)))

    project_data = load_project(project_path)
    author_intent = _build_author_intent_from_project(
        project_data["project"],
        project_data["world"],
        project_data["style"],
        project_data["plot_state"],
    )
    save_json(str(path), author_intent)
    return author_intent


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
            "created_at": utc_now(),
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
        plot_state["current_arc"] = "开篇阶段"
        plot_state["current_location"] = ""
        plot_state["current_time"] = ""
        plot_state["recent_events"] = []
        plot_state["open_threads"] = []
        plot_state["resolved_threads"] = []
        plot_state["foreshadowing"] = []
        plot_state["character_updates"] = []
        plot_state["active_characters"] = []
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
        default_value = "" if key in {"current_arc", "current_location", "current_time", "next_chapter_goal"} else []
        value = summary.get(key, default_value)
        if key in {"current_arc", "current_location", "current_time", "next_chapter_goal"}:
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
        "arc_summaries": [],
        "task_cards": [],
        "craft_briefs": [],
        "quality_reviews": [],
        "quality_drafts": [],
        "expert_reviews": [],
        "illustrations": [],
        "audiobook": [],
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

    for task_card_path in sorted((base / "task_cards").glob("chapter_*.json")):
        chapter_number = _parse_numbered_name(task_card_path.name, "chapter_", ".json")
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(task_card_path)
            removed["task_cards"].append(str(task_card_path.relative_to(base)).replace("\\", "/"))

    for craft_brief_path in sorted((base / "craft_briefs").glob("chapter_*.json")):
        chapter_number = _parse_numbered_name(craft_brief_path.name, "chapter_", ".json")
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(craft_brief_path)
            removed["craft_briefs"].append(str(craft_brief_path.relative_to(base)).replace("\\", "/"))

    for review_path in sorted((base / "quality_reviews").glob("chapter_*_attempt_*.json")):
        match = re.fullmatch(r"chapter_(\d{4})_attempt_\d+\.json", review_path.name)
        chapter_number = int(match.group(1)) if match else None
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(review_path)
            removed["quality_reviews"].append(str(review_path.relative_to(base)).replace("\\", "/"))

    for draft_path in sorted((base / "quality_drafts").glob("chapter_*_before_rewrite_*.md")):
        match = re.fullmatch(r"chapter_(\d{4})_before_rewrite_\d+\.md", draft_path.name)
        chapter_number = int(match.group(1)) if match else None
        if chapter_number is not None and chapter_number > keep_chapter_count:
            _remove_path(draft_path)
            removed["quality_drafts"].append(str(draft_path.relative_to(base)).replace("\\", "/"))

    expert_reviews_dir = base / "expert_reviews"
    if expert_reviews_dir.exists():
        for review_dir in sorted(expert_reviews_dir.glob("chapter_*")):
            chapter_number = _parse_numbered_name(review_dir.name, "chapter_", "")
            if chapter_number is not None and chapter_number > keep_chapter_count:
                _remove_path(review_dir)
                removed["expert_reviews"].append(str(review_dir.relative_to(base)).replace("\\", "/"))

    for arc_summary_path in sorted((base / "arc_summaries").glob("arc_*.json")):
        arc_index = _parse_numbered_name(arc_summary_path.name, "arc_", ".json")
        if arc_index is not None and arc_index * 5 > keep_chapter_count:
            _remove_path(arc_summary_path)
            removed["arc_summaries"].append(str(arc_summary_path.relative_to(base)).replace("\\", "/"))

    illustrations_dir = base / "illustrations"
    if illustrations_dir.exists():
        for record_dir in sorted(illustrations_dir.glob("chapter_*")):
            chapter_number = _parse_numbered_name(record_dir.name, "chapter_", "")
            if chapter_number is not None and chapter_number > keep_chapter_count:
                _remove_path(record_dir)
                removed["illustrations"].append(str(record_dir.relative_to(base)).replace("\\", "/"))

    audiobook_dir = base / "audiobook"
    if audiobook_dir.exists():
        for record_dir in sorted(audiobook_dir.glob("chapter_*")):
            chapter_number = _parse_numbered_name(record_dir.name, "chapter_", "")
            if chapter_number is not None and chapter_number > keep_chapter_count:
                _remove_path(record_dir)
                removed["audiobook"].append(str(record_dir.relative_to(base)).replace("\\", "/"))

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
    project_data["updated_at"] = utc_now()
    save_json(str(project_file), project_data)


def rollback_project(project_path: str, to_chapter: int) -> dict:
    base = Path(project_path)
    project_file = base / "project.json"
    if not project_file.exists():
        raise FileNotFoundError(f"项目目录中缺少 project.json: {project_path}")
    ensure_no_project_audio_lock(project_path, "回滚")

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
                "audiobook": [],
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
    project_data["updated_at"] = utc_now()
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
