"""Basic web UI for browsing and continuing novel projects."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from html import escape
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from app import run_next_chapter_from_progression, run_next_chapters
from chapter_context import peek_next_context_for_mode
from common_utils import utc_now
from context_builder import resolve_effective_chapter_task
from illustration_manager import get_illustration_record, illustrate_chapters, list_illustration_records
from progression_manager import (
    CUSTOM_PROGRESSION_OPTION_ID,
    ensure_fresh_progression_session,
    generate_progression_options,
    get_latest_active_progression_session,
    load_progression_session,
)
from project_manager import (
    DEFAULT_PLANNING_MODE,
    PLANNING_MODE_CHAPTER,
    PLANNING_MODE_NONE,
    PLANNING_MODE_VOLUME,
    get_latest_state_snapshot_chapter,
    init_project,
    load_json,
    load_project,
    normalize_planning_mode,
    rollback_project,
)
from runtime_config import (
    WEB_SELECTABLE_PROVIDERS,
    api_key_for_provider as shared_api_key_for_provider,
    build_runtime_config as shared_build_runtime_config,
    default_api_base_for_provider as shared_default_api_base_for_provider,
    default_model_for_provider as shared_default_model_for_provider,
    default_timeout_for_provider as shared_default_timeout_for_provider,
    load_runtime_config as shared_load_runtime_config,
    load_model_presets as shared_load_model_presets,
    normalize_provider as shared_normalize_provider,
    provider_requires_api_key as shared_provider_requires_api_key,
    resolve_timeout_for_provider as shared_resolve_timeout_for_provider,
    sanitize_runtime_overrides,
)
from version import APP_NAME, DISPLAY_VERSION, HTTP_SERVER_TOKEN, WEBUI_NAME
from web_auth import AuthService, LoginAttemptGuard, WebAuthSettings, load_auth_settings


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
API_KEYS_PATH = BASE_DIR / "api_keys.sh"
PROJECT_DIR_PATTERN = re.compile(r"^novel_project_")
MOJIBAKE_HINT_CHARS = set("闆皝绌归《鍙鍦鏄鐨勪簡鍚庡墠闂閿璇浠绗锛銆鈥€")
WEBUI_SERVICE_NAME = os.getenv("NOVEL_WRITER_WEBUI_SERVICE", "novel-writer-webui.service")
ADMIN_LOCALHOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
ADMIN_ACTION_STATUS_FILENAME = "admin_action_status.json"
PUBLIC_PATHS = {"/login", "/healthz"}
API_PATH_PREFIX = "/api/"

JOB_ACTIVE_STATUSES = {"queued", "running"}
JOB_FINISHED_STATUSES = {"succeeded", "failed"}
PROGRESSION_JOB_KINDS = {"progression_options", "progression_options_auto"}

_LOGIN_ATTEMPT_GUARDS: dict[tuple[int, int, int], LoginAttemptGuard] = {}
_LOGIN_ATTEMPT_GUARDS_LOCK = threading.Lock()


class BackgroundJobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def create_job(
        self,
        *,
        kind: str,
        title: str,
        project_id: str = "",
        project_path: str = "",
        busy_message: str = "",
        blocks_project: bool = True,
    ) -> dict:
        with self._lock:
            if project_path and blocks_project:
                active = self._find_active_project_job_locked(project_path, blocking_only=True)
                if active is not None:
                    raise RuntimeError(
                        busy_message
                        or f"当前项目已有后台任务在运行：{active.get('title', active.get('id', 'unknown'))}"
                    )
            job_id = f"job_{uuid4().hex[:10]}"
            job = {
                "id": job_id,
                "kind": kind,
                "title": title,
                "status": "queued",
                "stage": "queued",
                "message": "任务已加入队列，等待后台线程启动",
                "project_id": project_id,
                "project_path": project_path,
                "blocks_project": blocks_project,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "current": 0,
                "total": 0,
                "result_url": "",
                "result_label": "",
                "error": "",
                "events": [],
            }
            self._append_event_locked(job, "queued", job["message"])
            self._jobs[job_id] = job
            return self._copy_job(job)

    def _find_active_project_job_locked(
        self,
        project_path: str,
        *,
        blocking_only: bool = False,
        kinds: set[str] | None = None,
    ) -> dict | None:
        normalized = str(Path(project_path).resolve())
        for job in self._jobs.values():
            if job.get("project_path") != normalized:
                continue
            if job.get("status") not in JOB_ACTIVE_STATUSES:
                continue
            if blocking_only and not bool(job.get("blocks_project", True)):
                continue
            if kinds and job.get("kind") not in kinds:
                continue
            return job
        return None

    def _append_event_locked(self, job: dict, stage: str, message: str) -> None:
        events = job.setdefault("events", [])
        events.append(
            {
                "time": utc_now(),
                "stage": stage,
                "message": message,
            }
        )
        if len(events) > 40:
            del events[:-40]

    def _copy_job(self, job: dict) -> dict:
        copied = dict(job)
        copied["events"] = [dict(item) for item in job.get("events", [])]
        return copied

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            stage = fields.pop("stage", None)
            message = fields.pop("message", None)
            event = fields.pop("event", True)
            if stage is not None:
                job["stage"] = stage
            if message is not None:
                job["message"] = message
            for key, value in fields.items():
                if value is not None:
                    job[key] = value
            job["updated_at"] = utc_now()
            if event and (stage is not None or message is not None):
                self._append_event_locked(job, job.get("stage", ""), job.get("message", ""))

    def mark_running(self, job_id: str, message: str = "后台任务已启动") -> None:
        self.update(job_id, status="running", stage="running", message=message)

    def progress(self, job_id: str, payload: dict) -> None:
        self.update(
            job_id,
            stage=payload.get("stage"),
            message=payload.get("message"),
            current=payload.get("current"),
            total=payload.get("total"),
        )

    def finish_success(
        self,
        job_id: str,
        *,
        message: str,
        result_url: str = "",
        result_label: str = "",
        project_id: str | None = None,
        project_path: str | None = None,
    ) -> None:
        self.update(
            job_id,
            status="succeeded",
            stage="succeeded",
            message=message,
            result_url=result_url,
            result_label=result_label,
            project_id=project_id,
            project_path=str(Path(project_path).resolve()) if project_path else None,
        )

    def finish_failure(self, job_id: str, error: str) -> None:
        self.update(
            job_id,
            status="failed",
            stage="failed",
            message="任务执行失败",
            error=error,
        )

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else self._copy_job(job)

    def list_jobs(
        self,
        *,
        project_id: str = "",
        project_path: str = "",
        active_only: bool = False,
        blocking_only: bool = False,
        kinds: set[str] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        with self._lock:
            jobs = []
            normalized_path = str(Path(project_path).resolve()) if project_path else ""
            for job in self._jobs.values():
                if project_id and job.get("project_id") != project_id:
                    continue
                if normalized_path and job.get("project_path") != normalized_path:
                    continue
                if active_only and job.get("status") not in JOB_ACTIVE_STATUSES:
                    continue
                if blocking_only and not bool(job.get("blocks_project", True)):
                    continue
                if kinds and job.get("kind") not in kinds:
                    continue
                jobs.append(self._copy_job(job))
        jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return jobs[:limit]

    def has_active_project_job(
        self,
        project_path: Path,
        *,
        blocking_only: bool = True,
        kinds: set[str] | None = None,
    ) -> bool:
        return bool(
            self.list_jobs(
                project_path=str(project_path.resolve()),
                active_only=True,
                blocking_only=blocking_only,
                kinds=kinds,
                limit=1,
            )
        )


JOB_REGISTRY = BackgroundJobRegistry()


def _looks_like_mojibake(text: str) -> bool:
  if not text:
    return False
  hint_count = sum(1 for ch in text if ch in MOJIBAKE_HINT_CHARS)
  return hint_count >= max(1, len(text) // 4)


def _mojibake_score(text: str) -> int:
  return sum(1 for ch in text if ch in MOJIBAKE_HINT_CHARS) + text.count("�") * 2


def _repair_display_text(text: str) -> str:
  if not isinstance(text, str) or not _looks_like_mojibake(text):
    return text
  best = text
  best_score = _mojibake_score(text)
  for encoding in ("gb18030", "gbk", "cp1252", "latin1"):
    try:
      repaired = text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
      continue
    if not repaired:
      continue
    repaired_score = _mojibake_score(repaired)
    if repaired_score < best_score:
      best = repaired
      best_score = repaired_score
  return best


def _load_api_keys() -> dict[str, str]:
    env_keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GROK_API_KEY": os.environ.get("GROK_API_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "DOUBAO_API_KEY": os.environ.get("DOUBAO_API_KEY", ""),
        "OLLAMA_API_KEY": os.environ.get("OLLAMA_API_KEY", ""),
    }
    if all(env_keys.values()) or not API_KEYS_PATH.exists():
        return env_keys

    content = API_KEYS_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r'export\s+([A-Z0-9_]+)=("([^"]*)"|\'([^\']*)\')')
    for match in pattern.finditer(content):
        key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(4) or ""
        if key in env_keys and not env_keys[key]:
            env_keys[key] = value
    return env_keys


def _auth_settings() -> WebAuthSettings:
    return load_auth_settings()


def _login_attempt_guard(settings: WebAuthSettings) -> LoginAttemptGuard:
    key = (
        int(settings.login_max_attempts),
        int(settings.login_window_seconds),
        int(settings.login_lockout_seconds),
    )
    with _LOGIN_ATTEMPT_GUARDS_LOCK:
        guard = _LOGIN_ATTEMPT_GUARDS.get(key)
        if guard is None:
            guard = LoginAttemptGuard(
                max_attempts=settings.login_max_attempts,
                window_seconds=settings.login_window_seconds,
                lockout_seconds=settings.login_lockout_seconds,
            )
            _LOGIN_ATTEMPT_GUARDS[key] = guard
        return guard


def _auth_service(settings: WebAuthSettings) -> AuthService:
    return AuthService(settings)


def _safe_next_path(raw_path: str) -> str:
    candidate = str(raw_path or "").strip()
    if not candidate.startswith("/"):
        return "/projects"
    if candidate.startswith("//"):
        return "/projects"
    return candidate


def _is_public_path(path: str) -> bool:
    normalized = str(path or "").strip() or "/"
    return normalized in PUBLIC_PATHS


def _admin_action_status_path() -> Path:
    return OUTPUT_DIR / ADMIN_ACTION_STATUS_FILENAME


def _write_admin_action_status(payload: dict) -> None:
    path = _admin_action_status_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_admin_action_status() -> dict:
    path = _admin_action_status_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _systemd_user_command_env() -> dict[str, str]:
    env = dict(os.environ)
    runtime_dir = str(env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}").strip()
    if runtime_dir:
        env["XDG_RUNTIME_DIR"] = runtime_dir
        env.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
    return env


def _run_checked_command(command: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if completed.returncode != 0:
        raise RuntimeError(stderr or stdout or f"command failed: {' '.join(command)}")
    return stdout


def _client_can_manage_server(
    client_host: str,
    *,
    auth_settings: WebAuthSettings,
    authenticated: bool,
) -> bool:
    host = str(client_host or "").strip().split("%", 1)[0].lower()
    if auth_settings.enabled:
        return authenticated
    return host in ADMIN_LOCALHOSTS or host == "localhost"


def _get_repo_admin_info() -> dict:
    info = {
        "repo_root": str(REPO_ROOT),
        "service_name": WEBUI_SERVICE_NAME,
        "git_available": bool(shutil.which("git")),
        "systemd_run_available": bool(shutil.which("systemd-run")),
        "systemctl_available": bool(shutil.which("systemctl")),
        "branch": "",
        "commit": "",
        "upstream": "",
        "dirty": False,
        "error": "",
    }
    if not info["git_available"]:
        info["error"] = "git 不可用"
        return info
    try:
        info["branch"] = _run_checked_command(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"]
        )
        info["commit"] = _run_checked_command(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]
        )
        try:
            info["upstream"] = _run_checked_command(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
            )
        except Exception:
            info["upstream"] = ""
        dirty_output = _run_checked_command(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"]
        )
        info["dirty"] = bool(dirty_output.strip())
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _launch_admin_task(task: str) -> str:
    if task not in {"restart", "update"}:
        raise RuntimeError("不支持的管理操作。")
    if not shutil.which("systemd-run"):
        raise RuntimeError("当前环境缺少 systemd-run，无法从 Web UI 触发维护操作。")
    status_path = _admin_action_status_path()
    _write_admin_action_status(
        {
            "action": task,
            "status": "queued",
            "message": "维护任务已排队，等待 systemd 启动。",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    unit_name = f"novel-writer-admin-{task}-{uuid4().hex[:8]}"
    command = [
        "systemd-run",
        "--user",
        "--collect",
        "--unit",
        unit_name,
        sys.executable,
        str(Path(__file__).resolve()),
        "--admin-task",
        task,
        "--repo-root",
        str(REPO_ROOT),
        "--service-name",
        WEBUI_SERVICE_NAME,
        "--status-path",
        str(status_path),
    ]
    _run_checked_command(
        command,
        cwd=str(BASE_DIR),
        env=_systemd_user_command_env(),
    )
    return unit_name


def _run_admin_task(task: str, *, repo_root: str, service_name: str, status_path: str) -> int:
    repo_path = Path(repo_root).resolve()
    status_file = Path(status_path).resolve()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    def write_status(status: str, message: str, **extra) -> None:
        payload = {
            "action": task,
            "status": status,
            "message": message,
            "updated_at": utc_now(),
        }
        payload.update(extra)
        status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    started_at = utc_now()
    write_status("running", "维护任务已启动。", started_at=started_at)
    time.sleep(1.0)

    try:
        if task == "restart":
            _run_checked_command(
                ["systemctl", "--user", "restart", service_name],
                env=_systemd_user_command_env(),
            )
            write_status(
                "succeeded",
                "Web UI 已重启。请刷新页面重新连接。",
                started_at=started_at,
                finished_at=utc_now(),
                service_name=service_name,
            )
            return 0

        if task == "update":
            branch = _run_checked_command(["git", "-C", str(repo_path), "branch", "--show-current"])
            if not branch:
                raise RuntimeError("当前仓库不在可更新的分支上（detached HEAD）。")
            upstream = _run_checked_command(
                ["git", "-C", str(repo_path), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"]
            )
            dirty_output = _run_checked_command(["git", "-C", str(repo_path), "status", "--porcelain"])
            if dirty_output.strip():
                raise RuntimeError("当前仓库有未提交改动，已拒绝自动更新。")
            before_commit = _run_checked_command(["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"])
            pull_output = _run_checked_command(["git", "-C", str(repo_path), "pull", "--ff-only"])
            after_commit = _run_checked_command(["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"])
            _run_checked_command(
                ["systemctl", "--user", "restart", service_name],
                env=_systemd_user_command_env(),
            )
            changed = before_commit != after_commit
            message = "代码已更新并重启 Web UI。" if changed else "代码已是最新版本，Web UI 已重启。"
            write_status(
                "succeeded",
                message,
                started_at=started_at,
                finished_at=utc_now(),
                branch=branch,
                upstream=upstream,
                before_commit=before_commit,
                after_commit=after_commit,
                changed=changed,
                pull_output=pull_output,
                service_name=service_name,
            )
            return 0

        raise RuntimeError("不支持的管理操作。")
    except Exception as exc:
        write_status(
            "failed",
            str(exc),
            started_at=started_at,
            finished_at=utc_now(),
            service_name=service_name,
            repo_root=str(repo_path),
        )
        return 1


def _api_key_for_provider(provider: str, api_keys: dict[str, str]) -> str:
    return shared_api_key_for_provider(provider, api_keys)


def _default_model_for_provider(provider: str) -> str:
    return shared_default_model_for_provider(provider)


def _default_api_base_for_provider(provider: str) -> str:
    return shared_default_api_base_for_provider(provider)


def _default_timeout_for_provider(provider: str) -> int:
    return shared_default_timeout_for_provider(provider)


def _load_model_presets() -> dict[str, list[dict[str, str]]]:
    return shared_load_model_presets()


def _normalize_provider_for_ui(provider: object, default: str = "gemini") -> str:
    return shared_normalize_provider(provider, default=default)


def _resolve_model_name_from_form(form: dict[str, str]) -> str:
    custom_model = (form.get("model_name_custom") or "").strip()
    if custom_model:
        return custom_model
    preset_model = (form.get("model_preset") or "").strip()
    if preset_model:
        return preset_model
    return (form.get("model_name") or "").strip()


def _model_blank_label(provider: str, *, base_model: str, provider_explicit: bool) -> str:
    effective_provider = _normalize_provider_for_ui(provider, default="gemini")
    default_model = _default_model_for_provider(effective_provider)
    if provider_explicit:
        if default_model:
            return f"使用 {effective_provider} 默认模型（{default_model}）"
        return f"使用 {effective_provider} 默认模型"
    if base_model:
        return f"沿用项目当前模型（{base_model}）"
    return "沿用项目当前模型"


def _render_model_preset_options(provider: str, *, blank_label: str) -> str:
    presets = _load_model_presets().get(
        _normalize_provider_for_ui(provider, default="gemini"),
        [],
    )
    options = [f'<option value="" selected>{escape(blank_label)}</option>']
    seen_values: set[str] = set()
    for item in presets:
        value = str(item.get("value") or "").strip()
        if not value or value in seen_values:
            continue
        seen_values.add(value)
        label = str(item.get("label") or value).strip() or value
        options.append(f'<option value="{escape(value)}">{escape(label)}</option>')
    return "".join(options)


def _planning_mode_label(mode: str) -> str:
    mapping = {
        PLANNING_MODE_NONE: "无大纲模式",
        PLANNING_MODE_VOLUME: "仅卷纲模式",
        PLANNING_MODE_CHAPTER: "章纲模式",
    }
    return mapping.get(normalize_planning_mode(mode), "仅卷纲模式")


def _planning_mode_help(mode: str) -> str:
    mapping = {
        PLANNING_MODE_NONE: "只参考正文与剧情状态，自由度最高。",
        PLANNING_MODE_VOLUME: "保留长线方向，但不锁死每章任务。",
        PLANNING_MODE_CHAPTER: "严格按分章大纲推进，控制力最强。",
    }
    return mapping.get(normalize_planning_mode(mode), mapping[DEFAULT_PLANNING_MODE])


def _render_planning_mode_options(selected: str, *, include_project_default: bool = False) -> str:
    normalized = normalize_planning_mode(selected)
    options = []
    if include_project_default:
        selected_attr = ' selected' if not selected else ""
        options.append(f'<option value=""{selected_attr}>沿用项目设置</option>')
    for value in (PLANNING_MODE_NONE, PLANNING_MODE_VOLUME, PLANNING_MODE_CHAPTER):
        selected_attr = ' selected' if normalized == value and not include_project_default else ""
        options.append(f'<option value="{value}"{selected_attr}>{_planning_mode_label(value)}</option>')
    return "".join(options)


def _resolve_timeout_for_provider(provider: str, raw_value: object) -> int:
    return shared_resolve_timeout_for_provider(provider, raw_value)


def _provider_requires_api_key(provider: str) -> bool:
    return shared_provider_requires_api_key(provider)


def _list_projects() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for path in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not path.is_dir() or not PROJECT_DIR_PATTERN.match(path.name):
            continue
        project_file = path / "project.json"
        if not project_file.exists():
            continue
        try:
            project = load_json(str(project_file))
        except Exception:
            continue
        projects.append(
            {
                "dir_name": path.name,
                "path": path,
                "project_id": project.get("project_id", path.name),
            "name": _repair_display_text(project.get("name", path.name)),
                "description": project.get("description", ""),
                "chapter_count": project.get("chapter_count", 0),
                "updated_at": project.get("updated_at", ""),
                "provider": (project.get("llm_config") or {}).get("model_provider", ""),
            }
        )
    return projects


def _find_project(project_id: str) -> Path | None:
    for project in _list_projects():
        if project["project_id"] == project_id or project["dir_name"] == project_id:
            return project["path"]
    return None


def _build_runtime_config(project_path: Path, overrides: dict[str, str], api_keys: dict[str, str]) -> dict:
    return shared_build_runtime_config(str(project_path), overrides, api_keys)


def _load_saved_runtime_config(project_path: Path) -> dict:
    return shared_load_runtime_config(str(project_path))


def _runtime_overrides_from_form(form: dict[str, str]) -> dict[str, str]:
    log_llm_payload = bool(form.get("log_llm_payload"))
    return sanitize_runtime_overrides(
        {
            "provider": form.get("provider"),
            "model_name": _resolve_model_name_from_form(form),
            "planning_mode": form.get("planning_mode"),
            "temperature": form.get("temperature"),
            "max_tokens": form.get("max_tokens"),
            "timeout": form.get("timeout"),
            "api_base": form.get("api_base"),
            "log_llm_payload": "1" if log_llm_payload else "",
        }
    )


def _enqueue_progression_job(
    project_id: str,
    project_path: Path,
    runtime_config: dict,
    *,
    user_request: str = "",
    option_count: int = 4,
    runtime_overrides: dict | None = None,
    title: str | None = None,
    auto_generated: bool = False,
) -> dict | None:
    if JOB_REGISTRY.has_active_project_job(
        project_path,
        blocking_only=False,
        kinds=PROGRESSION_JOB_KINDS,
    ):
        return None

    kind = "progression_options_auto" if auto_generated else "progression_options"
    job = JOB_REGISTRY.create_job(
        kind=kind,
        title=title or (f"生成《{project_id}》的下一章推进选项"),
        project_id=project_id,
        project_path=str(project_path.resolve()),
        blocks_project=False,
    )

    def runner(progress_callback):
        session = generate_progression_options(
            str(project_path),
            runtime_config,
            user_request=user_request,
            option_count=option_count,
            runtime_overrides=runtime_overrides,
            progress_callback=progress_callback,
        )
        return {
            "message": f"已生成 {len(session.get('options', []))} 个下一章推进选项。",
            "result_url": "/project/" + urllib.parse.quote(project_id),
            "result_label": "查看项目页",
            "project_id": project_id,
            "project_path": str(project_path.resolve()),
        }

    _start_background_job(job["id"], runner)
    return job


def _render_provider_options(selected: str = "") -> str:
    options = ['<option value="">沿用项目设置</option>']
    for provider in sorted(WEB_SELECTABLE_PROVIDERS):
        selected_attr = ' selected' if selected == provider else ""
        options.append(f'<option value="{provider}"{selected_attr}>{provider}</option>')
    return "".join(options)


def _render_runtime_override_fields(base_provider: str = "gemini", base_model: str = "") -> str:
    effective_provider = _normalize_provider_for_ui(base_provider, default="gemini")
    initial_blank_label = _model_blank_label(
        effective_provider,
        base_model=base_model,
        provider_explicit=False,
    )
    return f"""
    <div class="two-col">
      <label>临时后端覆盖
        <select name="provider" data-model-provider-select data-base-provider="{escape(effective_provider)}">
          {_render_provider_options()}
        </select>
      </label>
      <label>Planning Mode
        <select name="planning_mode">
          {_render_planning_mode_options("", include_project_default=True)}
        </select>
      </label>
    </div>
    <div class="muted">留空则沿用项目设置。none 最自由，volume 更平衡，chapter 控制最强。</div>
    <div class="two-col">
      <label>模型预设
        <select
          name="model_preset"
          data-model-preset-select
          data-base-model="{escape(base_model)}"
        >
          {_render_model_preset_options(effective_provider, blank_label=initial_blank_label)}
        </select>
      </label>
      <label>自定义模型名（可选）
        <input
          type="text"
          name="model_name_custom"
          data-model-custom-input
          placeholder="如需未预设的 Model ID，可在这里手填覆盖"
        >
      </label>
    </div>
    <div class="muted">优先用预设下拉；只有模型不在预设里时，才需要手填自定义 Model ID。</div>
    <div class="two-col">
      <label>API Base（可选）
        <input type="text" name="api_base" placeholder="留空则沿用项目设置">
      </label>
      <label>Timeout
        <input type="number" name="timeout" placeholder="沿用项目设置">
      </label>
    </div>
    <div class="two-col">
      <label>Temperature
        <input type="number" step="0.1" name="temperature" placeholder="沿用项目设置">
      </label>
      <label>Max Tokens
        <input type="number" name="max_tokens" placeholder="沿用项目设置">
      </label>
    </div>
    <label class="muted">
      <input type="checkbox" name="log_llm_payload" value="1">
      启用模型调用落盘（请求与返回将写入项目下 llm_logs，便于排查问题）
    </label>
    """


def _create_project(form: dict[str, str], api_keys: dict[str, str], progress_callback=None) -> str:
    provider = (form.get("provider") or "gemini").strip().lower()
    if provider not in {"gemini", "grok", "deepseek", "doubao", "ollama"}:
        raise RuntimeError(f"unsupported provider: {provider}")

    api_key = _api_key_for_provider(provider, api_keys)
    if not api_key and _provider_requires_api_key(provider):
        raise RuntimeError(f"provider={provider} missing API key, please fill api_keys.sh")

    resolved_model_name = _resolve_model_name_from_form(form)
    config = {
        "project_name": (form.get("project_name") or "Novel Project").strip(),
        "project_description": (form.get("project_description") or "").strip(),
        "project_path": str(OUTPUT_DIR / "novel_project_{project_id}"),
        "init_with_llm": True,
        "story_request": (form.get("story_request") or "").strip(),
        "planning_mode": normalize_planning_mode(form.get("planning_mode")),
        "model_provider": provider,
        "model_name": (resolved_model_name or _default_model_for_provider(provider)).strip(),
        "api_base": (form.get("api_base") or _default_api_base_for_provider(provider)).strip(),
        "api_key": api_key,
        "temperature": float(form.get("temperature") or 0.9),
        "max_tokens": int(form.get("max_tokens") or 4000),
        "timeout": _resolve_timeout_for_provider(provider, form.get("timeout") or _default_timeout_for_provider(provider)),
    }

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    try:
        return init_project(tmp_path, progress_callback=progress_callback)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _read_chapters(project_path: Path) -> list[dict]:
    chapters_dir = project_path / "chapters"
    chapters = []
    for chapter_file in sorted(chapters_dir.glob("chapter_*.md")):
        text = chapter_file.read_text(encoding="utf-8")
        chapters.append(
            {
                "name": chapter_file.name,
                "slug": chapter_file.stem,
                "text": text,
            }
        )
    return chapters


def _chapter_number_from_slug(chapter_slug: str) -> int | None:
    match = re.fullmatch(r"chapter_(\d{4})", (chapter_slug or "").strip())
    if not match:
        return None
    return int(match.group(1))


def _illustration_overrides_from_form(form: dict[str, str]) -> dict:
    mapping = {
        "checkpoint": (form.get("checkpoint") or "").strip(),
        "width": (form.get("width") or "").strip(),
        "height": (form.get("height") or "").strip(),
        "steps": (form.get("steps") or "").strip(),
        "cfg": (form.get("cfg") or "").strip(),
        "comfyui_api_base": (form.get("comfyui_api_base") or "").strip(),
    }
    return {key: value for key, value in mapping.items() if value}


def _job_status_label(status: str) -> str:
    mapping = {
        "queued": "排队中",
        "running": "运行中",
        "succeeded": "已完成",
        "failed": "失败",
    }
    return mapping.get(status, status or "unknown")


def _job_status_class(status: str) -> str:
    mapping = {
        "queued": "status-queued",
        "running": "status-running",
        "succeeded": "status-succeeded",
        "failed": "status-failed",
    }
    return mapping.get(status, "status-neutral")


def _render_job_cards(jobs: list[dict], empty_text: str) -> str:
    if not jobs:
        return f'<p class="muted">{escape(empty_text)}</p>'

    cards = []
    for job in jobs:
        action = ""
        if job.get("result_url"):
            action = (
                f'<a class="job-link" href="{escape(job["result_url"])}">'
                f'{escape(job.get("result_label") or "查看结果")}</a>'
            )
        progress = ""
        current = int(job.get("current") or 0)
        total = int(job.get("total") or 0)
        if total > 0:
            percent = max(0, min(100, int(current * 100 / total)))
            progress = (
                '<div class="job-progress">'
                f'<div class="job-progress-bar" style="width:{percent}%"></div>'
                "</div>"
                f'<div class="muted">进度：{current}/{total}</div>'
            )
        cards.append(
            f"""
            <div class="job-card">
              <div class="job-card-head">
                <a href="/job/{escape(job['id'])}"><strong>{escape(job.get('title', job['id']))}</strong></a>
                <span class="status-pill {escape(_job_status_class(job.get('status', '')))}">{escape(_job_status_label(job.get('status', '')))}</span>
              </div>
              <div class="muted">{escape(job.get('message', '') or '等待状态更新')}</div>
              {progress}
              <div class="muted">更新时间：{escape(job.get('updated_at', ''))}</div>
              {action}
            </div>
            """
        )
    return "".join(cards)


def _render_progression_session(
    project_id: str,
    session: dict | None,
    *,
    disabled: bool,
    active_job: dict | None = None,
) -> str:
    if not session:
        if active_job:
            return (
                '<div class="option-empty-state">'
                '<strong>下一章推进选项正在后台生成</strong>'
                f'<div class="muted">{escape(active_job.get("message", "") or "正在分析当前剧情并生成候选方案。")}</div>'
                f'<div class="muted">更新时间：{escape(active_job.get("updated_at", ""))}</div>'
                f'<div class="muted"><a href="/job/{escape(active_job.get("id", ""))}">查看任务详情</a></div>'
                "</div>"
            )
        return '<p class="muted">还没有已生成的下一章推进选项。先填写偏好并生成 3-5 个候选方案；系统会额外附带 1 个空白自定义项。</p>'

    options_html = []
    recommended_option_id = str(session.get("recommended_option_id", "") or "").strip()
    for index, option in enumerate(session.get("options", []), start=1):
        option_id = str(option.get("option_id", "") or "").strip()
        checked_attr = ' checked' if option_id == recommended_option_id else ""
        if option.get("custom"):
            badge = '<span class="option-badge">自定义</span>'
        else:
            badge = '<span class="option-badge">推荐</span>' if option.get("recommended") else ""
        key_events = "".join(f"<li>{escape(item)}</li>" for item in option.get("key_events", []))
        chapter_outline = option.get("chapter_outline") or {}
        card_class = "option-card custom-option-card" if option.get("custom") else "option-card"
        options_html.append(
            f"""
            <label class="{card_class}">
              <input type="radio" name="progression_option" value="{escape(option_id)}"{checked_attr}>
              <div class="option-card-head">
                <strong>{index}. {escape(option.get('title', ''))}</strong>
                {badge}
              </div>
              <div class="muted">{escape(option.get('summary', ''))}</div>
              <div class="muted">为什么现在：{escape(option.get('why_now', ''))}</div>
              <div class="muted">本章纲要：{escape(chapter_outline.get('goal', '') or chapter_outline.get('summary', ''))}</div>
              <ul class="option-list">{key_events}</ul>
            </label>
            """
        )

    disabled_attr = " disabled" if disabled else ""
    return f"""
    <div class="option-session-meta">
      <div class="muted">当前会话：{escape(session.get('session_id', ''))}</div>
      <div class="muted">目标第 {escape(str(session.get('target_chapter_number', '')))} 章</div>
    </div>
    <form method="post" action="/project/{escape(project_id)}/continue-guided">
      <fieldset{disabled_attr}>
        <input type="hidden" name="progression_session" value="{escape(session.get('session_id', ''))}">
        <div class="option-grid">
          {''.join(options_html)}
        </div>
        <label>补充修改 / 自定义创意
          <textarea name="progression_feedback" placeholder="如果你选了上面的空白自定义项，请在这里直接写这一章想看的创意与情节；如果你选的是普通方案，这里就作为微调补充。"></textarea>
        </label>
        <div class="muted">选择“空白自定义项”后，这段输入会直接作为当前章的主任务；选择普通方案时，它只会作为微调补充。</div>
        <button type="submit">按所选方案续写下一章</button>
      </fieldset>
    </form>
    """


def _render_effective_task_summary(task_card: dict | None) -> str:
    if not task_card:
        return '<p class="muted">当前还没有可用的下一章任务卡。</p>'

    source = str(task_card.get("source", "") or "").strip()
    source_label_map = {
        "progression_selected": "已由推进选项细化",
        "chapter_outline": "来自分章大纲",
        "volume_outline": "来自分卷大纲",
        "plot_state": "来自 live state 的下一目标",
        "freeform": "来自自由续写兜底任务",
    }
    source_label = source_label_map.get(source, source or "未知来源")
    key_events = "".join(f"<li>{escape(item)}</li>" for item in task_card.get("key_events", []))
    derived = task_card.get("derived_from") or {}
    derived_text = ""
    if source == "progression_selected" and derived:
        option_id = str(derived.get("option_id", "") or "").strip()
        session_id = str(derived.get("session_id", "") or "").strip()
        details = " / ".join(part for part in (option_id, session_id) if part)
        if details:
            derived_text = f'<div class="muted">来源记录：{escape(details)}</div>'

    return f"""
    <div class="task-summary-card">
      <div class="option-panel-head">
        <div>
          <h3>有效当前章任务</h3>
          <div class="muted">{escape(source_label)}</div>
          {derived_text}
        </div>
        <span class="pill">第 {escape(str(task_card.get('chapter_number', '')))} 章</span>
      </div>
      <p><strong>当前章目标：</strong>{escape(task_card.get("goal", "") or "暂无")}</p>
      <p><strong>当前章摘要：</strong>{escape(task_card.get("summary", "") or "暂无")}</p>
      <p><strong>卷目标：</strong>{escape(task_card.get("volume_goal", "") or "暂无")}</p>
      <div class="muted"><strong>本章关键事件：</strong></div>
      <ul class="option-list">{key_events or '<li>暂无</li>'}</ul>
    </div>
    """


def _render_admin_panel(
    *,
    client_host: str,
    auth_settings: WebAuthSettings,
    authenticated: bool,
) -> str:
    repo_info = _get_repo_admin_info()
    admin_status = _read_admin_action_status()
    can_manage = _client_can_manage_server(
        client_host,
        auth_settings=auth_settings,
        authenticated=authenticated,
    )

    branch = escape(repo_info.get("branch", "") or "未知")
    commit = escape(repo_info.get("commit", "") or "未知")
    upstream = escape(repo_info.get("upstream", "") or "未配置")
    repo_state = "有未提交改动" if repo_info.get("dirty") else "干净"
    if auth_settings.enabled:
        action_hint = "这些操作会影响整个 Web UI 服务。当前已启用登录鉴权，登录用户可远程执行。"
    else:
        action_hint = "这些操作会影响整个 Web UI 服务。当前未启用登录鉴权，因此仍只允许本机访问。"
    update_disabled = ""
    restart_disabled = ""
    disabled_reason = ""
    if not can_manage:
        update_disabled = " disabled"
        restart_disabled = " disabled"
        if auth_settings.enabled:
            disabled_reason = "当前会话未登录，已禁用管理按钮。"
        else:
            disabled_reason = "当前请求不是本机访问，已禁用管理按钮。"
    elif repo_info.get("error"):
        update_disabled = " disabled"
        disabled_reason = f"仓库状态读取失败：{repo_info['error']}"
    elif repo_info.get("dirty"):
        update_disabled = " disabled"
        disabled_reason = "仓库当前有未提交改动，已禁用自动更新。"
    elif not repo_info.get("upstream"):
        update_disabled = " disabled"
        disabled_reason = "当前分支未配置上游远端，无法自动 pull。"

    status_html = ""
    if admin_status:
        status_class = admin_status.get("status", "")
        status_html = f"""
        <div class="job-card">
          <div class="job-card-head">
            <strong>最近维护结果</strong>
            <span class="status-pill {escape(_job_status_class(status_class))}">{escape(_job_status_label(status_class) if status_class in JOB_ACTIVE_STATUSES | JOB_FINISHED_STATUSES else status_class or "unknown")}</span>
          </div>
          <div class="muted">{escape(admin_status.get("message", "") or "暂无说明")}</div>
          <div class="muted">更新时间：{escape(admin_status.get("updated_at", "") or "未知")}</div>
        </div>
        """

    notice_html = f'<div class="warning-box">{escape(disabled_reason or action_hint)}</div>'
    return f"""
    <section class="panel">
      <div class="option-panel-head">
        <div>
          <h2>维护操作</h2>
          <p class="muted">在页面里直接重启 Web UI，或从远端拉取最新代码后自动重启。</p>
        </div>
        <span class="pill">{escape(WEBUI_SERVICE_NAME)}</span>
      </div>
      <p><strong>当前分支：</strong>{branch}</p>
      <p><strong>当前提交：</strong>{commit}</p>
      <p><strong>跟踪远端：</strong>{upstream}</p>
      <p><strong>仓库状态：</strong>{escape(repo_state)}</p>
      {notice_html}
      <div class="two-col">
        <form method="post" action="/admin/restart">
          <button type="submit"{restart_disabled}>重启 Web UI</button>
        </form>
        <form method="post" action="/admin/update">
          <button type="submit"{update_disabled}>拉取更新并重启</button>
        </form>
      </div>
      <div class="muted">更新使用 `git pull --ff-only`，如果仓库有本地改动或没有配置上游分支，系统会拒绝自动更新。</div>
      {status_html}
    </section>
    """


def _start_background_job(job_id: str, runner) -> None:
    def _target() -> None:
        JOB_REGISTRY.mark_running(job_id)
        try:
            result = runner(lambda payload: JOB_REGISTRY.progress(job_id, payload))
        except Exception as exc:
            error_text = f"{exc}\n\n{traceback.format_exc()}"
            JOB_REGISTRY.finish_failure(job_id, error_text)
            return

        JOB_REGISTRY.finish_success(
            job_id,
            message=result.get("message") or "任务执行完成",
            result_url=result.get("result_url", ""),
            result_label=result.get("result_label", ""),
            project_id=result.get("project_id"),
            project_path=result.get("project_path"),
        )

    thread = threading.Thread(target=_target, name=f"webui-job-{job_id}", daemon=True)
    thread.start()


def _render_page(
    title: str,
    body: str,
    notice: str = "",
    error: str = "",
    *,
    auth_enabled: bool = False,
    authenticated: bool = False,
) -> str:
    flash = ""
    if notice:
        flash += f'<div class="flash notice">{escape(notice)}</div>'
    if error:
        flash += f'<div class="flash error">{escape(error)}</div>'
    topbar_action = '<a href="/projects" class="ghost-button">返回首页</a>'
    if auth_enabled and authenticated:
        topbar_action = """
        <a href="/projects" class="ghost-button">返回首页</a>
        <form method="post" action="/logout" class="inline-form">
          <button type="submit" class="ghost-button">退出登录</button>
        </form>
        """
    model_presets = _load_model_presets()
    default_models = {
        provider: _default_model_for_provider(provider)
        for provider in sorted(model_presets)
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: rgba(255, 251, 244, 0.9);
      --ink: #1d1a16;
      --muted: #6f6254;
      --accent: #b44f2f;
      --accent-dark: #7f331c;
      --line: #d9cdbf;
      --shadow: 0 18px 45px rgba(75, 46, 24, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Noto Serif SC", "Songti SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(234, 191, 138, 0.28), transparent 28%),
        linear-gradient(180deg, #f2ece1 0%, #f7f4ee 40%, #efe6d7 100%);
      min-height: 100vh;
    }}
    a {{ color: var(--accent-dark); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .shell {{
      width: min(1160px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .topbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 22px;
    }}
    .brand {{
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: 0.04em;
      margin: 0;
    }}
    .sub {{
      color: var(--muted);
      margin: 4px 0 0;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(124, 91, 62, 0.15);
      box-shadow: var(--shadow);
      border-radius: 22px;
      padding: 20px;
      backdrop-filter: blur(10px);
    }}
    h2, h3 {{ margin-top: 0; }}
    .flash {{
      margin: 0 0 18px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid transparent;
    }}
    .notice {{
      background: rgba(120, 164, 112, 0.14);
      border-color: rgba(120, 164, 112, 0.28);
    }}
    .error {{
      background: rgba(185, 72, 72, 0.12);
      border-color: rgba(185, 72, 72, 0.28);
    }}
    .project-card {{
      padding: 14px 0;
      border-top: 1px solid var(--line);
    }}
    .project-card:first-of-type {{ border-top: 0; padding-top: 0; }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 6px;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid rgba(180, 79, 47, 0.28);
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      color: var(--accent-dark);
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    form {{
      display: grid;
      gap: 12px;
    }}
    fieldset {{
      border: 0;
      margin: 0;
      padding: 0;
      min-width: 0;
      display: grid;
      gap: 12px;
    }}
    fieldset[disabled] {{
      opacity: 0.62;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 14px;
      color: var(--muted);
    }}
    input, textarea, select, button {{
      font: inherit;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 11px 12px;
      background: rgba(255,255,255,0.88);
      color: var(--ink);
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: linear-gradient(135deg, var(--accent) 0%, #cf7c52 100%);
      color: #fff9f3;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{
      filter: brightness(0.97);
    }}
    .chapter-list a {{
      display: block;
      padding: 10px 12px;
      border-radius: 12px;
      margin-bottom: 8px;
      background: rgba(255,255,255,0.5);
    }}
    .chapter-view {{
      white-space: pre-wrap;
      line-height: 1.9;
      font-size: 17px;
    }}
    .chapter-nav {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 20px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}
    .chapter-nav-link, .chapter-nav-disabled {{
      display: inline-flex;
      align-items: center;
      min-width: 160px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(124, 91, 62, 0.16);
    }}
    .chapter-nav-link.next, .chapter-nav-disabled.next {{
      margin-left: auto;
      justify-content: flex-end;
      text-align: right;
    }}
    .chapter-nav-disabled {{
      color: var(--muted);
      background: rgba(255,255,255,0.38);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .hero {{
      margin-bottom: 18px;
      padding: 18px 20px;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(255, 242, 224, 0.92), rgba(250, 235, 215, 0.78));
      border: 1px solid rgba(180, 79, 47, 0.16);
    }}
    .hero h2 {{
      margin-bottom: 8px;
      font-size: clamp(24px, 3vw, 34px);
    }}
    .stack > * + * {{ margin-top: 18px; }}
    .project-layout {{
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .project-main {{
      min-width: 0;
    }}
    .project-sidebar {{
      min-width: 0;
    }}
    .project-snapshot {{
      display: grid;
      gap: 10px;
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .thumb {{
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 16px;
      border: 1px solid rgba(124, 91, 62, 0.15);
      background: rgba(255,255,255,0.56);
    }}
    .thumb img {{
      display: block;
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid rgba(124, 91, 62, 0.12);
      background: rgba(255,255,255,0.8);
    }}
    .muted {{ color: var(--muted); font-size: 14px; }}
    .warning-box {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(180, 79, 47, 0.08);
      border: 1px solid rgba(180, 79, 47, 0.18);
      color: var(--accent-dark);
      font-size: 14px;
      line-height: 1.6;
    }}
    .inline-form {{
      display: inline;
    }}
    .ghost-button {{
      background: rgba(255,255,255,0.72);
      color: var(--accent-dark);
      border: 1px solid rgba(124, 91, 62, 0.18);
      padding: 10px 14px;
    }}
    .ghost-button:hover {{
      background: rgba(255,255,255,0.9);
    }}
    .topbar-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .brand-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .version-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      color: var(--accent-dark);
      background: rgba(255,255,255,0.76);
      border: 1px solid rgba(124, 91, 62, 0.18);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
    }}
    .login-shell {{
      width: min(560px, 100%);
      margin: 8vh auto 0;
    }}
    .login-card {{
      display: grid;
      gap: 14px;
    }}
    .mono {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    .job-card {{
      display: grid;
      gap: 8px;
      padding: 12px 0;
      border-top: 1px solid var(--line);
    }}
    .job-card:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .job-card-head {{
      display: flex;
      gap: 10px;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid rgba(124, 91, 62, 0.18);
      background: rgba(255,255,255,0.8);
    }}
    .status-queued {{
      color: #855c24;
      background: rgba(239, 214, 169, 0.45);
    }}
    .status-running {{
      color: #2f5c7d;
      background: rgba(168, 206, 236, 0.4);
    }}
    .status-succeeded {{
      color: #35643b;
      background: rgba(170, 214, 176, 0.42);
    }}
    .status-failed {{
      color: #7f2d2d;
      background: rgba(232, 177, 177, 0.4);
    }}
    .job-progress {{
      width: 100%;
      height: 8px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(124, 91, 62, 0.12);
    }}
    .job-progress-bar {{
      height: 100%;
      background: linear-gradient(135deg, var(--accent) 0%, #cf7c52 100%);
    }}
    .job-link {{
      font-size: 14px;
    }}
    .option-panel-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .option-panel-head p {{
      margin: 0;
      max-width: 760px;
    }}
    .option-session-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .option-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }}
    .option-card {{
      display: grid;
      gap: 8px;
      align-content: start;
      min-height: 100%;
      padding: 16px;
      border-radius: 16px;
      border: 1px solid rgba(124, 91, 62, 0.16);
      background: rgba(255,255,255,0.62);
      box-shadow: 0 14px 28px rgba(75, 46, 24, 0.06);
    }}
    .option-card.custom-option-card {{
      border-style: dashed;
      background: rgba(250, 243, 232, 0.92);
    }}
    .option-card input[type="radio"] {{
      width: auto;
      margin: 0;
    }}
    .option-card-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .option-badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      color: #35643b;
      background: rgba(170, 214, 176, 0.42);
      border: 1px solid rgba(120, 164, 112, 0.28);
    }}
    .option-list {{
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
    }}
    .option-empty-state {{
      display: grid;
      gap: 8px;
      padding: 16px 18px;
      border-radius: 16px;
      background: rgba(255,255,255,0.66);
      border: 1px solid rgba(124, 91, 62, 0.14);
    }}
    .status-shell {{
      display: grid;
      gap: 18px;
    }}
    .status-main {{
      display: grid;
      gap: 14px;
    }}
    .status-box {{
      padding: 14px 16px;
      border-radius: 16px;
      background: rgba(255,255,255,0.68);
      border: 1px solid rgba(124, 91, 62, 0.14);
    }}
    .status-message {{
      font-size: 18px;
      line-height: 1.7;
    }}
    .status-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
    }}
    .job-log {{
      margin: 0;
      padding-left: 18px;
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .mono {{
      font-family: Consolas, "SFMono-Regular", monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .button-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    @media (max-width: 920px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .project-layout {{ grid-template-columns: 1fr; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .shell {{ width: min(100% - 20px, 1160px); }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <div class="brand-row">
          <h1 class="brand">{escape(WEBUI_NAME)}</h1>
          <span class="version-pill">版本 {escape(DISPLAY_VERSION)}</span>
        </div>
        <p class="sub">浏览项目、在线阅读章节、直接续写。</p>
      </div>
      <div class="topbar-actions">{topbar_action}</div>
    </div>
    {flash}
    {body}
  </div>
  <script>
    (() => {{
      const modelPresets = {json.dumps(model_presets, ensure_ascii=False)};
      const defaultModels = {json.dumps(default_models, ensure_ascii=False)};
      const normalizeProvider = (provider, fallback = "gemini") => {{
        const value = String(provider || "").trim().toLowerCase();
        return Object.prototype.hasOwnProperty.call(modelPresets, value) ? value : fallback;
      }};
      const buildBlankLabel = (providerSelect, presetSelect) => {{
        const baseProvider = normalizeProvider(providerSelect.dataset.baseProvider || "gemini", "gemini");
        const explicitProvider = String(providerSelect.value || "").trim().toLowerCase();
        const effectiveProvider = normalizeProvider(explicitProvider || baseProvider, baseProvider);
        const baseModel = String(presetSelect.dataset.baseModel || "").trim();
        const defaultModel = String(defaultModels[effectiveProvider] || "").trim();
        if (explicitProvider) {{
          return defaultModel
            ? `使用 ${{effectiveProvider}} 默认模型（${{defaultModel}}）`
            : `使用 ${{effectiveProvider}} 默认模型`;
        }}
        return baseModel ? `沿用项目当前模型（${{baseModel}}）` : "沿用项目当前模型";
      }};
      const bindPresetForm = (providerSelect) => {{
        const form = providerSelect.form || providerSelect.closest("form");
        if (!form) return;
        const presetSelect = form.querySelector("[data-model-preset-select]");
        const customInput = form.querySelector("[data-model-custom-input]");
        if (!presetSelect || !customInput) return;

        const rebuildOptions = () => {{
          const baseProvider = normalizeProvider(providerSelect.dataset.baseProvider || "gemini", "gemini");
          const explicitProvider = String(providerSelect.value || "").trim().toLowerCase();
          const effectiveProvider = normalizeProvider(explicitProvider || baseProvider, baseProvider);
          const entries = Array.isArray(modelPresets[effectiveProvider]) ? modelPresets[effectiveProvider] : [];
          const previousValue = String(presetSelect.value || "");
          const hasCustomValue = Boolean(String(customInput.value || "").trim());
          const seenValues = new Set();

          presetSelect.innerHTML = "";
          const blankOption = document.createElement("option");
          blankOption.value = "";
          blankOption.textContent = buildBlankLabel(providerSelect, presetSelect);
          presetSelect.appendChild(blankOption);

          for (const entry of entries) {{
            const value = String((entry && entry.value) || "").trim();
            if (!value || seenValues.has(value)) continue;
            seenValues.add(value);
            const option = document.createElement("option");
            option.value = value;
            option.textContent = String((entry && entry.label) || value).trim() || value;
            presetSelect.appendChild(option);
          }}

          if (hasCustomValue) {{
            presetSelect.value = "";
            return;
          }}
          presetSelect.value = seenValues.has(previousValue) ? previousValue : "";
        }};

        providerSelect.addEventListener("change", rebuildOptions);
        presetSelect.addEventListener("change", () => {{
          if (presetSelect.value) {{
            customInput.value = "";
          }}
        }});
        customInput.addEventListener("input", () => {{
          if (String(customInput.value || "").trim()) {{
            presetSelect.value = "";
          }}
        }});
        rebuildOptions();
      }};

      document.querySelectorAll("[data-model-provider-select]").forEach(bindPresetForm);
    }})();
  </script>
</body>
</html>
"""


def _render_login_page(
    *,
    error: str = "",
    next_path: str = "/projects",
    auth_settings: WebAuthSettings,
) -> str:
    warnings = []
    auth_service = _auth_service(auth_settings)
    if auth_service.should_warn_default_credentials():
        warnings.append("当前仍在使用默认账号密码，建议你尽快修改配置。")
    if auth_service.should_warn_default_secret_key():
        warnings.append("当前仍在使用默认 secret key，建议你尽快修改配置。")
    warning_html = "".join(f'<div class="warning-box">{escape(item)}</div>' for item in warnings)
    body = f"""
    <div class="login-shell">
      <section class="panel login-card">
        <div class="hero">
          <h2>登录后继续</h2>
          <p class="sub">当前 Web UI 已启用简单鉴权。登录成功后，你就可以继续远程访问项目、续写正文和执行维护操作。</p>
        </div>
        <form method="post" action="/login">
          <input type="hidden" name="next" value="{escape(_safe_next_path(next_path))}">
          <label>用户名
            <input type="text" name="username" autocomplete="username" autofocus>
          </label>
          <label>密码
            <input type="password" name="password" autocomplete="current-password">
          </label>
          <button type="submit">登录</button>
        </form>
        <div class="muted">认证配置文件：<span class="mono">{escape(auth_settings.config_path)}</span></div>
        {warning_html}
      </section>
    </div>
    """
    return _render_page(
        "登录",
        body,
        error=error,
        auth_enabled=auth_settings.enabled,
        authenticated=False,
    )


class NovelWriterHandler(BaseHTTPRequestHandler):
    server_version = HTTP_SERVER_TOKEN

    def _current_auth_settings(self) -> WebAuthSettings:
        return _auth_settings()

    def _auth_cookie_value(self) -> str:
        raw_cookie = self.headers.get("Cookie", "")
        if not raw_cookie:
            return ""
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return ""
        morsel = cookie.get(self._current_auth_settings().cookie_name)
        if morsel is None:
            return ""
        return morsel.value or ""

    def _is_authenticated(self, auth_settings: WebAuthSettings | None = None) -> bool:
        settings = auth_settings or self._current_auth_settings()
        if not settings.enabled:
            return True
        return _auth_service(settings).verify_token(self._auth_cookie_value())

    def _login_attempt_key(self) -> str:
        forwarded = str(self.headers.get("X-Forwarded-For", "") or "").split(",", 1)[0].strip()
        host = forwarded or str(self.client_address[0] or "").strip()
        return host or "unknown"

    def _requires_auth(self, path: str) -> bool:
        settings = self._current_auth_settings()
        return settings.enabled and not _is_public_path(path)

    def _send_standard_headers(self) -> None:
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")

    def _cookie_header_value(
        self,
        name: str,
        value: str,
        *,
        max_age: int | None = None,
        secure: bool = False,
    ) -> str:
        cookie = SimpleCookie()
        cookie[name] = value
        cookie[name]["path"] = "/"
        cookie[name]["httponly"] = True
        cookie[name]["samesite"] = "Lax"
        if secure:
            cookie[name]["secure"] = True
        if max_age is not None:
            cookie[name]["max-age"] = str(max_age)
        return cookie[name].OutputString()

    def _auth_redirect_target(self) -> str:
        path = self.path or "/projects"
        return _safe_next_path(path)

    def _enforce_auth(self, path: str) -> bool:
        settings = self._current_auth_settings()
        if not settings.enabled or _is_public_path(path):
            return True
        if self._is_authenticated(settings):
            return True
        if path.startswith(API_PATH_PREFIX):
            self._write_json({"error": "Unauthorized."}, status=HTTPStatus.UNAUTHORIZED)
            return False
        self._redirect(
            "/login?next=" + urllib.parse.quote(self._auth_redirect_target(), safe="/?=&"),
        )
        return False

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        auth_settings = self._current_auth_settings()
        if parsed.path == "/login":
            self._handle_login_page(parsed)
            return
        if parsed.path == "/healthz":
            self._write_json({"ok": True, "auth_enabled": auth_settings.enabled})
            return
        if not self._enforce_auth(parsed.path):
            return
        params = urllib.parse.parse_qs(parsed.query)
        notice = params.get("notice", [""])[0]
        error = params.get("error", [""])[0]

        if parsed.path in {"/", "/projects"}:
            self._handle_projects(notice=notice, error=error)
            return

        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) == 2 and parts[0] == "job":
            self._handle_job_page(parts[1], notice=notice, error=error)
            return
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "jobs":
            self._handle_job_api(parts[2])
            return
        if len(parts) == 2 and parts[0] == "project":
            self._handle_project_page(parts[1], notice=notice, error=error)
            return
        if len(parts) == 4 and parts[0] == "project" and parts[2] == "chapter":
            self._handle_chapter(parts[1], parts[3], notice=notice, error=error)
            return
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "illustration-file":
            self._handle_illustration_file(parts[1], parts[3], parts[4])
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/login":
            self._handle_login_submit(self._read_form())
            return
        if parsed.path == "/logout":
            self._handle_logout()
            return
        if not self._enforce_auth(parsed.path):
            return
        form = self._read_form()

        if parsed.path == "/projects/create":
            self._handle_create_project_async(form)
            return
        if parsed.path == "/admin/restart":
            self._handle_admin_restart()
            return
        if parsed.path == "/admin/update":
            self._handle_admin_update()
            return

        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "continue":
            self._handle_continue_async(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "progression-options":
            self._handle_progression_options(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "continue-guided":
            self._handle_continue_guided_async(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "rollback":
            self._handle_rollback(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "illustrate":
            self._handle_illustrate_async(parts[1], form)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        parsed = urllib.parse.parse_qs(raw, keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items()}

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self._send_standard_headers()
        self.end_headers()

    def _write_html(self, html: str, *, status: HTTPStatus = HTTPStatus.OK, extra_headers: dict[str, str] | None = None) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._send_standard_headers()
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_admin_restart(self) -> None:
        settings = self._current_auth_settings()
        if not _client_can_manage_server(
            self.client_address[0],
            auth_settings=settings,
            authenticated=self._is_authenticated(settings),
        ):
            reason = "请先登录后再执行管理操作。" if settings.enabled else "管理操作默认只允许本机访问。"
            self._redirect("/projects?error=" + urllib.parse.quote(reason))
            return
        try:
            _launch_admin_task("restart")
            self._redirect(
                "/projects?notice="
                + urllib.parse.quote("已开始重启 Web UI。页面会短暂断开，请在 3 秒后刷新。")
            )
        except Exception as exc:
            self._redirect("/projects?error=" + urllib.parse.quote(str(exc)))

    def _handle_admin_update(self) -> None:
        settings = self._current_auth_settings()
        if not _client_can_manage_server(
            self.client_address[0],
            auth_settings=settings,
            authenticated=self._is_authenticated(settings),
        ):
            reason = "请先登录后再执行管理操作。" if settings.enabled else "管理操作默认只允许本机访问。"
            self._redirect("/projects?error=" + urllib.parse.quote(reason))
            return
        try:
            repo_info = _get_repo_admin_info()
            if repo_info.get("error"):
                raise RuntimeError(f"仓库状态读取失败：{repo_info['error']}")
            if repo_info.get("dirty"):
                raise RuntimeError("当前仓库有未提交改动，请先提交或清理后再更新。")
            if not repo_info.get("upstream"):
                raise RuntimeError("当前分支未配置上游远端，无法自动更新。")
            _launch_admin_task("update")
            self._redirect(
                "/projects?notice="
                + urllib.parse.quote("已开始拉取更新并重启 Web UI。若更新成功，页面会在几秒后恢复。")
            )
        except Exception as exc:
            self._redirect("/projects?error=" + urllib.parse.quote(str(exc)))

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._send_standard_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_error(self, code, message=None, explain=None):
        safe_message = message
        if isinstance(safe_message, str):
            try:
                safe_message.encode("latin-1")
            except UnicodeEncodeError:
                safe_message = None
        return super().send_error(code, safe_message, explain)

    def _write_file(self, file_path: Path) -> None:
        data = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self._send_standard_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_login_page(self, parsed) -> None:
        settings = self._current_auth_settings()
        if not settings.enabled:
            self._redirect("/projects")
            return
        params = urllib.parse.parse_qs(parsed.query)
        next_path = _safe_next_path(params.get("next", ["/projects"])[0])
        if self._is_authenticated(settings):
            self._redirect(next_path)
            return
        self._write_html(
            _render_login_page(
                next_path=next_path,
                auth_settings=settings,
            ),
            extra_headers={"Cache-Control": "no-store"},
        )

    def _handle_login_submit(self, form: dict[str, str]) -> None:
        settings = self._current_auth_settings()
        if not settings.enabled:
            self._redirect("/projects")
            return
        next_path = _safe_next_path(form.get("next", "/projects"))
        attempt_key = self._login_attempt_key()
        guard = _login_attempt_guard(settings)
        if guard.is_locked(attempt_key):
            retry_after = max(1, guard.retry_after_seconds(attempt_key))
            self._write_html(
                _render_login_page(
                    error=f"登录失败次数过多，请 {retry_after} 秒后再试。",
                    next_path=next_path,
                    auth_settings=settings,
                ),
                status=HTTPStatus.TOO_MANY_REQUESTS,
                extra_headers={
                    "Retry-After": str(retry_after),
                    "Cache-Control": "no-store",
                },
            )
            return

        auth_service = _auth_service(settings)
        username = (form.get("username") or "").strip()
        password = form.get("password") or ""
        if not auth_service.verify(username, password):
            guard.register_failure(attempt_key)
            time.sleep(0.8)
            self._write_html(
                _render_login_page(
                    error="账号或密码错误。",
                    next_path=next_path,
                    auth_settings=settings,
                ),
                status=HTTPStatus.UNAUTHORIZED,
                extra_headers={"Cache-Control": "no-store"},
            )
            return

        guard.register_success(attempt_key)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", next_path)
        self._send_standard_headers()
        self.send_header(
            "Set-Cookie",
            self._cookie_header_value(
                settings.cookie_name,
                auth_service.issue_token(),
                max_age=settings.session_max_age_seconds,
                secure=settings.cookie_secure,
            ),
        )
        self.end_headers()

    def _handle_logout(self) -> None:
        settings = self._current_auth_settings()
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", "/login" if settings.enabled else "/projects")
        self._send_standard_headers()
        self.send_header(
            "Set-Cookie",
            self._cookie_header_value(
                settings.cookie_name,
                "",
                max_age=0,
                secure=settings.cookie_secure,
            ),
        )
        self.end_headers()

    def _handle_job_api(self, job_id: str) -> None:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            self._write_json({"error": "job not found"}, status=HTTPStatus.NOT_FOUND)
            return
        self._write_json(job)

    def _handle_job(self, job_id: str, notice: str = "", error: str = "") -> None:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        action_html = ""
        if job.get("result_url"):
            action_html += (
                f'<a href="{escape(job["result_url"])}">{escape(job.get("result_label") or "查看结果")}</a>'
            )
        elif job.get("project_id"):
            action_html += f'<a href="/project/{escape(job["project_id"])}">返回项目页</a>'
        else:
            action_html += '<a href="/projects">返回项目列表</a>'

        body = f"""
        <div class="status-shell">
          <section class="panel status-main">
            <div class="job-card-head">
              <h2>{escape(job.get("title", job_id))}</h2>
              <span id="job-status" class="status-pill {escape(_job_status_class(job.get('status', '')))}">{escape(_job_status_label(job.get('status', '')))}</span>
            </div>
            <div class="status-box">
              <div id="job-message" class="status-message">{escape(job.get("message", ""))}</div>
              <div id="job-progress-wrap">{_render_job_cards([job], "") if int(job.get("total") or 0) > 0 else ""}</div>
            </div>
            <div id="job-meta" class="status-meta">
              <span>创建时间：{escape(job.get("created_at", ""))}</span>
              <span>更新时间：<span id="job-updated">{escape(job.get("updated_at", ""))}</span></span>
            </div>
            <div class="button-row" id="job-actions">{action_html}</div>
            <div id="job-error-box" class="status-box" style="display:{'block' if job.get('error') else 'none'}">
              <strong>错误信息</strong>
              <div id="job-error" class="mono">{escape(job.get("error", ""))}</div>
            </div>
          </section>
          <section class="panel">
            <h3>任务日志</h3>
            <ol id="job-events" class="job-log">
              {''.join(f"<li><span class='mono'>{escape(item.get('time', ''))}</span> {escape(item.get('message', ''))}</li>" for item in job.get("events", []))}
            </ol>
          </section>
        </div>
        <script>
        (() => {{
          const jobId = {json.dumps(job_id)};
          const statusEl = document.getElementById("job-status");
          const messageEl = document.getElementById("job-message");
          const updatedEl = document.getElementById("job-updated");
          const eventsEl = document.getElementById("job-events");
          const actionsEl = document.getElementById("job-actions");
          const errorBoxEl = document.getElementById("job-error-box");
          const errorEl = document.getElementById("job-error");
          const progressWrapEl = document.getElementById("job-progress-wrap");
          const labelMap = {json.dumps({"queued": "排队中", "running": "运行中", "succeeded": "已完成", "failed": "失败"}, ensure_ascii=False)};
          const classMap = {json.dumps({"queued": "status-pill status-queued", "running": "status-pill status-running", "succeeded": "status-pill status-succeeded", "failed": "status-pill status-failed"})};
          const escapeHtml = (value) => String(value ?? "").replace(/[&<>\"']/g, (ch) => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\\"":"&quot;","'":"&#39;"}})[ch]);
          const renderActions = (job) => {{
            if (job.result_url) {{
              return `<a href="${{escapeHtml(job.result_url)}}">${{escapeHtml(job.result_label || "查看结果")}}</a>`;
            }}
            if (job.project_id) {{
              return `<a href="/project/${{encodeURIComponent(job.project_id)}}">返回项目页</a>`;
            }}
            return '<a href="/projects">返回项目列表</a>';
          }};
          const renderProgress = (job) => {{
            const total = Number(job.total || 0);
            const current = Number(job.current || 0);
            if (!total) return "";
            const percent = Math.max(0, Math.min(100, Math.round(current * 100 / total)));
            return `<div class="job-progress"><div class="job-progress-bar" style="width:${{percent}}%"></div></div><div class="muted">进度：${{current}}/${{total}}</div>`;
          }};
          const renderEvents = (events) => events.map((item) => `<li><span class="mono">${{escapeHtml(item.time || "")}}</span> ${{escapeHtml(item.message || "")}}</li>`).join("");
          const update = async () => {{
            const resp = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}`, {{ cache: "no-store" }});
            if (!resp.ok) return;
            const job = await resp.json();
            statusEl.textContent = labelMap[job.status] || job.status || "unknown";
            statusEl.className = classMap[job.status] || "status-pill";
            messageEl.textContent = job.message || "";
            updatedEl.textContent = job.updated_at || "";
            eventsEl.innerHTML = renderEvents(job.events || []);
            actionsEl.innerHTML = renderActions(job);
            progressWrapEl.innerHTML = renderProgress(job);
            if (job.error) {{
              errorBoxEl.style.display = "block";
              errorEl.textContent = job.error;
            }} else {{
              errorBoxEl.style.display = "none";
              errorEl.textContent = "";
            }}
            if (["succeeded", "failed"].includes(job.status)) {{
              clearInterval(timer);
            }}
          }};
          const timer = setInterval(update, 1500);
          update().catch(() => undefined);
        }})();
        </script>
        """
        self._write_html(
            _render_page(
                f"任务状态 - {job.get('title', job_id)}",
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_projects(self, notice: str = "", error: str = "") -> None:
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)
        projects = _list_projects()
        cards = []
        for item in projects:
            cards.append(
                f"""
                <div class="project-card">
                  <div><a href="/project/{escape(item['project_id'])}"><strong>{escape(item['name'])}</strong></a></div>
                  <div class="meta">{escape(item['description'] or '暂无简介')}</div>
                  <div class="meta">
                    <span class="pill">{escape(item['provider'] or 'unknown')}</span>
                    <span class="pill">{item['chapter_count']} 章</span>
                    <span class="pill">{escape(item['updated_at'] or '')}</span>
                  </div>
                </div>
                """
            )
        project_html = "".join(cards) or "<p>当前还没有项目，先在左侧创建一个新项目吧。</p>"
        recent_jobs_html = _render_job_cards(
            JOB_REGISTRY.list_jobs(limit=6),
            "当前还没有后台任务。",
        )
        admin_panel_html = _render_admin_panel(
            client_host=self.client_address[0],
            auth_settings=auth_settings,
            authenticated=authenticated,
        )

        body = f"""
        <div class="grid">
          <div class="stack">
            <section class="panel">
              <h2>新建项目</h2>
              <form method="post" action="/projects/create">
                <div class="two-col">
                  <label>模型后端
                    <select name="provider" data-model-provider-select data-base-provider="gemini">
                      <option value="gemini" selected>gemini</option>
                      <option value="grok">grok</option>
                      <option value="deepseek">deepseek</option>
                      <option value="doubao">doubao</option>
                      <option value="ollama">ollama</option>
                    </select>
                  </label>
                  <label>模型预设
                    <select name="model_preset" data-model-preset-select data-base-model="">
                      {_render_model_preset_options("gemini", blank_label=_model_blank_label("gemini", base_model="", provider_explicit=True))}
                    </select>
                  </label>
                </div>
                <label>自定义模型名（可选）
                  <input type="text" name="model_name_custom" data-model-custom-input placeholder="如需未预设的 Model ID，可在这里手填覆盖">
                </label>
                <div class="muted">默认可直接使用预设下拉，不需要每次手填 Model ID。</div>
                <label>项目名
                  <input type="text" name="project_name" value="雪封穹顶">
                </label>
                <label>项目简介
                  <input type="text" name="project_description" value="由模型根据需求自动生成设定的长篇小说项目。">
                </label>
                <label>故事需求
                  <textarea name="story_request" placeholder="把你想写的题材、角色、世界观、节奏偏好写在这里"></textarea>
                </label>
                <div class="two-col">
                  <label>Temperature
                    <input type="number" step="0.1" name="temperature" value="0.9">
                  </label>
                  <label>Max Tokens
                    <input type="number" name="max_tokens" value="4000">
                  </label>
                </div>
                <div class="two-col">
                  <label>Timeout
                    <input type="number" name="timeout" value="120">
                  </label>
                  <label>API Base（可选）
                    <input type="text" name="api_base" placeholder="如需自定义接口地址可填写">
                  </label>
                </div>
                <label>Planning Mode
                  <select name="planning_mode">
                    {_render_planning_mode_options(DEFAULT_PLANNING_MODE)}
                  </select>
                </label>
                <div class="muted">{escape(_planning_mode_help(DEFAULT_PLANNING_MODE))}</div>
                <button type="submit">创建项目</button>
              </form>
            </section>
            {admin_panel_html}
          </div>
          <section class="panel">
            <div class="hero">
              <h2>项目书架</h2>
              <p class="sub">这里会列出 `output/` 目录中的全部小说项目。点击即可阅读和续写。</p>
            </div>
            {project_html}
            <div class="project-card">
              <h3>后台任务</h3>
              {recent_jobs_html}
            </div>
          </section>
        </div>
        """
        self._write_html(
            _render_page(
                "项目列表",
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_project(self, project_id: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        data = load_project(str(project_path))
        project = data["project"]
        project_name = _repair_display_text(project.get("name", project_id))
        plot_state = data["plot_state"]
        planning_mode = normalize_planning_mode(project.get("planning_mode", DEFAULT_PLANNING_MODE))
        effective_task = resolve_effective_chapter_task(
            str(project_path),
            data,
            peek_next_context_for_mode(data, planning_mode),
            planning_mode=planning_mode,
            persist=False,
        )
        effective_task_html = _render_effective_task_summary(effective_task)
        chapters = _read_chapters(project_path)
        latest_snapshot = get_latest_state_snapshot_chapter(str(project_path))
        stats = (project.get("stats") or {}).get("total", {})
        illustration_records = list_illustration_records(str(project_path))
        active_jobs = JOB_REGISTRY.list_jobs(project_id=project_id, active_only=True, limit=6)
        active_jobs_html = _render_job_cards(active_jobs, "当前没有运行中的后台任务。")
        project_busy = bool(active_jobs)
        busy_attr = " disabled" if project_busy else ""
        busy_notice = (
            '<div class="warning-box">当前项目有后台任务正在运行。为避免并发写入冲突，续写、回滚和插图表单已暂时禁用。你可以打开下方任务卡片查看实时进度。</div>'
            if project_busy
            else ""
        )

        chapter_links = "".join(
            f"<a href=\"/project/{escape(project_id)}/chapter/{escape(chapter['slug'])}\">{escape(chapter['name'])}</a>"
            for chapter in chapters
        ) or "<p>还没有章节。</p>"

        chapter_options = ['<option value="latest">最新章节</option>'] + [
            f"<option value=\"{escape(chapter['slug'])}\">{escape(chapter['name'])}</option>"
            for chapter in chapters
        ]

        illustration_cards = []
        for record in illustration_records[:6]:
            chapter_slug = str(record.get("chapter_slug", ""))
            images = record.get("images") or []
            if not chapter_slug or not images:
                continue
            image = images[0]
            image_url = (
                f"/project/{urllib.parse.quote(project_id)}/illustration-file/"
                f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(image.get('file_name', ''))}"
            )
            illustration_cards.append(
                f"""
                <div class="thumb">
                  <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}"><img src="{image_url}" alt="{escape(chapter_slug)}"></a>
                  <div><strong>{escape(chapter_slug)}</strong></div>
                  <div class="muted">{escape(record.get('scene_summary', '') or '已生成插图')}</div>
                </div>
                """
            )
        illustration_gallery = "".join(illustration_cards) or "<p>当前还没有章节插图。</p>"

        body = f"""
        <div class="grid">
          <aside class="stack">
            <section class="panel">
              <h2>{escape(project_name)}</h2>
              <p class="meta">{escape(project.get("description", ""))}</p>
              <p class="meta"><span class="pill">{escape((project.get("llm_config") or {}).get("model_provider", ""))}</span><span class="pill">{project.get("chapter_count", 0)} 章</span></p>
              <p><strong>状态快照：</strong>{escape(f'已保存到第 {latest_snapshot} 章' if latest_snapshot is not None else '暂无')}</p>
              <p><strong>下章目标：</strong>{escape(plot_state.get("next_chapter_goal", "") or "暂无")}</p>
              <p><strong>当前地点：</strong>{escape(plot_state.get("current_location", "") or "未知")}</p>
              <p><strong>当前时间：</strong>{escape(plot_state.get("current_time", "") or "未知")}</p>
              <p><strong>请求：</strong>{stats.get("requests", 0)} 次</p>
              <p><strong>Token：</strong>{stats.get("total_tokens", 0)}</p>
              <p><strong>Planning:</strong>{escape(_planning_mode_label(project.get("planning_mode", DEFAULT_PLANNING_MODE)))}</p>
            </section>
            <section class="panel">
              <h3>后台任务</h3>
              {active_jobs_html}
            </section>
            {busy_notice}
            <section class="panel">
              <h3>续写</h3>
              <form method="post" action="/project/{escape(project_id)}/continue">
                <fieldset{busy_attr}>
                <div class="two-col">
                  <label>续写章节数
                    <input type="number" name="count" value="1" min="1" max="20">
                  </label>
                  <label>临时后端覆盖
                    <select name="provider">
                      <option value="">沿用项目设置</option>
                      <option value="gemini">gemini</option>
                      <option value="grok">grok</option>
                      <option value="deepseek">deepseek</option>
                      <option value="doubao">doubao</option>
                      <option value="ollama">ollama</option>
                    </select>
                  </label>
                </div>
                <label>想看的内容 / 情节走向
                  <textarea name="user_request" placeholder="例如：先推进食堂据点建设，再增加一点轻松互怼的互动。"></textarea>
                </label>
                <div class="two-col">
                  <label>模型名（可选）
                    <input type="text" name="model_name" placeholder="留空则沿用项目设置；切换后端时自动改为该后端默认模型 / Model ID">
                  </label>
                  <label>API Base（可选）
                    <input type="text" name="api_base" placeholder="留空则沿用项目设置">
                  </label>
                </div>
                <label>Planning Mode
                  <select name="planning_mode">
                    {_render_planning_mode_options("", include_project_default=True)}
                  </select>
                </label>
                <div class="muted">Leave blank to use the project default. none is the freest, volume is balanced, chapter is the most controlled.</div>
                <div class="two-col">
                  <label>Temperature
                    <input type="number" step="0.1" name="temperature" placeholder="沿用项目设置">
                  </label>
                  <label>Max Tokens
                    <input type="number" name="max_tokens" placeholder="沿用项目设置">
                  </label>
                </div>
                <label>Timeout
                  <input type="number" name="timeout" placeholder="沿用项目设置">
                </label>
                <label><input type="checkbox" name="illustrate_generated" value="1"> 续写完成后立即调用 ComfyUI 生成插图</label>
                <label>插图额外要求（可选）
                  <input type="text" name="illustration_request" placeholder="例如：突出雪夜窗景与室内暖光反差。">
                </label>
                <button type="submit">开始续写</button>
              </form>
            </section>
            <section class="panel">
              <h3>状态回滚</h3>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>状态回滚</h3>
              <form method="post" action="/project/{escape(project_id)}/rollback">
                <fieldset{busy_attr}>
                <label>保留到第几章
                  <input type="number" name="to_chapter" value="{max(0, int(project.get('chapter_count', 0) or 0))}" min="0" max="{max(0, int(project.get('chapter_count', 0) or 0))}">
                </label>
                <div class="warning-box">
                  回滚会删除目标章节之后的正文、摘要、章节插图和更晚的状态快照。回滚完成后，可以直接继续续写，从保留章节的状态往后写新版本。
                </div>
                <button class="ghost-button" type="submit">回滚到该章节</button>
              </form>
            </section>
            <section class="panel">
              <h3>生成插图</h3>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>生成插图</h3>
              <form method="post" action="/project/{escape(project_id)}/illustrate">
                <fieldset{busy_attr}>
                <label>目标章节
                  <select name="chapter_slug">
                    {''.join(chapter_options)}
                  </select>
                </label>
                <label>插图要求（可选）
                  <textarea name="user_request" placeholder="例如：更强调角色站位、情绪与镜头感。"></textarea>
                </label>
                <div class="two-col">
                  <label>Checkpoint（可选）
                    <input type="text" name="checkpoint" placeholder="illusious/illustrij_v21.safetensors">
                  </label>
                  <label>ComfyUI API（可选）
                    <input type="text" name="comfyui_api_base" placeholder="http://127.0.0.1:8188">
                  </label>
                </div>
                <div class="two-col">
                  <label>宽度
                    <input type="number" name="width" placeholder="832">
                  </label>
                  <label>高度
                    <input type="number" name="height" placeholder="1216">
                  </label>
                </div>
                <div class="two-col">
                  <label>Steps
                    <input type="number" name="steps" placeholder="28">
                  </label>
                  <label>CFG
                    <input type="number" step="0.1" name="cfg" placeholder="6.5">
                  </label>
                </div>
                <label><input type="checkbox" name="force" value="1"> 强制重绘</label>
                <button type="submit">为章节生成插图</button>
              </form>
            </section>
            <section class="panel">
              <h3>章节目录</h3>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>章节目录</h3>
              <div class="chapter-list">{chapter_links}</div>
            </section>
          </aside>
          <main class="stack">
            <section class="panel">
              <h2>剧情状态</h2>
              <div class="chapter-view">{escape(json.dumps(plot_state, ensure_ascii=False, indent=2))}</div>
            </section>
            <section class="panel">
              <h2>最近一章</h2>
              <div class="chapter-view">{escape(chapters[-1]["text"]) if chapters else "还没有正文。"}</div>
            </section>
            <section class="panel">
              <h2>最近插图</h2>
              <div class="gallery">{illustration_gallery}</div>
            </section>
          </main>
        </div>
        """
        self._write_html(
            _render_page(
                project_name,
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_chapter(self, project_id: str, chapter_slug: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        chapter_file = project_path / "chapters" / f"{chapter_slug}.md"
        if not chapter_file.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "章节不存在")
            return

        project = load_json(str(project_path / "project.json"))
        project_name = _repair_display_text(project.get("name", project_id))
        chapters = _read_chapters(project_path)
        chapter_number = _chapter_number_from_slug(chapter_slug)
        current_index = next((idx for idx, chapter in enumerate(chapters) if chapter["slug"] == chapter_slug), -1)
        previous_chapter = chapters[current_index - 1] if current_index > 0 else None
        next_chapter = chapters[current_index + 1] if 0 <= current_index < len(chapters) - 1 else None
        previous_link = (
            f'<a class="chapter-nav-link prev" href="/project/{escape(project_id)}/chapter/{escape(previous_chapter["slug"])}">← 上一章：{escape(previous_chapter["name"])}</a>'
            if previous_chapter
            else '<span class="chapter-nav-disabled prev">← 已是第一章</span>'
        )
        next_link = (
            f'<a class="chapter-nav-link next" href="/project/{escape(project_id)}/chapter/{escape(next_chapter["slug"])}">下一章：{escape(next_chapter["name"])} →</a>'
            if next_chapter
            else '<span class="chapter-nav-disabled next">已是最后一章 →</span>'
        )
        illustration_record = get_illustration_record(str(project_path), chapter_slug)
        illustration_gallery = "<p>当前还没有本章插图。</p>"
        if illustration_record and illustration_record.get("images"):
            cards = []
            for image in illustration_record.get("images", []):
                image_url = (
                    f"/project/{urllib.parse.quote(project_id)}/illustration-file/"
                    f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(image.get('file_name', ''))}"
                )
                cards.append(
                    f"""
                    <div class="thumb">
                      <a href="{image_url}"><img src="{image_url}" alt="{escape(chapter_slug)}"></a>
                      <div class="muted">{escape(illustration_record.get('scene_summary', '') or '章节插图')}</div>
                    </div>
                    """
                )
            illustration_gallery = "".join(cards)

        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}">返回项目</a>
            <h2>{escape(chapter_file.name)}</h2>
            <div class="chapter-view">{escape(chapter_file.read_text(encoding="utf-8"))}</div>
            <div class="chapter-nav">
              {previous_link}
              {next_link}
            </div>
            <div class="warning-box">
              如果你希望以本章作为新的分叉点继续写，可以直接回滚到本章状态。这样会删除后续章节及其摘要、插图和快照。
            </div>
            <form method="post" action="/project/{escape(project_id)}/rollback">
              <input type="hidden" name="to_chapter" value="{chapter_number if chapter_number is not None else 0}">
              <input type="hidden" name="return_to_chapter" value="{escape(chapter_slug)}">
              <button class="ghost-button" type="submit">回滚到本章并从这里继续写</button>
            </form>
          </section>
          <section class="panel">
            <h2>本章插图</h2>
            <div class="gallery">{illustration_gallery}</div>
            <form method="post" action="/project/{escape(project_id)}/illustrate">
              <input type="hidden" name="chapter_slug" value="{escape(chapter_slug)}">
              <label>插图要求（可选）
                <input type="text" name="user_request" placeholder="例如：强调人物表情与空间纵深。">
              </label>
              <div class="two-col">
                <label>Checkpoint（可选）
                  <input type="text" name="checkpoint" placeholder="illusious/illustrij_v21.safetensors">
                </label>
                <label>ComfyUI API（可选）
                  <input type="text" name="comfyui_api_base" placeholder="http://127.0.0.1:8188">
                </label>
              </div>
              <label><input type="checkbox" name="force" value="1"> 强制重绘</label>
              <button type="submit">为本章生成插图</button>
            </form>
          </section>
        </div>
        """
        self._write_html(
            _render_page(
                f"{project_name} - {chapter_file.name}",
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_illustration_file(self, project_id: str, chapter_slug: str, file_name: str) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        illustrations_root = (project_path / "illustrations").resolve()
        file_path = (illustrations_root / chapter_slug / file_name).resolve()
        if not file_path.exists() or illustrations_root not in file_path.parents:
            self.send_error(HTTPStatus.NOT_FOUND, "插图不存在")
            return
        self._write_file(file_path)

    def _handle_illustrate(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            llm_runtime_config = None
            try:
                llm_runtime_config = _build_runtime_config(project_path, {}, api_keys)
            except Exception:
                llm_runtime_config = None
            chapter_slug = (form.get("chapter_slug") or "latest").strip() or "latest"
            results = illustrate_chapters(
                str(project_path),
                chapter_refs=[chapter_slug],
                llm_config=llm_runtime_config,
                user_request=(form.get("user_request") or "").strip(),
                force=bool(form.get("force")),
                overrides=_illustration_overrides_from_form(form),
            )
            result = results[0]
            chapter_target = result.get("chapter_slug", chapter_slug)
            state = "已复用现有插图。" if result.get("reused") else "插图生成完成。"
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "/chapter/"
                + urllib.parse.quote(chapter_target)
                + "?notice="
                + urllib.parse.quote(state)
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )

    def _handle_create_project(self, form: dict[str, str]) -> None:
        api_keys = _load_api_keys()
        try:
            if not (form.get("story_request") or "").strip():
                raise RuntimeError("故事需求不能为空。")
            project_path = _create_project(form, api_keys)
            project_id = load_json(str(Path(project_path) / "project.json")).get("project_id", Path(project_path).name)
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote("项目创建成功。")
            )
        except Exception as exc:
            self._redirect("/projects?error=" + urllib.parse.quote(str(exc)))

    def _handle_continue(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            count = int(form.get("count") or "1")
            if count < 1:
                raise RuntimeError("续写章节数必须至少为 1。")
            runtime_config = _build_runtime_config(project_path, _runtime_overrides_from_form(form), api_keys)
            chapter_paths = run_next_chapters(
                str(project_path),
                runtime_config,
                count,
                user_request=(form.get("user_request") or "").strip(),
            )
            notice = f"续写完成，共生成 {count} 章。"
            if form.get("illustrate_generated"):
                illustration_results = illustrate_chapters(
                    str(project_path),
                    chapter_refs=chapter_paths,
                    llm_config=runtime_config,
                    user_request=(form.get("illustration_request") or "").strip(),
                    overrides=None,
                )
                new_count = sum(0 if item.get("reused") else 1 for item in illustration_results)
                notice += f" 插图处理完成（新生成 {new_count} 章插图）。"
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote(notice)
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )

    def _handle_rollback(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        try:
            to_chapter = int((form.get("to_chapter") or "").strip())
        except ValueError:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote("回滚目标章节必须是非负整数。")
            )
            return

        try:
            result = rollback_project(str(project_path), to_chapter)
            removed = result.get("removed") or {}
            notice = (
                f"项目已回滚到第 {result.get('target_chapter_count', 0)} 章。"
                f" 已清理 {len(removed.get('chapters', []))} 个后续章节。"
            )
            return_to_chapter = (form.get("return_to_chapter") or "").strip()
            if to_chapter > 0 and return_to_chapter and _chapter_number_from_slug(return_to_chapter) == to_chapter:
                self._redirect(
                    "/project/"
                    + urllib.parse.quote(project_id)
                    + "/chapter/"
                    + urllib.parse.quote(return_to_chapter)
                    + "?notice="
                    + urllib.parse.quote(notice)
                )
                return
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote(notice)
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )


    def _handle_job_page(self, job_id: str, notice: str = "", error: str = "") -> None:
        job = JOB_REGISTRY.get(job_id)
        if job is None:
            self.send_error(HTTPStatus.NOT_FOUND, "任务不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        body = f"""
        <div class="status-shell">
          <section class="panel status-main">
            <div class="job-card-head">
              <h2>{escape(job.get("title", job_id))}</h2>
              <span id="job-status" class="status-pill {escape(_job_status_class(job.get('status', '')))}">{escape(_job_status_label(job.get('status', '')))}</span>
            </div>
            <div class="status-box">
              <div id="job-message" class="status-message">{escape(job.get("message", ""))}</div>
              <div id="job-progress-wrap"></div>
            </div>
            <div class="status-meta">
              <span>创建时间：{escape(job.get("created_at", ""))}</span>
              <span>更新时间：<span id="job-updated">{escape(job.get("updated_at", ""))}</span></span>
            </div>
            <div class="button-row" id="job-actions"></div>
            <div id="job-error-box" class="status-box" style="display:none">
              <strong>错误信息</strong>
              <div id="job-error" class="mono"></div>
            </div>
          </section>
          <section class="panel">
            <h3>任务日志</h3>
            <ol id="job-events" class="job-log"></ol>
          </section>
        </div>
        <script>
        (() => {{
          const jobId = {json.dumps(job_id)};
          const labelMap = {json.dumps({"queued": "排队中", "running": "运行中", "succeeded": "已完成", "failed": "失败"}, ensure_ascii=False)};
          const classMap = {json.dumps({"queued": "status-pill status-queued", "running": "status-pill status-running", "succeeded": "status-pill status-succeeded", "failed": "status-pill status-failed"})};
          const statusEl = document.getElementById("job-status");
          const messageEl = document.getElementById("job-message");
          const updatedEl = document.getElementById("job-updated");
          const eventsEl = document.getElementById("job-events");
          const actionsEl = document.getElementById("job-actions");
          const errorBoxEl = document.getElementById("job-error-box");
          const errorEl = document.getElementById("job-error");
          const progressWrapEl = document.getElementById("job-progress-wrap");
          let redirectTimer = null;
          const escapeHtml = (value) => String(value ?? "").replace(/[&<>\"']/g, (ch) => {{
            if (ch === "&") return "&amp;";
            if (ch === "<") return "&lt;";
            if (ch === ">") return "&gt;";
            if (ch === "\"") return "&quot;";
            return "&#39;";
          }});
          const renderEvents = (events) => (events || []).map((item) => `<li><span class="mono">${{escapeHtml(item.time || "")}}</span> ${{escapeHtml(item.message || "")}}</li>`).join("");
          const renderActions = (job) => {{
            if (job.result_url) {{
              return `<a href="${{escapeHtml(job.result_url)}}">${{escapeHtml(job.result_label || "查看结果")}}</a>`;
            }}
            if (job.project_id) {{
              return `<a href="/project/${{encodeURIComponent(job.project_id)}}">返回项目页</a>`;
            }}
            return '<a href="/projects">返回项目列表</a>';
          }};
          const renderProgress = (job) => {{
            const total = Number(job.total || 0);
            const current = Number(job.current || 0);
            if (!total) return "";
            const percent = Math.max(0, Math.min(100, Math.round(current * 100 / total)));
            return `<div class="job-progress"><div class="job-progress-bar" style="width:${{percent}}%"></div></div><div class="muted">进度：${{current}}/${{total}}</div>`;
          }};
          const update = async () => {{
            const resp = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}`, {{ cache: "no-store" }});
            if (!resp.ok) return;
            const job = await resp.json();
            statusEl.textContent = labelMap[job.status] || job.status || "unknown";
            statusEl.className = classMap[job.status] || "status-pill";
            messageEl.textContent = job.message || "";
            updatedEl.textContent = job.updated_at || "";
            eventsEl.innerHTML = renderEvents(job.events);
            actionsEl.innerHTML = renderActions(job);
            progressWrapEl.innerHTML = renderProgress(job);
            if (job.error) {{
              errorBoxEl.style.display = "block";
              errorEl.textContent = job.error;
            }} else {{
              errorBoxEl.style.display = "none";
              errorEl.textContent = "";
            }}
            if (job.status === "succeeded" && job.result_url) {{
              if (redirectTimer === null) {{
                redirectTimer = window.setTimeout(() => {{
                  window.location.href = job.result_url;
                }}, 1200);
              }}
            }} else if (redirectTimer !== null) {{
              window.clearTimeout(redirectTimer);
              redirectTimer = null;
            }}
            if (["succeeded", "failed"].includes(job.status)) {{
              clearInterval(timer);
            }}
          }};
          const timer = setInterval(update, 1500);
          update().catch(() => undefined);
        }})();
        </script>
        """
        self._write_html(
            _render_page(
                f"任务状态 - {job.get('title', job_id)}",
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_project_page(self, project_id: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        data = load_project(str(project_path))
        project = data["project"]
        project_name = _repair_display_text(project.get("name", project_id))
        plot_state = data["plot_state"]
        planning_mode = normalize_planning_mode(project.get("planning_mode", DEFAULT_PLANNING_MODE))
        effective_task = resolve_effective_chapter_task(
            str(project_path),
            data,
            peek_next_context_for_mode(data, planning_mode),
            planning_mode=planning_mode,
            persist=False,
        )
        effective_task_html = _render_effective_task_summary(effective_task)
        chapters = _read_chapters(project_path)
        latest_snapshot = get_latest_state_snapshot_chapter(str(project_path))
        stats = (project.get("stats") or {}).get("total", {})
        illustration_records = list_illustration_records(str(project_path))
        active_jobs = JOB_REGISTRY.list_jobs(project_id=project_id, active_only=True, limit=8)
        active_jobs_html = _render_job_cards(active_jobs, "当前没有运行中的后台任务。")
        blocking_jobs = [job for job in active_jobs if job.get("blocks_project", True)]
        progression_jobs = [job for job in active_jobs if job.get("kind") in PROGRESSION_JOB_KINDS]
        project_busy = bool(blocking_jobs)
        progression_session = get_latest_active_progression_session(str(project_path))
        project_llm_config = project.get("llm_config") or {}
        runtime_override_fields_html = _render_runtime_override_fields(
            str(project_llm_config.get("model_provider") or ""),
            str(project_llm_config.get("model_name") or project_llm_config.get("model") or ""),
        )
        guided_session_html = _render_progression_session(
            project_id,
            progression_session,
            disabled=project_busy,
            active_job=progression_jobs[0] if progression_jobs else None,
        )
        busy_attr = " disabled" if project_busy else ""
        busy_notice = (
            '<div class="warning-box">当前项目有后台任务正在运行。为避免并发写入冲突，续写、回滚和插图表单已暂时禁用。你可以打开上方任务卡片查看实时进度。</div>'
            if project_busy
            else ""
        )
        progression_notice = (
            '<div class="warning-box">下一章推进选项正在后台刷新。你现在就可以阅读最新正文，候选方案生成完成后会出现在下方。</div>'
            if progression_jobs
            else ""
        )

        chapter_links = "".join(
            f"<a href=\"/project/{escape(project_id)}/chapter/{escape(chapter['slug'])}\">{escape(chapter['name'])}</a>"
            for chapter in chapters
        ) or "<p>还没有章节。</p>"
        chapter_options = ['<option value="latest">最新章节</option>'] + [
            f"<option value=\"{escape(chapter['slug'])}\">{escape(chapter['name'])}</option>"
            for chapter in chapters
        ]

        illustration_cards = []
        for record in illustration_records[:6]:
            chapter_slug = str(record.get("chapter_slug", ""))
            images = record.get("images") or []
            if not chapter_slug or not images:
                continue
            image = images[0]
            image_url = (
                f"/project/{urllib.parse.quote(project_id)}/illustration-file/"
                f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(image.get('file_name', ''))}"
            )
            illustration_cards.append(
                f"""
                <div class="thumb">
                  <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}"><img src="{image_url}" alt="{escape(chapter_slug)}"></a>
                  <div><strong>{escape(chapter_slug)}</strong></div>
                  <div class="muted">{escape(record.get('scene_summary', '') or '已生成插图')}</div>
                </div>
                """
            )
        illustration_gallery = "".join(illustration_cards) or "<p>当前还没有章节插图。</p>"
        latest_chapter_text = escape(chapters[-1]["text"]) if chapters else "还没有正文。"
        snapshot_text = f"已保存到第 {latest_snapshot} 章" if latest_snapshot is not None else "暂无"
        project_live_script = ""
        if progression_jobs:
            project_live_script = f"""
            <script>
            (() => {{
              const tick = () => {{
                const active = document.activeElement;
                if (active && ["INPUT", "TEXTAREA", "SELECT"].includes(active.tagName)) {{
                  return;
                }}
                window.location.reload();
              }};
              window.setTimeout(tick, 3500);
            }})();
            </script>
            """

        body = f"""
        <div class="project-layout">
          <aside class="stack project-sidebar">
            <section class="panel">
              <h2>{escape(project_name)}</h2>
              <p class="meta">{escape(project.get("description", ""))}</p>
              <p class="meta">
                <span class="pill">{escape((project.get("llm_config") or {}).get("model_provider", ""))}</span>
                <span class="pill">{project.get("chapter_count", 0)} 章</span>
              </p>
              <div class="project-snapshot">
                <p><strong>状态快照：</strong>{escape(snapshot_text)}</p>
                <p><strong>当前章任务：</strong>{escape(effective_task.get("goal", "") or "暂无")}</p>
                <p><strong>卷目标：</strong>{escape(effective_task.get("volume_goal", "") or "暂无")}</p>
                <p><strong>live-state 下一目标：</strong>{escape(plot_state.get("next_chapter_goal", "") or "暂无")}</p>
                <p><strong>当前位置：</strong>{escape(plot_state.get("current_location", "") or "未知")}</p>
                <p><strong>当前时间：</strong>{escape(plot_state.get("current_time", "") or "未知")}</p>
                <p><strong>请求：</strong>{stats.get("requests", 0)} 次</p>
                <p><strong>Token：</strong>{stats.get("total_tokens", 0)}</p>
                <p><strong>Planning:</strong>{escape(_planning_mode_label(planning_mode))}</p>
              </div>
            </section>
            <section class="panel">
              <h3>后台任务</h3>
              {active_jobs_html}
            </section>
            {busy_notice}
            <section class="panel">
              <h3>续写</h3>
              <form method="post" action="/project/{escape(project_id)}/continue">
                <fieldset{busy_attr}>
                  <div class="two-col">
                    <label>续写章节数
                      <input type="number" name="count" value="1" min="1" max="20">
                    </label>
                    <label>模式
                      <input type="text" value="直接续写，可一次写多章" disabled>
                    </label>
                  </div>
                  <label>想看的内容 / 情节走向
                    <textarea name="user_request" placeholder="例如：先推进食堂据点建设，再增加一点轻松互怼的互动。"></textarea>
                  </label>
                  {runtime_override_fields_html}
                  <label><input type="checkbox" name="illustrate_generated" value="1"> 续写完成后立即调用 ComfyUI 生成插图</label>
                  <label>插图额外要求（可选）
                    <input type="text" name="illustration_request" placeholder="例如：突出雪夜窗景与室内暖光反差。">
                  </label>
                  <button type="submit">开始续写</button>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>状态回滚</h3>
              <form method="post" action="/project/{escape(project_id)}/rollback">
                <fieldset{busy_attr}>
                  <label>保留到第几章
                    <input type="number" name="to_chapter" value="{max(0, int(project.get('chapter_count', 0) or 0))}" min="0" max="{max(0, int(project.get('chapter_count', 0) or 0))}">
                  </label>
                  <div class="warning-box">
                    回滚会删除目标章节之后的正文、摘要、章节插图和更晚的状态快照。回滚完成后，可以直接继续写，从保留章节的状态往后写新版本。
                  </div>
                  <button class="ghost-button" type="submit">回滚到该章节</button>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>生成插图</h3>
              <form method="post" action="/project/{escape(project_id)}/illustrate">
                <fieldset{busy_attr}>
                  <label>目标章节
                    <select name="chapter_slug">
                      {''.join(chapter_options)}
                    </select>
                  </label>
                  <label>插图要求（可选）
                    <textarea name="user_request" placeholder="例如：更强调角色站位、情绪与镜头感。"></textarea>
                  </label>
                  <div class="two-col">
                    <label>Checkpoint（可选）
                      <input type="text" name="checkpoint" placeholder="illusious/illustrij_v21.safetensors">
                    </label>
                    <label>ComfyUI API（可选）
                      <input type="text" name="comfyui_api_base" placeholder="http://127.0.0.1:8188">
                    </label>
                  </div>
                  <div class="two-col">
                    <label>宽度
                      <input type="number" name="width" placeholder="832">
                    </label>
                    <label>高度
                      <input type="number" name="height" placeholder="1216">
                    </label>
                  </div>
                  <div class="two-col">
                    <label>Steps
                      <input type="number" name="steps" placeholder="28">
                    </label>
                    <label>CFG
                      <input type="number" step="0.1" name="cfg" placeholder="6.5">
                    </label>
                  </div>
                  <label><input type="checkbox" name="force" value="1"> 强制重绘</label>
                  <button type="submit">为章节生成插图</button>
                </fieldset>
              </form>
            </section>
            <section class="panel">
              <h3>章节目录</h3>
              <div class="chapter-list">{chapter_links}</div>
            </section>
          </aside>
          <main class="stack project-main">
            <section class="panel">
              <h2>最近一章</h2>
              <div class="chapter-view">{latest_chapter_text}</div>
            </section>
            <section class="panel">
              {effective_task_html}
            </section>
            <section class="panel">
              <div class="option-panel-head">
                <div>
                  <h2>剧情推进选项</h2>
                  <p class="muted">围绕当前小说状态生成下一章的多个推进方案。系统会生成 3-5 个候选方案，并额外附带 1 个空白自定义项。现在会在项目创建完成、正文写完后自动后台刷新，你也可以在这里手动再生成一组。</p>
                </div>
                <span class="pill">固定作用于下一章</span>
              </div>
              <form method="post" action="/project/{escape(project_id)}/progression-options">
                <fieldset{busy_attr}>
                  <div class="two-col">
                    <label>选项数量
                      <select name="option_count">
                        <option value="3">3 个</option>
                        <option value="4" selected>4 个</option>
                        <option value="5">5 个</option>
                      </select>
                    </label>
                    <label>作用范围
                      <input type="text" value="固定为下一章" disabled>
                    </label>
                  </div>
                  <label>想看的方向 / 倾向
                    <textarea name="user_request" placeholder="例如：我想看一次更主动的外出搜集，但不要一下把大事件写完。"></textarea>
                  </label>
                  {runtime_override_fields_html}
                  <button type="submit">重新生成下一章推进选项</button>
                </fieldset>
              </form>
              <div class="warning-box">
                这组推进选项只对应当前“下一章”。如果你先直接续写、回滚，或项目章节数发生变化，这组候选方案会自动失效，需要重新生成。
              </div>
              {progression_notice}
              <div class="stack">
                {guided_session_html}
              </div>
            </section>
            <section class="panel">
              <h2>剧情状态</h2>
              <div class="chapter-view">{escape(json.dumps(plot_state, ensure_ascii=False, indent=2))}</div>
            </section>
            <section class="panel">
              <h2>最近插图</h2>
              <div class="gallery">{illustration_gallery}</div>
            </section>
          </main>
        </div>
        {project_live_script}
        """
        self._write_html(
            _render_page(
                project_name,
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_create_project_async(self, form: dict[str, str]) -> None:
        api_keys = _load_api_keys()
        try:
            if not (form.get("story_request") or "").strip():
                raise RuntimeError("故事需求不能为空。")
            job = JOB_REGISTRY.create_job(kind="create_project", title="创建新项目")
        except Exception as exc:
            self._redirect("/projects?error=" + urllib.parse.quote(str(exc)))
            return

        def runner(progress_callback):
            project_path = _create_project(form, api_keys, progress_callback=progress_callback)
            project_path_obj = Path(project_path)
            project_meta = load_json(str(project_path_obj / "project.json"))
            new_project_id = project_meta.get("project_id", project_path_obj.name)
            auto_job = _enqueue_progression_job(
                new_project_id,
                project_path_obj,
                _load_saved_runtime_config(project_path_obj),
                title=f"生成《{new_project_id}》的开篇推进选项",
                auto_generated=True,
            )
            message = "项目创建完成。"
            if auto_job is not None:
                message += " 下一章推进选项已在后台开始生成。"
            return {
                "message": message,
                "result_url": "/project/" + urllib.parse.quote(new_project_id),
                "result_label": "打开项目",
                "project_id": new_project_id,
                "project_path": project_path,
            }

        _start_background_job(job["id"], runner)
        self._redirect("/job/" + urllib.parse.quote(job["id"]))

    def _handle_continue_async(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            count = int(form.get("count") or "1")
            if count < 1:
                raise RuntimeError("续写章节数必须至少为 1。")
            runtime_overrides = _runtime_overrides_from_form(form)
            runtime_config = _build_runtime_config(project_path, runtime_overrides, api_keys)
            job = JOB_REGISTRY.create_job(
                kind="continue",
                title=f"续写《{project_id}》",
                project_id=project_id,
                project_path=str(project_path.resolve()),
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )
            return

        def runner(progress_callback):
            chapter_paths = run_next_chapters(
                str(project_path),
                runtime_config,
                count,
                user_request=(form.get("user_request") or "").strip(),
                progress_callback=progress_callback,
            )
            message = f"续写完成，共生成 {count} 章。"
            auto_job = _enqueue_progression_job(
                project_id,
                project_path,
                runtime_config,
                runtime_overrides=runtime_overrides,
                title=f"生成《{project_id}》的下一章推进选项",
                auto_generated=True,
            )
            if auto_job is not None:
                message += " 下一章推进选项已在后台开始生成。"
            if form.get("illustrate_generated"):
                progress_callback({"stage": "illustration_batch", "message": "正在为新章节生成插图"})
                illustration_results = illustrate_chapters(
                    str(project_path),
                    chapter_refs=chapter_paths,
                    llm_config=runtime_config,
                    user_request=(form.get("illustration_request") or "").strip(),
                    overrides=None,
                    progress_callback=progress_callback,
                )
                new_count = sum(0 if item.get("reused") else 1 for item in illustration_results)
                message += f" 插图处理完成，新生成 {new_count} 张。"
            return {
                "message": message,
                "result_url": "/project/" + urllib.parse.quote(project_id),
                "result_label": "返回项目页",
                "project_id": project_id,
                "project_path": str(project_path.resolve()),
            }

        _start_background_job(job["id"], runner)
        self._redirect("/job/" + urllib.parse.quote(job["id"]))

    def _handle_progression_options(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        if JOB_REGISTRY.has_active_project_job(project_path):
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote("当前项目有后台任务正在运行，请稍后再生成推进选项。")
            )
            return

        try:
            runtime_overrides = _runtime_overrides_from_form(form)
            runtime_config = _build_runtime_config(project_path, runtime_overrides, api_keys)
            job = _enqueue_progression_job(
                project_id,
                project_path,
                runtime_config,
                user_request=(form.get("user_request") or "").strip(),
                option_count=int(form.get("option_count") or "4"),
                runtime_overrides=runtime_overrides,
                title=f"生成《{project_id}》的下一章推进选项",
            )
            if job is None:
                notice = "下一章推进选项已经在后台生成中，请稍候刷新项目页。"
            else:
                notice = "已开始在后台生成下一章推进选项。你可以先继续阅读当前正文。"
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote(notice)
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )

    def _handle_continue_guided_async(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        session_id = (form.get("progression_session") or "").strip()
        option_ref = (form.get("progression_option") or "").strip()
        feedback = (form.get("progression_feedback") or "").strip()
        try:
            if not session_id or not option_ref:
                raise RuntimeError("请选择一个推进选项后再继续写。")
            if option_ref == CUSTOM_PROGRESSION_OPTION_ID and not feedback:
                raise RuntimeError("选择空白自定义项后，请先填写你自己的创意与想看的情节。")
            session = ensure_fresh_progression_session(
                str(project_path),
                load_progression_session(str(project_path), session_id),
            )
            if session.get("status") == "stale":
                raise RuntimeError("当前推进选项已过期，请重新生成推进选项。")
            runtime_overrides = session.get("runtime_overrides") or {}
            runtime_config = _build_runtime_config(
                project_path,
                runtime_overrides,
                api_keys,
            )
            job = JOB_REGISTRY.create_job(
                kind="continue_guided",
                title=f"按推进选项续写《{project_id}》",
                project_id=project_id,
                project_path=str(project_path.resolve()),
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )
            return

        def runner(progress_callback):
            chapter_path = run_next_chapter_from_progression(
                str(project_path),
                runtime_config,
                progression_session=session_id,
                progression_option=option_ref,
                progression_feedback=feedback,
                progress_callback=progress_callback,
            )
            auto_job = _enqueue_progression_job(
                project_id,
                project_path,
                runtime_config,
                runtime_overrides=runtime_overrides,
                title=f"刷新《{project_id}》的下一章推进选项",
                auto_generated=True,
            )
            message = "已按所选推进方案生成下一章。"
            if auto_job is not None:
                message += " 下一章推进选项已在后台开始生成。"
            return {
                "message": message,
                "result_url": "/project/" + urllib.parse.quote(project_id),
                "result_label": "返回项目页",
                "project_id": project_id,
                "project_path": str(project_path.resolve()),
            }

        _start_background_job(job["id"], runner)
        self._redirect("/job/" + urllib.parse.quote(job["id"]))

    def _handle_illustrate_async(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        chapter_slug = (form.get("chapter_slug") or "latest").strip() or "latest"
        try:
            llm_runtime_config = None
            try:
                llm_runtime_config = _build_runtime_config(project_path, {}, api_keys)
            except Exception:
                llm_runtime_config = None
            job = JOB_REGISTRY.create_job(
                kind="illustrate",
                title=f"生成插图：{chapter_slug}",
                project_id=project_id,
                project_path=str(project_path.resolve()),
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )
            return

        def runner(progress_callback):
            results = illustrate_chapters(
                str(project_path),
                chapter_refs=[chapter_slug],
                llm_config=llm_runtime_config,
                user_request=(form.get("user_request") or "").strip(),
                force=bool(form.get("force")),
                overrides=_illustration_overrides_from_form(form),
                progress_callback=progress_callback,
            )
            result = results[0]
            chapter_target = result.get("chapter_slug", chapter_slug)
            message = "已复用现有插图。" if result.get("reused") else "插图生成完成。"
            return {
                "message": message,
                "result_url": "/project/"
                + urllib.parse.quote(project_id)
                + "/chapter/"
                + urllib.parse.quote(chapter_target),
                "result_label": "打开章节",
                "project_id": project_id,
                "project_path": str(project_path.resolve()),
            }

        _start_background_job(job["id"], runner)
        self._redirect("/job/" + urllib.parse.quote(job["id"]))

def main() -> None:
    parser = argparse.ArgumentParser(description=f"Basic web UI for {APP_NAME}")
    parser.add_argument("--version", action="version", version=f"{WEBUI_NAME} {DISPLAY_VERSION}")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, use 0.0.0.0 for remote access")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    parser.add_argument("--admin-task", choices=("restart", "update"), help="Run an admin maintenance task and exit")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help=argparse.SUPPRESS)
    parser.add_argument("--service-name", default=WEBUI_SERVICE_NAME, help=argparse.SUPPRESS)
    parser.add_argument("--status-path", default=str(_admin_action_status_path()), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.admin_task:
        raise SystemExit(
            _run_admin_task(
                args.admin_task,
                repo_root=args.repo_root,
                service_name=args.service_name,
                status_path=args.status_path,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), NovelWriterHandler)
    print(f"[{utc_now()}] Web UI listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{utc_now()}] Web UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
