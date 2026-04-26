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
from email.parser import BytesParser
from email.policy import default as email_policy_default
from html import escape
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

from app import run_next_chapter_from_progression, run_next_chapters
from audiobook_manager import (
    UploadedVoiceFile,
    audiobook_file_path,
    ensure_voice_config,
    generate_audiobook_chapters,
    get_audiobook_record,
    list_audiobook_records,
    narrator_preset_options,
    save_uploaded_voice_reference,
)
from chapter_context import peek_next_context_for_mode
from common_utils import utc_now
from context_builder import resolve_effective_chapter_task
from external_services import AudioFrameClient, ImageFrameClient, load_audio_frame_runtime, load_image_frame_runtime
from illustration_manager import get_illustration_record, illustrate_chapters, list_illustration_records
from polish_manager import POLISH_PRESETS, run_chapter_polish
from progression_manager import (
    CUSTOM_PROGRESSION_OPTION_ID,
    SELECTION_MODE_RECOMMENDED,
    ensure_fresh_progression_session,
    generate_progression_options,
    get_latest_active_progression_session,
    load_progression_session,
    validate_selection_mode,
)
from quality_manager import list_quality_artifacts
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
    DEFAULT_REVIEW_MODE,
    DEFAULT_WRITING_QUALITY_MODE,
    REVIEW_MODE_AUTO,
    REVIEW_MODE_MANUAL,
    WEB_SELECTABLE_PROVIDERS,
    WRITING_QUALITY_BALANCED,
    WRITING_QUALITY_HIGH,
    WRITING_QUALITY_LIGHT,
    api_key_for_provider as shared_api_key_for_provider,
    build_runtime_config as shared_build_runtime_config,
    default_api_base_for_provider as shared_default_api_base_for_provider,
    default_model_for_provider as shared_default_model_for_provider,
    default_timeout_for_provider as shared_default_timeout_for_provider,
    load_runtime_config as shared_load_runtime_config,
    load_model_presets as shared_load_model_presets,
    normalize_review_mode,
    normalize_writing_quality_mode,
    normalize_provider as shared_normalize_provider,
    provider_requires_api_key as shared_provider_requires_api_key,
    resolve_timeout_for_provider as shared_resolve_timeout_for_provider,
    sanitize_runtime_overrides,
)
from version import APP_NAME, DISPLAY_VERSION, HTTP_SERVER_TOKEN, WEBUI_NAME
from web_auth import AuthService, LoginAttemptGuard, WebAuthSettings, load_auth_settings

if not hasattr(os, "getuid"):
    def _windows_getuid() -> int:
        try:
            return int(os.environ.get("UID") or 0)
        except ValueError:
            return 0

    os.getuid = _windows_getuid  # type: ignore[attr-defined]


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
OUTPUT_DIR = BASE_DIR / "output"
API_KEYS_PATH = BASE_DIR / "api_keys.sh"
PROJECT_DIR_PATTERN = re.compile(r"^novel_project_")
MOJIBAKE_HINT_CHARS = set("闆皝绌归《鍙鍦鏄鐨勪簡鍚庡墠闂閿璇浠绗锛銆鈥€")
WEBUI_SERVICE_NAME = os.getenv("NOVEL_WRITER_WEBUI_SERVICE", "novel-writer-webui.service")
WEBUI_SERVICE_SCOPE = os.getenv("NOVEL_WRITER_WEBUI_SERVICE_SCOPE", "auto").strip().lower() or "auto"
ADMIN_LOCALHOSTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
ADMIN_ACTION_STATUS_FILENAME = "admin_action_status.json"
PUBLIC_PATHS = {"/login", "/healthz"}
API_PATH_PREFIX = "/api/"
EXTERNAL_SERVICE_HEALTH_TIMEOUT_SECONDS = 3.0

JOB_ACTIVE_STATUSES = {"queued", "running"}
JOB_FINISHED_STATUSES = {"succeeded", "failed"}
PROGRESSION_JOB_KINDS = {"progression_options", "progression_options_auto"}

_LOGIN_ATTEMPT_GUARDS: dict[tuple[int, int, int], LoginAttemptGuard] = {}
_LOGIN_ATTEMPT_GUARDS_LOCK = threading.Lock()
_EXTERNAL_SERVICE_HEALTH_LOCK = threading.Lock()
_EXTERNAL_SERVICE_HEALTH_CACHE: dict = {}


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
    if runtime_dir != "/":
        runtime_dir = runtime_dir.rstrip("/")
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


def _systemctl_scope_args(scope: str) -> list[str]:
    return ["--user"] if scope == "user" else []


def _systemctl_scope_env(scope: str) -> dict[str, str] | None:
    if scope == "user":
        return _systemd_user_command_env()
    return None


def _systemctl_query_scope(service_name: str, scope: str) -> bool:
    command = ["systemctl", *_systemctl_scope_args(scope), "show", service_name, "--property=LoadState", "--value"]
    try:
        output = _run_checked_command(command, env=_systemctl_scope_env(scope))
    except Exception:
        return False
    return str(output).strip().lower() not in {"", "not-found", "masked"}


def _resolve_service_scope(service_name: str) -> str:
    explicit_scope = WEBUI_SERVICE_SCOPE
    if explicit_scope in {"user", "system"}:
        return explicit_scope
    if _systemctl_query_scope(service_name, "user"):
        return "user"
    if _systemctl_query_scope(service_name, "system"):
        return "system"
    return "user"


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
    service_scope = _resolve_service_scope(WEBUI_SERVICE_NAME)
    info = {
        "repo_root": str(REPO_ROOT),
        "service_name": WEBUI_SERVICE_NAME,
        "service_scope": service_scope,
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
    service_scope = _resolve_service_scope(WEBUI_SERVICE_NAME)
    status_path = _admin_action_status_path()
    _write_admin_action_status(
        {
            "action": task,
            "status": "queued",
            "message": "维护任务已排队，等待 systemd 启动。",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "service_scope": service_scope,
        }
    )
    unit_name = f"novel-writer-admin-{task}-{uuid4().hex[:8]}"
    command = [
        "systemd-run",
        *_systemctl_scope_args(service_scope),
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
        "--service-scope",
        service_scope,
        "--status-path",
        str(status_path),
    ]
    _run_checked_command(
        command,
        cwd=str(BASE_DIR),
        env=_systemctl_scope_env(service_scope),
    )
    return unit_name


def _run_admin_task(task: str, *, repo_root: str, service_name: str, service_scope: str, status_path: str) -> int:
    repo_path = Path(repo_root).resolve()
    status_file = Path(status_path).resolve()
    status_file.parent.mkdir(parents=True, exist_ok=True)

    def write_status(status: str, message: str, **extra) -> None:
        payload = {
            "action": task,
            "status": status,
            "message": message,
            "updated_at": utc_now(),
            "service_scope": service_scope,
        }
        payload.update(extra)
        status_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    started_at = utc_now()
    write_status("running", "维护任务已启动。", started_at=started_at)
    time.sleep(1.0)

    try:
        if task == "restart":
            _run_checked_command(
                ["systemctl", *_systemctl_scope_args(service_scope), "restart", service_name],
                env=_systemctl_scope_env(service_scope),
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
                ["systemctl", *_systemctl_scope_args(service_scope), "restart", service_name],
                env=_systemctl_scope_env(service_scope),
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


def _quality_model_from_form(form: dict[str, str], api_keys: dict[str, str] | None = None) -> dict:
    provider = _normalize_provider_for_ui(form.get("quality_provider"), default="")
    model_name = (form.get("quality_model_name") or "").strip()
    api_base = (form.get("quality_api_base") or "").strip()
    quality_model: dict[str, object] = {}
    if provider:
        quality_model["model_provider"] = provider
    if model_name:
        quality_model["model_name"] = model_name
        quality_model["model"] = model_name
    if api_base:
        quality_model["api_base"] = api_base
    for form_key, config_key in (
        ("quality_max_tokens", "max_tokens"),
        ("quality_timeout", "timeout"),
    ):
        value = (form.get(form_key) or "").strip()
        if value:
            quality_model[config_key] = value
    if provider and api_keys is not None:
        quality_model["api_key"] = _api_key_for_provider(provider, api_keys)
    return quality_model


def _quality_model_label(llm_config: dict) -> str:
    quality_model = llm_config.get("quality_model") if isinstance(llm_config.get("quality_model"), dict) else {}
    if not quality_model:
        return "inherit main"
    provider = str(quality_model.get("model_provider") or llm_config.get("model_provider") or "").strip()
    model = str(quality_model.get("model_name") or quality_model.get("model") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    if provider:
        return f"{provider}/default"
    if model:
        return f"main/{model}"
    return "inherit main"


def _stats_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _stats_float(value: object) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _format_tokens(value: object) -> str:
    return f"{_stats_int(value):,}"


def _format_usd(value: object) -> str:
    amount = _stats_float(value)
    if amount == 0:
        return "$0.0000"
    if amount < 0.0001:
        return "$<0.0001"
    return f"${amount:.4f}"


def _usage_cost_summary(stats: dict | None) -> dict[str, int | float | dict]:
    stats = stats if isinstance(stats, dict) else {}
    total = stats.get("total") if isinstance(stats.get("total"), dict) else {}
    cost = stats.get("cost") if isinstance(stats.get("cost"), dict) else {}
    priced_tokens = _stats_int(cost.get("priced_tokens"))
    unpriced_tokens = _stats_int(cost.get("unpriced_tokens"))
    total_tokens = _stats_int(total.get("total_tokens"))
    legacy_tokens = max(0, total_tokens - priced_tokens - unpriced_tokens)
    return {
        "total": total,
        "cost": cost,
        "estimated_total_usd": _stats_float(cost.get("estimated_total_usd")),
        "priced_tokens": priced_tokens,
        "unpriced_tokens": unpriced_tokens,
        "legacy_tokens": legacy_tokens,
    }


def _render_cost_meta(stats: dict | None) -> str:
    summary = _usage_cost_summary(stats)
    parts = [f"估算费用：{_format_usd(summary['estimated_total_usd'])}"]
    if summary["unpriced_tokens"]:
        parts.append(f"未定价：{_format_tokens(summary['unpriced_tokens'])} tokens")
    if summary["legacy_tokens"]:
        parts.append(f"历史未估价：{_format_tokens(summary['legacy_tokens'])} tokens")
    return f'<div class="meta">{" · ".join(escape(part) for part in parts)}</div>'


def _pricing_status_label(status: object) -> str:
    mapping = {
        "priced": "已定价",
        "local": "本地 $0",
        "unpriced": "未定价",
    }
    return mapping.get(str(status or "").strip(), "未知")


def _render_sidebar_usage_stats(stats: dict | None) -> str:
    summary = _usage_cost_summary(stats)
    total = summary["total"]
    return f"""
                <p><strong>请求：</strong>{_format_tokens(total.get("requests"))} 次（成功 {_format_tokens(total.get("successes"))} / 失败 {_format_tokens(total.get("failures"))}）</p>
                <p><strong>Token：</strong>{_format_tokens(total.get("total_tokens"))}</p>
                <p><strong>Prompt：</strong>{_format_tokens(total.get("prompt_tokens"))}</p>
                <p><strong>Output：</strong>{_format_tokens(total.get("completion_tokens"))}</p>
                <p><strong>Cached：</strong>{_format_tokens(total.get("cached_tokens"))}</p>
                <p><strong>Reasoning：</strong>{_format_tokens(total.get("reasoning_tokens"))}</p>
                <p><strong>Thought：</strong>{_format_tokens(total.get("thought_tokens"))}</p>
                <p><strong>估算费用：</strong>{_format_usd(summary["estimated_total_usd"])}</p>
                <p><strong>未定价 Token：</strong>{_format_tokens(summary["unpriced_tokens"])}</p>
                <p><strong>历史未估价：</strong>{_format_tokens(summary["legacy_tokens"])}</p>
    """


def _render_token_cost_panel(stats: dict | None) -> str:
    stats = stats if isinstance(stats, dict) else {}
    by_phase = stats.get("by_phase") if isinstance(stats.get("by_phase"), dict) else {}
    cost = stats.get("cost") if isinstance(stats.get("cost"), dict) else {}
    cost_by_phase = cost.get("by_phase") if isinstance(cost.get("by_phase"), dict) else {}
    cost_by_model = cost.get("by_model") if isinstance(cost.get("by_model"), dict) else {}

    phase_rows = []
    for phase in sorted(set(by_phase) | set(cost_by_phase)):
        usage = by_phase.get(phase) if isinstance(by_phase.get(phase), dict) else {}
        phase_cost = cost_by_phase.get(phase) if isinstance(cost_by_phase.get(phase), dict) else {}
        if (
            _stats_int(usage.get("requests")) == 0
            and _stats_int(usage.get("total_tokens")) == 0
            and _stats_int(phase_cost.get("requests")) == 0
        ):
            continue
        phase_rows.append(
            f"""
            <tr>
              <td>{escape(str(phase))}</td>
              <td>{_format_tokens(usage.get("requests"))}</td>
              <td>{_format_tokens(usage.get("prompt_tokens"))}</td>
              <td>{_format_tokens(usage.get("completion_tokens"))}</td>
              <td>{_format_tokens(usage.get("cached_tokens"))}</td>
              <td>{_format_tokens(usage.get("total_tokens"))}</td>
              <td>{_format_usd(phase_cost.get("estimated_usd"))}</td>
              <td>{_format_tokens(phase_cost.get("unpriced_tokens"))}</td>
            </tr>
            """
        )
    if not phase_rows:
        phase_rows.append('<tr><td colspan="8" class="muted">暂无 token 统计。</td></tr>')

    model_rows = []
    sorted_models = sorted(
        cost_by_model.values(),
        key=lambda item: _stats_float(item.get("estimated_usd")) if isinstance(item, dict) else 0.0,
        reverse=True,
    )
    for item in sorted_models:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        source_label = source.get("name") or item.get("reason") or ""
        model_rows.append(
            f"""
            <tr>
              <td>{escape(str(item.get("provider") or ""))}</td>
              <td>{escape(str(item.get("model") or ""))}</td>
              <td>{_format_tokens(item.get("requests"))}</td>
              <td>{_format_tokens(item.get("total_tokens"))}</td>
              <td>{_format_usd(item.get("estimated_usd"))}</td>
              <td>{escape(_pricing_status_label(item.get("pricing_status")))}</td>
              <td>{escape(str(source_label))}</td>
            </tr>
            """
        )
    if not model_rows:
        model_rows.append('<tr><td colspan="7" class="muted">暂无新调用费用统计。</td></tr>')

    return f"""
            <section class="panel">
              <h2>Token / 费用统计</h2>
              {_render_cost_meta(stats)}
              <h3>按阶段</h3>
              <table class="quality-table">
                <thead>
                  <tr><th>阶段</th><th>请求</th><th>Prompt</th><th>Output</th><th>Cached</th><th>Total</th><th>估算费用</th><th>未定价</th></tr>
                </thead>
                <tbody>{''.join(phase_rows)}</tbody>
              </table>
              <h3>按模型</h3>
              <table class="quality-table">
                <thead>
                  <tr><th>Provider</th><th>Model</th><th>请求</th><th>Total</th><th>估算费用</th><th>价格状态</th><th>来源</th></tr>
                </thead>
                <tbody>{''.join(model_rows)}</tbody>
              </table>
            </section>
    """


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


def _quality_mode_label(mode: str) -> str:
    mapping = {
        WRITING_QUALITY_LIGHT: "轻量：只增强写作提示",
        WRITING_QUALITY_BALANCED: "平衡：蓝图 + 质检",
        WRITING_QUALITY_HIGH: "高质量：严格质检 + 可重写",
    }
    return mapping.get(normalize_writing_quality_mode(mode), mapping[DEFAULT_WRITING_QUALITY_MODE])


def _render_quality_mode_options(selected: str, *, include_project_default: bool = False) -> str:
    normalized = normalize_writing_quality_mode(selected)
    options = []
    if include_project_default:
        selected_attr = ' selected' if not selected else ""
        options.append(f'<option value=""{selected_attr}>沿用项目设置</option>')
    for value in (WRITING_QUALITY_LIGHT, WRITING_QUALITY_BALANCED, WRITING_QUALITY_HIGH):
        selected_attr = ' selected' if normalized == value and not include_project_default else ""
        options.append(f'<option value="{value}"{selected_attr}>{_quality_mode_label(value)}</option>')
    return "".join(options)


def _review_mode_label(mode: str) -> str:
    mapping = {
        REVIEW_MODE_AUTO: "自动：高质量失败时重写一次",
        REVIEW_MODE_MANUAL: "手动：只保存质检报告",
    }
    return mapping.get(normalize_review_mode(mode), mapping[DEFAULT_REVIEW_MODE])


def _render_review_mode_options(selected: str, *, include_project_default: bool = False) -> str:
    normalized = normalize_review_mode(selected)
    options = []
    if include_project_default:
        selected_attr = ' selected' if not selected else ""
        options.append(f'<option value=""{selected_attr}>沿用项目设置</option>')
    for value in (REVIEW_MODE_AUTO, REVIEW_MODE_MANUAL):
        selected_attr = ' selected' if normalized == value and not include_project_default else ""
        options.append(f'<option value="{value}"{selected_attr}>{_review_mode_label(value)}</option>')
    return "".join(options)


def _auto_selection_mode_label(mode: str) -> str:
    return "随机模式" if str(mode or "").strip().lower() == "random" else "推荐模式"


def _render_auto_selection_mode_options(selected: str) -> str:
    normalized = validate_selection_mode(selected, allow_manual=False)
    options = []
    for value in ("recommended", "random"):
        selected_attr = ' selected' if normalized == value else ""
        options.append(f'<option value="{value}"{selected_attr}>{_auto_selection_mode_label(value)}</option>')
    return "".join(options)


def _auto_continue_help(mode: str) -> str:
    normalized_mode = normalize_planning_mode(mode)
    if normalized_mode == PLANNING_MODE_NONE:
        return "自动续写时，每一章都会先提炼 objective，再生成多个 plan；你这次填写的目标 / 倾向会同时影响每章 objective 与 plan。"
    return "自动续写时，每一章都会先围绕当前 objective 生成多个 plan，再按所选策略自动挑一个执行；你这次填写的目标 / 倾向只影响 plan，不改写 objective。"


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
                "stats": project.get("stats") or {},
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


def _runtime_overrides_from_form(form: dict[str, str]) -> dict[str, object]:
    log_llm_payload = bool(form.get("log_llm_payload"))
    return sanitize_runtime_overrides(
        {
            "provider": form.get("provider"),
            "model_name": _resolve_model_name_from_form(form),
            "planning_mode": form.get("planning_mode"),
            "writing_quality_mode": form.get("writing_quality_mode"),
            "review_mode": form.get("review_mode"),
            "max_tokens": form.get("max_tokens"),
            "timeout": form.get("timeout"),
            "api_base": form.get("api_base"),
            "log_llm_payload": "1" if log_llm_payload else "",
            "quality_model": _quality_model_from_form(form),
        }
    )


def _polish_preset_ids_from_form(form: dict[str, str]) -> list[str]:
    return [
        preset["id"]
        for preset in POLISH_PRESETS
        if form.get(f"polish_preset_{preset['id']}")
    ]


def _render_polish_preset_checkboxes() -> str:
    return "".join(
        f"""
        <label class="pill-check">
          <input type="checkbox" name="polish_preset_{escape(preset['id'])}" value="1">
          {escape(preset['label'])}
        </label>
        """
        for preset in POLISH_PRESETS
    )


def _enqueue_progression_job(
    project_id: str,
    project_path: Path,
    runtime_config: dict,
    *,
    user_request: str = "",
    objective_override: str = "",
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
            objective_override=objective_override,
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


def _render_provider_options(selected: str = "", *, include_project_default: bool = True) -> str:
    options = ['<option value="">沿用项目设置</option>'] if include_project_default else []
    for provider in sorted(WEB_SELECTABLE_PROVIDERS):
        selected_attr = ' selected' if selected == provider else ""
        options.append(f'<option value="{provider}"{selected_attr}>{provider}</option>')
    return "".join(options)


def _render_quality_provider_options() -> str:
    options = ['<option value="">inherit main provider</option>']
    for provider in sorted(WEB_SELECTABLE_PROVIDERS):
        options.append(f'<option value="{provider}">{provider}</option>')
    return "".join(options)


def _render_runtime_override_fields(
    base_provider: str = "gemini",
    base_model: str = "",
    *,
    include_planning_mode: bool = True,
    include_quality_fields: bool = True,
) -> str:
    effective_provider = _normalize_provider_for_ui(base_provider, default="gemini")
    initial_blank_label = _model_blank_label(
        effective_provider,
        base_model=base_model,
        provider_explicit=False,
    )
    planning_field_html = (
        f"""
      <label>Planning Mode
        <select name="planning_mode">
          {_render_planning_mode_options("", include_project_default=True)}
        </select>
      </label>
        """
        if include_planning_mode
        else "<div></div>"
    )
    planning_help_html = (
        '<div class="muted">留空则沿用项目设置。none 最自由，volume 更平衡，chapter 控制最强。</div>'
        if include_planning_mode
        else '<div class="muted">留空则沿用项目设置；这些覆盖只对本次调用生效。</div>'
    )
    quality_fields_html = (
        f"""
    <div class="two-col">
      <label>写作质量模式
        <select name="writing_quality_mode">
          {_render_quality_mode_options("", include_project_default=True)}
        </select>
      </label>
      <label>审稿模式
        <select name="review_mode">
          {_render_review_mode_options("", include_project_default=True)}
        </select>
      </label>
    </div>
    <div class="muted">留空则沿用项目设置。平衡模式会先生成本章创作蓝图，写后保存质检报告；高质量模式在自动审稿失败时最多重写一次。</div>
        """
        if include_quality_fields
        else ""
    )
    quality_model_fields_html = (
        """
    <div class="two-col">
      <label>Quality Provider
        <select name="quality_provider">
          {_render_quality_provider_options()}
        </select>
      </label>
      <label>Quality Model
        <input type="text" name="quality_model_name" placeholder="inherit main model">
      </label>
    </div>
    <div class="two-col">
      <label>Quality API Base
        <input type="text" name="quality_api_base" placeholder="inherit or provider default">
      </label>
      <label>Quality Timeout
        <input type="number" name="quality_timeout" placeholder="inherit or provider default">
      </label>
    </div>
    <label>Quality Max Tokens
      <input type="number" name="quality_max_tokens" placeholder="inherit main">
    </label>
    <div class="muted">Optional advanced model used only for craft brief, quality review, and rewrite.</div>
        """
        if include_quality_fields
        else ""
    )
    return f"""
    <div class="two-col">
      <label>临时后端覆盖
        <select name="provider" data-model-provider-select data-base-provider="{escape(effective_provider)}">
          {_render_provider_options()}
        </select>
      </label>
      {planning_field_html}
    </div>
    {planning_help_html}
    {quality_fields_html}
    {quality_model_fields_html}
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
    <label>Max Tokens
      <input type="number" name="max_tokens" placeholder="沿用项目设置">
    </label>
    <label class="muted">
      <input type="checkbox" name="log_llm_payload" value="1">
      启用模型调用落盘（请求与返回将写入项目下 llm_logs，便于排查问题）
    </label>
    """


def _create_project(form: dict[str, str], api_keys: dict[str, str], progress_callback=None) -> str:
    provider = (form.get("provider") or "gemini").strip().lower()
    provider = _normalize_provider_for_ui(provider, default="")
    if provider not in WEB_SELECTABLE_PROVIDERS:
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
        "writing_quality_mode": normalize_writing_quality_mode(form.get("writing_quality_mode")),
        "review_mode": normalize_review_mode(form.get("review_mode")),
        "model_provider": provider,
        "model_name": (resolved_model_name or _default_model_for_provider(provider)).strip(),
        "api_base": (form.get("api_base") or _default_api_base_for_provider(provider)).strip(),
        "api_key": api_key,
        "max_tokens": int(form.get("max_tokens") or 4000),
        "timeout": _resolve_timeout_for_provider(provider, form.get("timeout") or _default_timeout_for_provider(provider)),
    }
    quality_model = _quality_model_from_form(form, api_keys)
    if quality_model:
        config["quality_model"] = quality_model

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


def _quality_status_label(report: dict) -> str:
    if bool(report.get("review_unavailable")):
        return "审稿不可用"
    return "通过" if bool(report.get("passed")) else "未通过"


def _render_chapter_quality_panel(project_id: str, chapter_slug: str, artifacts: dict) -> str:
    reports = artifacts.get("reports") or []
    drafts = artifacts.get("pre_rewrite_drafts") or []
    rewrite_count = int(artifacts.get("rewrite_count", 0) or 0)
    latest_report = reports[-1].get("report", {}) if reports else {}
    latest_status = _quality_status_label(latest_report) if latest_report else "暂无"
    report_button = (
        f'<a class="ghost-button" href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}/quality-report">查看质量报告</a>'
        if reports
        else ""
    )
    draft_button = (
        f'<a class="ghost-button" href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}/pre-rewrite">查看重写前文本</a>'
        if drafts
        else ""
    )
    empty_note = ""
    if not reports and not drafts:
        empty_note = '<p class="muted">暂无质量报告或自动重写记录。light 模式和未触发重写的历史章节可能没有相关产物。</p>'
    elif not drafts:
        empty_note = '<p class="muted">暂无重写前文本。历史章节如果生成时未保存原稿，无法补回。</p>'
    return f"""
    <section class="panel">
      <h2>质量优化</h2>
      <p><strong>已迭代优化次数：</strong>{rewrite_count}</p>
      <p><strong>质量报告：</strong>{len(reports)} 份</p>
      <p><strong>重写前文本：</strong>{len(drafts)} 份</p>
      <p><strong>最新审稿状态：</strong>{escape(latest_status)}</p>
      <div class="button-row">
        {report_button}
        {draft_button}
      </div>
      {empty_note}
    </section>
    """


def _render_issue_items(items: object) -> str:
    if not isinstance(items, list) or not items:
        return '<p class="muted">暂无</p>'
    rendered = []
    for item in items:
        if isinstance(item, dict):
            severity = str(item.get("severity") or "").strip()
            category = str(item.get("category") or "").strip()
            issue = str(item.get("issue") or "").strip()
            evidence = str(item.get("evidence") or "").strip()
            fix = str(item.get("fix") or "").strip()
            severity_html = f' <span class="pill">{escape(severity)}</span>' if severity else ""
            category_html = f' <span class="pill">{escape(category)}</span>' if category else ""
            evidence_html = f'<div class="muted">证据：{escape(evidence)}</div>' if evidence else ""
            fix_html = f'<div class="muted">修复：{escape(fix)}</div>' if fix else ""
            rendered.append(
                "<li>"
                f"<strong>{escape(issue or '未命名问题')}</strong>"
                f"{severity_html}"
                f"{category_html}"
                f"{evidence_html}"
                f"{fix_html}"
                "</li>"
            )
        else:
            rendered.append(f"<li>{escape(str(item))}</li>")
    return f"<ul>{''.join(rendered)}</ul>"


def _render_string_items(items: object) -> str:
    if not isinstance(items, list) or not items:
        return '<p class="muted">暂无</p>'
    return "<ul>" + "".join(f"<li>{escape(str(item))}</li>" for item in items) + "</ul>"


def _render_score_rows(scores: object) -> str:
    if not isinstance(scores, dict) or not scores:
        return '<p class="muted">暂无评分</p>'
    rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in scores.items()
    )
    return f'<table class="quality-table"><tbody>{rows}</tbody></table>'


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


def _render_narrator_preset_options(project_path: Path, selected: str = "") -> str:
    config = ensure_voice_config(str(project_path))
    active = selected or str(config.get("selected_narrator_id") or "")
    options = []
    for preset in narrator_preset_options(str(project_path)):
        value = str(preset.get("id") or "").strip()
        if not value:
            continue
        selected_attr = ' selected' if value == active else ""
        label = str(preset.get("label") or value).strip()
        options.append(f'<option value="{escape(value)}"{selected_attr}>{escape(label)}</option>')
    return "".join(options)


def _project_character_names(project_path: Path) -> list[str]:
    data = load_project(str(project_path))
    characters = data.get("characters") or {}
    names = []
    for group in ("protagonists", "supporting"):
        for character in characters.get(group) or []:
            name = str(character.get("name", "") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def _render_character_voice_options(project_path: Path) -> str:
    names = _project_character_names(project_path)
    options = ['<option value="">不上传角色参考音频</option>']
    for name in names:
        options.append(f'<option value="{escape(name)}">{escape(name)}</option>')
    return "".join(options)


def _audiobook_audio_url(project_id: str, record: dict) -> str:
    chapter_slug = str(record.get("chapter_slug", "") or "")
    combined = str(record.get("combined_audio", "") or "")
    file_name = Path(combined).name
    if not chapter_slug or not file_name:
        return ""
    return (
        f"/project/{urllib.parse.quote(project_id)}/audiobook-file/"
        f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(file_name)}"
    )


def _render_audiobook_player(project_id: str, record: dict | None) -> str:
    if not record:
        return "<p>当前还没有本章有声版本。</p>"
    audio_url = _audiobook_audio_url(project_id, record)
    if not audio_url:
        return "<p>当前还没有本章有声版本。</p>"
    duration = record.get("combined_duration_seconds", 0)
    duration_text = f"{duration} 秒" if duration else "未知时长"
    return f"""
    <div class="audiobook-player">
      <audio controls src="{escape(audio_url)}"></audio>
      <div class="muted">
        {escape(str(record.get("segment_count", len(record.get("segments", [])) or 0)))} 个片段，
        {escape(duration_text)}，
        {escape(record.get("generated_at", ""))}
      </div>
    </div>
    """


def _render_audiobook_records(project_id: str, records: list[dict]) -> str:
    if not records:
        return "<p>当前还没有有声章节。</p>"
    items = []
    for record in records[:6]:
        chapter_slug = str(record.get("chapter_slug", "") or "")
        audio_url = _audiobook_audio_url(project_id, record)
        if not chapter_slug or not audio_url:
            continue
        items.append(
            f"""
            <div class="audio-record">
              <div><strong>{escape(chapter_slug)}</strong></div>
              <audio controls src="{escape(audio_url)}"></audio>
              <div class="muted">{escape(str(record.get("segment_count", len(record.get("segments", [])) or 0)))} 个片段</div>
              <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}">打开章节</a>
            </div>
            """
        )
    return "".join(items) or "<p>当前还没有有声章节。</p>"


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


def _external_service_definitions() -> tuple[dict, ...]:
    return (
        {
            "id": "image_frame",
            "label": "Image Frame",
            "health_path": "/healthz",
            "runtime_loader": load_image_frame_runtime,
            "client_class": ImageFrameClient,
        },
        {
            "id": "audio_frame",
            "label": "Audio Frame API",
            "health_path": "/healthz",
            "runtime_loader": load_audio_frame_runtime,
            "client_class": AudioFrameClient,
        },
    )


def _external_service_payload_summary(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    summary = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[str(key)] = value
    return summary


def _external_service_status_label(status: str) -> str:
    mapping = {
        "succeeded": "连通",
        "failed": "失败",
        "unknown": "未检测",
    }
    return mapping.get(status, status or "unknown")


def _external_service_placeholder() -> dict:
    services = []
    for definition in _external_service_definitions():
        api_base = ""
        message = "启动检测尚未完成"
        try:
            runtime = definition["runtime_loader"]()
            api_base = str(runtime.get("api_base") or "").strip()
        except Exception as exc:
            message = f"配置读取失败：{exc}"
        services.append(
            {
                "id": definition["id"],
                "label": definition["label"],
                "api_base": api_base,
                "health_path": definition["health_path"],
                "ok": None,
                "status": "unknown",
                "message": message,
                "latency_ms": None,
                "details": {},
            }
        )
    return {"ok": False, "checked_at": "", "services": services}


def _check_external_service(definition: dict) -> dict:
    started = time.monotonic()
    api_base = ""
    try:
        runtime = definition["runtime_loader"]()
        api_base = str(runtime.get("api_base") or "").strip()
        client = definition["client_class"](api_base)
        payload = client.request_json(
            definition["health_path"],
            timeout=EXTERNAL_SERVICE_HEALTH_TIMEOUT_SECONDS,
        )
        ok = bool(payload.get("ok", True)) if isinstance(payload, dict) else True
        message = "连通正常" if ok else "健康检查返回异常"
        if isinstance(payload, dict) and payload.get("message"):
            message = str(payload.get("message"))
        return {
            "id": definition["id"],
            "label": definition["label"],
            "api_base": api_base,
            "health_path": definition["health_path"],
            "ok": ok,
            "status": "succeeded" if ok else "failed",
            "message": message,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "details": _external_service_payload_summary(payload),
        }
    except Exception as exc:
        return {
            "id": definition["id"],
            "label": definition["label"],
            "api_base": api_base,
            "health_path": definition["health_path"],
            "ok": False,
            "status": "failed",
            "message": str(exc),
            "latency_ms": int((time.monotonic() - started) * 1000),
            "details": {},
        }


def _check_external_services() -> dict:
    services = [_check_external_service(definition) for definition in _external_service_definitions()]
    return {
        "ok": all(bool(service.get("ok")) for service in services),
        "checked_at": utc_now(),
        "services": services,
    }


def _refresh_external_service_health() -> dict:
    snapshot = _check_external_services()
    with _EXTERNAL_SERVICE_HEALTH_LOCK:
        _EXTERNAL_SERVICE_HEALTH_CACHE.clear()
        _EXTERNAL_SERVICE_HEALTH_CACHE.update(snapshot)
    return snapshot


def _external_service_health_snapshot() -> dict:
    with _EXTERNAL_SERVICE_HEALTH_LOCK:
        if _EXTERNAL_SERVICE_HEALTH_CACHE:
            return {
                "ok": bool(_EXTERNAL_SERVICE_HEALTH_CACHE.get("ok")),
                "checked_at": str(_EXTERNAL_SERVICE_HEALTH_CACHE.get("checked_at") or ""),
                "services": [dict(service) for service in _EXTERNAL_SERVICE_HEALTH_CACHE.get("services", [])],
            }
    return _external_service_placeholder()


def _render_external_service_panel() -> str:
    snapshot = _external_service_health_snapshot()
    service_rows = []
    for service in snapshot.get("services", []):
        status = str(service.get("status") or "unknown")
        latency = service.get("latency_ms")
        latency_text = f" / {latency} ms" if isinstance(latency, int) else ""
        service_rows.append(
            f"""
            <div class="job-card" data-external-service-row="{escape(str(service.get('id') or ''))}">
              <div class="job-card-head">
                <strong>{escape(str(service.get("label") or ""))}</strong>
                <span class="status-pill {escape(_job_status_class(status))}" data-external-service-status>
                  {escape(_external_service_status_label(status))}
                </span>
              </div>
              <div class="muted" data-external-service-base>{escape(str(service.get("api_base") or ""))}{escape(str(service.get("health_path") or ""))}</div>
              <div class="muted" data-external-service-message>{escape(str(service.get("message") or ""))}{escape(latency_text)}</div>
            </div>
            """
        )
    checked_at = str(snapshot.get("checked_at") or "未检测")
    return f"""
    <section class="panel" data-external-service-panel>
      <div class="option-panel-head">
        <div>
          <h2>外部服务</h2>
          <p class="muted">Image Frame / Audio Frame 连通性</p>
        </div>
        <button type="button" class="ghost-button" data-external-service-check>测试连通性</button>
      </div>
      <div class="muted" data-external-service-checked>最近检测：{escape(checked_at)}</div>
      <div class="stack" data-external-service-list>
        {''.join(service_rows)}
      </div>
    </section>
    <script>
    (() => {{
      const panel = document.querySelector("[data-external-service-panel]");
      if (!panel || panel.dataset.bound === "1") return;
      panel.dataset.bound = "1";
      const button = panel.querySelector("[data-external-service-check]");
      const checked = panel.querySelector("[data-external-service-checked]");
      const list = panel.querySelector("[data-external-service-list]");
      const statusLabel = (status) => ({{succeeded: "连通", failed: "失败", unknown: "未检测"}}[status] || status || "unknown");
      const statusClass = (status) => {{
        const map = {{succeeded: "status-succeeded", failed: "status-failed", unknown: "status-neutral"}};
        return `status-pill ${{map[status] || "status-neutral"}}`;
      }};
      const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }}[char]));
      const render = (payload) => {{
        if (checked) checked.textContent = `最近检测：${{payload.checked_at || "未检测"}}`;
        if (!list) return;
        list.innerHTML = (payload.services || []).map((service) => {{
          const latency = Number.isInteger(service.latency_ms) ? ` / ${{service.latency_ms}} ms` : "";
          return `
            <div class="job-card" data-external-service-row="${{escapeHtml(service.id)}}">
              <div class="job-card-head">
                <strong>${{escapeHtml(service.label)}}</strong>
                <span class="${{statusClass(service.status)}}" data-external-service-status>${{statusLabel(service.status)}}</span>
              </div>
              <div class="muted" data-external-service-base>${{escapeHtml(service.api_base)}}${{escapeHtml(service.health_path)}}</div>
              <div class="muted" data-external-service-message>${{escapeHtml(service.message)}}${{escapeHtml(latency)}}</div>
            </div>`;
        }}).join("");
      }};
      button?.addEventListener("click", async () => {{
        button.disabled = true;
        const previousText = button.textContent;
        button.textContent = "检测中...";
        try {{
          const response = await fetch("/api/external-services/check", {{headers: {{"Accept": "application/json"}}}});
          const payload = await response.json();
          render(payload);
        }} catch (error) {{
          if (checked) checked.textContent = `检测失败：${{error}}`;
        }} finally {{
          button.disabled = false;
          button.textContent = previousText || "测试连通性";
        }}
      }});
    }})();
    </script>
    """


def _render_job_events(events: object) -> str:
    if not isinstance(events, list) or not events:
        return '<li class="muted">暂无任务日志。</li>'
    rows = []
    for item in events:
        if not isinstance(item, dict):
            continue
        event_time = str(item.get("time") or "").strip()
        stage = str(item.get("stage") or "").strip()
        message = str(item.get("message") or stage or "").strip()
        stage_html = f' <span class="pill">{escape(stage)}</span>' if stage else ""
        rows.append(
            f'<li><span class="mono">{escape(event_time)}</span>{stage_html} {escape(message)}</li>'
        )
    return "".join(rows) or '<li class="muted">暂无任务日志。</li>'


def _job_error_summary(job: dict) -> str:
    error = str(job.get("error") or "").strip()
    if not error:
        return ""
    first_line = ""
    for line in error.splitlines():
        line = line.strip()
        if not line:
            continue
        if not first_line:
            first_line = line
        if "Error" in line or "Exception" in line or line.startswith("project_manager."):
            first_line = line
            break
    if len(first_line) > 360:
        return first_line[:357].rstrip() + "..."
    return first_line


def _render_job_error_box(job: dict) -> str:
    error = str(job.get("error") or "").strip()
    display = "block" if error else "none"
    return (
        f'<div id="job-error-box" class="status-box" style="display:{display}">'
        "<strong>错误信息</strong>"
        f'<div id="job-error" class="mono">{escape(error)}</div>'
        "</div>"
    )


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
        error_summary = _job_error_summary(job)
        error_html = (
            f'<div class="warning-box"><strong>失败原因：</strong>{escape(error_summary)}</div>'
            if error_summary
            else ""
        )
        cards.append(
            f"""
            <div class="job-card">
              <div class="job-card-head">
                <a href="/job/{escape(job['id'])}"><strong>{escape(job.get('title', job['id']))}</strong></a>
                <span class="status-pill {escape(_job_status_class(job.get('status', '')))}">{escape(_job_status_label(job.get('status', '')))}</span>
              </div>
              <div class="muted">{escape(job.get('message', '') or '等待状态更新')}</div>
              {error_html}
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
    session_objective = str(session.get("objective", "") or "").strip()
    for index, option in enumerate(session.get("options", []), start=1):
        option_id = str(option.get("option_id", "") or "").strip()
        checked_attr = ' checked' if option_id == recommended_option_id else ""
        if option.get("custom"):
            badge = '<span class="option-badge">自定义</span>'
        else:
            badge = '<span class="option-badge">推荐</span>' if option.get("recommended") else ""
        plan_summary = str(option.get("plan_summary", "") or option.get("summary", "") or "").strip()
        plan_steps = option.get("plan_steps") or option.get("key_events") or []
        key_events = "".join(f"<li>{escape(item)}</li>" for item in plan_steps)
        card_class = "option-card custom-option-card" if option.get("custom") else "option-card"
        options_html.append(
            f"""
            <label class="{card_class}">
              <input type="radio" name="progression_option" value="{escape(option_id)}"{checked_attr}>
              <div class="option-card-head">
                <strong>{index}. {escape(option.get('title', ''))}</strong>
                {badge}
              </div>
              <div class="muted">{escape(plan_summary)}</div>
              <ul class="option-list">{key_events}</ul>
            </label>
            """
        )

    disabled_attr = " disabled" if disabled else ""
    return f"""
    <div class="option-session-meta">
      <div class="muted">当前会话：{escape(session.get('session_id', ''))}</div>
      <div class="muted">目标第 {escape(str(session.get('target_chapter_number', '')))} 章</div>
      <div class="muted">本组 plan 基于 objective：{escape(session_objective or '暂无')}</div>
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
        <div class="muted">选择“空白自定义项”后，这段输入会直接作为当前章的执行 plan；选择普通方案时，它只会作为微调补充，不能改写 objective。</div>
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
    objective = str(task_card.get("objective", "") or task_card.get("goal", "") or "").strip()
    plan_summary = str(task_card.get("plan_summary", "") or task_card.get("summary", "") or "").strip()
    plan_steps = task_card.get("plan_steps") or task_card.get("key_events") or []
    key_events = "".join(f"<li>{escape(item)}</li>" for item in plan_steps)
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
          <h3>有效当前章任务卡</h3>
          <div class="muted">{escape(source_label)}</div>
          {derived_text}
        </div>
        <span class="pill">第 {escape(str(task_card.get('chapter_number', '')))} 章</span>
      </div>
      <p><strong>当前章 objective：</strong>{escape(objective or "暂无")}</p>
      <p><strong>当前章 plan：</strong>{escape(plan_summary or "暂无")}</p>
      <p><strong>卷目标：</strong>{escape(task_card.get("volume_goal", "") or "暂无")}</p>
      <div class="muted"><strong>计划步骤：</strong></div>
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
    service_scope = escape(repo_info.get("service_scope", "") or "未知")
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
      <p><strong>服务作用域：</strong>{service_scope}</p>
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
    .quality-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0 14px;
      background: rgba(255,255,255,0.56);
      border: 1px solid rgba(124, 91, 62, 0.12);
      border-radius: 12px;
      overflow: hidden;
    }}
    .quality-table th,
    .quality-table td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid rgba(124, 91, 62, 0.1);
      vertical-align: top;
    }}
    .quality-table th {{
      width: 220px;
      color: var(--accent-dark);
      background: rgba(255,255,255,0.48);
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
    audio {{
      width: 100%;
    }}
    .audio-record, .audiobook-player {{
      display: grid;
      gap: 8px;
      padding: 12px;
      border-radius: 16px;
      border: 1px solid rgba(124, 91, 62, 0.15);
      background: rgba(255,255,255,0.56);
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
    .status-neutral {{
      color: #5f5a55;
      background: rgba(255,255,255,0.72);
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
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "external-services" and parts[2] == "check":
            self._handle_external_services_check()
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
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "chapter" and parts[4] == "quality-report":
            self._handle_chapter_quality_report(parts[1], parts[3], notice=notice, error=error)
            return
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "chapter" and parts[4] == "pre-rewrite":
            self._handle_chapter_pre_rewrite(parts[1], parts[3], notice=notice, error=error)
            return
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "illustration-file":
            self._handle_illustration_file(parts[1], parts[3], parts[4])
            return
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "audiobook-file":
            self._handle_audiobook_file(parts[1], parts[3], parts[4])
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
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "chapter" and parts[4] == "polish":
            self._handle_polish_chapter_async(parts[1], parts[3], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "rollback":
            self._handle_rollback(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "illustrate":
            self._handle_illustrate_async(parts[1], form)
            return
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "audiobook":
            self._handle_audiobook_async(parts[1], form)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_bytes = self.rfile.read(length)
        self._uploaded_files = {}
        content_type = str(self.headers.get("Content-Type", "") or "")
        if content_type.lower().startswith("multipart/form-data"):
            return self._read_multipart_form(raw_bytes, content_type)

        raw = raw_bytes.decode("utf-8")
        parsed = urllib.parse.parse_qs(raw, keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items()}

    def _read_multipart_form(self, raw_bytes: bytes, content_type: str) -> dict[str, str]:
        message = BytesParser(policy=email_policy_default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + raw_bytes
        )
        form: dict[str, str] = {}
        files: dict[str, UploadedVoiceFile] = {}
        for part in message.iter_parts():
            disposition = part.get("Content-Disposition", "")
            if "form-data" not in disposition:
                continue
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            filename = part.get_filename()
            if filename:
                files[str(name)] = UploadedVoiceFile(
                    filename=filename,
                    content=payload,
                    content_type=str(part.get_content_type() or ""),
                )
            else:
                charset = part.get_content_charset() or "utf-8"
                form[str(name)] = payload.decode(charset, errors="replace")
        self._uploaded_files = files
        return form

    def _uploaded_file(self, name: str) -> UploadedVoiceFile | None:
        return getattr(self, "_uploaded_files", {}).get(name)

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

    def _handle_external_services_check(self) -> None:
        self._write_json(_refresh_external_service_health())

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
              {_render_job_events(job.get("events"))}
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
          const renderEvents = (events) => {{
            const items = Array.isArray(events) ? events : [];
            if (!items.length) return '<li class="muted">暂无任务日志。</li>';
            return items.map((item) => {{
              const stage = item.stage ? ` <span class="pill">${{escapeHtml(item.stage)}}</span>` : "";
              const message = item.message || item.stage || "";
              return `<li><span class="mono">${{escapeHtml(item.time || "")}}</span>${{stage}} ${{escapeHtml(message)}}</li>`;
            }}).join("");
          }};
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
                  {_render_cost_meta(item.get("stats") or {})}
                </div>
                """
            )
        project_html = "".join(cards) or "<p>当前还没有项目，先在左侧创建一个新项目吧。</p>"
        recent_jobs_html = _render_job_cards(
            JOB_REGISTRY.list_jobs(limit=6),
            "当前还没有后台任务。",
        )
        external_service_panel_html = _render_external_service_panel()
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
                      {_render_provider_options("gemini", include_project_default=False)}
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
                  <label>Max Tokens
                    <input type="number" name="max_tokens" value="4000">
                  </label>
                  <label>Timeout
                    <input type="number" name="timeout" value="120">
                  </label>
                </div>
                <label>API Base（可选）
                  <input type="text" name="api_base" placeholder="如需自定义接口地址可填写">
                </label>
                <label>Planning Mode
                  <select name="planning_mode">
                    {_render_planning_mode_options(DEFAULT_PLANNING_MODE)}
                  </select>
                </label>
                <div class="muted">{escape(_planning_mode_help(DEFAULT_PLANNING_MODE))}</div>
                <div class="two-col">
                  <label>写作质量模式
                    <select name="writing_quality_mode">
                      {_render_quality_mode_options(DEFAULT_WRITING_QUALITY_MODE)}
                    </select>
                  </label>
                  <label>审稿模式
                    <select name="review_mode">
                      {_render_review_mode_options(DEFAULT_REVIEW_MODE)}
                    </select>
                  </label>
                </div>
                <div class="muted">默认平衡模式会先生成短蓝图，写后保存质检报告；高质量模式可在自动审稿失败时重写一次。</div>
                <div class="two-col">
                  <label>Quality Provider
                    <select name="quality_provider">
                      {_render_quality_provider_options()}
                    </select>
                  </label>
                  <label>Quality Model
                    <input type="text" name="quality_model_name" placeholder="inherit main model">
                  </label>
                </div>
                <div class="two-col">
                  <label>Quality API Base
                    <input type="text" name="quality_api_base" placeholder="inherit or provider default">
                  </label>
                  <label>Quality Timeout
                    <input type="number" name="quality_timeout" placeholder="inherit or provider default">
                  </label>
                </div>
                <label>Quality Max Tokens
                  <input type="number" name="quality_max_tokens" placeholder="inherit main">
                </label>
                <div class="muted">Optional advanced model used only for craft brief, quality review, and rewrite.</div>
                <button type="submit">创建项目</button>
              </form>
            </section>
            {external_service_panel_html}
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
        project_stats = project.get("stats") or {}
        stats = project_stats.get("total", {})
        illustration_records = list_illustration_records(str(project_path))
        active_jobs = JOB_REGISTRY.list_jobs(project_id=project_id, active_only=True, limit=6)
        active_jobs_html = _render_job_cards(active_jobs, "当前没有运行中的后台任务。")
        external_service_panel_html = _render_external_service_panel()
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
              {_render_sidebar_usage_stats(project_stats)}
              <p><strong>Planning:</strong>{escape(_planning_mode_label(project.get("planning_mode", DEFAULT_PLANNING_MODE)))}</p>
              <p><strong>Quality:</strong>{escape(_quality_mode_label((project.get("llm_config") or {}).get("writing_quality_mode", DEFAULT_WRITING_QUALITY_MODE)))}</p>
              <p><strong>Review:</strong>{escape(_review_mode_label((project.get("llm_config") or {}).get("review_mode", DEFAULT_REVIEW_MODE)))}</p>
              <p><strong>Quality Model:</strong>{escape(_quality_model_label(project.get("llm_config") or {}))}</p>
            </section>
            <section class="panel">
              <h3>后台任务</h3>
              {active_jobs_html}
            </section>
            {external_service_panel_html}
            {busy_notice}
            <section class="panel">
              <h3>续写</h3>
              <form method="post" action="/project/{escape(project_id)}/continue">
                <fieldset{busy_attr}>
                <div class="two-col">
                  <label>续写章节数
                    <input type="number" name="count" value="1" min="1" max="20">
                  </label>
                  <label>选 plan 策略
                    <select name="selection_mode">
                      {_render_auto_selection_mode_options(SELECTION_MODE_RECOMMENDED)}
                    </select>
                  </label>
                </div>
                <div class="muted">{escape(_auto_continue_help(normalize_planning_mode(project.get("planning_mode", DEFAULT_PLANNING_MODE))))}</div>
                <label>想看的内容 / 情节走向
                  <textarea name="user_request" placeholder="例如：先推进食堂据点建设，再增加一点轻松互怼的互动。"></textarea>
                </label>
                <div class="two-col">
                  <label>临时后端覆盖
                    <select name="provider">
                      {_render_provider_options()}
                    </select>
                  </label>
                  <div></div>
                </div>
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
                  <label>Max Tokens
                    <input type="number" name="max_tokens" placeholder="沿用项目设置">
                  </label>
                  <label>Timeout
                    <input type="number" name="timeout" placeholder="沿用项目设置">
                  </label>
                </div>
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
        chapter_text = chapter_file.read_text(encoding="utf-8")
        chapter_number = _chapter_number_from_slug(chapter_slug)
        quality_artifacts = (
            list_quality_artifacts(str(project_path), chapter_number)
            if chapter_number is not None
            else {"reports": [], "pre_rewrite_drafts": [], "rewrite_count": 0}
        )
        quality_panel_html = _render_chapter_quality_panel(project_id, chapter_slug, quality_artifacts)
        current_index = next((idx for idx, chapter in enumerate(chapters) if chapter["slug"] == chapter_slug), -1)
        previous_chapter = chapters[current_index - 1] if current_index > 0 else None
        next_chapter = chapters[current_index + 1] if 0 <= current_index < len(chapters) - 1 else None
        active_jobs = JOB_REGISTRY.list_jobs(project_id=project_id, active_only=True, limit=4)
        blocking_jobs = [job for job in active_jobs if job.get("blocks_project", True)]
        project_busy = bool(blocking_jobs)
        busy_attr = " disabled" if project_busy else ""
        busy_notice = (
            '<div class="warning-box">当前项目有后台任务正在运行。为避免并发写入冲突，章节润色暂时禁用。</div>'
            if project_busy
            else ""
        )
        project_llm_config = project.get("llm_config") or {}
        polish_runtime_fields_html = _render_runtime_override_fields(
            str(project_llm_config.get("model_provider") or ""),
            str(project_llm_config.get("model_name") or project_llm_config.get("model") or ""),
            include_planning_mode=False,
            include_quality_fields=False,
        )
        polish_preset_html = _render_polish_preset_checkboxes()
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
        audiobook_record = get_audiobook_record(str(project_path), chapter_slug)
        audiobook_player_html = _render_audiobook_player(project_id, audiobook_record)
        narrator_options_html = _render_narrator_preset_options(project_path)
        character_voice_options_html = _render_character_voice_options(project_path)

        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}">返回项目</a>
            <h2>{escape(chapter_file.name)}</h2>
            <div class="chapter-view">{escape(chapter_text)}</div>
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
          {quality_panel_html}
          {busy_notice}
          <section class="panel">
            <h2>章节润色</h2>
            <p class="muted">对当前章节做表达、节奏和细节层面的润色。完成后会直接覆盖本章正文，并在项目下自动备份原文。</p>
            <form method="post" action="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}/polish">
              <fieldset{busy_attr}>
                <label>润色方向（可多选）
                  <div class="button-row polish-preset-row">
                    {polish_preset_html}
                  </div>
                </label>
                <label>自定义润色要求（可选）
                  <textarea name="polish_custom_request" placeholder="例如：多一点轻松互怼，但不要改变本章事件结果。"></textarea>
                </label>
                {polish_runtime_fields_html}
                <button type="submit">开始润色本章</button>
              </fieldset>
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
          <section class="panel">
            <h2>本章有声小说</h2>
            {audiobook_player_html}
            <form method="post" action="/project/{escape(project_id)}/audiobook" enctype="multipart/form-data">
              <fieldset{busy_attr}>
                <input type="hidden" name="chapter_slug" value="{escape(chapter_slug)}">
                <label>旁白音色
                  <select name="narrator_preset">
                    {narrator_options_html}
                  </select>
                </label>
                <label>旁白参考 WAV（可选）
                  <input type="file" name="narrator_reference_audio" accept=".wav,audio/wav,audio/x-wav">
                </label>
                <label>旁白参考文本（可选）
                  <input type="text" name="narrator_prompt_text" placeholder="如上传参考音频，可填写对应文本以增强克隆相似度">
                </label>
                <div class="two-col">
                  <label>角色参考目标
                    <select name="character_voice_name">
                      {character_voice_options_html}
                    </select>
                  </label>
                  <label>角色参考 WAV（可选）
                    <input type="file" name="character_reference_audio" accept=".wav,audio/wav,audio/x-wav">
                  </label>
                </div>
                <label>角色参考文本（可选）
                  <input type="text" name="character_prompt_text" placeholder="如上传角色参考音频，可填写对应文本">
                </label>
                <label><input type="checkbox" name="force" value="1"> 强制重新生成</label>
                <button type="submit">生成本章有声版</button>
              </fieldset>
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

    def _handle_chapter_quality_report(self, project_id: str, chapter_slug: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return
        chapter_number = _chapter_number_from_slug(chapter_slug)
        if chapter_number is None:
            self.send_error(HTTPStatus.NOT_FOUND, "章节不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        project = load_json(str(project_path / "project.json"))
        project_name = _repair_display_text(project.get("name", project_id))
        artifacts = list_quality_artifacts(str(project_path), chapter_number)
        report_sections = []
        for item in artifacts.get("reports") or []:
            report = item.get("report") if isinstance(item.get("report"), dict) else {}
            attempt = item.get("attempt") or "?"
            if item.get("error"):
                report_sections.append(
                    f"""
                    <section class="panel">
                      <h2>Attempt {escape(str(attempt))}</h2>
                      <div class="warning-box">报告读取失败：{escape(str(item.get("error")))}</div>
                    </section>
                    """
                )
                continue
            raw_json = json.dumps(report, ensure_ascii=False, indent=2)
            report_sections.append(
                f"""
                <section class="panel">
                  <h2>Attempt {escape(str(attempt))}</h2>
                  <p><strong>状态：</strong>{escape(_quality_status_label(report))}</p>
                  <p><strong>平均分：</strong>{escape(str(report.get("average_score", "暂无")))}</p>
                  <h3>分项评分</h3>
                  {_render_score_rows(report.get("scores"))}
                  <h3>阻断问题</h3>
                  {_render_issue_items(report.get("blocking_issues"))}
                  <h3>主要问题</h3>
                  {_render_string_items(report.get("issues"))}
                  <h3>重写方案</h3>
                  {_render_string_items(report.get("rewrite_plan"))}
                  <h3>修订建议</h3>
                  <div class="chapter-view">{escape(str(report.get("revision_guidance") or "暂无"))}</div>
                  <h3>原始 JSON</h3>
                  <div class="chapter-view">{escape(raw_json)}</div>
                </section>
                """
            )
        if not report_sections:
            report_sections.append('<section class="panel"><h2>质量报告</h2><p class="muted">暂无本章质量报告。</p></section>')

        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}">返回章节</a>
            <h2>{escape(chapter_slug)} 质量报告</h2>
            <p class="muted">这里展示本章所有审稿 attempt，包含自动重写后的二审报告。</p>
          </section>
          {''.join(report_sections)}
        </div>
        """
        self._write_html(
            _render_page(
                f"{project_name} - {chapter_slug} 质量报告",
                body,
                notice=notice,
                error=error,
                auth_enabled=auth_settings.enabled,
                authenticated=authenticated,
            )
        )

    def _handle_chapter_pre_rewrite(self, project_id: str, chapter_slug: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return
        chapter_number = _chapter_number_from_slug(chapter_slug)
        if chapter_number is None:
            self.send_error(HTTPStatus.NOT_FOUND, "章节不存在")
            return
        auth_settings = self._current_auth_settings()
        authenticated = self._is_authenticated(auth_settings)

        project = load_json(str(project_path / "project.json"))
        project_name = _repair_display_text(project.get("name", project_id))
        artifacts = list_quality_artifacts(str(project_path), chapter_number)
        draft_sections = []
        for item in artifacts.get("pre_rewrite_drafts") or []:
            path = Path(str(item.get("path") or ""))
            text = ""
            error_text = ""
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as exc:  # pragma: no cover - damaged local artifact
                error_text = str(exc)
            attempt = item.get("rewrite_attempt") or "?"
            if error_text:
                draft_sections.append(
                    f"""
                    <section class="panel">
                      <h2>重写前文本 {escape(str(attempt))}</h2>
                      <div class="warning-box">文本读取失败：{escape(error_text)}</div>
                    </section>
                    """
                )
                continue
            draft_sections.append(
                f"""
                <section class="panel">
                  <h2>重写前文本 {escape(str(attempt))}</h2>
                  <div class="chapter-view">{escape(text)}</div>
                </section>
                """
            )
        if not draft_sections:
            draft_sections.append('<section class="panel"><h2>重写前文本</h2><p class="muted">暂无本章重写前文本。历史章节如果生成时未保存原稿，无法补回。</p></section>')

        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}">返回章节</a>
            <h2>{escape(chapter_slug)} 重写前文本</h2>
            <p class="muted">这些文本来自高质量自动审稿失败后、执行重写前保存的草稿。</p>
          </section>
          {''.join(draft_sections)}
        </div>
        """
        self._write_html(
            _render_page(
                f"{project_name} - {chapter_slug} 重写前文本",
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

    def _handle_audiobook_file(self, project_id: str, chapter_slug: str, file_name: str) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        try:
            file_path = audiobook_file_path(project_path, chapter_slug, file_name)
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND, "有声章节音频不存在")
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
            selection_mode = validate_selection_mode(form.get("selection_mode"), allow_manual=False)
            runtime_overrides = _runtime_overrides_from_form(form)
            runtime_config = _build_runtime_config(project_path, runtime_overrides, api_keys)
            chapter_paths = run_next_chapters(
                str(project_path),
                runtime_config,
                count,
                user_request=(form.get("user_request") or "").strip(),
                selection_mode=selection_mode,
                runtime_overrides=runtime_overrides,
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
            <div id="job-error-box" class="status-box" style="display:{'block' if job.get('error') else 'none'}">
              <strong>错误信息</strong>
              <div id="job-error" class="mono">{escape(job.get("error", ""))}</div>
            </div>
          </section>
          <section class="panel">
            <h3>任务日志</h3>
            <ol id="job-events" class="job-log">
              {_render_job_events(job.get("events"))}
            </ol>
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
            if (ch === '"') return "&quot;";
            return "&#39;";
          }});
          const renderEvents = (events) => {{
            const items = Array.isArray(events) ? events : [];
            if (!items.length) return '<li class="muted">暂无任务日志。</li>';
            return items.map((item) => {{
              const stage = item.stage ? ` <span class="pill">${{escapeHtml(item.stage)}}</span>` : "";
              const message = item.message || item.stage || "";
              return `<li><span class="mono">${{escapeHtml(item.time || "")}}</span>${{stage}} ${{escapeHtml(message)}}</li>`;
            }}).join("");
          }};
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
        project_stats = project.get("stats") or {}
        stats = project_stats.get("total", {})
        illustration_records = list_illustration_records(str(project_path))
        audiobook_records = list_audiobook_records(str(project_path))
        active_jobs = JOB_REGISTRY.list_jobs(project_id=project_id, active_only=True, limit=8)
        active_jobs_html = _render_job_cards(active_jobs, "当前没有运行中的后台任务。")
        external_service_panel_html = _render_external_service_panel()
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
        audiobook_gallery = _render_audiobook_records(project_id, audiobook_records)
        narrator_options_html = _render_narrator_preset_options(project_path)
        character_voice_options_html = _render_character_voice_options(project_path)
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
                <p><strong>当前章 objective：</strong>{escape(effective_task.get("objective", "") or effective_task.get("goal", "") or "暂无")}</p>
                <p><strong>卷目标：</strong>{escape(effective_task.get("volume_goal", "") or "暂无")}</p>
                <p><strong>live-state 下一目标：</strong>{escape(plot_state.get("next_chapter_goal", "") or "暂无")}</p>
                <p><strong>当前位置：</strong>{escape(plot_state.get("current_location", "") or "未知")}</p>
                <p><strong>当前时间：</strong>{escape(plot_state.get("current_time", "") or "未知")}</p>
                {_render_sidebar_usage_stats(project_stats)}
                <p><strong>Planning:</strong>{escape(_planning_mode_label(planning_mode))}</p>
                <p><strong>Quality:</strong>{escape(_quality_mode_label(project_llm_config.get("writing_quality_mode", DEFAULT_WRITING_QUALITY_MODE)))}</p>
                <p><strong>Review:</strong>{escape(_review_mode_label(project_llm_config.get("review_mode", DEFAULT_REVIEW_MODE)))}</p>
                <p><strong>Quality Model:</strong>{escape(_quality_model_label(project_llm_config))}</p>
              </div>
            </section>
            <section class="panel">
              <h3>后台任务</h3>
              {active_jobs_html}
            </section>
            {external_service_panel_html}
            {busy_notice}
            <section class="panel">
              <h3>续写</h3>
              <form method="post" action="/project/{escape(project_id)}/continue">
                <fieldset{busy_attr}>
                  <div class="two-col">
                    <label>续写章节数
                      <input type="number" name="count" value="1" min="1" max="20">
                    </label>
                    <label>选 plan 策略
                      <select name="selection_mode">
                        {_render_auto_selection_mode_options(SELECTION_MODE_RECOMMENDED)}
                      </select>
                    </label>
                  </div>
                  <div class="muted">{escape(_auto_continue_help(planning_mode))}</div>
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
              <h3>有声小说</h3>
              <form method="post" action="/project/{escape(project_id)}/audiobook" enctype="multipart/form-data">
                <fieldset{busy_attr}>
                  <label>目标章节
                    <select name="chapter_slug">
                      {''.join(chapter_options)}
                    </select>
                  </label>
                  <label>旁白音色
                    <select name="narrator_preset">
                      {narrator_options_html}
                    </select>
                  </label>
                  <label>旁白参考 WAV（可选）
                    <input type="file" name="narrator_reference_audio" accept=".wav,audio/wav,audio/x-wav">
                  </label>
                  <label>旁白参考文本（可选）
                    <input type="text" name="narrator_prompt_text" placeholder="如上传参考音频，可填写对应文本以增强克隆相似度">
                  </label>
                  <div class="two-col">
                    <label>角色参考目标
                      <select name="character_voice_name">
                        {character_voice_options_html}
                      </select>
                    </label>
                    <label>角色参考 WAV（可选）
                      <input type="file" name="character_reference_audio" accept=".wav,audio/wav,audio/x-wav">
                    </label>
                  </div>
                  <label>角色参考文本（可选）
                    <input type="text" name="character_prompt_text" placeholder="如上传角色参考音频，可填写对应文本">
                  </label>
                  <label><input type="checkbox" name="force" value="1"> 强制重新生成</label>
                  <button type="submit">生成有声章节</button>
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
            {_render_token_cost_panel(project_stats)}
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
                  <label>本章 objective（可修改）
                    <textarea name="objective" placeholder="例如：建立临时安全区，并确认是否需要离开隔离区搜集物资。">{escape(effective_task.get("objective", "") or effective_task.get("goal", "") or "")}</textarea>
                  </label>
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
            <section class="panel">
              <h2>最近有声章节</h2>
              <div class="gallery">{audiobook_gallery}</div>
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
            selection_mode = validate_selection_mode(form.get("selection_mode"), allow_manual=False)
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
                selection_mode=selection_mode,
                runtime_overrides=runtime_overrides,
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
                objective_override=(form.get("objective") or "").strip(),
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

    def _handle_polish_chapter_async(self, project_id: str, chapter_slug: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            runtime_overrides = _runtime_overrides_from_form(form)
            runtime_config = _build_runtime_config(project_path, runtime_overrides, api_keys)
            preset_ids = _polish_preset_ids_from_form(form)
            custom_request = (form.get("polish_custom_request") or "").strip()
            job = JOB_REGISTRY.create_job(
                kind="polish_chapter",
                title=f"润色章节：{chapter_slug}",
                project_id=project_id,
                project_path=str(project_path.resolve()),
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "/chapter/"
                + urllib.parse.quote(chapter_slug)
                + "?error="
                + urllib.parse.quote(str(exc))
            )
            return

        def runner(progress_callback):
            result = run_chapter_polish(
                str(project_path),
                runtime_config,
                chapter_slug,
                preset_ids=preset_ids,
                custom_request=custom_request,
                progress_callback=progress_callback,
            )
            backup_label = ""
            backup_path = Path(str(result.get("backup_path", "")))
            try:
                backup_label = str(backup_path.relative_to(project_path)).replace("\\", "/")
            except ValueError:
                backup_label = str(backup_path)
            message = f"章节润色完成，原文已备份到 {backup_label}。"
            if int(result.get("staled_progression_sessions", 0) or 0):
                message += " 已让旧的下一章推进选项失效。"
            result_url = (
                "/project/"
                + urllib.parse.quote(project_id)
                + "/chapter/"
                + urllib.parse.quote(chapter_slug)
                + "?notice="
                + urllib.parse.quote(message)
            )
            return {
                "message": message,
                "result_url": result_url,
                "result_label": "打开章节",
                "project_id": project_id,
                "project_path": str(project_path.resolve()),
            }

        _start_background_job(job["id"], runner)
        self._redirect("/job/" + urllib.parse.quote(job["id"]))

    def _handle_audiobook_async(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        chapter_slug = (form.get("chapter_slug") or "latest").strip() or "latest"
        narrator_upload = self._uploaded_file("narrator_reference_audio")
        character_upload = self._uploaded_file("character_reference_audio")
        character_name = (form.get("character_voice_name") or "").strip()
        try:
            if character_upload and not character_name:
                raise RuntimeError("上传角色参考音频前，请先选择角色。")
            job = JOB_REGISTRY.create_job(
                kind="audiobook",
                title=f"生成有声章节：{chapter_slug}",
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
            if narrator_upload:
                progress_callback({"stage": "audiobook_voice_ref", "message": "正在保存旁白参考音频"})
                save_uploaded_voice_reference(
                    str(project_path),
                    target="narrator",
                    uploaded_file=narrator_upload,
                    prompt_text=(form.get("narrator_prompt_text") or "").strip(),
                )
            if character_upload:
                progress_callback({"stage": "audiobook_voice_ref", "message": f"正在保存 {character_name} 的参考音频"})
                save_uploaded_voice_reference(
                    str(project_path),
                    target=character_name,
                    uploaded_file=character_upload,
                    prompt_text=(form.get("character_prompt_text") or "").strip(),
                )
            results = generate_audiobook_chapters(
                str(project_path),
                chapter_refs=[chapter_slug],
                force=bool(form.get("force")),
                narrator_preset=(form.get("narrator_preset") or "").strip(),
                progress_callback=progress_callback,
            )
            result = results[0]
            chapter_target = result.get("chapter_slug", chapter_slug)
            message = "已复用现有有声章节。" if result.get("reused") else "有声章节生成完成。"
            if narrator_upload or character_upload:
                message = "参考音频已保存，" + message
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
    parser.add_argument("--service-scope", default=WEBUI_SERVICE_SCOPE, choices=("auto", "user", "system"), help=argparse.SUPPRESS)
    parser.add_argument("--status-path", default=str(_admin_action_status_path()), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.admin_task:
        service_scope = args.service_scope
        if service_scope == "auto":
            service_scope = _resolve_service_scope(args.service_name)
        raise SystemExit(
            _run_admin_task(
                args.admin_task,
                repo_root=args.repo_root,
                service_name=args.service_name,
                service_scope=service_scope,
                status_path=args.status_path,
            )
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    health_snapshot = _refresh_external_service_health()
    status_text = "ok" if health_snapshot.get("ok") else "degraded"
    print(f"[{utc_now()}] External service health check: {status_text}")
    for service in health_snapshot.get("services", []):
        print(
            "[{time}] - {label} {status} {base}: {message}".format(
                time=utc_now(),
                label=service.get("label") or service.get("id"),
                status=service.get("status"),
                base=(service.get("api_base") or "") + (service.get("health_path") or ""),
                message=service.get("message") or "",
            )
        )
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
