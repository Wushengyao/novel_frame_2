from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


DEFAULT_AUTH_CONFIG_PATH = Path.home() / ".config" / "novel-writer" / "webui_auth.env"
DEFAULT_AUTH_COOKIE_NAME = "novel_writer_webui_session"


@dataclass(frozen=True)
class WebAuthSettings:
    enabled: bool
    username: str
    password: str
    secret_key: str
    cookie_name: str
    cookie_secure: bool
    session_max_age_seconds: int
    login_max_attempts: int
    login_window_seconds: int
    login_lockout_seconds: int
    config_path: str


class AuthService:
    DEFAULT_SECRET_KEY = "change-this-session-secret"
    DEFAULT_USERNAME = "admin"
    DEFAULT_PASSWORD = "ChangeThisPassword!"

    def __init__(self, settings: WebAuthSettings) -> None:
        self.settings = settings

    def verify(self, username: str, password: str) -> bool:
        if not self.settings.enabled:
            return True
        return hmac.compare_digest(username.strip(), self.settings.username) and hmac.compare_digest(
            password,
            self.settings.password,
        )

    def issue_token(self) -> str:
        payload = f"{self.settings.username}:{self.settings.password}".encode("utf-8")
        secret = self.settings.secret_key.encode("utf-8")
        return hmac.new(secret, payload, hashlib.sha256).hexdigest()

    def verify_token(self, token: str | None) -> bool:
        if not self.settings.enabled:
            return True
        if not token:
            return False
        return hmac.compare_digest(token, self.issue_token())

    def should_warn_default_credentials(self) -> bool:
        if not self.settings.enabled:
            return False
        return (
            self.settings.username == self.DEFAULT_USERNAME
            and self.settings.password == self.DEFAULT_PASSWORD
        )

    def should_warn_default_secret_key(self) -> bool:
        if not self.settings.enabled:
            return False
        return self.settings.secret_key == self.DEFAULT_SECRET_KEY


@dataclass
class _AttemptBucket:
    attempts: deque[float] = field(default_factory=deque)
    locked_until: float = 0.0


class LoginAttemptGuard:
    def __init__(
        self,
        *,
        max_attempts: int,
        window_seconds: int,
        lockout_seconds: int,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.window_seconds = max(1, int(window_seconds))
        self.lockout_seconds = max(1, int(lockout_seconds))
        self._now_fn = now_fn or time.time
        self._buckets: dict[str, _AttemptBucket] = {}

    def is_locked(self, key: str) -> bool:
        bucket = self._bucket(key)
        now = self._now()
        self._prune(bucket, now)
        if bucket.locked_until <= now:
            bucket.locked_until = 0.0
            return False
        return True

    def retry_after_seconds(self, key: str) -> int:
        bucket = self._bucket(key)
        now = self._now()
        remain = int(bucket.locked_until - now)
        return remain if remain > 0 else 0

    def register_failure(self, key: str) -> None:
        bucket = self._bucket(key)
        now = self._now()
        self._prune(bucket, now)
        bucket.attempts.append(now)
        if len(bucket.attempts) >= self.max_attempts:
            bucket.locked_until = now + self.lockout_seconds
            bucket.attempts.clear()

    def register_success(self, key: str) -> None:
        bucket = self._bucket(key)
        bucket.attempts.clear()
        bucket.locked_until = 0.0

    def _bucket(self, key: str) -> _AttemptBucket:
        return self._buckets.setdefault(key, _AttemptBucket())

    def _now(self) -> float:
        return float(self._now_fn())

    def _prune(self, bucket: _AttemptBucket, now: float) -> None:
        threshold = now - self.window_seconds
        while bucket.attempts and bucket.attempts[0] < threshold:
            bucket.attempts.popleft()


_ENV_LINE_PATTERN = re.compile(
    r"""^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^#\n]+?))\s*$"""
)


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_PATTERN.match(raw_line)
        if not match:
            continue
        key = match.group(1)
        value = match.group(2)
        if value is None:
            value = match.group(3)
        if value is None:
            value = (match.group(4) or "").strip()
        values[key] = value
    return values


def _parse_bool(value: object, *, default: bool) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(value: object, *, default: int) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return int(text)
    except Exception:
        return default


def resolve_auth_config_path() -> Path:
    configured = os.getenv("NOVEL_WRITER_AUTH_CONFIG", "").strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_AUTH_CONFIG_PATH


def load_auth_settings() -> WebAuthSettings:
    config_path = resolve_auth_config_path()
    file_values = _read_env_file(config_path)

    def resolve(key: str, default: str = "") -> str:
        if key in os.environ:
            return os.environ[key]
        return str(file_values.get(key, default))

    return WebAuthSettings(
        enabled=_parse_bool(resolve("NOVEL_WRITER_AUTH_ENABLED", "0"), default=False),
        username=(resolve("NOVEL_WRITER_AUTH_USERNAME", AuthService.DEFAULT_USERNAME).strip() or AuthService.DEFAULT_USERNAME),
        password=resolve("NOVEL_WRITER_AUTH_PASSWORD", AuthService.DEFAULT_PASSWORD) or AuthService.DEFAULT_PASSWORD,
        secret_key=(resolve("NOVEL_WRITER_AUTH_SECRET_KEY", AuthService.DEFAULT_SECRET_KEY).strip() or AuthService.DEFAULT_SECRET_KEY),
        cookie_name=(resolve("NOVEL_WRITER_AUTH_COOKIE_NAME", DEFAULT_AUTH_COOKIE_NAME).strip() or DEFAULT_AUTH_COOKIE_NAME),
        cookie_secure=_parse_bool(resolve("NOVEL_WRITER_AUTH_COOKIE_SECURE", "0"), default=False),
        session_max_age_seconds=_parse_int(
            resolve("NOVEL_WRITER_AUTH_SESSION_MAX_AGE_SECONDS", "604800"),
            default=604800,
        ),
        login_max_attempts=_parse_int(
            resolve("NOVEL_WRITER_AUTH_LOGIN_MAX_ATTEMPTS", "5"),
            default=5,
        ),
        login_window_seconds=_parse_int(
            resolve("NOVEL_WRITER_AUTH_LOGIN_WINDOW_SECONDS", "300"),
            default=300,
        ),
        login_lockout_seconds=_parse_int(
            resolve("NOVEL_WRITER_AUTH_LOGIN_LOCKOUT_SECONDS", "900"),
            default=900,
        ),
        config_path=str(config_path),
    )
