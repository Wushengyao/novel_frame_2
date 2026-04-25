"""External service configuration and clients.

This module keeps deployment-specific endpoints, local model paths, and worker
launch details out of feature code. Local deployments can copy
``external_services.example.json`` to ``external_services.json`` or point
``NOVEL_EXTERNAL_SERVICES_CONFIG`` at another JSON file.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, parse, request


EXTERNAL_SERVICES_CONFIG_ENV = "NOVEL_EXTERNAL_SERVICES_CONFIG"
DEFAULT_EXTERNAL_SERVICES_CONFIG_PATH = Path(__file__).resolve().with_name("external_services.json")

DEFAULT_WORKFLOW_TEMPLATE_NAME = "image_z_image_turbo (2).json"
DEFAULT_ILLUSTRATION_NEGATIVE_PROMPT = ""
DEFAULT_ILLUSTRATION_STYLE_PRESET = (
    "clean subject separation, layered depth, expressive body language, atmospheric perspective"
)
DEFAULT_COMFYUI_PREFERRED_CHECKPOINTS = (
    "illusious/illustrij_v21.safetensors",
    "illusious/illustrij_v20.safetensors",
    "illusious/illustrij_v19.safetensors",
    "illusious/illustrij_v18.safetensors",
    "illusious/illustrij_v17.safetensors",
    "illusious/prefectIllustriousXL_v70.safetensors",
)

DEFAULT_VOXCPM2_ROOT = "/home/wsy/VoxCPM2"
DEFAULT_VOXCPM2_PYTHON = "/home/wsy/VoxCPM2/.venv/bin/python"
DEFAULT_VOXCPM2_MODEL_ID = "openbmb/VoxCPM2"
DEFAULT_IMAGE_FRAME_API_BASE = "http://127.0.0.1:8010"
DEFAULT_AUDIO_FRAME_API_BASE = "http://127.0.0.1:8808"

DEFAULT_EXTERNAL_SERVICES_CONFIG: dict[str, Any] = {
    "version": 1,
    "comfyui": {
        "api_base": "http://127.0.0.1:8188",
        "root": "",
        "workflow_template": "",
        "checkpoint": "",
        "width": 1280,
        "height": 1280,
        "steps": 8,
        "cfg": 1.0,
        "sampler_name": "res_multistep",
        "scheduler": "simple",
        "timeout": 600,
        "poll_interval": 1.5,
        "negative_prompt": DEFAULT_ILLUSTRATION_NEGATIVE_PROMPT,
        "style_preset": DEFAULT_ILLUSTRATION_STYLE_PRESET,
        "seed": 0,
        "preferred_checkpoints": list(DEFAULT_COMFYUI_PREFERRED_CHECKPOINTS),
    },
    "voxcpm2": {
        "root": DEFAULT_VOXCPM2_ROOT,
        "python": DEFAULT_VOXCPM2_PYTHON,
        "model_id": DEFAULT_VOXCPM2_MODEL_ID,
        "device": "auto",
        "load_denoiser": False,
        "optimize": True,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "normalize": True,
        "denoise": False,
        "silence_ms": 260,
        "timeout_seconds": 0,
    },
    "image_frame": {
        "api_base": DEFAULT_IMAGE_FRAME_API_BASE,
        "provider": "google",
        "model": "",
        "size": "",
        "aspect_ratio": "1:1",
        "google_image_size": "",
        "quality": "",
        "background": "",
        "moderation": "",
        "num_outputs": 1,
        "timeout": 600,
        "poll_interval": 2.0,
        "auth_username": "",
        "auth_password": "",
    },
    "audio_frame": {
        "api_base": DEFAULT_AUDIO_FRAME_API_BASE,
        "timeout": 0,
    },
    "audiobook": {
        "backend": "local_worker",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def external_services_config_path() -> Path:
    configured = str(os.environ.get(EXTERNAL_SERVICES_CONFIG_ENV, "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_EXTERNAL_SERVICES_CONFIG_PATH


def load_external_services_overrides(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else external_services_config_path()
    if not path.exists() or not path.is_file():
        return {}
    return _load_json_file(path)


def load_external_services_config(config_path: str | Path | None = None) -> dict[str, Any]:
    overrides = load_external_services_overrides(config_path)
    return _deep_merge(DEFAULT_EXTERNAL_SERVICES_CONFIG, overrides)


def load_service_config(service_name: str, *, include_defaults: bool = True) -> dict[str, Any]:
    source = load_external_services_config() if include_defaults else load_external_services_overrides()
    service_config = source.get(service_name) or {}
    return deepcopy(service_config) if isinstance(service_config, dict) else {}


def normalize_http_base(value: object, default: str = "http://127.0.0.1:8188") -> str:
    text = str(value or default).strip().rstrip("/")
    if not text:
        text = default.rstrip("/")
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    return text


def coerce_bool(raw_value: object, default: bool = False) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if raw_value is None or raw_value == "":
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def coerce_int(raw_value: object, default: int) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def coerce_float(raw_value: object, default: float) -> float:
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


class ComfyUIClient:
    """Small client for the ComfyUI HTTP API used by image generation."""

    def __init__(self, api_base: str):
        self.api_base = normalize_http_base(api_base)

    def request_json(
        self,
        path_or_url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
        allow_404: bool = False,
    ) -> dict[str, Any]:
        url = self._url(path_or_url)
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if allow_404 and exc.code == 404:
                return {}
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Failed to connect to ComfyUI: {reason}") from exc

    def request_bytes(self, path_or_url: str, *, timeout: int = 60) -> bytes:
        url = self._url(path_or_url)
        try:
            with request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI file download failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Failed to download ComfyUI image: {reason}") from exc

    def queue_prompt(self, workflow: dict[str, Any], *, client_id: str, timeout: int = 60) -> str:
        payload = self.request_json(
            "/prompt",
            payload={"prompt": workflow, "client_id": client_id},
            timeout=timeout,
        )
        prompt_id = str(payload.get("prompt_id", "")).strip()
        if not prompt_id:
            raise RuntimeError(f"ComfyUI 未返回 prompt_id: {payload}")
        return prompt_id

    def wait_for_prompt(self, prompt_id: str, *, timeout: int, poll_interval: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_payload: dict[str, Any] = {}
        while time.time() < deadline:
            payload = self.request_json(
                f"/history/{parse.quote(prompt_id)}",
                timeout=max(10, int(poll_interval * 4)),
                allow_404=True,
            )
            if payload:
                item = payload.get(prompt_id) or {}
                if item:
                    last_payload = item
                    status = item.get("status") or {}
                    status_str = str(status.get("status_str", "")).lower()
                    if item.get("outputs"):
                        return item
                    if status.get("completed") and status_str not in {"error", "execution_error"}:
                        return item
                    if status_str in {"error", "execution_error"}:
                        raise RuntimeError(f"ComfyUI 生成失败: {json.dumps(status, ensure_ascii=False)}")
            time.sleep(poll_interval)
        raise RuntimeError(f"等待 ComfyUI 生成超时。prompt_id={prompt_id}，最后状态={last_payload}")

    def download_image(self, image_info: dict[str, Any], *, timeout: int) -> bytes:
        query = parse.urlencode(
            {
                "filename": image_info.get("filename", ""),
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
        )
        return self.request_bytes(f"/view?{query}", timeout=timeout)

    def _url(self, path_or_url: str) -> str:
        text = str(path_or_url or "").strip()
        if text.startswith(("http://", "https://")):
            return text
        if not text.startswith("/"):
            text = "/" + text
        return self.api_base + text


def normalize_voxcpm2_runtime(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_config if isinstance(raw_config, dict) else {}
    return {
        "voxcpm_root": str(raw.get("voxcpm_root") or raw.get("root") or "").strip(),
        "voxcpm_python": str(raw.get("voxcpm_python") or raw.get("python") or "").strip(),
        "model_id": str(raw.get("model_id") or "").strip(),
        "device": str(raw.get("device") or "auto").strip() or "auto",
        "load_denoiser": coerce_bool(raw.get("load_denoiser"), False),
        "optimize": coerce_bool(raw.get("optimize"), True),
        "cfg_value": coerce_float(raw.get("cfg_value"), 2.0),
        "inference_timesteps": coerce_int(raw.get("inference_timesteps"), 10),
        "normalize": coerce_bool(raw.get("normalize"), True),
        "denoise": coerce_bool(raw.get("denoise"), False),
        "silence_ms": coerce_int(raw.get("silence_ms"), 260),
        "timeout_seconds": coerce_int(raw.get("timeout_seconds"), 0),
    }


def _voxcpm2_env_overrides() -> dict[str, Any]:
    mapping = {
        "root": "NOVEL_VOXCPM2_ROOT",
        "python": "NOVEL_VOXCPM2_PYTHON",
        "model_id": "NOVEL_VOXCPM2_MODEL_ID",
        "device": "NOVEL_VOXCPM2_DEVICE",
        "load_denoiser": "NOVEL_VOXCPM2_LOAD_DENOISER",
        "optimize": "NOVEL_VOXCPM2_OPTIMIZE",
        "cfg_value": "NOVEL_VOXCPM2_CFG_VALUE",
        "inference_timesteps": "NOVEL_VOXCPM2_INFERENCE_TIMESTEPS",
        "normalize": "NOVEL_VOXCPM2_NORMALIZE",
        "denoise": "NOVEL_VOXCPM2_DENOISE",
        "silence_ms": "NOVEL_VOXCPM2_SILENCE_MS",
        "timeout_seconds": "NOVEL_VOXCPM2_TIMEOUT_SECONDS",
    }
    return {key: os.environ[env_name] for key, env_name in mapping.items() if os.environ.get(env_name, "") != ""}


def load_voxcpm2_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = deepcopy(DEFAULT_EXTERNAL_SERVICES_CONFIG["voxcpm2"])
    for layer in (
        load_service_config("voxcpm2", include_defaults=False),
        _voxcpm2_env_overrides(),
        runtime_overrides or {},
    ):
        if isinstance(layer, dict):
            raw.update(layer)
    return normalize_voxcpm2_runtime(raw)


def normalize_image_frame_runtime(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_config if isinstance(raw_config, dict) else {}
    return {
        "api_base": normalize_http_base(raw.get("api_base"), DEFAULT_IMAGE_FRAME_API_BASE),
        "provider": str(raw.get("provider") or "google").strip() or "google",
        "model": str(raw.get("model") or "").strip(),
        "size": str(raw.get("size") or "").strip(),
        "aspect_ratio": str(raw.get("aspect_ratio") or "1:1").strip() or "1:1",
        "google_image_size": str(raw.get("google_image_size") or "").strip(),
        "quality": str(raw.get("quality") or "").strip(),
        "background": str(raw.get("background") or "").strip(),
        "moderation": str(raw.get("moderation") or "").strip(),
        "num_outputs": coerce_int(raw.get("num_outputs"), 1),
        "timeout": coerce_int(raw.get("timeout"), 600),
        "poll_interval": coerce_float(raw.get("poll_interval"), 2.0),
        "auth_username": str(raw.get("auth_username") or "").strip(),
        "auth_password": str(raw.get("auth_password") or "").strip(),
    }


def normalize_audio_frame_runtime(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_config if isinstance(raw_config, dict) else {}
    return {
        "api_base": normalize_http_base(raw.get("api_base"), DEFAULT_AUDIO_FRAME_API_BASE),
        "timeout": coerce_int(raw.get("timeout"), 0),
    }


def _image_frame_env_overrides() -> dict[str, Any]:
    mapping = {
        "api_base": "NOVEL_IMAGE_FRAME_API_BASE",
        "provider": "NOVEL_IMAGE_FRAME_PROVIDER",
        "model": "NOVEL_IMAGE_FRAME_MODEL",
        "size": "NOVEL_IMAGE_FRAME_SIZE",
        "aspect_ratio": "NOVEL_IMAGE_FRAME_ASPECT_RATIO",
        "google_image_size": "NOVEL_IMAGE_FRAME_GOOGLE_IMAGE_SIZE",
        "quality": "NOVEL_IMAGE_FRAME_QUALITY",
        "background": "NOVEL_IMAGE_FRAME_BACKGROUND",
        "moderation": "NOVEL_IMAGE_FRAME_MODERATION",
        "num_outputs": "NOVEL_IMAGE_FRAME_NUM_OUTPUTS",
        "timeout": "NOVEL_IMAGE_FRAME_TIMEOUT",
        "poll_interval": "NOVEL_IMAGE_FRAME_POLL_INTERVAL",
        "auth_username": "NOVEL_IMAGE_FRAME_AUTH_USERNAME",
        "auth_password": "NOVEL_IMAGE_FRAME_AUTH_PASSWORD",
    }
    return {key: os.environ[env_name] for key, env_name in mapping.items() if os.environ.get(env_name, "") != ""}


def _audio_frame_env_overrides() -> dict[str, Any]:
    mapping = {
        "api_base": "NOVEL_AUDIO_FRAME_API_BASE",
        "timeout": "NOVEL_AUDIO_FRAME_TIMEOUT",
    }
    return {key: os.environ[env_name] for key, env_name in mapping.items() if os.environ.get(env_name, "") != ""}


def load_image_frame_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = deepcopy(DEFAULT_EXTERNAL_SERVICES_CONFIG["image_frame"])
    for layer in (
        load_service_config("image_frame", include_defaults=False),
        _image_frame_env_overrides(),
        runtime_overrides or {},
    ):
        if isinstance(layer, dict):
            raw.update(layer)
    return normalize_image_frame_runtime(raw)


def load_audio_frame_runtime(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = deepcopy(DEFAULT_EXTERNAL_SERVICES_CONFIG["audio_frame"])
    for layer in (
        load_service_config("audio_frame", include_defaults=False),
        _audio_frame_env_overrides(),
        runtime_overrides or {},
    ):
        if isinstance(layer, dict):
            raw.update(layer)
    return normalize_audio_frame_runtime(raw)


class JsonHttpClient:
    def __init__(self, api_base: str):
        self.api_base = normalize_http_base(api_base)
        self.cookie = ""

    def request_json(
        self,
        path_or_url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: int = 60,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path_or_url)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = {"Content-Type": "application/json"} if payload is not None else {}
        request_headers.update(headers or {})
        if self.cookie:
            request_headers["Cookie"] = self.cookie
        req = request.Request(url, data=body, headers=request_headers, method="POST" if payload is not None else "GET")
        try:
            with request.urlopen(req, timeout=timeout) as response:
                self._store_cookie(response)
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP request failed with {exc.code}: {detail}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Failed to connect to service: {reason}") from exc

    def request_bytes(self, path_or_url: str, *, timeout: int = 60) -> bytes:
        req = request.Request(self._url(path_or_url), method="GET")
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        try:
            with request.urlopen(req, timeout=timeout) as response:
                self._store_cookie(response)
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"File download failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Failed to download file: {reason}") from exc

    def _store_cookie(self, response) -> None:
        cookie = response.headers.get("Set-Cookie")
        if cookie:
            self.cookie = cookie.split(";", 1)[0]

    def _url(self, path_or_url: str) -> str:
        text = str(path_or_url or "").strip()
        if text.startswith(("http://", "https://")):
            return text
        if not text.startswith("/"):
            text = "/" + text
        return self.api_base + text


class AudioFrameClient(JsonHttpClient):
    def synthesize(
        self,
        *,
        text: str,
        control_instruction: str = "",
        reference_audio: str = "",
        prompt_text: str = "",
        cfg_value: float = 2.0,
        normalize: bool = True,
        denoise: bool = False,
        inference_timesteps: int = 10,
        timeout: int = 0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": text,
            "control_instruction": control_instruction,
            "prompt_text": prompt_text,
            "cfg_value": cfg_value,
            "normalize": normalize,
            "denoise": denoise,
            "inference_timesteps": inference_timesteps,
        }
        if reference_audio:
            path = Path(reference_audio)
            payload["reference_audio_filename"] = path.name
            payload["reference_audio_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
        return self.request_json("/api/tts", payload=payload, timeout=timeout if timeout > 0 else 3600)


class ImageFrameClient(JsonHttpClient):
    def login(self, username: str, password: str, *, timeout: int = 30) -> None:
        if not username and not password:
            return
        form = parse.urlencode({"username": username, "password": password, "next": "/"}).encode("utf-8")
        req = request.Request(
            self._url("/login"),
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            class NoRedirect(request.HTTPRedirectHandler):
                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    return None

            opener = request.build_opener(NoRedirect)
            with opener.open(req, timeout=timeout) as response:
                self._store_cookie(response)
        except error.HTTPError as exc:
            if 300 <= exc.code < 400:
                self._store_cookie(exc)
                if self.cookie:
                    return
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Image Frame login failed with HTTP {exc.code}: {detail}") from exc

    def create_text_to_image_task(self, runtime: dict[str, Any], prompt: str, *, timeout: int = 60) -> dict[str, Any]:
        fields = {
            "provider": runtime["provider"],
            "mode": "text_to_image",
            "prompt": prompt,
            "model": runtime.get("model", ""),
            "size": runtime.get("size", ""),
            "aspect_ratio": runtime.get("aspect_ratio", ""),
            "google_image_size": runtime.get("google_image_size", ""),
            "quality": runtime.get("quality", ""),
            "background": runtime.get("background", ""),
            "moderation": runtime.get("moderation", ""),
            "num_outputs": str(max(1, coerce_int(runtime.get("num_outputs"), 1))),
        }
        return self.request_multipart("/api/tasks", fields=fields, timeout=timeout)

    def request_multipart(self, path_or_url: str, *, fields: dict[str, str], timeout: int = 60) -> dict[str, Any]:
        boundary = "----novel-frame-boundary"
        parts: list[bytes] = []
        for name, value in fields.items():
            if value is None or str(value) == "":
                continue
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode("utf-8")
            )
        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        req = request.Request(
            self._url(path_or_url),
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        if self.cookie:
            req.add_header("Cookie", self.cookie)
        try:
            with request.urlopen(req, timeout=timeout) as response:
                self._store_cookie(response)
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Image Frame request failed with HTTP {exc.code}: {detail}") from exc

    def wait_for_task(self, task_id: str, *, timeout: int, poll_interval: float) -> dict[str, Any]:
        deadline = time.time() + timeout
        last_payload: dict[str, Any] = {}
        while time.time() < deadline:
            payload = self.request_json(f"/api/tasks/{parse.quote(task_id)}", timeout=max(10, int(poll_interval * 4)))
            last_payload = payload
            status = str(payload.get("status", "")).lower()
            if status in {"succeeded", "failed"}:
                return payload
            time.sleep(poll_interval)
        raise RuntimeError(f"Image Frame task timed out: {task_id}, last state: {last_payload}")


class VoxCPM2Service:
    """Launches the isolated VoxCPM2 worker process."""

    def __init__(self, runtime: dict[str, Any]):
        self.runtime = normalize_voxcpm2_runtime(runtime)

    def worker_python(self) -> str:
        configured = str(self.runtime.get("voxcpm_python") or "").strip()
        if configured and Path(configured).exists():
            return configured
        return sys.executable

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        voxcpm_root = str(self.runtime.get("voxcpm_root") or "").strip()
        if voxcpm_root:
            src_path = str(Path(voxcpm_root).expanduser() / "src")
            env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return env

    def run_worker(
        self,
        request_path: str | Path,
        *,
        worker_script_path: str | Path,
        cwd: str | Path,
    ) -> subprocess.CompletedProcess:
        command = [
            self.worker_python(),
            str(worker_script_path),
            "--request",
            str(request_path),
        ]
        timeout_seconds = coerce_int(self.runtime.get("timeout_seconds"), 0)
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=self.build_env(),
            text=True,
            capture_output=True,
            timeout=timeout_seconds if timeout_seconds > 0 else None,
            check=False,
        )
