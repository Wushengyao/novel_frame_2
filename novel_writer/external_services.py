"""External service configuration and clients.

This module keeps deployment-specific endpoints, local model paths, and worker
launch details out of feature code. Local deployments can copy
``external_services.example.json`` to ``external_services.json`` or point
``NOVEL_EXTERNAL_SERVICES_CONFIG`` at another JSON file.
"""

from __future__ import annotations

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
