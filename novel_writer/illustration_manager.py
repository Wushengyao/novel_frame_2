"""ComfyUI-powered illustration helpers for novel chapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import os
import random
import re
from pathlib import Path
from threading import Lock
from typing import Any

from common_utils import emit_progress, extract_json_object, safe_int, utc_now
from external_services import (
    ComfyUIClient,
    DEFAULT_COMFYUI_PREFERRED_CHECKPOINTS,
    DEFAULT_ILLUSTRATION_NEGATIVE_PROMPT,
    DEFAULT_ILLUSTRATION_STYLE_PRESET,
    DEFAULT_WORKFLOW_TEMPLATE_NAME,
    ImageFrameClient,
    load_image_frame_runtime,
    load_service_config,
    normalize_http_base,
)
from llm_client import generate_text_with_metadata
from project_manager import load_json, load_project, save_json, update_project_stats
from prompt_builder import build_illustration_prompt, build_system_prompt


LEGACY_DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, bad anatomy, extra fingers, malformed hands, malformed face, "
    "deformed body, duplicate, multiple views, split panels, comic page, text, watermark, logo, caption, "
    "jpeg artifacts, cropped, out of frame"
)
DEFAULT_NEGATIVE_PROMPT = DEFAULT_ILLUSTRATION_NEGATIVE_PROMPT
LEGACY_DEFAULT_STYLE_PRESET = (
    "masterpiece, best quality, detailed light novel illustration, cinematic composition, expressive characters, "
    "rich environmental storytelling, dramatic winter atmosphere, soft volumetric lighting"
)
DEFAULT_STYLE_PRESET = DEFAULT_ILLUSTRATION_STYLE_PRESET
PREFERRED_CHECKPOINTS = DEFAULT_COMFYUI_PREFERRED_CHECKPOINTS

PROJECT_STATS_LOCK = Lock()

def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _update_project_stats_threadsafe(
    project_path: str,
    phase: str,
    success: bool,
    usage: dict | None = None,
    metadata: dict | None = None,
    chapter_slug: str = "",
) -> None:
    with PROJECT_STATS_LOCK:
        update_project_stats(
            project_path,
            phase=phase,
            success=success,
            usage=usage,
            metadata=metadata,
            chapter_slug=chapter_slug,
        )


def _coerce_worker_count(requested_workers: Any, task_count: int) -> int:
    if task_count <= 1:
        return 1
    workers = safe_int(requested_workers, 0)
    if workers <= 0:
        workers = min(4, task_count)
    return max(1, min(task_count, workers))


def _normalize_checkpoint_name(name: str) -> str:
    normalized = str(name or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    return normalized


def _trim_text(text: str, limit: int = 2600) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    half = max(400, limit // 2)
    return f"{clean[:half].rstrip()}\n...\n{clean[-half:].lstrip()}"


def _compact_scene_summary(text: str, *, limit: int = 56) -> str:
    raw = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if not raw:
        return ""
    pieces = re.split(r"[。！？!?；;，,]+", raw)
    selected = []
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip(" ,，。；：、")
        if clean:
            selected.append(clean)
        if len(selected) >= 1:
            break
    summary = "；".join(selected) if selected else raw
    if len(summary) > limit:
        summary = summary[:limit].rstrip(" ,，。；：、")
    return summary


def _split_paragraphs(text: str) -> list[str]:
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    return paragraphs


def _character_name_variants(character: dict) -> list[str]:
    name = str(character.get("name", "")).strip()
    if not name:
        return []
    variants = [name]
    if len(name) >= 2:
        variants.append(name[-2:])
    return list(dict.fromkeys(item for item in variants if item))


def _count_character_mentions(text: str, character: dict) -> int:
    haystack = str(text or "")
    total = 0
    for variant in _character_name_variants(character):
        total += haystack.count(variant)
    return total


def _select_scene_window(chapter_text: str, characters: list[dict], *, window_size: int = 1) -> str:
    paragraphs = _split_paragraphs(chapter_text)
    if not paragraphs:
        return str(chapter_text or "").strip()
    if len(paragraphs) <= window_size:
        return "\n\n".join(paragraphs)

    best_text = "\n\n".join(paragraphs[-window_size:])
    best_score = -1.0
    for start in range(len(paragraphs)):
        window = paragraphs[start : start + window_size]
        if not window:
            continue
        window_text = "\n\n".join(window)
        mention_counts = [_count_character_mentions(window_text, character) for character in characters]
        total_mentions = sum(mention_counts)
        unique_mentions = sum(1 for count in mention_counts if count > 0)
        narrative_weight = min(len(window_text), 500) / 500.0
        recency_weight = start / max(1, len(paragraphs) - 1)
        score = total_mentions * 10 + unique_mentions * 4 + narrative_weight + recency_weight
        if score > best_score:
            best_score = score
            best_text = window_text
    return best_text


def _select_scene_characters(scene_text: str, chapter_text: str, characters: list[dict], *, limit: int = 3) -> list[dict]:
    if not characters:
        return []

    scored = []
    for index, character in enumerate(characters):
        scene_mentions = _count_character_mentions(scene_text, character)
        chapter_mentions = _count_character_mentions(chapter_text, character)
        score = scene_mentions * 100 + chapter_mentions * 5 - index
        scored.append((score, scene_mentions, chapter_mentions, character))

    present = [item for item in scored if item[1] > 0]
    if not present:
        present = [item for item in scored if item[2] > 0]
    if not present:
        present = scored[:1]

    present.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [item[3] for item in present[:limit]]


def _prompt_fragments(text: str, *, max_parts: int = 6, part_limit: int = 36) -> list[str]:
    raw = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if not raw:
        return []

    pieces = re.split(r"[，,。；;：:\|/]+", raw)
    results: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        clean = re.sub(r"\s+", " ", piece).strip(" .,!?:;，。；：、")
        if not clean:
            continue
        if len(clean) > part_limit:
            clean = clean[:part_limit].rstrip()
        lowered = clean.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        results.append(clean)
        if len(results) >= max_parts:
            break
    return results


def _merge_prompt_parts(*groups: list[str] | tuple[str, ...]) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            clean = str(item or "").strip().strip(",")
            if not clean:
                continue
            lowered = clean.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            merged.append(clean)
    return ", ".join(merged)


def _normalize_prompt_text(text: str, *, limit: int = 220) -> str:
    clean = re.sub(r"\s+", " ", str(text or "").replace("\r", " ").replace("\n", " ")).strip(" ，。；;,.!")
    if len(clean) > limit:
        truncated = clean[:limit]
        cut_candidates = [truncated.rfind(sep) for sep in ("。", "；", "，", ",", " ")]
        cut_at = max(cut_candidates)
        if cut_at >= max(24, limit // 3):
            truncated = truncated[:cut_at]
        clean = truncated.rstrip(" ，。；;,.!")
    return clean


def _character_prompt_sentence(character: dict, *, action: str = "", expression: str = "", outfit: str = "") -> str:
    name = _normalize_prompt_text(character.get("name", ""), limit=24)
    appearance = _normalize_prompt_text(character.get("appearance", ""), limit=220)
    outfit_text = _normalize_prompt_text(outfit, limit=140)
    action_text = _normalize_prompt_text(action, limit=140)
    expression_text = _normalize_prompt_text(expression, limit=100)

    clauses = []
    if name:
        clauses.append(name)
    if appearance:
        clauses.append(appearance)
    if expression_text:
        clauses.append(f"表情为{expression_text}")
    if action_text:
        clauses.append(f"动作/姿态为{action_text}")
    if outfit_text:
        clauses.append(f"穿着{outfit_text}")
    return "，".join(item for item in clauses if item)


def _compose_structured_positive_prompt(
    *,
    runtime_config: dict,
    scene_summary: str,
    characters: list[dict] | None = None,
    environment: str = "",
    composition: str = "",
    lighting: str = "",
    extra_parts: list[str] | None = None,
) -> str:
    style_text = _normalize_prompt_text(runtime_config.get("style_preset", DEFAULT_STYLE_PRESET), limit=180)
    composition_text = _normalize_prompt_text(composition, limit=180)
    lighting_text = _normalize_prompt_text(lighting, limit=180)
    environment_text = _normalize_prompt_text(environment, limit=280)
    scene_text = _normalize_prompt_text(_compact_scene_summary(scene_summary, limit=72), limit=72)

    character_parts: list[str] = []
    for character in characters or []:
        if isinstance(character, dict):
            sentence = _character_prompt_sentence(
                character,
                action=str(character.get("action", "") or ""),
                expression=str(character.get("expression", "") or ""),
                outfit=str(character.get("outfit", "") or ""),
            )
            if sentence:
                character_parts.append(sentence)

    long_detail_parts: list[str] = []
    short_detail_parts: list[str] = []
    for item in extra_parts or []:
        raw_text = _normalize_prompt_text(item, limit=600)
        if not raw_text:
            continue
        if len(raw_text) > 80:
            long_detail_parts.append(raw_text)
        else:
            short_detail_parts.append(raw_text)

    sections = []
    if style_text:
        sections.append(style_text)
    if scene_text:
        sections.append(f"场景主题：{scene_text}")
    if character_parts:
        sections.append("人物描写：" + "；".join(character_parts))
    if composition_text:
        sections.append(f"构图与镜头：{composition_text}")
    if lighting_text:
        sections.append(f"光线氛围：{lighting_text}")
    if environment_text:
        sections.append(f"环境描写：{environment_text}")
    if long_detail_parts:
        sections.append("核心画面描述：" + "；".join(long_detail_parts))
    if short_detail_parts:
        sections.append("补充细节：" + "，".join(short_detail_parts))
    sections.append("画面需要清晰呈现人物皮肤质感、衣物褶皱、材质纹理、空间层次与环境细节，整体适合作为高质量文生图提示词")
    return "。".join(section for section in sections if section)


def _resolve_candidate_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.name.lower() != "comfyui" and (candidate / "ComfyUI").exists():
        candidate = candidate / "ComfyUI"
    if candidate.exists():
        return candidate.resolve()
    return None


def _candidate_comfyui_roots(service_config: dict | None = None) -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    workspace_root = base_dir.parent.parent
    candidates: list[Path] = []
    seen: set[str] = set()
    service_config = service_config or {}

    for raw in (
        os.environ.get("NOVEL_COMFYUI_ROOT", ""),
        str(service_config.get("root", "") or ""),
        str(workspace_root / "ComfyUI_cu128_50XX" / "ComfyUI"),
    ):
        path = _resolve_candidate_path(raw)
        if path is not None and str(path) not in seen:
            seen.add(str(path))
            candidates.append(path)

    try:
        for item in workspace_root.iterdir():
            if not item.is_dir() or not item.name.lower().startswith("comfyui"):
                continue
            path = _resolve_candidate_path(str(item))
            if path is not None and str(path) not in seen:
                seen.add(str(path))
                candidates.append(path)
    except OSError:
        pass

    return candidates


def _resolve_comfyui_root(saved_config: dict, overrides: dict, service_overrides: dict, service_defaults: dict) -> Path | None:
    for raw in (
        overrides.get("comfyui_root", ""),
        os.environ.get("NOVEL_COMFYUI_ROOT", ""),
        service_overrides.get("root", ""),
        saved_config.get("comfyui_root", ""),
        service_defaults.get("root", ""),
    ):
        path = _resolve_candidate_path(str(raw))
        if path is not None:
            return path
    for candidate in _candidate_comfyui_roots(service_defaults):
        return candidate
    return None


def _resolve_workflow_template_path(
    saved_config: dict,
    overrides: dict,
    service_overrides: dict,
    comfyui_root: Path | None,
) -> str:
    candidates = [
        str(overrides.get("workflow_template", "") or "").strip(),
        str(os.environ.get("NOVEL_COMFYUI_WORKFLOW_TEMPLATE", "") or "").strip(),
        str(service_overrides.get("workflow_template", "") or "").strip(),
    ]

    if comfyui_root is not None:
        candidates.append(str(comfyui_root.parent / "workflow" / DEFAULT_WORKFLOW_TEMPLATE_NAME))

    workspace_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(str(workspace_root / "ComfyUI_cu128_50XX" / "workflow" / DEFAULT_WORKFLOW_TEMPLATE_NAME))
    candidates.append(str(saved_config.get("workflow_template", "") or "").strip())

    for raw in candidates:
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
    return ""


def _relative_checkpoint_name(checkpoint_path: Path, checkpoints_dir: Path) -> str:
    relative = checkpoint_path.resolve().relative_to(checkpoints_dir.resolve())
    return _normalize_checkpoint_name(str(relative))


def _preferred_checkpoints(service_config: dict | None = None) -> tuple[str, ...]:
    raw_items = (service_config or {}).get("preferred_checkpoints")
    if not isinstance(raw_items, list):
        return tuple(PREFERRED_CHECKPOINTS)
    items = [str(item or "").strip() for item in raw_items]
    return tuple(item for item in items if item) or tuple(PREFERRED_CHECKPOINTS)


def _find_default_checkpoint(comfyui_root: Path | None, service_config: dict | None = None) -> str:
    if comfyui_root is None:
        return ""
    checkpoints_dir = comfyui_root / "models" / "checkpoints"
    if not checkpoints_dir.exists():
        return ""

    for name in _preferred_checkpoints(service_config):
        path = checkpoints_dir / Path(name)
        if path.exists():
            return _normalize_checkpoint_name(name)

    for pattern in ("*.safetensors", "*.ckpt"):
        for path in sorted(checkpoints_dir.rglob(pattern)):
            if path.is_file():
                return _relative_checkpoint_name(path, checkpoints_dir)
    return ""


def _resolve_checkpoint(
    saved_config: dict,
    overrides: dict,
    service_overrides: dict,
    service_defaults: dict,
    comfyui_root: Path | None,
) -> str:
    raw_values = (
        str(overrides.get("checkpoint", "") or "").strip(),
        str(os.environ.get("NOVEL_COMFYUI_CHECKPOINT", "") or "").strip(),
        str(service_overrides.get("checkpoint", "") or "").strip(),
        str(saved_config.get("checkpoint", "") or "").strip(),
    )
    checkpoints_dir = (comfyui_root / "models" / "checkpoints") if comfyui_root else None

    for raw in raw_values:
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute() and checkpoints_dir and candidate.exists():
            return _relative_checkpoint_name(candidate, checkpoints_dir)
        return _normalize_checkpoint_name(raw)

    return _find_default_checkpoint(comfyui_root, service_defaults)


def _extract_workflow_template_defaults(template_path: str) -> dict[str, Any]:
    if not template_path:
        return {}
    try:
        workflow = load_json(template_path)
    except Exception:
        return {}

    defaults: dict[str, Any] = {}
    latent_node = _find_first_node_by_class(workflow, "EmptySD3LatentImage") or _find_first_node_by_class(workflow, "EmptyLatentImage")
    if latent_node is not None:
        _, node = latent_node
        inputs = node.get("inputs") or {}
        defaults["width"] = inputs.get("width")
        defaults["height"] = inputs.get("height")

    sampler_node = _find_first_node_by_class(workflow, "KSampler")
    if sampler_node is not None:
        _, node = sampler_node
        inputs = node.get("inputs") or {}
        defaults["steps"] = inputs.get("steps")
        defaults["cfg"] = inputs.get("cfg")
        defaults["sampler_name"] = inputs.get("sampler_name")
        defaults["scheduler"] = inputs.get("scheduler")
    return defaults


def _build_runtime_config(project_path: str, overrides: dict | None = None) -> dict:
    project = load_json(str(Path(project_path) / "project.json"))
    saved = project.get("illustration_config") or {}
    merged_overrides = overrides or {}
    backend = str(
        merged_overrides.get("backend")
        or os.environ.get("NOVEL_ILLUSTRATION_BACKEND")
        or saved.get("backend")
        or "image_frame"
    ).strip().lower()
    if backend not in {"image_frame", "comfyui"}:
        backend = "image_frame"
    image_frame_config = {
        "api_base": saved.get("image_frame_api_base"),
        "provider": saved.get("image_frame_provider"),
        "model": saved.get("image_frame_model"),
        "size": saved.get("image_frame_size"),
        "aspect_ratio": saved.get("image_frame_aspect_ratio"),
        "google_image_size": saved.get("image_frame_google_image_size"),
        "quality": saved.get("image_frame_quality"),
        "background": saved.get("image_frame_background"),
        "moderation": saved.get("image_frame_moderation"),
        "num_outputs": saved.get("image_frame_num_outputs"),
        "timeout": saved.get("image_frame_timeout"),
        "poll_interval": saved.get("image_frame_poll_interval"),
        "auth_username": saved.get("image_frame_auth_username"),
    }
    image_frame_config.update(
        {
            key: value
            for key, value in {
                "api_base": merged_overrides.get("image_frame_api_base"),
                "provider": merged_overrides.get("image_frame_provider"),
                "model": merged_overrides.get("image_frame_model"),
                "size": merged_overrides.get("image_frame_size"),
                "aspect_ratio": merged_overrides.get("image_frame_aspect_ratio"),
                "google_image_size": merged_overrides.get("image_frame_google_image_size"),
                "quality": merged_overrides.get("image_frame_quality"),
                "background": merged_overrides.get("image_frame_background"),
                "moderation": merged_overrides.get("image_frame_moderation"),
                "num_outputs": merged_overrides.get("image_frame_num_outputs"),
                "timeout": merged_overrides.get("image_frame_timeout"),
                "poll_interval": merged_overrides.get("image_frame_poll_interval"),
                "auth_username": merged_overrides.get("image_frame_auth_username"),
                "auth_password": merged_overrides.get("image_frame_auth_password"),
            }.items()
            if value not in (None, "")
        }
    )
    image_runtime = load_image_frame_runtime(
        {key: value for key, value in image_frame_config.items() if value not in (None, "")}
    )
    service_defaults = load_service_config("comfyui", include_defaults=True)
    service_overrides = load_service_config("comfyui", include_defaults=False)
    comfyui_root = _resolve_comfyui_root(saved, merged_overrides, service_overrides, service_defaults)
    workflow_template = _resolve_workflow_template_path(saved, merged_overrides, service_overrides, comfyui_root)
    workflow_defaults = _extract_workflow_template_defaults(workflow_template)
    saved_workflow_template = str(saved.get("workflow_template", "") or "").strip()
    saved_matches_template = bool(workflow_template and saved_workflow_template == workflow_template)
    checkpoint = _resolve_checkpoint(saved, merged_overrides, service_overrides, service_defaults, comfyui_root)
    saved_negative_prompt = str(saved.get("negative_prompt", "") or "").strip()
    if saved_negative_prompt == LEGACY_DEFAULT_NEGATIVE_PROMPT:
        saved_negative_prompt = ""
    saved_style_preset = str(saved.get("style_preset", "") or "").strip()
    if saved_style_preset == LEGACY_DEFAULT_STYLE_PRESET:
        saved_style_preset = ""

    config = {
        "backend": backend,
        "image_frame_api_base": image_runtime["api_base"],
        "image_frame_provider": image_runtime["provider"],
        "image_frame_model": image_runtime["model"],
        "image_frame_size": image_runtime["size"],
        "image_frame_aspect_ratio": image_runtime["aspect_ratio"],
        "image_frame_google_image_size": image_runtime["google_image_size"],
        "image_frame_quality": image_runtime["quality"],
        "image_frame_background": image_runtime["background"],
        "image_frame_moderation": image_runtime["moderation"],
        "image_frame_num_outputs": image_runtime["num_outputs"],
        "image_frame_timeout": image_runtime["timeout"],
        "image_frame_poll_interval": image_runtime["poll_interval"],
        "image_frame_auth_username": image_runtime["auth_username"],
        "image_frame_auth_password": image_runtime["auth_password"],
        "comfyui_api_base": normalize_http_base(
            str(
                merged_overrides.get("comfyui_api_base")
                or os.environ.get("NOVEL_COMFYUI_API_BASE")
                or service_overrides.get("api_base")
                or saved.get("comfyui_api_base")
                or service_defaults.get("api_base")
            )
        ),
        "comfyui_root": str(comfyui_root) if comfyui_root else "",
        "workflow_template": workflow_template,
        "checkpoint": checkpoint,
        "width": safe_int(
            merged_overrides.get("width")
            or os.environ.get("NOVEL_COMFYUI_WIDTH")
            or service_overrides.get("width")
            or (saved.get("width") if saved_matches_template else workflow_defaults.get("width")),
            safe_int(service_defaults.get("width"), 1280),
        ),
        "height": safe_int(
            merged_overrides.get("height")
            or os.environ.get("NOVEL_COMFYUI_HEIGHT")
            or service_overrides.get("height")
            or (saved.get("height") if saved_matches_template else workflow_defaults.get("height")),
            safe_int(service_defaults.get("height"), 1280),
        ),
        "steps": safe_int(
            merged_overrides.get("steps")
            or os.environ.get("NOVEL_COMFYUI_STEPS")
            or service_overrides.get("steps")
            or (saved.get("steps") if saved_matches_template else workflow_defaults.get("steps")),
            safe_int(service_defaults.get("steps"), 8),
        ),
        "cfg": _safe_float(
            merged_overrides.get("cfg")
            or os.environ.get("NOVEL_COMFYUI_CFG")
            or service_overrides.get("cfg")
            or (saved.get("cfg") if saved_matches_template else workflow_defaults.get("cfg")),
            _safe_float(service_defaults.get("cfg"), 1.0),
        ),
        "sampler_name": str(
            merged_overrides.get("sampler_name")
            or os.environ.get("NOVEL_COMFYUI_SAMPLER")
            or service_overrides.get("sampler_name")
            or (saved.get("sampler_name") if saved_matches_template else workflow_defaults.get("sampler_name"))
            or service_defaults.get("sampler_name")
            or "euler"
        ).strip(),
        "scheduler": str(
            merged_overrides.get("scheduler")
            or os.environ.get("NOVEL_COMFYUI_SCHEDULER")
            or service_overrides.get("scheduler")
            or (saved.get("scheduler") if saved_matches_template else workflow_defaults.get("scheduler"))
            or service_defaults.get("scheduler")
            or "normal"
        ).strip(),
        "timeout": safe_int(
            merged_overrides.get("timeout")
            or os.environ.get("NOVEL_COMFYUI_TIMEOUT")
            or service_overrides.get("timeout")
            or saved.get("timeout"),
            safe_int(service_defaults.get("timeout"), 600),
        ),
        "poll_interval": _safe_float(
            merged_overrides.get("poll_interval")
            or os.environ.get("NOVEL_COMFYUI_POLL_INTERVAL")
            or service_overrides.get("poll_interval")
            or saved.get("poll_interval"),
            _safe_float(service_defaults.get("poll_interval"), 1.5),
        ),
        "negative_prompt": str(
            merged_overrides.get("negative_prompt")
            or os.environ.get("NOVEL_COMFYUI_NEGATIVE_PROMPT")
            or service_overrides.get("negative_prompt")
            or saved_negative_prompt
            or service_defaults.get("negative_prompt")
            or DEFAULT_NEGATIVE_PROMPT
        ).strip(),
        "style_preset": str(
            merged_overrides.get("style_preset")
            or os.environ.get("NOVEL_COMFYUI_STYLE_PRESET")
            or service_overrides.get("style_preset")
            or saved_style_preset
            or service_defaults.get("style_preset")
            or DEFAULT_STYLE_PRESET
        ).strip(),
        "seed": safe_int(
            merged_overrides.get("seed")
            or os.environ.get("NOVEL_COMFYUI_SEED")
            or service_overrides.get("seed")
            or saved.get("seed"),
            safe_int(service_defaults.get("seed"), 0),
        ),
    }

    if backend == "comfyui" and not config["workflow_template"] and not config["checkpoint"]:
        raise RuntimeError(
            "未找到可用的 ComfyUI 工作流模板或 checkpoint。请确认 external_services.json 中的 comfyui.workflow_template / comfyui.checkpoint，或设置 NOVEL_COMFYUI_WORKFLOW_TEMPLATE。"
        )
    return config


def _persist_runtime_config(project_path: str, runtime_config: dict) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    project_data["illustration_config"] = {
        "backend": runtime_config.get("backend", "image_frame"),
        "image_frame_api_base": runtime_config.get("image_frame_api_base", ""),
        "image_frame_provider": runtime_config.get("image_frame_provider", ""),
        "image_frame_model": runtime_config.get("image_frame_model", ""),
        "image_frame_size": runtime_config.get("image_frame_size", ""),
        "image_frame_aspect_ratio": runtime_config.get("image_frame_aspect_ratio", ""),
        "image_frame_google_image_size": runtime_config.get("image_frame_google_image_size", ""),
        "image_frame_quality": runtime_config.get("image_frame_quality", ""),
        "image_frame_background": runtime_config.get("image_frame_background", ""),
        "image_frame_moderation": runtime_config.get("image_frame_moderation", ""),
        "image_frame_num_outputs": int(runtime_config.get("image_frame_num_outputs", 1)),
        "image_frame_timeout": int(runtime_config.get("image_frame_timeout", 600)),
        "image_frame_poll_interval": float(runtime_config.get("image_frame_poll_interval", 2.0)),
        "image_frame_auth_username": runtime_config.get("image_frame_auth_username", ""),
        "comfyui_api_base": runtime_config.get("comfyui_api_base", ""),
        "comfyui_root": runtime_config.get("comfyui_root", ""),
        "workflow_template": runtime_config.get("workflow_template", ""),
        "checkpoint": runtime_config.get("checkpoint", ""),
        "width": int(runtime_config.get("width", 832)),
        "height": int(runtime_config.get("height", 1216)),
        "steps": int(runtime_config.get("steps", 28)),
        "cfg": float(runtime_config.get("cfg", 6.5)),
        "sampler_name": runtime_config.get("sampler_name", "euler"),
        "scheduler": runtime_config.get("scheduler", "normal"),
        "timeout": int(runtime_config.get("timeout", 600)),
        "poll_interval": float(runtime_config.get("poll_interval", 1.5)),
        "negative_prompt": runtime_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "style_preset": runtime_config.get("style_preset", DEFAULT_STYLE_PRESET),
        "seed": int(runtime_config.get("seed", 0) or 0),
    }
    project_data["updated_at"] = utc_now()
    save_json(str(project_file), project_data)


def _resolve_chapter_file(project_path: str, chapter_ref: str | None) -> Path:
    chapters_dir = Path(project_path) / "chapters"
    if not chapters_dir.exists():
        raise RuntimeError("项目中还没有章节，无法生成插图。")

    normalized = (chapter_ref or "latest").strip()
    if not normalized or normalized == "latest":
        chapters = sorted(chapters_dir.glob("chapter_*.md"))
        if not chapters:
            raise RuntimeError("项目中还没有章节，无法生成插图。")
        return chapters[-1]

    candidate = Path(normalized)
    if candidate.exists():
        return candidate.resolve()

    if not normalized.endswith(".md"):
        normalized += ".md"
    chapter_file = chapters_dir / normalized
    if chapter_file.exists():
        return chapter_file.resolve()
    raise RuntimeError(f"找不到章节文件: {chapter_ref}")


def _default_prompt_payload(project_data: dict, chapter_text: str, runtime_config: dict, user_request: str) -> dict:
    project = project_data.get("project", {})
    world = project_data.get("world", {})
    plot_state = project_data.get("plot_state", {})
    style = project_data.get("style", {})
    protagonists = (project_data.get("characters", {}) or {}).get("protagonists") or []
    scene_text = _select_scene_window(chapter_text, protagonists)
    present_characters = _select_scene_characters(scene_text, chapter_text, protagonists, limit=3)
    location_text = str(plot_state.get("current_location", "")).strip() or str(world.get("setting", "")).strip()
    characters_payload = []
    for character in present_characters:
        characters_payload.append(
            {
                "name": str(character.get("name", "")).strip(),
                "appearance": str(character.get("appearance", "")).strip(),
                "outfit": "符合角色设定的冬季生存穿搭，保暖且保留角色个人风格",
                "action": "围绕当前章节关键事件展开自然互动与动作表现",
                "expression": "符合当前剧情氛围的自然表情，情绪清晰可读",
            }
        )

    environment_text = ", ".join(
        item
        for item in (
            location_text,
            _compact_scene_summary(scene_text, limit=96),
            "前景、中景、远景都要有明确可视化环境信息，带出生活痕迹、道具与空间层次",
        )
        if item
    )
    scene_summary = _compact_scene_summary(scene_text, limit=48)
    composition_text = "中景或中近景构图，明确主体与陪体关系，镜头高度自然，画面重心清晰，人物动作完整可读"
    lighting_text = "根据场景使用自然光或室内暖光，强调冷暖对比、体积光、皮肤与衣物的真实质感"

    positive_prompt = _compose_structured_positive_prompt(
        runtime_config=runtime_config,
        scene_summary=scene_summary,
        characters=characters_payload,
        environment=environment_text,
        composition=composition_text,
        lighting=lighting_text,
        extra_parts=[
            str(project.get("name", "novel illustration")).strip(),
            str(world.get("genre", "")).strip(),
            _compact_scene_summary(str(style.get("tone", "")).strip(), limit=36),
            _compact_scene_summary(str(user_request or "").strip(), limit=40),
        ],
    )
    return {
        "scene_summary": scene_summary,
        "positive_prompt": positive_prompt,
        "negative_prompt": runtime_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "prompt_source": "fallback",
    }


def _generate_prompt_payload(
    project_path: str,
    chapter_text: str,
    llm_config: dict | None,
    runtime_config: dict,
    user_request: str = "",
    chapter_slug: str = "",
    progress_callback=None,
) -> dict:
    project_data = load_project(project_path)
    provider = str((llm_config or {}).get("model_provider", "") or "").strip().lower()
    has_remote_credentials = bool((llm_config or {}).get("api_key"))
    has_local_llm = provider in {"ollama", "llama_cpp"} and bool((llm_config or {}).get("api_base")) and bool(
        (llm_config or {}).get("model") or (llm_config or {}).get("model_name")
    )
    if llm_config and llm_config.get("model_provider") and (has_remote_credentials or has_local_llm):
        prompt = build_illustration_prompt(project_data, chapter_text, user_request=user_request)
        try:
            emit_progress(progress_callback, "illustration_prompt", "正在生成插图提示词")
            response_text, metadata = generate_text_with_metadata(
                prompt,
                llm_config,
                log_context={"phase": "illustration_prompt", "chapter_slug": chapter_slug},
                system_prompt=build_system_prompt("illustration"),
                response_format="json",
            )
            _update_project_stats_threadsafe(
                project_path,
                phase="illustration_prompt",
                success=True,
                usage=metadata.get("usage"),
                metadata=metadata,
                chapter_slug=chapter_slug,
            )
            payload = extract_json_object(response_text, "Could not parse JSON from illustration prompt response.")
            llm_positive_prompt = _normalize_prompt_text(str(payload.get("positive_prompt", "") or ""), limit=1200)
            positive_prompt = _compose_structured_positive_prompt(
                runtime_config=runtime_config,
                scene_summary=str(payload.get("scene_summary", "") or "").strip() or _compact_scene_summary(chapter_text, limit=48),
                characters=payload.get("characters") if isinstance(payload.get("characters"), list) else None,
                environment=str(payload.get("environment", "") or "").strip(),
                composition=str(payload.get("composition", "") or "").strip(),
                lighting=str(payload.get("lighting", "") or "").strip(),
                extra_parts=[llm_positive_prompt] if llm_positive_prompt else None,
            ).strip()
            if positive_prompt:
                return {
                    "scene_summary": _compact_scene_summary(str(payload.get("scene_summary", "")).strip() or chapter_text, limit=48),
                    "positive_prompt": positive_prompt,
                    "negative_prompt": str(payload.get("negative_prompt", "")).strip()
                    or runtime_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
                    "prompt_source": "llm",
                }
        except Exception:
            _update_project_stats_threadsafe(
                project_path,
                phase="illustration_prompt",
                success=False,
                usage=None,
                chapter_slug=chapter_slug,
            )

    emit_progress(progress_callback, "illustration_prompt_fallback", "插图提示词回退到本地规则生成")
    return _default_prompt_payload(project_data, chapter_text, runtime_config, user_request)


def _build_workflow(
    *,
    checkpoint: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    filename_prefix: str,
) -> dict[str, Any]:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": int(width),
                "height": int(height),
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": positive_prompt,
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": negative_prompt,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": filename_prefix,
                "images": ["8", 0],
            },
        },
    }


def _find_first_node_by_class(workflow: dict[str, Any], class_type: str) -> tuple[str, dict[str, Any]] | None:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return str(node_id), node
    return None


def _set_first_node_input(
    workflow: dict[str, Any],
    class_type: str,
    input_name: str,
    value: Any,
    *,
    required: bool = True,
) -> None:
    match = _find_first_node_by_class(workflow, class_type)
    if match is None:
        if required:
            raise RuntimeError(f"ComfyUI workflow missing required node: {class_type}")
        return
    _, node = match
    inputs = node.setdefault("inputs", {})
    inputs[input_name] = value


def _build_workflow_from_template(
    *,
    template_path: str,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    filename_prefix: str,
    checkpoint: str = "",
) -> dict[str, Any]:
    workflow = deepcopy(load_json(template_path))
    _set_first_node_input(workflow, "CLIPTextEncode", "text", positive_prompt)
    _set_first_node_input(workflow, "KSampler", "seed", int(seed))
    _set_first_node_input(workflow, "KSampler", "steps", int(steps))
    _set_first_node_input(workflow, "KSampler", "cfg", float(cfg))
    _set_first_node_input(workflow, "KSampler", "sampler_name", sampler_name)
    _set_first_node_input(workflow, "KSampler", "scheduler", scheduler)
    _set_first_node_input(workflow, "SaveImage", "filename_prefix", filename_prefix)

    if _find_first_node_by_class(workflow, "EmptySD3LatentImage") is not None:
        _set_first_node_input(workflow, "EmptySD3LatentImage", "width", int(width))
        _set_first_node_input(workflow, "EmptySD3LatentImage", "height", int(height))
        _set_first_node_input(workflow, "EmptySD3LatentImage", "batch_size", 1, required=False)
    else:
        _set_first_node_input(workflow, "EmptyLatentImage", "width", int(width), required=False)
        _set_first_node_input(workflow, "EmptyLatentImage", "height", int(height), required=False)
        _set_first_node_input(workflow, "EmptyLatentImage", "batch_size", 1, required=False)

    if _find_first_node_by_class(workflow, "CheckpointLoaderSimple") is not None:
        _set_first_node_input(workflow, "CheckpointLoaderSimple", "ckpt_name", checkpoint, required=False)
    return workflow


def _collect_output_images(history_item: dict) -> list[dict]:
    images: list[dict] = []
    for node_output in (history_item.get("outputs") or {}).values():
        for image in node_output.get("images", []) or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(image)
    return images


def _chapter_record_dir(project_path: str, chapter_slug: str) -> Path:
    return Path(project_path) / "illustrations" / chapter_slug


def _asset_record_dir(project_path: str, *parts: str) -> Path:
    return Path(project_path) / "illustrations" / Path(*parts)


def _load_existing_record(project_path: str, metadata_path: Path) -> dict | None:
    if not metadata_path.exists():
        return None
    try:
        existing = load_json(str(metadata_path))
    except Exception:
        return None

    images = existing.get("images") or []
    if images and all((Path(project_path) / image.get("relative_path", "")).exists() for image in images):
        existing["reused"] = True
        return existing
    return None


def _render_illustration_images(
    project_path: str,
    *,
    asset_slug: str,
    record_dir: Path,
    prompt_payload: dict,
    runtime_config: dict,
    progress_callback=None,
) -> tuple[str, int, list[dict]]:
    if str(runtime_config.get("backend") or "").lower() == "image_frame":
        return _render_image_frame_images(
            project_path,
            asset_slug=asset_slug,
            record_dir=record_dir,
            prompt_payload=prompt_payload,
            runtime_config=runtime_config,
            progress_callback=progress_callback,
        )

    seed = int(runtime_config.get("seed") or 0) or random.randint(1, 2**31 - 1)
    project_id = load_json(str(Path(project_path) / "project.json")).get("project_id", Path(project_path).name)
    filename_prefix = f"novel_writer/{project_id}/{asset_slug}"
    workflow_template = str(runtime_config.get("workflow_template", "") or "").strip()
    if workflow_template:
        workflow = _build_workflow_from_template(
            template_path=workflow_template,
            positive_prompt=prompt_payload["positive_prompt"],
            negative_prompt=prompt_payload["negative_prompt"],
            width=int(runtime_config["width"]),
            height=int(runtime_config["height"]),
            steps=int(runtime_config["steps"]),
            cfg=float(runtime_config["cfg"]),
            sampler_name=str(runtime_config["sampler_name"]),
            scheduler=str(runtime_config["scheduler"]),
            seed=seed,
            filename_prefix=filename_prefix,
            checkpoint=str(runtime_config.get("checkpoint", "") or ""),
        )
    else:
        workflow = _build_workflow(
            checkpoint=runtime_config["checkpoint"],
            positive_prompt=prompt_payload["positive_prompt"],
            negative_prompt=prompt_payload["negative_prompt"],
            width=int(runtime_config["width"]),
            height=int(runtime_config["height"]),
            steps=int(runtime_config["steps"]),
            cfg=float(runtime_config["cfg"]),
            sampler_name=str(runtime_config["sampler_name"]),
            scheduler=str(runtime_config["scheduler"]),
            seed=seed,
            filename_prefix=filename_prefix,
        )

    comfyui_client = ComfyUIClient(runtime_config["comfyui_api_base"])
    emit_progress(progress_callback, "illustration_queue", "正在提交到 ComfyUI")
    prompt_id = comfyui_client.queue_prompt(
        workflow,
        client_id=f"novel-writer-{random.randint(1000, 9999)}",
        timeout=60,
    )
    emit_progress(progress_callback, "illustration_wait", "ComfyUI 正在生成图片")
    history_item = comfyui_client.wait_for_prompt(
        prompt_id,
        timeout=int(runtime_config["timeout"]),
        poll_interval=float(runtime_config["poll_interval"]),
    )
    output_images = _collect_output_images(history_item)
    if not output_images:
        raise RuntimeError("ComfyUI 已完成执行，但没有返回图片输出。")

    emit_progress(progress_callback, "illustration_download", "正在下载并保存插图结果")
    record_dir.mkdir(parents=True, exist_ok=True)
    for old_file in record_dir.glob("image_*"):
        old_file.unlink(missing_ok=True)

    saved_images = []
    for index, image_info in enumerate(output_images, start=1):
        suffix = Path(str(image_info.get("filename", "image.png"))).suffix or ".png"
        local_name = f"image_{index:02d}{suffix}"
        local_path = record_dir / local_name
        local_path.write_bytes(
            comfyui_client.download_image(
                image_info,
                timeout=max(30, int(runtime_config["timeout"])),
            )
        )
        saved_images.append(
            {
                "file_name": local_name,
                "relative_path": str(local_path.relative_to(Path(project_path))).replace("\\", "/"),
                "source": {
                    "filename": image_info.get("filename", ""),
                    "subfolder": image_info.get("subfolder", ""),
                    "type": image_info.get("type", "output"),
                },
            }
        )

    emit_progress(progress_callback, "illustration_saved", "插图文件已保存")
    return prompt_id, seed, saved_images


def _image_frame_prompt(prompt_payload: dict) -> str:
    prompt = str(prompt_payload.get("positive_prompt", "") or "").strip()
    negative = str(prompt_payload.get("negative_prompt", "") or "").strip()
    if negative:
        prompt = f"{prompt}\n\nAvoid: {negative}"
    return prompt


def _render_image_frame_images(
    project_path: str,
    *,
    asset_slug: str,
    record_dir: Path,
    prompt_payload: dict,
    runtime_config: dict,
    progress_callback=None,
) -> tuple[str, int, list[dict]]:
    seed = int(runtime_config.get("seed") or 0) or random.randint(1, 2**31 - 1)
    runtime = load_image_frame_runtime(
        {
            "api_base": runtime_config.get("image_frame_api_base"),
            "provider": runtime_config.get("image_frame_provider"),
            "model": runtime_config.get("image_frame_model"),
            "size": runtime_config.get("image_frame_size"),
            "aspect_ratio": runtime_config.get("image_frame_aspect_ratio"),
            "google_image_size": runtime_config.get("image_frame_google_image_size"),
            "quality": runtime_config.get("image_frame_quality"),
            "background": runtime_config.get("image_frame_background"),
            "moderation": runtime_config.get("image_frame_moderation"),
            "num_outputs": runtime_config.get("image_frame_num_outputs"),
            "timeout": runtime_config.get("image_frame_timeout"),
            "poll_interval": runtime_config.get("image_frame_poll_interval"),
            "auth_username": runtime_config.get("image_frame_auth_username"),
            "auth_password": runtime_config.get("image_frame_auth_password"),
        }
    )
    runtime["seed"] = seed
    client = ImageFrameClient(runtime["api_base"])
    client.login(runtime.get("auth_username", ""), runtime.get("auth_password", ""))
    emit_progress(progress_callback, "illustration_queue", "正在提交到 Image Frame")
    created = client.create_text_to_image_task(runtime, _image_frame_prompt(prompt_payload), timeout=60)
    task_id = str(created.get("id") or "").strip()
    if not task_id:
        raise RuntimeError(f"Image Frame 未返回任务 ID: {created}")

    emit_progress(progress_callback, "illustration_wait", "Image Frame 正在生成图片")
    completed = client.wait_for_task(
        task_id,
        timeout=int(runtime["timeout"]),
        poll_interval=float(runtime["poll_interval"]),
    )
    if str(completed.get("status", "")).lower() != "succeeded":
        raise RuntimeError(f"Image Frame 生成失败: {completed.get('error') or completed}")

    assets = completed.get("output_assets") or []
    if not assets:
        raise RuntimeError("Image Frame 已完成任务，但没有返回图片输出。")

    emit_progress(progress_callback, "illustration_download", "正在下载并保存 Image Frame 结果")
    record_dir.mkdir(parents=True, exist_ok=True)
    for old_file in record_dir.glob("image_*"):
        old_file.unlink(missing_ok=True)

    saved_images = []
    for index, asset in enumerate(assets, start=1):
        url = str(asset.get("url") or "")
        filename = str(asset.get("filename") or f"image_{index:02d}.png")
        suffix = Path(filename).suffix or ".png"
        local_name = f"image_{index:02d}{suffix}"
        local_path = record_dir / local_name
        local_path.write_bytes(client.request_bytes(url, timeout=max(30, int(runtime["timeout"]))))
        saved_images.append(
            {
                "file_name": local_name,
                "relative_path": str(local_path.relative_to(Path(project_path))).replace("\\", "/"),
                "source": {
                    "task_id": task_id,
                    "url": url,
                    "filename": filename,
                    "provider": runtime.get("provider", ""),
                },
            }
        )

    emit_progress(progress_callback, "illustration_saved", "插图文件已保存")
    return task_id, seed, saved_images


def _slugify_name(text: str, default: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", str(text or "").strip(), flags=re.UNICODE).strip("_")
    return slug or default


def _default_cover_prompt_payload(project_data: dict, runtime_config: dict, user_request: str) -> dict:
    project = project_data.get("project", {})
    world = project_data.get("world", {})
    plot_state = project_data.get("plot_state", {})
    style = project_data.get("style", {})
    protagonists = (project_data.get("characters", {}) or {}).get("protagonists") or []

    protagonist_focus = []
    for item in protagonists[:4]:
        protagonist_focus.append(
            {
                "name": str(item.get("name", "")).strip(),
                "appearance": str(item.get("appearance", "")).strip(),
                "outfit": "story-consistent signature outfit",
                "action": "posed for key visual",
                "expression": "emotionally readable expression",
            }
        )

    positive_prompt = _compose_structured_positive_prompt(
        runtime_config=runtime_config,
        scene_summary=f"《{project.get('name', '小说')}》封面主视觉",
        characters=protagonist_focus,
        environment=", ".join(
            item for item in (
                str(plot_state.get("current_location", "")).strip(),
                "winter survival campus backdrop",
            ) if item
        ),
        composition="wide cover shot, centered hero composition, clear focal hierarchy",
        lighting="dramatic natural key light, readable silhouettes, atmospheric background glow",
        extra_parts=[
            "light novel cover",
            "key visual",
            "single image no text",
            str(project.get("name", "")).strip(),
            str(world.get("genre", "")).strip(),
            _compact_scene_summary(str(style.get("tone", "")).strip(), limit=28),
            _compact_scene_summary(str(user_request or "").strip(), limit=28),
        ],
    )
    return {
        "scene_summary": f"《{project.get('name', '小说')}》封面主视觉",
        "positive_prompt": positive_prompt,
        "negative_prompt": runtime_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "prompt_source": "fallback",
    }


def _default_character_portrait_prompt_payload(
    project_data: dict,
    character: dict,
    runtime_config: dict,
    user_request: str,
) -> dict:
    project = project_data.get("project", {})
    world = project_data.get("world", {})
    style = project_data.get("style", {})
    name = str(character.get("name", "角色")).strip() or "角色"
    role = str(character.get("role", "")).strip()
    positive_prompt = _compose_structured_positive_prompt(
        runtime_config=runtime_config,
        scene_summary=f"{name} 人物立绘",
        characters=[
            {
                "name": name,
                "appearance": str(character.get("appearance", "")).strip(),
                "outfit": "signature outfit",
                "action": "standing pose",
                "expression": "clear readable expression",
            }
        ],
        environment=", ".join(
            item for item in (
                str(world.get("setting", "")).strip(),
                "subtle background hint",
            ) if item
        ),
        composition="full body portrait, three-quarter view, clean framing, clear silhouette separation",
        lighting="soft readable portrait lighting, gentle rim light, balanced background contrast",
        extra_parts=[
            "full body character portrait",
            "character sheet style",
            "single character",
            role,
            str(project.get("name", "")).strip(),
            str(world.get("genre", "")).strip(),
            _compact_scene_summary(str(style.get("tone", "")).strip(), limit=28),
            _compact_scene_summary(str(user_request or "").strip(), limit=28),
        ],
    )
    return {
        "scene_summary": f"{name} 人物立绘",
        "positive_prompt": positive_prompt,
        "negative_prompt": runtime_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        "prompt_source": "fallback",
    }


def get_illustration_record(project_path: str, chapter_slug: str) -> dict | None:
    metadata_path = _chapter_record_dir(project_path, chapter_slug) / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return load_json(str(metadata_path))
    except Exception:
        return None


def list_illustration_records(project_path: str) -> list[dict]:
    illustrations_dir = Path(project_path) / "illustrations"
    if not illustrations_dir.exists():
        return []
    records: list[dict] = []
    for metadata_path in illustrations_dir.glob("chapter_*/metadata.json"):
        try:
            records.append(load_json(str(metadata_path)))
        except Exception:
            continue
    return sorted(records, key=lambda item: item.get("chapter_slug", ""), reverse=True)


def illustrate_chapter(
    project_path: str,
    chapter_ref: str | None = None,
    *,
    llm_config: dict | None = None,
    user_request: str = "",
    force: bool = False,
    runtime_config: dict | None = None,
    persist_runtime_config: bool = True,
    progress_callback=None,
) -> dict:
    chapter_file = _resolve_chapter_file(project_path, chapter_ref)
    chapter_slug = chapter_file.stem
    record_dir = _chapter_record_dir(project_path, chapter_slug)
    metadata_path = record_dir / "metadata.json"
    emit_progress(progress_callback, "illustration_prepare", f"正在处理 {chapter_slug} 的插图")

    if not force:
        existing = _load_existing_record(project_path, metadata_path)
        if existing:
            emit_progress(progress_callback, "illustration_reused", f"{chapter_slug} 已复用已有插图")
            return existing

    resolved_runtime = dict(runtime_config or _build_runtime_config(project_path))
    if persist_runtime_config:
        _persist_runtime_config(project_path, resolved_runtime)

    chapter_text = chapter_file.read_text(encoding="utf-8")
    prompt_payload = _generate_prompt_payload(
        project_path,
        chapter_text,
        llm_config,
        resolved_runtime,
        user_request=user_request,
        chapter_slug=chapter_slug,
        progress_callback=progress_callback,
    )

    prompt_id, seed, saved_images = _render_illustration_images(
        project_path,
        asset_slug=chapter_slug,
        record_dir=record_dir,
        prompt_payload=prompt_payload,
        runtime_config=resolved_runtime,
        progress_callback=progress_callback,
    )

    record = {
        "chapter_slug": chapter_slug,
        "chapter_file": str(chapter_file.relative_to(Path(project_path))).replace("\\", "/"),
        "generated_at": utc_now(),
        "prompt_id": prompt_id,
        "seed": seed,
        "scene_summary": prompt_payload.get("scene_summary", ""),
        "positive_prompt": prompt_payload.get("positive_prompt", ""),
        "negative_prompt": prompt_payload.get("negative_prompt", ""),
        "prompt_source": prompt_payload.get("prompt_source", "fallback"),
        "user_request": user_request,
        "backend": resolved_runtime.get("backend", "image_frame"),
        "image_frame": {
            "api_base": resolved_runtime.get("image_frame_api_base", ""),
            "provider": resolved_runtime.get("image_frame_provider", ""),
            "model": resolved_runtime.get("image_frame_model", ""),
            "size": resolved_runtime.get("image_frame_size", ""),
            "aspect_ratio": resolved_runtime.get("image_frame_aspect_ratio", ""),
            "google_image_size": resolved_runtime.get("image_frame_google_image_size", ""),
            "quality": resolved_runtime.get("image_frame_quality", ""),
            "background": resolved_runtime.get("image_frame_background", ""),
            "moderation": resolved_runtime.get("image_frame_moderation", ""),
            "num_outputs": int(resolved_runtime.get("image_frame_num_outputs", 1)),
        },
        "comfyui": {
            "api_base": resolved_runtime.get("comfyui_api_base", ""),
            "workflow_template": resolved_runtime.get("workflow_template", ""),
            "checkpoint": resolved_runtime.get("checkpoint", ""),
            "width": int(resolved_runtime.get("width", 832)),
            "height": int(resolved_runtime.get("height", 1216)),
            "steps": int(resolved_runtime.get("steps", 28)),
            "cfg": float(resolved_runtime.get("cfg", 6.5)),
            "sampler_name": resolved_runtime.get("sampler_name", "euler"),
            "scheduler": resolved_runtime.get("scheduler", "normal"),
        },
        "images": saved_images,
        "reused": False,
    }
    save_json(str(metadata_path), record)
    emit_progress(progress_callback, "illustration_done", f"{chapter_slug} 的插图已完成")
    return record


def illustrate_chapters(
    project_path: str,
    *,
    chapter_refs: list[str] | None = None,
    llm_config: dict | None = None,
    user_request: str = "",
    force: bool = False,
    overrides: dict | None = None,
    max_workers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    runtime_config = _build_runtime_config(project_path, overrides=overrides)
    refs = chapter_refs or ["latest"]
    _persist_runtime_config(project_path, runtime_config)
    if progress_callback is not None:
        max_workers = 1
    worker_count = _coerce_worker_count(max_workers, len(refs))
    if worker_count == 1:
        results = []
        for index, chapter_ref in enumerate(refs):
            emit_progress(
                progress_callback,
                "illustration_batch",
                f"正在生成第 {index + 1}/{len(refs)} 个章节插图",
                current=index,
                total=len(refs),
            )
            results.append(
                illustrate_chapter(
                    project_path,
                    chapter_ref,
                    llm_config=llm_config,
                    user_request=user_request,
                    force=force,
                    runtime_config=runtime_config,
                    persist_runtime_config=False,
                    progress_callback=progress_callback,
                )
            )
            emit_progress(
                progress_callback,
                "illustration_batch_done",
                f"第 {index + 1}/{len(refs)} 个章节插图已完成",
                current=index + 1,
                total=len(refs),
            )
        return results

    results: list[dict | None] = [None] * len(refs)
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="illustrate") as executor:
        futures = {
            executor.submit(
                illustrate_chapter,
                project_path,
                chapter_ref,
                llm_config=llm_config,
                user_request=user_request,
                force=force,
                runtime_config=runtime_config,
                persist_runtime_config=False,
                progress_callback=progress_callback,
            ): index
            for index, chapter_ref in enumerate(refs)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()

    return [result for result in results if result is not None]


def illustrate_cover(
    project_path: str,
    *,
    user_request: str = "",
    force: bool = False,
    runtime_config: dict | None = None,
) -> dict:
    record_dir = _asset_record_dir(project_path, "cover")
    metadata_path = record_dir / "metadata.json"
    if not force:
        existing = _load_existing_record(project_path, metadata_path)
        if existing:
            return existing

    resolved_runtime = dict(runtime_config or _build_runtime_config(project_path))
    _persist_runtime_config(project_path, resolved_runtime)
    project_data = load_project(project_path)
    prompt_payload = _default_cover_prompt_payload(project_data, resolved_runtime, user_request)
    prompt_id, seed, saved_images = _render_illustration_images(
        project_path,
        asset_slug="cover",
        record_dir=record_dir,
        prompt_payload=prompt_payload,
        runtime_config=resolved_runtime,
    )

    record = {
        "asset_kind": "cover",
        "asset_slug": "cover",
        "generated_at": utc_now(),
        "prompt_id": prompt_id,
        "seed": seed,
        "scene_summary": prompt_payload.get("scene_summary", ""),
        "positive_prompt": prompt_payload.get("positive_prompt", ""),
        "negative_prompt": prompt_payload.get("negative_prompt", ""),
        "prompt_source": prompt_payload.get("prompt_source", "fallback"),
        "user_request": user_request,
        "backend": resolved_runtime.get("backend", "image_frame"),
        "image_frame": {
            "api_base": resolved_runtime.get("image_frame_api_base", ""),
            "provider": resolved_runtime.get("image_frame_provider", ""),
            "model": resolved_runtime.get("image_frame_model", ""),
            "size": resolved_runtime.get("image_frame_size", ""),
            "aspect_ratio": resolved_runtime.get("image_frame_aspect_ratio", ""),
            "google_image_size": resolved_runtime.get("image_frame_google_image_size", ""),
            "quality": resolved_runtime.get("image_frame_quality", ""),
            "background": resolved_runtime.get("image_frame_background", ""),
            "moderation": resolved_runtime.get("image_frame_moderation", ""),
            "num_outputs": int(resolved_runtime.get("image_frame_num_outputs", 1)),
        },
        "comfyui": {
            "api_base": resolved_runtime.get("comfyui_api_base", ""),
            "workflow_template": resolved_runtime.get("workflow_template", ""),
            "checkpoint": resolved_runtime.get("checkpoint", ""),
            "width": int(resolved_runtime.get("width", 832)),
            "height": int(resolved_runtime.get("height", 1216)),
            "steps": int(resolved_runtime.get("steps", 28)),
            "cfg": float(resolved_runtime.get("cfg", 6.5)),
            "sampler_name": resolved_runtime.get("sampler_name", "euler"),
            "scheduler": resolved_runtime.get("scheduler", "normal"),
        },
        "images": saved_images,
        "reused": False,
    }
    save_json(str(metadata_path), record)
    return record


def illustrate_character_portraits(
    project_path: str,
    *,
    user_request: str = "",
    force: bool = False,
    runtime_config: dict | None = None,
) -> list[dict]:
    resolved_runtime = dict(runtime_config or _build_runtime_config(project_path))
    _persist_runtime_config(project_path, resolved_runtime)
    project_data = load_project(project_path)
    characters = project_data.get("characters", {}) or {}
    portrait_targets = (characters.get("protagonists") or []) or (characters.get("supporting") or [])

    results = []
    for index, character in enumerate(portrait_targets, start=1):
        name = str(character.get("name", "角色")).strip() or f"character_{index:02d}"
        character_slug = f"character_{index:02d}_{_slugify_name(name, f'character_{index:02d}') }"
        record_dir = _asset_record_dir(project_path, "characters", character_slug)
        metadata_path = record_dir / "metadata.json"

        if not force:
            existing = _load_existing_record(project_path, metadata_path)
            if existing:
                results.append(existing)
                continue

        prompt_payload = _default_character_portrait_prompt_payload(project_data, character, resolved_runtime, user_request)
        prompt_id, seed, saved_images = _render_illustration_images(
            project_path,
            asset_slug=f"characters/{character_slug}",
            record_dir=record_dir,
            prompt_payload=prompt_payload,
            runtime_config=resolved_runtime,
        )
        record = {
            "asset_kind": "character_portrait",
            "asset_slug": character_slug,
            "character_index": index,
            "character_name": name,
            "character_role": str(character.get("role", "")).strip(),
            "generated_at": utc_now(),
            "prompt_id": prompt_id,
            "seed": seed,
            "scene_summary": prompt_payload.get("scene_summary", ""),
            "positive_prompt": prompt_payload.get("positive_prompt", ""),
            "negative_prompt": prompt_payload.get("negative_prompt", ""),
            "prompt_source": prompt_payload.get("prompt_source", "fallback"),
            "user_request": user_request,
            "backend": resolved_runtime.get("backend", "image_frame"),
            "image_frame": {
                "api_base": resolved_runtime.get("image_frame_api_base", ""),
                "provider": resolved_runtime.get("image_frame_provider", ""),
                "model": resolved_runtime.get("image_frame_model", ""),
                "size": resolved_runtime.get("image_frame_size", ""),
                "aspect_ratio": resolved_runtime.get("image_frame_aspect_ratio", ""),
                "google_image_size": resolved_runtime.get("image_frame_google_image_size", ""),
                "quality": resolved_runtime.get("image_frame_quality", ""),
                "background": resolved_runtime.get("image_frame_background", ""),
                "moderation": resolved_runtime.get("image_frame_moderation", ""),
                "num_outputs": int(resolved_runtime.get("image_frame_num_outputs", 1)),
            },
            "comfyui": {
                "api_base": resolved_runtime.get("comfyui_api_base", ""),
                "workflow_template": resolved_runtime.get("workflow_template", ""),
                "checkpoint": resolved_runtime.get("checkpoint", ""),
                "width": int(resolved_runtime.get("width", 832)),
                "height": int(resolved_runtime.get("height", 1216)),
                "steps": int(resolved_runtime.get("steps", 28)),
                "cfg": float(resolved_runtime.get("cfg", 6.5)),
                "sampler_name": resolved_runtime.get("sampler_name", "euler"),
                "scheduler": resolved_runtime.get("scheduler", "normal"),
            },
            "images": saved_images,
            "reused": False,
        }
        save_json(str(metadata_path), record)
        results.append(record)

    return results


def illustrate_project_assets(
    project_path: str,
    *,
    user_request: str = "",
    force: bool = False,
    overrides: dict | None = None,
) -> dict:
    runtime_config = _build_runtime_config(project_path, overrides=overrides)
    return {
        "cover": illustrate_cover(
            project_path,
            user_request=user_request,
            force=force,
            runtime_config=runtime_config,
        ),
        "portraits": illustrate_character_portraits(
            project_path,
            user_request=user_request,
            force=force,
            runtime_config=runtime_config,
        ),
    }
