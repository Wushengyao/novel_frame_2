"""ComfyUI-powered illustration helpers for novel chapters."""

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from llm_client import generate_text_with_metadata
from project_manager import load_json, load_project, save_json, update_project_stats
from prompt_builder import build_illustration_prompt


LEGACY_DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, bad anatomy, extra fingers, malformed hands, malformed face, "
    "deformed body, duplicate, multiple views, split panels, comic page, text, watermark, logo, caption, "
    "jpeg artifacts, cropped, out of frame"
)
DEFAULT_NEGATIVE_PROMPT = ""
LEGACY_DEFAULT_STYLE_PRESET = (
    "masterpiece, best quality, detailed light novel illustration, cinematic composition, expressive characters, "
    "rich environmental storytelling, dramatic winter atmosphere, soft volumetric lighting"
)
DEFAULT_STYLE_PRESET = (
    "clean subject separation, layered depth, expressive body language, atmospheric perspective"
)
DEFAULT_WORKFLOW_TEMPLATE_NAME = "image_z_image_turbo (2).json"
PREFERRED_CHECKPOINTS = (
    "illusious/illustrij_v21.safetensors",
    "illusious/illustrij_v20.safetensors",
    "illusious/illustrij_v19.safetensors",
    "illusious/illustrij_v18.safetensors",
    "illusious/illustrij_v17.safetensors",
    "illusious/prefectIllustriousXL_v70.safetensors",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_checkpoint_name(name: str) -> str:
    normalized = str(name or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    separator = "\\" if os.name == "nt" else "/"
    return normalized.replace("/", separator)


def _extract_json_object(text: str) -> dict:
    text = (text or "").strip()
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
    raise ValueError("Could not parse JSON from illustration prompt response.")


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


def _normalize_api_base(value: str) -> str:
    text = (value or "http://127.0.0.1:8188").strip().rstrip("/")
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    return text


def _request_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 60, allow_404: bool = False) -> dict:
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


def _request_bytes(url: str, *, timeout: int = 60) -> bytes:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI file download failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"Failed to download ComfyUI image: {reason}") from exc


def _resolve_candidate_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.name.lower() != "comfyui" and (candidate / "ComfyUI").exists():
        candidate = candidate / "ComfyUI"
    if candidate.exists():
        return candidate.resolve()
    return None


def _candidate_comfyui_roots() -> list[Path]:
    base_dir = Path(__file__).resolve().parent
    workspace_root = base_dir.parent.parent
    candidates: list[Path] = []
    seen: set[str] = set()

    for raw in (
        os.environ.get("NOVEL_COMFYUI_ROOT", ""),
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


def _resolve_comfyui_root(saved_config: dict, overrides: dict) -> Path | None:
    for raw in (
        overrides.get("comfyui_root", ""),
        os.environ.get("NOVEL_COMFYUI_ROOT", ""),
        saved_config.get("comfyui_root", ""),
    ):
        path = _resolve_candidate_path(str(raw))
        if path is not None:
            return path
    for candidate in _candidate_comfyui_roots():
        return candidate
    return None


def _resolve_workflow_template_path(saved_config: dict, overrides: dict, comfyui_root: Path | None) -> str:
    candidates = [
        str(overrides.get("workflow_template", "") or "").strip(),
        str(os.environ.get("NOVEL_COMFYUI_WORKFLOW_TEMPLATE", "") or "").strip(),
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


def _find_default_checkpoint(comfyui_root: Path | None) -> str:
    if comfyui_root is None:
        return ""
    checkpoints_dir = comfyui_root / "models" / "checkpoints"
    if not checkpoints_dir.exists():
        return ""

    for name in PREFERRED_CHECKPOINTS:
        path = checkpoints_dir / Path(name)
        if path.exists():
            return _normalize_checkpoint_name(name)

    for pattern in ("*.safetensors", "*.ckpt"):
        for path in sorted(checkpoints_dir.rglob(pattern)):
            if path.is_file():
                return _relative_checkpoint_name(path, checkpoints_dir)
    return ""


def _resolve_checkpoint(saved_config: dict, overrides: dict, comfyui_root: Path | None) -> str:
    raw_values = (
        str(overrides.get("checkpoint", "") or "").strip(),
        str(os.environ.get("NOVEL_COMFYUI_CHECKPOINT", "") or "").strip(),
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

    return _find_default_checkpoint(comfyui_root)


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
    comfyui_root = _resolve_comfyui_root(saved, merged_overrides)
    workflow_template = _resolve_workflow_template_path(saved, merged_overrides, comfyui_root)
    workflow_defaults = _extract_workflow_template_defaults(workflow_template)
    saved_workflow_template = str(saved.get("workflow_template", "") or "").strip()
    saved_matches_template = bool(workflow_template and saved_workflow_template == workflow_template)
    checkpoint = _resolve_checkpoint(saved, merged_overrides, comfyui_root)
    saved_negative_prompt = str(saved.get("negative_prompt", "") or "").strip()
    if saved_negative_prompt == LEGACY_DEFAULT_NEGATIVE_PROMPT:
        saved_negative_prompt = ""
    saved_style_preset = str(saved.get("style_preset", "") or "").strip()
    if saved_style_preset == LEGACY_DEFAULT_STYLE_PRESET:
        saved_style_preset = ""

    config = {
        "comfyui_api_base": _normalize_api_base(
            str(
                merged_overrides.get("comfyui_api_base")
                or os.environ.get("NOVEL_COMFYUI_API_BASE")
                or saved.get("comfyui_api_base")
                or "http://127.0.0.1:8188"
            )
        ),
        "comfyui_root": str(comfyui_root) if comfyui_root else "",
        "workflow_template": workflow_template,
        "checkpoint": checkpoint,
        "width": _safe_int(
            merged_overrides.get("width")
            or os.environ.get("NOVEL_COMFYUI_WIDTH")
            or (saved.get("width") if saved_matches_template else workflow_defaults.get("width")),
            1280,
        ),
        "height": _safe_int(
            merged_overrides.get("height")
            or os.environ.get("NOVEL_COMFYUI_HEIGHT")
            or (saved.get("height") if saved_matches_template else workflow_defaults.get("height")),
            1280,
        ),
        "steps": _safe_int(
            merged_overrides.get("steps")
            or os.environ.get("NOVEL_COMFYUI_STEPS")
            or (saved.get("steps") if saved_matches_template else workflow_defaults.get("steps")),
            8,
        ),
        "cfg": _safe_float(
            merged_overrides.get("cfg")
            or os.environ.get("NOVEL_COMFYUI_CFG")
            or (saved.get("cfg") if saved_matches_template else workflow_defaults.get("cfg")),
            1.0,
        ),
        "sampler_name": str(
            merged_overrides.get("sampler_name")
            or os.environ.get("NOVEL_COMFYUI_SAMPLER")
            or (saved.get("sampler_name") if saved_matches_template else workflow_defaults.get("sampler_name"))
            or "res_multistep"
        ).strip(),
        "scheduler": str(
            merged_overrides.get("scheduler")
            or os.environ.get("NOVEL_COMFYUI_SCHEDULER")
            or (saved.get("scheduler") if saved_matches_template else workflow_defaults.get("scheduler"))
            or "simple"
        ).strip(),
        "timeout": _safe_int(
            merged_overrides.get("timeout") or os.environ.get("NOVEL_COMFYUI_TIMEOUT") or saved.get("timeout"),
            600,
        ),
        "poll_interval": _safe_float(
            merged_overrides.get("poll_interval")
            or os.environ.get("NOVEL_COMFYUI_POLL_INTERVAL")
            or saved.get("poll_interval"),
            1.5,
        ),
        "negative_prompt": str(
            merged_overrides.get("negative_prompt")
            or os.environ.get("NOVEL_COMFYUI_NEGATIVE_PROMPT")
            or saved_negative_prompt
            or DEFAULT_NEGATIVE_PROMPT
        ).strip(),
        "style_preset": str(
            merged_overrides.get("style_preset")
            or os.environ.get("NOVEL_COMFYUI_STYLE_PRESET")
            or saved_style_preset
            or DEFAULT_STYLE_PRESET
        ).strip(),
        "seed": _safe_int(
            merged_overrides.get("seed") or os.environ.get("NOVEL_COMFYUI_SEED") or saved.get("seed"),
            0,
        ),
    }

    if not config["workflow_template"] and not config["checkpoint"]:
        raise RuntimeError(
            "未找到可用的 ComfyUI 工作流模板或 checkpoint。请确认 workflow/image_z_image_turbo (2).json 存在，或设置 NOVEL_COMFYUI_WORKFLOW_TEMPLATE。"
        )
    return config


def _persist_runtime_config(project_path: str, runtime_config: dict) -> None:
    project_file = Path(project_path) / "project.json"
    project_data = load_json(str(project_file))
    project_data["illustration_config"] = {
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
    }
    project_data["updated_at"] = _utc_now()
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
) -> dict:
    project_data = load_project(project_path)
    if llm_config and llm_config.get("api_key") and llm_config.get("model_provider"):
        prompt = build_illustration_prompt(project_data, chapter_text, user_request=user_request)
        try:
            response_text, metadata = generate_text_with_metadata(prompt, llm_config)
            update_project_stats(
                project_path,
                phase="illustration_prompt",
                success=True,
                usage=metadata.get("usage"),
            )
            payload = _extract_json_object(response_text)
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
            update_project_stats(project_path, phase="illustration_prompt", success=False, usage=None)

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


def _queue_prompt(api_base: str, workflow: dict[str, Any]) -> str:
    payload = _request_json(
        f"{api_base}/prompt",
        payload={"prompt": workflow, "client_id": f"novel-writer-{random.randint(1000, 9999)}"},
        timeout=60,
    )
    prompt_id = str(payload.get("prompt_id", "")).strip()
    if not prompt_id:
        raise RuntimeError(f"ComfyUI 未返回 prompt_id: {payload}")
    return prompt_id


def _wait_for_prompt(api_base: str, prompt_id: str, timeout: int, poll_interval: float) -> dict:
    deadline = time.time() + timeout
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        payload = _request_json(
            f"{api_base}/history/{parse.quote(prompt_id)}",
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


def _collect_output_images(history_item: dict) -> list[dict]:
    images: list[dict] = []
    for node_output in (history_item.get("outputs") or {}).values():
        for image in node_output.get("images", []) or []:
            if isinstance(image, dict) and image.get("filename"):
                images.append(image)
    return images


def _download_image(api_base: str, image_info: dict, timeout: int) -> bytes:
    query = parse.urlencode(
        {
            "filename": image_info.get("filename", ""),
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
    )
    return _request_bytes(f"{api_base}/view?{query}", timeout=timeout)


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
) -> tuple[str, int, list[dict]]:
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

    prompt_id = _queue_prompt(runtime_config["comfyui_api_base"], workflow)
    history_item = _wait_for_prompt(
        runtime_config["comfyui_api_base"],
        prompt_id,
        timeout=int(runtime_config["timeout"]),
        poll_interval=float(runtime_config["poll_interval"]),
    )
    output_images = _collect_output_images(history_item)
    if not output_images:
        raise RuntimeError("ComfyUI 已完成执行，但没有返回图片输出。")

    record_dir.mkdir(parents=True, exist_ok=True)
    for old_file in record_dir.glob("image_*"):
        old_file.unlink(missing_ok=True)

    saved_images = []
    for index, image_info in enumerate(output_images, start=1):
        suffix = Path(str(image_info.get("filename", "image.png"))).suffix or ".png"
        local_name = f"image_{index:02d}{suffix}"
        local_path = record_dir / local_name
        local_path.write_bytes(
            _download_image(
                runtime_config["comfyui_api_base"],
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

    return prompt_id, seed, saved_images


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
) -> dict:
    chapter_file = _resolve_chapter_file(project_path, chapter_ref)
    chapter_slug = chapter_file.stem
    record_dir = _chapter_record_dir(project_path, chapter_slug)
    metadata_path = record_dir / "metadata.json"

    if not force:
        existing = _load_existing_record(project_path, metadata_path)
        if existing:
            return existing

    resolved_runtime = dict(runtime_config or _build_runtime_config(project_path))
    _persist_runtime_config(project_path, resolved_runtime)

    chapter_text = chapter_file.read_text(encoding="utf-8")
    prompt_payload = _generate_prompt_payload(
        project_path,
        chapter_text,
        llm_config,
        resolved_runtime,
        user_request=user_request,
    )

    prompt_id, seed, saved_images = _render_illustration_images(
        project_path,
        asset_slug=chapter_slug,
        record_dir=record_dir,
        prompt_payload=prompt_payload,
        runtime_config=resolved_runtime,
    )

    record = {
        "chapter_slug": chapter_slug,
        "chapter_file": str(chapter_file.relative_to(Path(project_path))).replace("\\", "/"),
        "generated_at": _utc_now(),
        "prompt_id": prompt_id,
        "seed": seed,
        "scene_summary": prompt_payload.get("scene_summary", ""),
        "positive_prompt": prompt_payload.get("positive_prompt", ""),
        "negative_prompt": prompt_payload.get("negative_prompt", ""),
        "prompt_source": prompt_payload.get("prompt_source", "fallback"),
        "user_request": user_request,
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


def illustrate_chapters(
    project_path: str,
    *,
    chapter_refs: list[str] | None = None,
    llm_config: dict | None = None,
    user_request: str = "",
    force: bool = False,
    overrides: dict | None = None,
) -> list[dict]:
    runtime_config = _build_runtime_config(project_path, overrides=overrides)
    refs = chapter_refs or ["latest"]
    results = []
    for chapter_ref in refs:
        results.append(
            illustrate_chapter(
                project_path,
                chapter_ref,
                llm_config=llm_config,
                user_request=user_request,
                force=force,
                runtime_config=runtime_config,
            )
        )
    return results


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
        "generated_at": _utc_now(),
        "prompt_id": prompt_id,
        "seed": seed,
        "scene_summary": prompt_payload.get("scene_summary", ""),
        "positive_prompt": prompt_payload.get("positive_prompt", ""),
        "negative_prompt": prompt_payload.get("negative_prompt", ""),
        "prompt_source": prompt_payload.get("prompt_source", "fallback"),
        "user_request": user_request,
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
            "generated_at": _utc_now(),
            "prompt_id": prompt_id,
            "seed": seed,
            "scene_summary": prompt_payload.get("scene_summary", ""),
            "positive_prompt": prompt_payload.get("positive_prompt", ""),
            "negative_prompt": prompt_payload.get("negative_prompt", ""),
            "prompt_source": prompt_payload.get("prompt_source", "fallback"),
            "user_request": user_request,
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
