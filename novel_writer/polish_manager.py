"""Chapter polishing workflow for already generated chapters."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from common_utils import emit_progress, safe_int, utc_now
from console_logger import log_error, log_info, log_success
from llm_client import generate_text_with_metadata, raise_if_llm_response_truncated
from progression_manager import mark_active_progression_sessions_stale
from project_manager import (
    ensure_no_project_audio_lock,
    ensure_chapter_heading,
    extract_chapter_title,
    load_json,
    load_project,
    normalize_chapter_text,
    record_context_telemetry,
    save_json,
    update_project_stats,
)
from prompt_builder import build_chapter_polish_prompt, build_system_prompt


POLISH_PRESETS = [
    {
        "id": "details",
        "label": "细节增强",
        "prompt": "细节增强：强化动作、感官、环境、表情和内心细节，让场景更具体。",
    },
    {
        "id": "cheerful",
        "label": "更欢乐",
        "prompt": "更欢乐：增加轻松互怼、幽默节奏或温暖小互动，但不破坏当前处境。",
    },
    {
        "id": "longer",
        "label": "更长",
        "prompt": "更长：适度扩写段落、互动、过渡和描写，不靠重复原句凑长度。",
    },
    {
        "id": "smoother",
        "label": "节奏更顺",
        "prompt": "节奏更顺：优化段落衔接、情绪推进和句子节奏，让阅读更流畅。",
    },
    {
        "id": "interaction",
        "label": "人物互动更自然",
        "prompt": "人物互动更自然：让对白、反应、沉默和肢体动作更符合人物关系。",
    },
    {
        "id": "imagery",
        "label": "语言更有画面感",
        "prompt": "语言更有画面感：提升镜头感、空间层次、光影和氛围描写。",
    },
]
POLISH_PRESET_BY_ID = {preset["id"]: preset for preset in POLISH_PRESETS}
CHAPTER_SLUG_PATTERN = re.compile(r"^chapter_(\d{4})$")


def normalize_polish_preset_ids(preset_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in preset_ids or []:
        preset_id = str(raw_id or "").strip()
        if preset_id not in POLISH_PRESET_BY_ID or preset_id in seen:
            continue
        normalized.append(preset_id)
        seen.add(preset_id)
    return normalized


def _polish_directions(preset_ids: list[str], custom_request: str) -> list[str]:
    directions = [POLISH_PRESET_BY_ID[preset_id]["prompt"] for preset_id in preset_ids]
    if not directions and not custom_request.strip():
        directions.append("基础润色：优化语句、衔接、画面感和可读性。")
    return directions


def _chapter_path(project_path: str, chapter_slug: str) -> Path:
    match = CHAPTER_SLUG_PATTERN.fullmatch(str(chapter_slug or "").strip())
    if not match:
        raise ValueError("章节标识无效，必须形如 chapter_0001。")

    base = Path(project_path).resolve()
    path = (base / "chapters" / f"{chapter_slug}.md").resolve()
    chapters_dir = (base / "chapters").resolve()
    if chapters_dir not in path.parents:
        raise ValueError("章节路径无效。")
    if not path.exists():
        raise FileNotFoundError(f"章节不存在: {chapter_slug}")
    return path


def _strip_wrapping_code_fence(text: str) -> str:
    content = str(text or "").strip()
    match = re.fullmatch(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", content, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def _validate_polished_text(original_text: str, polished_text: str) -> None:
    if not polished_text.strip():
        raise ValueError("模型返回的润色正文为空，已保留原文。")

    original_len = len(original_text.strip())
    polished_len = len(polished_text.strip())
    if original_len >= 200 and polished_len < max(80, int(original_len * 0.35)):
        raise ValueError("模型返回的润色正文明显过短，已保留原文。")
    if original_len >= 80 and polished_len < 30:
        raise ValueError("模型返回的润色正文明显过短，已保留原文。")


def _backup_original_chapter(
    project_path: str,
    chapter_slug: str,
    original_text: str,
    *,
    preset_ids: list[str],
    custom_request: str,
    config: dict,
    polished_text: str,
) -> tuple[Path, Path]:
    safe_timestamp = utc_now().replace("+00:00", "Z").replace("-", "").replace(":", "")
    backup_dir = Path(project_path) / "polish_backups" / chapter_slug
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_timestamp}_{uuid4().hex[:8]}"
    backup_path = backup_dir / f"{stem}.md"
    metadata_path = backup_dir / f"{stem}.json"
    backup_path.write_text(original_text.rstrip() + "\n", encoding="utf-8")
    save_json(
        str(metadata_path),
        {
            "created_at": utc_now(),
            "chapter_slug": chapter_slug,
            "backup_file": backup_path.name,
            "preset_ids": preset_ids,
            "preset_labels": [POLISH_PRESET_BY_ID[preset_id]["label"] for preset_id in preset_ids],
            "custom_request": custom_request.strip(),
            "model_provider": config.get("model_provider", ""),
            "model": config.get("model") or config.get("model_name") or "",
            "original_chars": len(original_text.strip()),
            "polished_chars": len(polished_text.strip()),
        },
    )
    return backup_path, metadata_path


def _is_latest_chapter(project_path: str, chapter_slug: str) -> bool:
    project = load_json(str(Path(project_path) / "project.json"))
    chapter_count = safe_int(project.get("chapter_count"), 0)
    match = CHAPTER_SLUG_PATTERN.fullmatch(chapter_slug)
    if not match:
        return False
    return int(match.group(1)) == chapter_count and chapter_count > 0


def run_chapter_polish(
    project_path: str,
    config: dict,
    chapter_slug: str,
    preset_ids: list[str] | tuple[str, ...] | None = None,
    custom_request: str = "",
    *,
    progress_callback=None,
) -> dict:
    ensure_no_project_audio_lock(project_path, "润色章节")
    normalized_preset_ids = normalize_polish_preset_ids(preset_ids)
    custom_request = str(custom_request or "").strip()
    chapter_file = _chapter_path(project_path, chapter_slug)
    original_text = chapter_file.read_text(encoding="utf-8")
    if not original_text.strip():
        raise ValueError("当前章节正文为空，无法润色。")
    match = CHAPTER_SLUG_PATTERN.fullmatch(chapter_slug)
    chapter_number = int(match.group(1)) if match else 0
    original_title = extract_chapter_title(original_text, chapter_number=chapter_number) if chapter_number else ""

    log_info(f"polish_chapter: prepare project={project_path} chapter={chapter_slug}")
    emit_progress(progress_callback, "polish_prepare", "正在准备章节润色")
    project_data = load_project(project_path)
    directions = _polish_directions(normalized_preset_ids, custom_request)
    prompt = build_chapter_polish_prompt(
        project_data,
        original_text,
        polish_directions=directions,
        custom_request=custom_request,
    )
    record_context_telemetry(
        project_path,
        "polish",
        prompt_chars=len(prompt),
        section_chars={
            "chapter_text": len(original_text),
            "polish_directions": sum(len(item) for item in directions),
            "custom_request": len(custom_request),
        },
        planning_mode="",
        extra={
            "chapter_slug": chapter_slug,
            "preset_ids": normalized_preset_ids,
        },
    )

    log_context = {
        "phase": "polish",
        "project_id": str(project_data["project"].get("project_id") or "").strip(),
        "project_path": str(Path(project_path).resolve()),
        "chapter_slug": chapter_slug,
        "preset_ids": normalized_preset_ids,
        "has_custom_request": bool(custom_request),
    }

    try:
        emit_progress(progress_callback, "polish_request", "正在请求模型润色章节")
        response_text, metadata = generate_text_with_metadata(
            prompt,
            config,
            log_context=log_context,
            system_prompt=build_system_prompt("polish"),
        )
        raise_if_llm_response_truncated(metadata, phase="polish")
        stripped_response = _strip_wrapping_code_fence(response_text)
        polished_text = normalize_chapter_text(stripped_response)
        if original_title:
            polished_text = ensure_chapter_heading(polished_text, chapter_number, original_title)
        _validate_polished_text(original_text, polished_text)
    except Exception:
        update_project_stats(project_path, phase="polish", success=False, usage=None)
        log_error("polish_chapter: polish request or validation failed")
        raise

    update_project_stats(
        project_path,
        phase="polish",
        success=True,
        usage=metadata.get("usage"),
        metadata=metadata,
    )
    emit_progress(progress_callback, "polish_backup", "正在备份原章节正文")
    backup_path, metadata_path = _backup_original_chapter(
        project_path,
        chapter_slug,
        original_text,
        preset_ids=normalized_preset_ids,
        custom_request=custom_request,
        config=config,
        polished_text=polished_text,
    )
    emit_progress(progress_callback, "polish_save", "正在覆盖章节正文")
    chapter_file.write_text(polished_text.rstrip() + "\n", encoding="utf-8")

    stale_count = 0
    if _is_latest_chapter(project_path, chapter_slug):
        stale_count = mark_active_progression_sessions_stale(project_path)

    emit_progress(progress_callback, "polish_done", "章节润色完成")
    log_success(f"polish_chapter: saved {chapter_slug}, backup={backup_path}")
    return {
        "chapter_slug": chapter_slug,
        "chapter_path": str(chapter_file),
        "backup_path": str(backup_path),
        "metadata_path": str(metadata_path),
        "original_chars": len(original_text.strip()),
        "polished_chars": len(polished_text.strip()),
        "staled_progression_sessions": stale_count,
    }
