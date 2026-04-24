"""VoxCPM2-powered audiobook helpers for novel chapters."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common_utils import emit_progress, utc_now
from external_services import (
    DEFAULT_EXTERNAL_SERVICES_CONFIG,
    DEFAULT_VOXCPM2_MODEL_ID,
    DEFAULT_VOXCPM2_PYTHON,
    DEFAULT_VOXCPM2_ROOT,
    VoxCPM2Service,
    load_voxcpm2_runtime,
    normalize_voxcpm2_runtime,
)
from project_manager import load_json, load_project, save_json


AUDIOBOOK_DIR_NAME = "audiobook"
VOICE_REFS_DIR_NAME = "voice_refs"
VOICES_FILENAME = "voices.json"
NARRATOR_SPEAKER = "旁白"

DEFAULT_SPLIT_CONFIG = {
    "target_chars": 80,
    "max_chars": 120,
    "min_merge_chars": 24,
}

DEFAULT_AUDIOBOOK_RUNTIME = normalize_voxcpm2_runtime(DEFAULT_EXTERNAL_SERVICES_CONFIG["voxcpm2"])

NARRATOR_PRESETS = [
    {
        "id": "warm_female",
        "label": "温柔女声",
        "control_instruction": "成熟温柔的中文女声，叙述自然，情绪细腻，语速中等偏慢，适合长篇小说旁白",
    },
    {
        "id": "calm_male",
        "label": "沉稳男声",
        "control_instruction": "沉稳低缓的中文男声，吐字清楚，节奏从容，带一点故事感和可靠感",
    },
    {
        "id": "clear_female",
        "label": "清亮女声",
        "control_instruction": "清亮年轻的中文女声，声音干净，表达灵动，语速中等，适合轻小说叙事",
    },
    {
        "id": "magnetic_male",
        "label": "磁性男声",
        "control_instruction": "磁性成熟的中文男声，音色厚实，情绪克制，适合悬疑或末世氛围叙事",
    },
    {
        "id": "neutral_documentary",
        "label": "中性纪录片旁白",
        "control_instruction": "中性自然的中文纪录片旁白，发音标准，情绪稳定，信息表达清晰",
    },
]

SPEECH_VERBS = (
    "说",
    "道",
    "问",
    "答",
    "喊",
    "叫",
    "低声",
    "轻声",
    "喃喃",
    "开口",
    "补充",
    "解释",
    "提醒",
    "嘀咕",
    "笑",
    "叹",
)
INNER_MONOLOGUE_HINTS = (
    "心想",
    "暗想",
    "想道",
    "心道",
    "心里想",
    "在心里",
    "心底",
    "脑海里",
    "意识到",
)
QUOTE_PAIRS = {
    "“": "”",
    "「": "」",
    "『": "』",
    '"': '"',
}
SENTENCE_ENDINGS = "。！？!?；;"
MINOR_ENDINGS = "，,、：:"


@dataclass(frozen=True)
class UploadedVoiceFile:
    filename: str
    content: bytes
    content_type: str = ""


def _audiobook_root(project_path: str | Path) -> Path:
    return Path(project_path) / AUDIOBOOK_DIR_NAME


def _voices_path(project_path: str | Path) -> Path:
    return _audiobook_root(project_path) / VOICES_FILENAME


def _voice_refs_dir(project_path: str | Path) -> Path:
    return _audiobook_root(project_path) / VOICE_REFS_DIR_NAME


def _chapter_record_dir(project_path: str | Path, chapter_slug: str) -> Path:
    return _audiobook_root(project_path) / chapter_slug


def _relative_path(project_path: str | Path, path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(project_path).resolve())).replace("\\", "/")


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def _all_characters(characters: dict) -> list[dict]:
    result = []
    for group in ("protagonists", "supporting"):
        for item in characters.get(group) or []:
            if isinstance(item, dict) and str(item.get("name", "")).strip():
                result.append(item)
    return result


def _character_names(characters: dict | list[dict]) -> list[str]:
    items = characters if isinstance(characters, list) else _all_characters(characters)
    names = [str(item.get("name", "")).strip() for item in items if isinstance(item, dict)]
    return [name for name in names if name]


def _default_character_voice(character: dict) -> dict:
    name = str(character.get("name", "")).strip()
    role = str(character.get("role", "")).strip()
    description = _normalize_text(character.get("description", ""))
    appearance = _normalize_text(character.get("appearance", ""))
    traits = "，".join(item for item in (role, description[:80], appearance[:60]) if item)
    if traits:
        control = f"中文有声小说角色音色，符合{name}的人物设定：{traits}。台词自然，情绪贴合剧情，避免夸张播音腔"
    else:
        control = f"中文有声小说角色音色，符合{name}的人物设定，台词自然，情绪贴合剧情"
    return {
        "control_instruction": control,
        "reference_audio": "",
        "prompt_text": "",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    }


def _default_voice_config(project_data: dict) -> dict:
    characters = _all_characters(project_data.get("characters") or {})
    return {
        "version": 1,
        "updated_at": utc_now(),
        "selected_narrator_id": NARRATOR_PRESETS[0]["id"],
        "narrator_presets": NARRATOR_PRESETS,
        "narrator_reference": {
            "reference_audio": "",
            "prompt_text": "",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        },
        "character_voices": {
            str(character.get("name", "")).strip(): _default_character_voice(character)
            for character in characters
            if str(character.get("name", "")).strip()
        },
    }


def ensure_voice_config(project_path: str | Path) -> dict:
    project_data = load_project(str(project_path))
    path = _voices_path(project_path)
    if path.exists():
        try:
            config = load_json(str(path))
        except Exception:
            config = {}
    else:
        config = {}

    defaults = _default_voice_config(project_data)
    config.setdefault("version", defaults["version"])
    config.setdefault("selected_narrator_id", defaults["selected_narrator_id"])
    config["narrator_presets"] = defaults["narrator_presets"]
    config.setdefault("narrator_reference", defaults["narrator_reference"])
    config.setdefault("character_voices", {})
    for name, default_voice in defaults["character_voices"].items():
        existing = config["character_voices"].get(name)
        if isinstance(existing, dict):
            merged = dict(default_voice)
            merged.update(existing)
            config["character_voices"][name] = merged
        else:
            config["character_voices"][name] = default_voice
    config["updated_at"] = utc_now()
    save_json(str(path), config)
    return config


def save_voice_config(project_path: str | Path, config: dict) -> dict:
    config = dict(config)
    config["updated_at"] = utc_now()
    save_json(str(_voices_path(project_path)), config)
    return config


def narrator_preset_options(project_path: str | Path) -> list[dict]:
    return list(ensure_voice_config(project_path).get("narrator_presets") or NARRATOR_PRESETS)


def _selected_narrator(config: dict) -> dict:
    selected_id = str(config.get("selected_narrator_id") or "").strip()
    for preset in config.get("narrator_presets") or NARRATOR_PRESETS:
        if str(preset.get("id") or "").strip() == selected_id:
            return preset
    return (config.get("narrator_presets") or NARRATOR_PRESETS)[0]


def update_selected_narrator(project_path: str | Path, narrator_preset: str) -> dict:
    config = ensure_voice_config(project_path)
    requested = str(narrator_preset or "").strip()
    valid_ids = {str(item.get("id") or "").strip() for item in config.get("narrator_presets") or []}
    if requested and requested in valid_ids:
        config["selected_narrator_id"] = requested
        save_voice_config(project_path, config)
    return config


def _safe_voice_ref_name(prefix: str, filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix != ".wav":
        raise ValueError("参考音频只支持 .wav 文件。")
    safe_prefix = re.sub(r"[^\w\-]+", "_", prefix, flags=re.UNICODE).strip("_") or "voice"
    return f"{safe_prefix}_{utc_now().replace(':', '').replace('-', '').replace('+00:00', 'Z')}.wav"


def save_uploaded_voice_reference(
    project_path: str | Path,
    *,
    target: str,
    uploaded_file: UploadedVoiceFile,
    prompt_text: str = "",
) -> dict:
    if not uploaded_file or not uploaded_file.content:
        raise ValueError("上传的参考音频为空。")

    config = ensure_voice_config(project_path)
    refs_dir = _voice_refs_dir(project_path)
    refs_dir.mkdir(parents=True, exist_ok=True)

    normalized_target = str(target or "").strip()
    if not normalized_target:
        raise ValueError("缺少参考音频目标。")
    local_name = _safe_voice_ref_name(normalized_target, uploaded_file.filename)
    local_path = refs_dir / local_name
    local_path.write_bytes(uploaded_file.content)
    relative = _relative_path(project_path, local_path)

    payload = {
        "reference_audio": relative,
        "prompt_text": str(prompt_text or "").strip(),
    }
    if normalized_target == "narrator":
        config.setdefault("narrator_reference", {})
        config["narrator_reference"].update(payload)
    else:
        config.setdefault("character_voices", {})
        existing = config["character_voices"].setdefault(normalized_target, {})
        existing.update(payload)
    return save_voice_config(project_path, config)


def _resolve_chapter_file(project_path: str | Path, chapter_ref: str | None) -> Path:
    chapters_dir = Path(project_path) / "chapters"
    if not chapters_dir.exists():
        raise RuntimeError("项目中还没有章节，无法生成有声章节。")

    normalized = (chapter_ref or "latest").strip()
    if not normalized or normalized == "latest":
        chapters = sorted(chapters_dir.glob("chapter_*.md"))
        if not chapters:
            raise RuntimeError("项目中还没有章节，无法生成有声章节。")
        return chapters[-1].resolve()

    candidate = Path(normalized)
    if candidate.exists():
        return candidate.resolve()

    if not normalized.endswith(".md"):
        normalized += ".md"
    chapter_file = chapters_dir / normalized
    if chapter_file.exists():
        return chapter_file.resolve()
    raise RuntimeError(f"找不到章节文件: {chapter_ref}")


def _clean_markdown_text(text: str) -> str:
    lines = []
    for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        lines.append(line)
    return "\n".join(lines).strip()


def _find_quote_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        close = QUOTE_PAIRS.get(char)
        if not close:
            index += 1
            continue
        end = text.find(close, index + 1)
        if end == -1:
            index += 1
            continue
        content = text[index + 1 : end].strip()
        if content:
            spans.append((index, end + 1, content))
        index = end + 1
    return spans


def _contains_speech_verb(text: str) -> bool:
    return any(verb in text for verb in SPEECH_VERBS)


def _speaker_from_context(text: str, character_names: list[str], *, prefer_last: bool) -> str:
    window = text[-60:] if prefer_last else text[:60]
    candidates = []
    for name in character_names:
        position = window.rfind(name) if prefer_last else window.find(name)
        if position == -1:
            continue
        local = window[max(0, position - 12) : position + len(name) + 18]
        score = 10 if _contains_speech_verb(local) else 1
        if "：" in local or ":" in local:
            score += 3
        candidates.append((score, position, name))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=prefer_last)
    return candidates[-1][2] if not prefer_last else candidates[0][2]


def _resolve_quote_speaker(
    before: str,
    after: str,
    character_names: list[str],
    previous_speaker: str,
) -> str:
    speaker = _speaker_from_context(before, character_names, prefer_last=True)
    if speaker:
        return speaker
    speaker = _speaker_from_context(after, character_names, prefer_last=False)
    if speaker:
        return speaker
    if previous_speaker in character_names:
        return previous_speaker
    return NARRATOR_SPEAKER


def _inner_monologue_speaker(text: str, character_names: list[str]) -> str:
    if not any(hint in text for hint in INNER_MONOLOGUE_HINTS):
        return ""
    scored = []
    for name in character_names:
        position = text.find(name)
        if position == -1:
            continue
        nearest_hint = min((abs(text.find(hint) - position) for hint in INNER_MONOLOGUE_HINTS if hint in text), default=999)
        scored.append((nearest_hint, position, name))
    if not scored:
        return ""
    scored.sort()
    return scored[0][2]


def _split_by_punctuation(text: str, punctuations: str) -> list[str]:
    pieces = []
    start = 0
    for index, char in enumerate(text):
        if char in punctuations:
            piece = text[start : index + 1].strip()
            if piece:
                pieces.append(piece)
            start = index + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces or ([text.strip()] if text.strip() else [])


def _force_chunk(text: str, max_chars: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    return [clean[index : index + max_chars].strip() for index in range(0, len(clean), max_chars) if clean[index : index + max_chars].strip()]


def split_text_for_tts(text: str, split_config: dict | None = None) -> list[str]:
    config = dict(DEFAULT_SPLIT_CONFIG)
    config.update(split_config or {})
    target_chars = max(20, int(config.get("target_chars") or 80))
    max_chars = max(target_chars, int(config.get("max_chars") or 120))
    min_merge_chars = max(0, int(config.get("min_merge_chars") or 24))

    normalized = _normalize_text(text)
    if not normalized:
        return []

    pieces: list[str] = []
    for sentence in _split_by_punctuation(normalized, SENTENCE_ENDINGS):
        if len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for minor in _split_by_punctuation(sentence, MINOR_ENDINGS):
            if len(minor) <= max_chars:
                pieces.append(minor)
            else:
                pieces.extend(_force_chunk(minor, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        candidate = current + piece if current else piece
        if current and (len(candidate) > max_chars or len(current) >= target_chars):
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < min_merge_chars and len(chunks[-2] + chunks[-1]) <= max_chars:
        chunks[-2] += chunks[-1]
        chunks.pop()
    return chunks


def _append_segment(
    segments: list[dict],
    *,
    segment_type: str,
    speaker: str,
    text: str,
    split_config: dict,
) -> None:
    clean = _normalize_text(text)
    if not clean:
        return
    for chunk in split_text_for_tts(clean, split_config):
        if not chunk:
            continue
        previous = segments[-1] if segments else None
        if (
            previous
            and previous.get("type") == segment_type
            and previous.get("speaker") == speaker
            and len(str(previous.get("text", "")) + chunk) <= int(split_config.get("max_chars", 120))
        ):
            previous["text"] = str(previous.get("text", "")) + chunk
            continue
        segments.append(
            {
                "segment_id": f"segment_{len(segments) + 1:04d}",
                "index": len(segments) + 1,
                "type": segment_type,
                "speaker": speaker or NARRATOR_SPEAKER,
                "text": chunk,
            }
        )


def _split_non_dialogue_sentences(text: str) -> list[str]:
    pieces = []
    for sentence in _split_by_punctuation(text, SENTENCE_ENDINGS):
        if sentence.strip():
            pieces.append(sentence.strip())
    return pieces


def parse_chapter_segments(
    chapter_text: str,
    characters: dict | list[dict],
    *,
    split_config: dict | None = None,
) -> list[dict]:
    config = dict(DEFAULT_SPLIT_CONFIG)
    config.update(split_config or {})
    character_names = _character_names(characters)
    text = _clean_markdown_text(chapter_text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    segments: list[dict] = []
    previous_speaker = ""

    for paragraph in paragraphs:
        spans = _find_quote_spans(paragraph)
        if not spans:
            for sentence in _split_non_dialogue_sentences(paragraph):
                inner_speaker = _inner_monologue_speaker(sentence, character_names)
                if inner_speaker:
                    _append_segment(
                        segments,
                        segment_type="inner_monologue",
                        speaker=inner_speaker,
                        text=sentence,
                        split_config=config,
                    )
                    previous_speaker = inner_speaker
                else:
                    _append_segment(
                        segments,
                        segment_type="narration",
                        speaker=NARRATOR_SPEAKER,
                        text=sentence,
                        split_config=config,
                    )
            continue

        cursor = 0
        for start, end, quoted_text in spans:
            before = paragraph[cursor:start]
            if before.strip():
                _append_segment(
                    segments,
                    segment_type="narration",
                    speaker=NARRATOR_SPEAKER,
                    text=before,
                    split_config=config,
                )
            speaker = _resolve_quote_speaker(paragraph[:start], paragraph[end:], character_names, previous_speaker)
            _append_segment(
                segments,
                segment_type="dialogue",
                speaker=speaker,
                text=quoted_text,
                split_config=config,
            )
            previous_speaker = speaker if speaker != NARRATOR_SPEAKER else previous_speaker
            cursor = end
        tail = paragraph[cursor:]
        if tail.strip():
            _append_segment(
                segments,
                segment_type="narration",
                speaker=NARRATOR_SPEAKER,
                text=tail,
                split_config=config,
            )

    return segments


def _voice_for_segment(project_path: str | Path, voice_config: dict, segment: dict) -> dict:
    speaker = str(segment.get("speaker") or NARRATOR_SPEAKER).strip() or NARRATOR_SPEAKER
    if speaker == NARRATOR_SPEAKER:
        preset = _selected_narrator(voice_config)
        narrator_ref = voice_config.get("narrator_reference") or {}
        voice = {
            "voice_id": f"narrator:{preset.get('id', 'default')}",
            "speaker": NARRATOR_SPEAKER,
            "control_instruction": str(preset.get("control_instruction", "") or "").strip(),
            "reference_audio": str(narrator_ref.get("reference_audio", "") or "").strip(),
            "prompt_text": str(narrator_ref.get("prompt_text", "") or "").strip(),
            "cfg_value": float(narrator_ref.get("cfg_value") or DEFAULT_AUDIOBOOK_RUNTIME["cfg_value"]),
            "inference_timesteps": int(narrator_ref.get("inference_timesteps") or DEFAULT_AUDIOBOOK_RUNTIME["inference_timesteps"]),
        }
    else:
        character_voice = (voice_config.get("character_voices") or {}).get(speaker) or {}
        voice = {
            "voice_id": f"character:{speaker}",
            "speaker": speaker,
            "control_instruction": str(character_voice.get("control_instruction", "") or "").strip(),
            "reference_audio": str(character_voice.get("reference_audio", "") or "").strip(),
            "prompt_text": str(character_voice.get("prompt_text", "") or "").strip(),
            "cfg_value": float(character_voice.get("cfg_value") or DEFAULT_AUDIOBOOK_RUNTIME["cfg_value"]),
            "inference_timesteps": int(character_voice.get("inference_timesteps") or DEFAULT_AUDIOBOOK_RUNTIME["inference_timesteps"]),
        }

    if voice["reference_audio"]:
        reference_path = Path(voice["reference_audio"])
        if not reference_path.is_absolute():
            reference_path = Path(project_path) / reference_path
        voice["reference_audio"] = str(reference_path.resolve())
        voice["mode"] = "reference"
    else:
        voice["mode"] = "design"
    return voice


def _build_worker_segments(project_path: str | Path, voice_config: dict, segments: list[dict]) -> list[dict]:
    worker_segments = []
    for segment in segments:
        item = dict(segment)
        item["voice"] = _voice_for_segment(project_path, voice_config, item)
        worker_segments.append(item)
    return worker_segments


def get_audiobook_record(project_path: str | Path, chapter_slug: str) -> dict | None:
    metadata_path = _chapter_record_dir(project_path, chapter_slug) / "manifest.json"
    if not metadata_path.exists():
        return None
    try:
        manifest = load_json(str(metadata_path))
    except Exception:
        return None
    combined = str(manifest.get("combined_audio") or "")
    if combined and (Path(project_path) / combined).exists():
        return manifest
    return None


def list_audiobook_records(project_path: str | Path) -> list[dict]:
    root = _audiobook_root(project_path)
    if not root.exists():
        return []
    records = []
    for manifest_path in root.glob("chapter_*/manifest.json"):
        try:
            manifest = load_json(str(manifest_path))
        except Exception:
            continue
        combined = str(manifest.get("combined_audio") or "")
        if combined and (Path(project_path) / combined).exists():
            records.append(manifest)
    return sorted(records, key=lambda item: str(item.get("chapter_slug", "")), reverse=True)


def _existing_record(project_path: str | Path, chapter_slug: str) -> dict | None:
    record = get_audiobook_record(project_path, chapter_slug)
    if record:
        copied = dict(record)
        copied["reused"] = True
        return copied
    return None


def _worker_script_path() -> Path:
    return Path(__file__).resolve().with_name("voxcpm2_audiobook_worker.py")


def _resolve_worker_python(runtime: dict) -> str:
    return VoxCPM2Service(runtime).worker_python()


def _run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
    return VoxCPM2Service(runtime).run_worker(
        request_path,
        worker_script_path=_worker_script_path(),
        cwd=Path(__file__).resolve().parent,
    )


def generate_audiobook_chapter(
    project_path: str | Path,
    chapter_ref: str | None = None,
    *,
    force: bool = False,
    narrator_preset: str = "",
    runtime_overrides: dict | None = None,
    split_config: dict | None = None,
    progress_callback=None,
) -> dict:
    project_path = Path(project_path)
    chapter_file = _resolve_chapter_file(project_path, chapter_ref)
    chapter_slug = chapter_file.stem
    record_dir = _chapter_record_dir(project_path, chapter_slug)
    manifest_path = record_dir / "manifest.json"
    emit_progress(progress_callback, "audiobook_prepare", f"正在准备 {chapter_slug} 的有声章节")

    if not force:
        existing = _existing_record(project_path, chapter_slug)
        if existing:
            emit_progress(progress_callback, "audiobook_reused", f"{chapter_slug} 已复用已有音频")
            return existing

    voice_config = update_selected_narrator(project_path, narrator_preset) if narrator_preset else ensure_voice_config(project_path)
    project_data = load_project(str(project_path))
    split_config_resolved = dict(DEFAULT_SPLIT_CONFIG)
    split_config_resolved.update(split_config or {})
    chapter_text = chapter_file.read_text(encoding="utf-8")
    segments = parse_chapter_segments(chapter_text, project_data.get("characters") or {}, split_config=split_config_resolved)
    if not segments:
        raise RuntimeError("章节中没有可合成的文本片段。")

    runtime = load_voxcpm2_runtime(runtime_overrides)
    record_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = record_dir / "segments"
    if force and record_dir.exists():
        for old_path in record_dir.glob("*.wav"):
            old_path.unlink(missing_ok=True)
        if segments_dir.exists():
            shutil.rmtree(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "project_path": str(project_path.resolve()),
        "chapter_slug": chapter_slug,
        "chapter_file": _relative_path(project_path, chapter_file),
        "record_dir": str(record_dir.resolve()),
        "segments_dir": str(segments_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combined_audio_path": str((record_dir / f"{chapter_slug}.wav").resolve()),
        "narrator_id": str(voice_config.get("selected_narrator_id") or ""),
        "segments": _build_worker_segments(project_path, voice_config, segments),
        "runtime": runtime,
        "split_config": split_config_resolved,
    }
    request_path = record_dir / "request.json"
    save_json(str(request_path), request_payload)

    emit_progress(progress_callback, "audiobook_worker", f"正在调用 VoxCPM2 合成 {len(segments)} 个片段", current=0, total=len(segments))
    result = _run_worker(request_path, runtime)
    if result.returncode != 0:
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 有声章节生成失败，退出码 {result.returncode}。\n{detail}")
    if not manifest_path.exists():
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 worker 未生成 manifest.json。\n{detail}")

    manifest = load_json(str(manifest_path))
    manifest["reused"] = False
    emit_progress(progress_callback, "audiobook_done", f"{chapter_slug} 的有声章节已完成", current=len(segments), total=len(segments))
    return manifest


def generate_audiobook_chapters(
    project_path: str | Path,
    *,
    chapter_refs: list[str] | None = None,
    force: bool = False,
    narrator_preset: str = "",
    runtime_overrides: dict | None = None,
    progress_callback=None,
) -> list[dict]:
    refs = chapter_refs or ["latest"]
    results = []
    for index, chapter_ref in enumerate(refs):
        emit_progress(
            progress_callback,
            "audiobook_batch",
            f"正在生成第 {index + 1}/{len(refs)} 个有声章节",
            current=index,
            total=len(refs),
        )
        results.append(
            generate_audiobook_chapter(
                project_path,
                chapter_ref,
                force=force,
                narrator_preset=narrator_preset,
                runtime_overrides=runtime_overrides,
                progress_callback=progress_callback,
            )
        )
    return results


def chapter_refs_for_all(project_path: str | Path) -> list[str]:
    chapters_dir = Path(project_path) / "chapters"
    return [str(path) for path in sorted(chapters_dir.glob("chapter_*.md"))]


def audiobook_file_path(project_path: str | Path, chapter_slug: str, file_name: str) -> Path:
    root = _audiobook_root(project_path).resolve()
    file_path = (root / chapter_slug / file_name).resolve()
    if not file_path.exists() or root not in file_path.parents or file_path.suffix.lower() != ".wav":
        raise FileNotFoundError("有声章节音频不存在。")
    return file_path
