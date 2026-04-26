"""VoxCPM2-powered audiobook helpers for novel chapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common_utils import emit_progress, utc_now
from external_services import (
    AudioFrameClient,
    DEFAULT_EXTERNAL_SERVICES_CONFIG,
    DEFAULT_VOXCPM2_MODEL_ID,
    DEFAULT_VOXCPM2_PYTHON,
    DEFAULT_VOXCPM2_ROOT,
    VoxCPM2Service,
    load_audio_frame_runtime,
    load_service_config,
    load_voxcpm2_runtime,
    normalize_voxcpm2_runtime,
)
from project_manager import load_json, load_project, save_json


AUDIOBOOK_DIR_NAME = "audiobook"
VOICE_REFS_DIR_NAME = "voice_refs"
VOICES_FILENAME = "voices.json"
NARRATOR_SPEAKER = "旁白"
VOICE_CONFIG_VERSION = 2
GENERATION_MODE_ADVANCED = "advanced"
GENERATION_MODE_SIMPLE = "simple"
GENERATION_MODES = {GENERATION_MODE_ADVANCED, GENERATION_MODE_SIMPLE}

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


def normalize_generation_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in GENERATION_MODES else GENERATION_MODE_ADVANCED


def _signature(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def _reference_source(entry: dict) -> str:
    source = str((entry or {}).get("reference_source") or "").strip().lower()
    if source in {"uploaded", "auto"}:
        return source
    if str((entry or {}).get("reference_audio") or "").strip():
        return "uploaded"
    return ""


def _voice_defaults() -> dict:
    return {
        "reference_audio": "",
        "prompt_text": "",
        "reference_source": "",
        "reference_signature": "",
        "profile_signature": "",
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    }


def _reference_prompt_text(label: str) -> str:
    clean_label = _normalize_text(label) or "有声小说音色"
    return f"这是{clean_label}的有声小说参考音频，用于后续章节保持稳定音色。"


def _reference_abs_path(project_path: str | Path, reference_audio: str) -> Path:
    reference_path = Path(reference_audio)
    if not reference_path.is_absolute():
        reference_path = Path(project_path) / reference_path
    return reference_path.resolve()


def _reference_exists(project_path: str | Path, entry: dict) -> bool:
    reference_audio = str((entry or {}).get("reference_audio") or "").strip()
    return bool(reference_audio and _reference_abs_path(project_path, reference_audio).exists())


def _auto_reference_relative(voice_id: str, reference_signature: str) -> str:
    safe_voice_id = re.sub(r"[^\w\-]+", "_", voice_id, flags=re.UNICODE).strip("_") or "voice"
    return f"{AUDIOBOOK_DIR_NAME}/{VOICE_REFS_DIR_NAME}/auto_{safe_voice_id}_{reference_signature}.wav"


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
    profile_signature = _signature(
        {
            "kind": "character",
            "name": name,
            "role": role,
            "description": description,
            "appearance": appearance,
            "control_instruction": control,
        }
    )
    voice = _voice_defaults()
    voice.update(
        {
            "control_instruction": control,
            "profile_signature": profile_signature,
        }
    )
    return voice


def _narrator_profile_signature(preset: dict) -> str:
    return _signature(
        {
            "kind": "narrator",
            "id": str(preset.get("id") or "").strip(),
            "control_instruction": str(preset.get("control_instruction") or "").strip(),
        }
    )


def _default_simple_voice(preset: dict) -> dict:
    voice = _voice_defaults()
    control = str(preset.get("control_instruction") or "").strip()
    voice.update(
        {
            "control_instruction": control,
            "profile_signature": _signature(
                {
                    "kind": "simple",
                    "narrator_id": str(preset.get("id") or "").strip(),
                    "control_instruction": control,
                }
            ),
        }
    )
    return voice


def _default_narrator_reference(preset: dict) -> dict:
    voice = _voice_defaults()
    voice["profile_signature"] = _narrator_profile_signature(preset)
    return voice


def _merge_voice_entry(default_voice: dict, existing: dict | None, *, preserve_control: bool = False) -> dict:
    merged = dict(default_voice)
    if isinstance(existing, dict):
        merged.update(existing)
        if not preserve_control:
            merged["control_instruction"] = default_voice.get("control_instruction", merged.get("control_instruction", ""))
        if str(existing.get("reference_audio") or "").strip() and not str(existing.get("reference_source") or "").strip():
            merged["reference_source"] = "uploaded"
    if default_voice.get("profile_signature"):
        merged["profile_signature"] = default_voice["profile_signature"]
    merged.setdefault("reference_source", _reference_source(merged))
    merged.setdefault("reference_signature", "")
    merged.setdefault("profile_signature", default_voice.get("profile_signature", ""))
    merged.setdefault("prompt_text", "")
    merged.setdefault("reference_audio", "")
    merged.setdefault("cfg_value", default_voice.get("cfg_value", 2.0))
    merged.setdefault("inference_timesteps", default_voice.get("inference_timesteps", 10))
    return merged


def _refresh_simple_voice_for_selected_narrator(config: dict) -> None:
    preset = _selected_narrator(config)
    existing = config.get("simple_voice") if isinstance(config.get("simple_voice"), dict) else {}
    config["simple_voice"] = _merge_voice_entry(_default_simple_voice(preset), existing)


def _default_voice_config(project_data: dict) -> dict:
    characters = _all_characters(project_data.get("characters") or {})
    selected_preset = NARRATOR_PRESETS[0]
    return {
        "version": VOICE_CONFIG_VERSION,
        "updated_at": utc_now(),
        "generation_mode": GENERATION_MODE_ADVANCED,
        "selected_narrator_id": selected_preset["id"],
        "narrator_presets": NARRATOR_PRESETS,
        "narrator_reference": _default_narrator_reference(selected_preset),
        "simple_voice": _default_simple_voice(selected_preset),
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
    config["version"] = VOICE_CONFIG_VERSION
    config["generation_mode"] = normalize_generation_mode(config.get("generation_mode") or defaults["generation_mode"])
    config.setdefault("selected_narrator_id", defaults["selected_narrator_id"])
    config["narrator_presets"] = defaults["narrator_presets"]
    config["narrator_reference"] = _merge_voice_entry(
        _default_narrator_reference(_selected_narrator(config)),
        config.get("narrator_reference") if isinstance(config.get("narrator_reference"), dict) else {},
        preserve_control=True,
    )
    _refresh_simple_voice_for_selected_narrator(config)
    config.setdefault("character_voices", {})
    for name, default_voice in defaults["character_voices"].items():
        config["character_voices"][name] = _merge_voice_entry(default_voice, config["character_voices"].get(name))
    config["updated_at"] = utc_now()
    save_json(str(path), config)
    return config


def save_voice_config(project_path: str | Path, config: dict) -> dict:
    config = dict(config)
    config["version"] = VOICE_CONFIG_VERSION
    config["generation_mode"] = normalize_generation_mode(config.get("generation_mode"))
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
        config["narrator_reference"] = _merge_voice_entry(
            _default_narrator_reference(_selected_narrator(config)),
            config.get("narrator_reference") if isinstance(config.get("narrator_reference"), dict) else {},
            preserve_control=True,
        )
        _refresh_simple_voice_for_selected_narrator(config)
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
        "reference_source": "uploaded",
        "reference_signature": hashlib.sha1(uploaded_file.content).hexdigest()[:16],
    }
    if normalized_target == "narrator":
        config.setdefault("narrator_reference", {})
        config["narrator_reference"].update(payload)
    elif normalized_target == "simple":
        config.setdefault("simple_voice", {})
        config["simple_voice"].update(payload)
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


def _voice_float(entry: dict, key: str) -> float:
    try:
        return float((entry or {}).get(key) or DEFAULT_AUDIOBOOK_RUNTIME[key])
    except (TypeError, ValueError, KeyError):
        return float(DEFAULT_AUDIOBOOK_RUNTIME.get(key, 2.0))


def _voice_int(entry: dict, key: str) -> int:
    try:
        return int((entry or {}).get(key) or DEFAULT_AUDIOBOOK_RUNTIME[key])
    except (TypeError, ValueError, KeyError):
        return int(DEFAULT_AUDIOBOOK_RUNTIME.get(key, 10))


def _voice_reference_plan(
    project_path: str | Path,
    *,
    target_type: str,
    target_name: str,
    voice_id: str,
    label: str,
    control_instruction: str,
    entry: dict,
    profile_signature: str,
) -> tuple[dict, dict, dict | None]:
    entry = entry if isinstance(entry, dict) else {}
    source = _reference_source(entry)
    cfg_value = _voice_float(entry, "cfg_value")
    inference_timesteps = _voice_int(entry, "inference_timesteps")
    reference_prompt = str(entry.get("prompt_text") or "").strip() or _reference_prompt_text(label)
    reference_signature = _signature(
        {
            "voice_id": voice_id,
            "profile_signature": profile_signature,
            "prompt_text": reference_prompt,
            "control_instruction": control_instruction,
        }
    )

    task: dict | None = None
    if source == "uploaded":
        relative_reference = str(entry.get("reference_audio") or "").strip()
        if not relative_reference:
            raise RuntimeError(f"{label} 的上传参考音频路径为空，无法生成有声章节。")
        reference_path = _reference_abs_path(project_path, relative_reference)
        if not reference_path.exists():
            raise RuntimeError(f"{label} 的上传参考音频不存在：{relative_reference}")
        reference_signature = str(entry.get("reference_signature") or reference_signature)
    else:
        planned_relative = _auto_reference_relative(voice_id, reference_signature)
        entry_relative = str(entry.get("reference_audio") or "").strip()
        if (
            source == "auto"
            and entry_relative
            and str(entry.get("profile_signature") or "") == profile_signature
            and str(entry.get("reference_signature") or "") == reference_signature
            and _reference_exists(project_path, entry)
        ):
            relative_reference = entry_relative
        elif (Path(project_path) / planned_relative).exists():
            relative_reference = planned_relative
        else:
            relative_reference = planned_relative
            task = {
                "voice_id": voice_id,
                "target_type": target_type,
                "target_name": target_name,
                "label": label,
                "control_instruction": control_instruction,
                "reference_audio": str(_reference_abs_path(project_path, relative_reference)),
                "reference_audio_relative": relative_reference,
                "prompt_text": reference_prompt,
                "reference_source": "auto",
                "reference_signature": reference_signature,
                "profile_signature": profile_signature,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            }
        source = "auto"

    reference_path = _reference_abs_path(project_path, relative_reference)
    voice = {
        "voice_id": voice_id,
        "speaker": target_name or label,
        "control_instruction": control_instruction,
        "reference_audio": str(reference_path),
        "prompt_text": reference_prompt if source == "auto" else str(entry.get("prompt_text") or "").strip(),
        "reference_source": source,
        "reference_signature": reference_signature,
        "profile_signature": profile_signature,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
        "mode": "reference",
    }
    record = {
        "voice_id": voice_id,
        "target_type": target_type,
        "target_name": target_name,
        "label": label,
        "reference_audio": relative_reference,
        "prompt_text": voice["prompt_text"],
        "reference_source": source,
        "reference_signature": reference_signature,
        "profile_signature": profile_signature,
        "control_instruction": control_instruction,
        "generated": bool(task),
    }
    return voice, record, task


def _simple_voice_descriptor(project_path: str | Path, voice_config: dict) -> tuple[dict, dict, dict | None]:
    preset = _selected_narrator(voice_config)
    narrator_ref = voice_config.get("narrator_reference") if isinstance(voice_config.get("narrator_reference"), dict) else {}
    if _reference_source(narrator_ref) == "uploaded":
        entry = narrator_ref
        target_type = "narrator"
        target_name = NARRATOR_SPEAKER
        voice_id = f"simple:{preset.get('id', 'default')}:uploaded_narrator"
    else:
        entry = voice_config.get("simple_voice") if isinstance(voice_config.get("simple_voice"), dict) else {}
        target_type = "simple"
        target_name = "统一音色"
        voice_id = f"simple:{preset.get('id', 'default')}"
    control = str((entry or {}).get("control_instruction") or preset.get("control_instruction") or "").strip()
    profile_signature = _signature(
        {
            "kind": "simple",
            "narrator_id": str(preset.get("id") or "").strip(),
            "control_instruction": control,
        }
    )
    return _voice_reference_plan(
        project_path,
        target_type=target_type,
        target_name=target_name,
        voice_id=voice_id,
        label="统一音色",
        control_instruction=control,
        entry=entry,
        profile_signature=profile_signature,
    )


def _advanced_voice_descriptor(
    project_path: str | Path,
    voice_config: dict,
    speaker: str,
) -> tuple[dict, dict, dict | None]:
    if speaker == NARRATOR_SPEAKER:
        preset = _selected_narrator(voice_config)
        entry = voice_config.get("narrator_reference") if isinstance(voice_config.get("narrator_reference"), dict) else {}
        control = str(preset.get("control_instruction") or "").strip()
        return _voice_reference_plan(
            project_path,
            target_type="narrator",
            target_name=NARRATOR_SPEAKER,
            voice_id=f"narrator:{preset.get('id', 'default')}",
            label=NARRATOR_SPEAKER,
            control_instruction=control,
            entry=entry,
            profile_signature=_narrator_profile_signature(preset),
        )

    character_voice = (voice_config.get("character_voices") or {}).get(speaker) or {}
    if not isinstance(character_voice, dict):
        character_voice = {}
    control = str(character_voice.get("control_instruction") or f"中文有声小说角色音色，符合{speaker}的人物设定，台词自然，情绪贴合剧情").strip()
    profile_signature = str(character_voice.get("profile_signature") or "") or _signature(
        {"kind": "character", "name": speaker, "control_instruction": control}
    )
    return _voice_reference_plan(
        project_path,
        target_type="character",
        target_name=speaker,
        voice_id=f"character:{speaker}",
        label=speaker,
        control_instruction=control,
        entry=character_voice,
        profile_signature=profile_signature,
    )


def _build_voice_plan(
    project_path: str | Path,
    voice_config: dict,
    segments: list[dict],
    generation_mode: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    mode = normalize_generation_mode(generation_mode)
    voice_cache: dict[str, dict] = {}
    reference_records: dict[str, dict] = {}
    reference_tasks: dict[str, dict] = {}

    if mode == GENERATION_MODE_SIMPLE:
        voice, record, task = _simple_voice_descriptor(project_path, voice_config)
        voice_cache["__simple__"] = voice
        reference_records[voice["voice_id"]] = record
        if task:
            reference_tasks[voice["voice_id"]] = task

    worker_segments = []
    for segment in segments:
        item = dict(segment)
        if mode == GENERATION_MODE_SIMPLE:
            voice = voice_cache["__simple__"]
        else:
            speaker = str(item.get("speaker") or NARRATOR_SPEAKER).strip() or NARRATOR_SPEAKER
            cache_key = speaker
            if cache_key not in voice_cache:
                voice, record, task = _advanced_voice_descriptor(project_path, voice_config, speaker)
                voice_cache[cache_key] = voice
                reference_records[voice["voice_id"]] = record
                if task:
                    reference_tasks[voice["voice_id"]] = task
            voice = voice_cache[cache_key]
        item["voice"] = dict(voice)
        worker_segments.append(item)

    return worker_segments, list(reference_records.values()), list(reference_tasks.values())


def _apply_voice_reference_records(project_path: str | Path, voice_config: dict, voice_references: list[dict]) -> dict:
    changed = False
    for reference in voice_references or []:
        if str(reference.get("reference_source") or "").strip() != "auto":
            continue
        payload = {
            "reference_audio": str(reference.get("reference_audio") or "").strip(),
            "prompt_text": str(reference.get("prompt_text") or "").strip(),
            "reference_source": "auto",
            "reference_signature": str(reference.get("reference_signature") or "").strip(),
            "profile_signature": str(reference.get("profile_signature") or "").strip(),
        }
        if not payload["reference_audio"]:
            continue
        target_type = str(reference.get("target_type") or "").strip()
        target_name = str(reference.get("target_name") or "").strip()
        if target_type == "narrator":
            entry = voice_config.setdefault("narrator_reference", {})
        elif target_type == "simple":
            entry = voice_config.setdefault("simple_voice", {})
        elif target_type == "character" and target_name:
            entry = voice_config.setdefault("character_voices", {}).setdefault(target_name, {})
        else:
            continue
        if _reference_source(entry) == "uploaded":
            continue
        entry.update(payload)
        changed = True
    if changed:
        return save_voice_config(project_path, voice_config)
    return voice_config


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


def _audiobook_backend(runtime_overrides: dict | None = None) -> str:
    configured = str(
        (runtime_overrides or {}).get("backend")
        or (runtime_overrides or {}).get("audiobook_backend")
        or load_service_config("audiobook", include_defaults=True).get("backend")
        or "local_worker"
    ).strip().lower()
    return configured if configured in {"local_worker", "audio_frame"} else "local_worker"


def _audio_frame_overrides(runtime_overrides: dict | None) -> dict:
    raw = runtime_overrides or {}
    return {
        key: value
        for key, value in {
            "api_base": raw.get("audio_frame_api_base"),
            "timeout": raw.get("audio_frame_timeout"),
        }.items()
        if value not in (None, "")
    }


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        frame_count = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        return float(frame_count) / float(frame_rate) if frame_rate else 0.0


def _combine_wav_files(paths: list[Path], output_path: Path, *, silence_ms: int = 260) -> None:
    if not paths:
        raise RuntimeError("no audio chunks were generated")

    params = None
    chunks: list[bytes] = []
    for index, path in enumerate(paths):
        with wave.open(str(path), "rb") as wav_file:
            current_params = wav_file.getparams()
            if params is None:
                params = current_params
            elif (
                current_params.nchannels != params.nchannels
                or current_params.sampwidth != params.sampwidth
                or current_params.framerate != params.framerate
                or current_params.comptype != params.comptype
            ):
                raise RuntimeError("Audio Frame returned WAV segments with incompatible formats.")
            chunks.append(wav_file.readframes(wav_file.getnframes()))
            if index != len(paths) - 1 and silence_ms > 0:
                silence_frames = int(current_params.framerate * silence_ms / 1000)
                chunks.append(b"\x00" * silence_frames * current_params.nchannels * current_params.sampwidth)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out_file:
        out_file.setparams(params)
        for chunk in chunks:
            out_file.writeframes(chunk)


def _run_audio_frame_service(request_payload: dict, runtime_overrides: dict | None, progress_callback=None) -> None:
    runtime = load_audio_frame_runtime(_audio_frame_overrides(runtime_overrides))
    client = AudioFrameClient(runtime["api_base"])
    project_path = Path(request_payload["project_path"])
    segments_dir = Path(request_payload["segments_dir"])
    combined_audio_path = Path(request_payload["combined_audio_path"])
    manifest_path = Path(request_payload["manifest_path"])
    segments_dir.mkdir(parents=True, exist_ok=True)

    voice_references = [dict(item) for item in request_payload.get("voice_references") or []]
    reference_by_voice_id = {str(item.get("voice_id") or ""): item for item in voice_references}
    reference_tasks = request_payload.get("voice_reference_tasks") or []
    for index, task in enumerate(reference_tasks, start=1):
        emit_progress(
            progress_callback,
            "audiobook_voice_reference",
            f"Audio Frame 正在生成参考音色 {index}/{len(reference_tasks)}",
            current=index - 1,
            total=len(reference_tasks),
        )
        response = client.synthesize(
            text=str(task.get("prompt_text") or ""),
            control_instruction=str(task.get("control_instruction") or ""),
            reference_audio="",
            prompt_text="",
            cfg_value=float(task.get("cfg_value") or request_payload["runtime"].get("cfg_value") or 2.0),
            normalize=bool(request_payload["runtime"].get("normalize", True)),
            denoise=False,
            inference_timesteps=int(
                task.get("inference_timesteps") or request_payload["runtime"].get("inference_timesteps") or 10
            ),
            timeout=int(runtime.get("timeout") or 0),
        )
        reference_path = Path(str(task.get("reference_audio") or "")).resolve()
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_bytes(base64.b64decode(str(response.get("audio_base64") or "")))
        record = reference_by_voice_id.get(str(task.get("voice_id") or ""))
        if record is not None:
            record["generated"] = True
            record["duration_seconds"] = round(_wav_duration_seconds(reference_path), 3)

    generated_paths: list[Path] = []
    manifest_segments = []
    segments = request_payload.get("segments") or []
    for index, item in enumerate(segments, start=1):
        voice = item.get("voice") or {}
        emit_progress(
            progress_callback,
            "audiobook_audio_frame",
            f"Audio Frame 正在合成 {index}/{len(segments)}",
            current=index - 1,
            total=len(segments),
        )
        response = client.synthesize(
            text=str(item.get("text") or ""),
            control_instruction=str(voice.get("control_instruction") or ""),
            reference_audio=str(voice.get("reference_audio") or ""),
            prompt_text=str(voice.get("prompt_text") or ""),
            cfg_value=float(voice.get("cfg_value") or request_payload["runtime"].get("cfg_value") or 2.0),
            normalize=bool(request_payload["runtime"].get("normalize", True)),
            denoise=bool(request_payload["runtime"].get("denoise", False)),
            inference_timesteps=int(
                voice.get("inference_timesteps") or request_payload["runtime"].get("inference_timesteps") or 10
            ),
            timeout=int(runtime.get("timeout") or 0),
        )
        audio_bytes = base64.b64decode(str(response.get("audio_base64") or ""))
        local_path = segments_dir / f"{item.get('segment_id', f'segment_{index:04d}')}.wav"
        local_path.write_bytes(audio_bytes)
        item = dict(item)
        item["audio_file"] = _relative_path(project_path, local_path)
        item["duration_seconds"] = round(_wav_duration_seconds(local_path), 3)
        manifest_segments.append(item)
        generated_paths.append(local_path)

    _combine_wav_files(
        generated_paths,
        combined_audio_path,
        silence_ms=int(request_payload["runtime"].get("silence_ms") or DEFAULT_AUDIOBOOK_RUNTIME["silence_ms"]),
    )
    save_json(
        str(manifest_path),
        {
            "chapter_slug": request_payload["chapter_slug"],
            "chapter_file": request_payload["chapter_file"],
            "generated_at": utc_now(),
            "status": "succeeded",
            "combined_audio": _relative_path(project_path, combined_audio_path),
            "segment_count": len(manifest_segments),
            "segments": manifest_segments,
            "narrator_id": request_payload.get("narrator_id", ""),
            "generation_mode": request_payload.get("generation_mode", GENERATION_MODE_ADVANCED),
            "voice_references": voice_references,
            "split_config": request_payload.get("split_config", {}),
            "audio_frame": {
                "api_base": runtime["api_base"],
            },
        },
    )


def generate_audiobook_chapter(
    project_path: str | Path,
    chapter_ref: str | None = None,
    *,
    force: bool = False,
    narrator_preset: str = "",
    generation_mode: str = GENERATION_MODE_ADVANCED,
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

    resolved_generation_mode = normalize_generation_mode(generation_mode)
    voice_config = update_selected_narrator(project_path, narrator_preset) if narrator_preset else ensure_voice_config(project_path)
    if voice_config.get("generation_mode") != resolved_generation_mode:
        voice_config["generation_mode"] = resolved_generation_mode
        voice_config = save_voice_config(project_path, voice_config)
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
    worker_segments, voice_references, voice_reference_tasks = _build_voice_plan(
        project_path,
        voice_config,
        segments,
        resolved_generation_mode,
    )

    request_payload = {
        "project_path": str(project_path.resolve()),
        "chapter_slug": chapter_slug,
        "chapter_file": _relative_path(project_path, chapter_file),
        "record_dir": str(record_dir.resolve()),
        "segments_dir": str(segments_dir.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "combined_audio_path": str((record_dir / f"{chapter_slug}.wav").resolve()),
        "narrator_id": str(voice_config.get("selected_narrator_id") or ""),
        "generation_mode": resolved_generation_mode,
        "segments": worker_segments,
        "voice_references": voice_references,
        "voice_reference_tasks": voice_reference_tasks,
        "runtime": runtime,
        "split_config": split_config_resolved,
    }
    request_path = record_dir / "request.json"
    save_json(str(request_path), request_payload)

    emit_progress(progress_callback, "audiobook_worker", f"正在调用 VoxCPM2 合成 {len(segments)} 个片段", current=0, total=len(segments))
    if _audiobook_backend(runtime_overrides) == "audio_frame":
        _run_audio_frame_service(request_payload, runtime_overrides, progress_callback=progress_callback)
        result = subprocess.CompletedProcess(["audio_frame"], 0, stdout="ok", stderr="")
    else:
        result = _run_worker(request_path, runtime)
    if result.returncode != 0:
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 有声章节生成失败，退出码 {result.returncode}。\n{detail}")
    if not manifest_path.exists():
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 worker 未生成 manifest.json。\n{detail}")

    manifest = load_json(str(manifest_path))
    voice_config = _apply_voice_reference_records(project_path, voice_config, manifest.get("voice_references") or [])
    manifest["reused"] = False
    emit_progress(progress_callback, "audiobook_done", f"{chapter_slug} 的有声章节已完成", current=len(segments), total=len(segments))
    return manifest


def generate_audiobook_chapters(
    project_path: str | Path,
    *,
    chapter_refs: list[str] | None = None,
    force: bool = False,
    narrator_preset: str = "",
    generation_mode: str = GENERATION_MODE_ADVANCED,
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
                generation_mode=generation_mode,
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
