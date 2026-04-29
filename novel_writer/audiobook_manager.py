"""VoxCPM2-powered audiobook helpers for novel chapters."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from common_utils import emit_progress, extract_json_object, utc_now
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
from llm_client import generate_text_with_metadata
from project_manager import acquire_project_audio_lock, load_json, load_project, save_json, update_project_stats


AUDIOBOOK_DIR_NAME = "audiobook"
VOICE_REFS_DIR_NAME = "voice_refs"
VOICES_FILENAME = "voices.json"
NARRATOR_SPEAKER = "旁白"
VOICE_CONFIG_VERSION = 2
GENERATION_MODE_ADVANCED = "advanced"
GENERATION_MODE_SIMPLE = "simple"
GENERATION_MODES = {GENERATION_MODE_ADVANCED, GENERATION_MODE_SIMPLE}
VOICE_SEED_MODULUS = 2_147_483_647
CLONE_MODE_STYLE_CONTROL = "style_control"
CLONE_MODE_HIFI = "hifi"
CLONE_MODES = {CLONE_MODE_STYLE_CONTROL, CLONE_MODE_HIFI}
SEGMENT_TYPE_NARRATION = "narration"
SEGMENT_TYPE_DIALOGUE = "dialogue"
SEGMENT_TYPE_INNER_MONOLOGUE = "inner_monologue"
SEGMENT_TYPE_QUOTED_TEXT = "quoted_text"
AUDIOBOOK_SEGMENT_TYPES = {
    SEGMENT_TYPE_NARRATION,
    SEGMENT_TYPE_DIALOGUE,
    SEGMENT_TYPE_INNER_MONOLOGUE,
    SEGMENT_TYPE_QUOTED_TEXT,
}
LLM_SEGMENT_PROMPT_TARGET_CHARS = 12000

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


@dataclass(frozen=True)
class AudioFrameEndpointSlot:
    slot_id: str
    api_base: str
    kind: str
    speed: float
    max_chars: int = 0


@dataclass(frozen=True)
class AudioFrameSegmentJob:
    payload_index: int
    position: int
    item: dict
    text_chars: int


@dataclass
class PreparedAudiobookChapter:
    project_path: Path
    chapter_slug: str
    manifest_path: Path
    request_payload: dict | None
    voice_config: dict | None = None
    reused_manifest: dict | None = None


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


def normalize_clone_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in CLONE_MODES else CLONE_MODE_STYLE_CONTROL


def _signature(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def _voice_seed(*parts: object) -> int:
    joined = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % VOICE_SEED_MODULUS


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


def _normalize_performance_instruction(value: object, *, limit: int = 96) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    text = re.sub(r"^[（(]+|[）)]+$", "", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip("，,、；;。 ")
    return text


def _append_segment(
    segments: list[dict],
    *,
    segment_type: str,
    speaker: str,
    text: str,
    split_config: dict,
    performance_instruction: str = "",
    emotion: str = "",
    tone: str = "",
    delivery: str = "",
) -> None:
    clean = _normalize_text(text)
    if not clean:
        return
    performance = _normalize_performance_instruction(performance_instruction)
    emotion = _normalize_text(emotion)[:40]
    tone = _normalize_text(tone)[:40]
    delivery = _normalize_text(delivery)[:40]
    for chunk in split_text_for_tts(clean, split_config):
        if not chunk:
            continue
        previous = segments[-1] if segments else None
        if (
            previous
            and previous.get("type") == segment_type
            and previous.get("speaker") == speaker
            and str(previous.get("performance_instruction") or "") == performance
            and len(str(previous.get("text", "")) + chunk) <= int(split_config.get("max_chars", 120))
        ):
            previous["text"] = str(previous.get("text", "")) + chunk
            continue
        segment = {
            "segment_id": f"segment_{len(segments) + 1:04d}",
            "index": len(segments) + 1,
            "type": segment_type,
            "speaker": speaker or NARRATOR_SPEAKER,
            "text": chunk,
        }
        if performance:
            segment["performance_instruction"] = performance
        if emotion:
            segment["emotion"] = emotion
        if tone:
            segment["tone"] = tone
        if delivery:
            segment["delivery"] = delivery
        segments.append(segment)


def _split_non_dialogue_sentences(text: str) -> list[str]:
    pieces = []
    for sentence in _split_by_punctuation(text, SENTENCE_ENDINGS):
        if sentence.strip():
            pieces.append(sentence.strip())
    return pieces


def _character_brief(characters: dict | list[dict]) -> list[dict]:
    items = characters if isinstance(characters, list) else _all_characters(characters)
    briefs = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            continue
        briefs.append(
            {
                "name": name,
                "role": str(item.get("role", "") or "").strip(),
                "description": _normalize_text(item.get("description", ""))[:120],
            }
        )
    return briefs


def _make_text_unit(
    units: list[dict],
    *,
    kind: str,
    text: str,
    paragraph: str,
    paragraph_index: int,
    rule_type: str,
    rule_speaker: str,
    before: str = "",
    after: str = "",
) -> None:
    clean = _normalize_text(text)
    if not clean:
        return
    units.append(
        {
            "id": f"unit_{len(units) + 1:04d}",
            "kind": kind,
            "text": clean,
            "paragraph_index": paragraph_index,
            "paragraph": _normalize_text(paragraph)[:600],
            "before": _normalize_text(before)[-160:],
            "after": _normalize_text(after)[:160],
            "rule_type": rule_type,
            "rule_speaker": rule_speaker or NARRATOR_SPEAKER,
        }
    )


def _build_segment_units(chapter_text: str, characters: dict | list[dict]) -> list[dict]:
    character_names = _character_names(characters)
    text = _clean_markdown_text(chapter_text)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[dict] = []
    previous_speaker = ""

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        spans = _find_quote_spans(paragraph)
        if not spans:
            for sentence in _split_non_dialogue_sentences(paragraph):
                inner_speaker = _inner_monologue_speaker(sentence, character_names)
                if inner_speaker:
                    _make_text_unit(
                        units,
                        kind="text",
                        text=sentence,
                        paragraph=paragraph,
                        paragraph_index=paragraph_index,
                        rule_type=SEGMENT_TYPE_INNER_MONOLOGUE,
                        rule_speaker=inner_speaker,
                    )
                    previous_speaker = inner_speaker
                else:
                    _make_text_unit(
                        units,
                        kind="text",
                        text=sentence,
                        paragraph=paragraph,
                        paragraph_index=paragraph_index,
                        rule_type=SEGMENT_TYPE_NARRATION,
                        rule_speaker=NARRATOR_SPEAKER,
                    )
            continue

        cursor = 0
        for start, end, quoted_text in spans:
            before = paragraph[cursor:start]
            if before.strip():
                _make_text_unit(
                    units,
                    kind="text",
                    text=before,
                    paragraph=paragraph,
                    paragraph_index=paragraph_index,
                    rule_type=SEGMENT_TYPE_NARRATION,
                    rule_speaker=NARRATOR_SPEAKER,
                )
            speaker = _resolve_quote_speaker(paragraph[:start], paragraph[end:], character_names, previous_speaker)
            _make_text_unit(
                units,
                kind="quote",
                text=quoted_text,
                paragraph=paragraph,
                paragraph_index=paragraph_index,
                rule_type=SEGMENT_TYPE_DIALOGUE,
                rule_speaker=speaker,
                before=paragraph[:start],
                after=paragraph[end:],
            )
            previous_speaker = speaker if speaker != NARRATOR_SPEAKER else previous_speaker
            cursor = end
        tail = paragraph[cursor:]
        if tail.strip():
            _make_text_unit(
                units,
                kind="text",
                text=tail,
                paragraph=paragraph,
                paragraph_index=paragraph_index,
                rule_type=SEGMENT_TYPE_NARRATION,
                rule_speaker=NARRATOR_SPEAKER,
            )

    return units


def _append_units_with_assignments(
    units: list[dict],
    assignments_by_id: dict[str, dict],
    character_names: list[str],
    split_config: dict,
) -> list[dict]:
    aliases = {
        "narrator": SEGMENT_TYPE_NARRATION,
        "narration": SEGMENT_TYPE_NARRATION,
        "旁白": SEGMENT_TYPE_NARRATION,
        "dialog": SEGMENT_TYPE_DIALOGUE,
        "dialogue": SEGMENT_TYPE_DIALOGUE,
        "speech": SEGMENT_TYPE_DIALOGUE,
        "台词": SEGMENT_TYPE_DIALOGUE,
        "人物语言": SEGMENT_TYPE_DIALOGUE,
        "inner": SEGMENT_TYPE_INNER_MONOLOGUE,
        "inner_monologue": SEGMENT_TYPE_INNER_MONOLOGUE,
        "thought": SEGMENT_TYPE_INNER_MONOLOGUE,
        "心理活动": SEGMENT_TYPE_INNER_MONOLOGUE,
        "quoted_text": SEGMENT_TYPE_QUOTED_TEXT,
        "quote": SEGMENT_TYPE_QUOTED_TEXT,
        "citation": SEGMENT_TYPE_QUOTED_TEXT,
        "引用": SEGMENT_TYPE_QUOTED_TEXT,
    }

    def match_character(value: str) -> str:
        clean = str(value or "").strip().strip("：:，,。.!?！？“”\"'「」『』（）()[]【】")
        if clean in character_names:
            return clean
        matches = [name for name in character_names if name and name in clean]
        return matches[0] if len(matches) == 1 else ""

    segments: list[dict] = []
    for unit in units:
        assignment = assignments_by_id.get(str(unit.get("id") or ""), {})
        raw_type = str(assignment.get("type") or assignment.get("segment_type") or unit.get("rule_type") or "").strip()
        segment_type = aliases.get(raw_type.lower()) or aliases.get(raw_type) or str(unit.get("rule_type") or "")
        if segment_type not in AUDIOBOOK_SEGMENT_TYPES:
            segment_type = str(unit.get("rule_type") or SEGMENT_TYPE_NARRATION)

        raw_speaker = str(assignment.get("speaker") or "").strip()
        rule_speaker = str(unit.get("rule_speaker") or NARRATOR_SPEAKER).strip()
        if segment_type in {SEGMENT_TYPE_NARRATION, SEGMENT_TYPE_QUOTED_TEXT}:
            speaker = NARRATOR_SPEAKER
        elif match_character(raw_speaker):
            speaker = match_character(raw_speaker)
        elif match_character(rule_speaker):
            speaker = match_character(rule_speaker)
        else:
            speaker = NARRATOR_SPEAKER

        emotion = str(assignment.get("emotion") or "").strip()
        tone = str(assignment.get("tone") or "").strip()
        delivery = str(assignment.get("delivery") or "").strip()
        performance = str(
            assignment.get("performance_instruction")
            or assignment.get("style_instruction")
            or assignment.get("delivery_instruction")
            or ""
        ).strip()
        if not performance:
            performance = "，".join(item for item in (emotion, tone, delivery) if item)

        _append_segment(
            segments,
            segment_type=segment_type,
            speaker=speaker,
            text=str(unit.get("text") or ""),
            split_config=split_config,
            performance_instruction=performance,
            emotion=emotion,
            tone=tone,
            delivery=delivery,
        )
    return segments


def _build_audiobook_segment_prompt(units: list[dict], characters: dict | list[dict]) -> str:
    payload = {
        "task": "classify_audiobook_text_units",
        "allowed_types": [
            SEGMENT_TYPE_NARRATION,
            SEGMENT_TYPE_DIALOGUE,
            SEGMENT_TYPE_INNER_MONOLOGUE,
            SEGMENT_TYPE_QUOTED_TEXT,
        ],
        "speaker_rules": {
            SEGMENT_TYPE_NARRATION: "speaker 必须为 旁白",
            SEGMENT_TYPE_QUOTED_TEXT: "speaker 必须为 旁白；用于格言、标语、标题、转述原文、文件内容等只是被引号包住的引用，不是角色正在说话",
            SEGMENT_TYPE_DIALOGUE: "speaker 必须为 characters 中实际正在说话的人物名",
            SEGMENT_TYPE_INNER_MONOLOGUE: "speaker 必须为正在思考/心理活动的人物名",
        },
        "characters": _character_brief(characters),
        "units": [
            {
                "id": unit["id"],
                "kind": unit["kind"],
                "text": unit["text"],
                "paragraph_index": unit["paragraph_index"],
                "before": unit.get("before", ""),
                "after": unit.get("after", ""),
                "paragraph": unit.get("paragraph", ""),
                "rule_guess": {
                    "type": unit.get("rule_type", ""),
                    "speaker": unit.get("rule_speaker", ""),
                },
            }
            for unit in units
        ],
        "return_schema": {
            "segments": [
                {
                    "id": "unit_0001",
                    "type": "narration | dialogue | inner_monologue | quoted_text",
                    "speaker": "旁白 or exact character name",
                    "emotion": "短情绪标签，如 紧张、迟疑、冷静、疲惫；不确定可留空",
                    "tone": "短语气标签，如 低声、克制、急促、温柔；不确定可留空",
                    "delivery": "短朗读方式，如 语速略快、停顿明显、压低声音；不确定可留空",
                    "performance_instruction": "适合放入 TTS 控制括号的中文短语，合并 emotion/tone/delivery，不要超过 40 字",
                    "confidence": 0.0,
                    "reason": "short Chinese reason",
                }
            ]
        },
    }
    return (
        "你是有声小说分镜和说话人识别器。请只根据上下文判断每个 unit 的朗读归属，不要改写文本。\n"
        "重点区分：旁白、人物语言、人物心理活动、以及只是被引号包住的引用文本。\n"
        "只有角色在现场实际开口说话时才标为 dialogue；心想、暗想、意识流、内心判断标为 inner_monologue；"
        "标语、书名、文件内容、被回忆或转述的原句、术语强调等不是现场发言的引号内容标为 quoted_text。\n"
        "同时为每个 unit 给出简短表演提示：情绪、语气、朗读方式。提示必须来自文本上下文，避免夸张，不要加入角色名。\n"
        "输出必须是 JSON object，且 segments 覆盖每一个输入 id。\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _llm_unit_batches(units: list[dict], *, target_chars: int = LLM_SEGMENT_PROMPT_TARGET_CHARS) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for unit in units:
        unit_chars = len(json.dumps(unit, ensure_ascii=False))
        if current and current_chars + unit_chars > target_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += unit_chars
    if current:
        batches.append(current)
    return batches


def _merge_llm_metadata(metadata_items: list[dict]) -> dict:
    if not metadata_items:
        return {}
    merged = dict(metadata_items[-1])
    usage: dict[str, int] = {}
    for metadata in metadata_items:
        item_usage = metadata.get("usage") if isinstance(metadata, dict) else {}
        if not isinstance(item_usage, dict):
            continue
        for key, value in item_usage.items():
            try:
                usage[key] = int(usage.get(key, 0)) + int(value or 0)
            except (TypeError, ValueError):
                continue
    if usage:
        merged["usage"] = usage
    return merged


def _llm_config_available(config: dict | None) -> bool:
    if not isinstance(config, dict) or not config.get("model_provider"):
        return False
    provider = str(config.get("model_provider") or "").strip().lower()
    if provider in {"ollama", "llama_cpp"}:
        return bool(config.get("api_base") and (config.get("model") or config.get("model_name")))
    if str(config.get("api_key") or "").strip():
        return True
    api_base = str(config.get("api_base") or "").strip().lower()
    return provider == "openai_compatible" and (
        "127.0.0.1" in api_base or "localhost" in api_base or "0.0.0.0" in api_base
    )


def _parse_segments_with_llm(
    chapter_text: str,
    characters: dict | list[dict],
    *,
    llm_config: dict,
    split_config: dict,
) -> tuple[list[dict], dict]:
    units = _build_segment_units(chapter_text, characters)
    if not units:
        return [], {}
    assignments_by_id: dict[str, dict] = {}
    metadata_items: list[dict] = []
    for batch_index, batch in enumerate(_llm_unit_batches(units), start=1):
        response_text, metadata = generate_text_with_metadata(
            _build_audiobook_segment_prompt(batch, characters),
            llm_config,
            log_context={"phase": "audiobook", "task": "segment_classification", "batch_index": batch_index},
            system_prompt=(
                "你只输出 JSON，不输出解释。你擅长中文小说文本中旁白、台词、心理活动、引号引用的精确归类。"
            ),
            response_format="json",
        )
        metadata_items.append(metadata)
        payload = extract_json_object(response_text, "Could not parse JSON from audiobook segment classification response.")
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            raise ValueError("Audiobook segment classification response is missing segments.")
        assignments_by_id.update(
            {
                str(item.get("id") or "").strip(): item
                for item in raw_segments
                if isinstance(item, dict) and str(item.get("id") or "").strip()
            }
        )
    if not assignments_by_id:
        raise ValueError("Audiobook segment classification response did not include any usable segment ids.")
    return _append_units_with_assignments(
        units,
        assignments_by_id,
        _character_names(characters),
        split_config,
    ), _merge_llm_metadata(metadata_items)


def _update_audiobook_llm_stats(project_path: str | Path | None, *, success: bool, metadata: dict | None = None) -> None:
    if not project_path:
        return
    try:
        update_project_stats(
            str(project_path),
            phase="audiobook",
            success=success,
            usage=(metadata or {}).get("usage") if isinstance(metadata, dict) else None,
            metadata=metadata,
        )
    except Exception:
        return


def _parse_chapter_segments_by_rules(
    chapter_text: str,
    characters: dict | list[dict],
    *,
    split_config: dict | None = None,
) -> list[dict]:
    config = dict(DEFAULT_SPLIT_CONFIG)
    config.update(split_config or {})
    units = _build_segment_units(chapter_text, characters)
    return _append_units_with_assignments(units, {}, _character_names(characters), config)


def parse_chapter_segments(
    chapter_text: str,
    characters: dict | list[dict],
    *,
    split_config: dict | None = None,
    llm_config: dict | None = None,
    progress_callback=None,
    project_path: str | Path | None = None,
) -> list[dict]:
    config = dict(DEFAULT_SPLIT_CONFIG)
    config.update(split_config or {})

    if _llm_config_available(llm_config):
        try:
            emit_progress(progress_callback, "audiobook_segment_llm", "正在用 LLM 分辨旁白、台词、心理活动和引号引用")
            segments, metadata = _parse_segments_with_llm(
                chapter_text,
                characters,
                llm_config=llm_config or {},
                split_config=config,
            )
            _update_audiobook_llm_stats(project_path, success=True, metadata=metadata)
            if segments:
                return segments
        except Exception:
            _update_audiobook_llm_stats(project_path, success=False, metadata=None)
            emit_progress(progress_callback, "audiobook_segment_rules", "LLM 识别失败，已回退到本地规则分辨")

    return _parse_chapter_segments_by_rules(chapter_text, characters, split_config=config)


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


def _compose_segment_control_instruction(base_control: str, performance_instruction: str) -> str:
    base = _normalize_text(base_control)[:240].rstrip("，,、；;。 ")
    performance = _normalize_performance_instruction(performance_instruction, limit=80)
    if base and performance:
        return f"{base}；本句表演：{performance}"
    return base or performance


def _voice_for_segment(base_voice: dict, segment: dict, clone_mode: str) -> dict:
    voice = dict(base_voice)
    clone_mode = normalize_clone_mode(clone_mode)
    reference_prompt = str(voice.get("prompt_text") or "").strip()
    performance = str(segment.get("performance_instruction") or "").strip()
    voice["clone_mode"] = clone_mode
    voice["base_control_instruction"] = str(voice.get("control_instruction") or "").strip()
    voice["reference_prompt_text"] = reference_prompt
    voice["performance_instruction"] = _normalize_performance_instruction(performance)
    voice["control_instruction"] = _compose_segment_control_instruction(
        voice["base_control_instruction"],
        voice["performance_instruction"],
    )
    if clone_mode == CLONE_MODE_STYLE_CONTROL:
        voice["prompt_text"] = ""
    return voice


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
    seed = _voice_seed(voice_id, profile_signature, reference_signature)

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
                "seed": seed,
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
        "seed": seed,
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
        "seed": seed,
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
    runtime: dict | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    mode = normalize_generation_mode(generation_mode)
    clone_mode = normalize_clone_mode((runtime or {}).get("clone_mode"))
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
        item["voice"] = _voice_for_segment(voice, item, clone_mode)
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
            "api_bases": raw.get("audio_frame_api_bases"),
            "endpoints": raw.get("audio_frame_endpoints") or raw.get("endpoints"),
            "workers": raw.get("audio_frame_workers"),
            "timeout": raw.get("audio_frame_timeout"),
        }.items()
        if value not in (None, "")
    }


def _audio_frame_api_bases(runtime: dict) -> list[str]:
    bases = runtime.get("api_bases") or [runtime.get("api_base")]
    result = [str(base or "").strip() for base in bases if str(base or "").strip()]
    return result or [str(runtime["api_base"])]


def _audio_frame_runtime_endpoints(runtime: dict) -> list[dict]:
    endpoints = runtime.get("endpoints")
    if isinstance(endpoints, list) and endpoints:
        return [dict(item) for item in endpoints if isinstance(item, dict)]
    return [
        {
            "api_base": api_base,
            "kind": "gpu",
            "capacity": 1,
            "max_chars": 0,
            "speed": 1.0,
        }
        for api_base in _audio_frame_api_bases(runtime)
    ]


def _audio_frame_endpoint_slots(runtime: dict) -> list[AudioFrameEndpointSlot]:
    slots: list[AudioFrameEndpointSlot] = []
    for endpoint_index, endpoint in enumerate(_audio_frame_runtime_endpoints(runtime)):
        api_base = str(endpoint.get("api_base") or "").strip()
        if not api_base:
            continue
        kind = str(endpoint.get("kind") or "gpu").strip().lower()
        if kind not in {"gpu", "cpu"}:
            kind = "gpu"
        try:
            capacity = int(endpoint.get("capacity") or 1)
        except (TypeError, ValueError):
            capacity = 1
        try:
            speed = float(endpoint.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        try:
            max_chars = int(endpoint.get("max_chars") or 0)
        except (TypeError, ValueError):
            max_chars = 0
        for slot_index in range(max(1, capacity)):
            slots.append(
                AudioFrameEndpointSlot(
                    slot_id=f"{kind}:{endpoint_index}:{slot_index}",
                    api_base=api_base,
                    kind=kind,
                    speed=max(0.01, speed),
                    max_chars=max(0, max_chars),
                )
            )
    if slots:
        return slots
    return [AudioFrameEndpointSlot("gpu:0:0", str(runtime["api_base"]), "gpu", 1.0, 0)]


def _audio_frame_manifest_endpoints(runtime: dict) -> list[dict]:
    return [
        {
            "api_base": str(endpoint.get("api_base") or "").strip(),
            "kind": str(endpoint.get("kind") or "gpu").strip().lower(),
            "capacity": max(1, int(endpoint.get("capacity") or 1)),
            "max_chars": max(0, int(endpoint.get("max_chars") or 0)),
            "speed": float(endpoint.get("speed") or 1.0),
        }
        for endpoint in _audio_frame_runtime_endpoints(runtime)
        if str(endpoint.get("api_base") or "").strip()
    ]


def _audio_frame_text_chars(value: object) -> int:
    return len(_normalize_text(value))


def _slot_estimated_seconds(slot: AudioFrameEndpointSlot, chars: int) -> float:
    return max(1, chars) / max(0.01, slot.speed)


def _assign_audio_frame_jobs(
    jobs: list[AudioFrameSegmentJob],
    slots: list[AudioFrameEndpointSlot],
) -> dict[str, list[AudioFrameSegmentJob]]:
    if not jobs:
        return {}
    gpu_slots = [slot for slot in slots if slot.kind != "cpu"] or list(slots)
    cpu_slots = [slot for slot in slots if slot.kind == "cpu" and slot.max_chars > 0]
    assignments: dict[str, list[AudioFrameSegmentJob]] = {slot.slot_id: [] for slot in slots}
    slot_by_id = {slot.slot_id: slot for slot in slots}
    job_slot: dict[tuple[int, int], str] = {}
    gpu_loads = {slot.slot_id: 0.0 for slot in gpu_slots}

    for job in sorted(jobs, key=lambda item: item.text_chars, reverse=True):
        slot_id = min(gpu_loads, key=gpu_loads.get)
        gpu_loads[slot_id] += _slot_estimated_seconds(slot_by_id[slot_id], job.text_chars)
        assignments[slot_id].append(job)
        job_slot[(job.payload_index, job.position)] = slot_id

    gpu_baseline = max(gpu_loads.values(), default=0.0)
    if not cpu_slots:
        return {slot_id: items for slot_id, items in assignments.items() if items}

    cpu_loads = {slot.slot_id: 0.0 for slot in cpu_slots}
    for job in sorted(jobs, key=lambda item: item.text_chars):
        eligible_cpu_slots = [slot for slot in cpu_slots if job.text_chars <= slot.max_chars]
        if not eligible_cpu_slots:
            continue
        cpu_slot = min(eligible_cpu_slots, key=lambda slot: cpu_loads[slot.slot_id])
        projected_cpu_load = cpu_loads[cpu_slot.slot_id] + _slot_estimated_seconds(cpu_slot, job.text_chars)
        if projected_cpu_load > gpu_baseline:
            continue
        old_slot_id = job_slot[(job.payload_index, job.position)]
        if old_slot_id == cpu_slot.slot_id:
            continue
        assignments[old_slot_id] = [
            item for item in assignments[old_slot_id] if (item.payload_index, item.position) != (job.payload_index, job.position)
        ]
        assignments[cpu_slot.slot_id].append(job)
        job_slot[(job.payload_index, job.position)] = cpu_slot.slot_id
        cpu_loads[cpu_slot.slot_id] = projected_cpu_load

    return {slot_id: items for slot_id, items in assignments.items() if items}


def _assign_audio_frame_retry_jobs(
    jobs: list[AudioFrameSegmentJob],
    slots: list[AudioFrameEndpointSlot],
) -> tuple[dict[str, list[AudioFrameSegmentJob]], list[AudioFrameSegmentJob]]:
    if not jobs:
        return {}, []
    if any(slot.kind != "cpu" for slot in slots):
        return _assign_audio_frame_jobs(jobs, slots), []

    cpu_slots = [slot for slot in slots if slot.kind == "cpu" and slot.max_chars > 0]
    if not cpu_slots:
        return {}, list(jobs)

    assignments: dict[str, list[AudioFrameSegmentJob]] = {slot.slot_id: [] for slot in cpu_slots}
    cpu_loads = {slot.slot_id: 0.0 for slot in cpu_slots}
    unassigned: list[AudioFrameSegmentJob] = []
    for job in sorted(jobs, key=lambda item: item.text_chars):
        eligible_slots = [slot for slot in cpu_slots if job.text_chars <= slot.max_chars]
        if not eligible_slots:
            unassigned.append(job)
            continue
        slot = min(eligible_slots, key=lambda item: cpu_loads[item.slot_id])
        assignments[slot.slot_id].append(job)
        cpu_loads[slot.slot_id] += _slot_estimated_seconds(slot, job.text_chars)

    return {slot_id: items for slot_id, items in assignments.items() if items}, unassigned


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


def _audio_frame_reference_api_bases(runtime: dict) -> list[str]:
    bases = []
    seen: set[str] = set()
    for endpoint in _audio_frame_runtime_endpoints(runtime):
        if str(endpoint.get("kind") or "gpu").strip().lower() == "cpu":
            continue
        base = str(endpoint.get("api_base") or "").strip()
        if base and base not in seen:
            bases.append(base)
            seen.add(base)
    return bases or _audio_frame_api_bases(runtime)


def _run_audio_frame_reference_tasks(
    request_payloads: list[dict],
    runtime: dict,
    progress_callback=None,
) -> None:
    reference_api_bases = _audio_frame_reference_api_bases(runtime)
    timeout = int(runtime.get("timeout") or 0)
    total = sum(len(payload.get("voice_reference_tasks") or []) for payload in request_payloads)
    if total <= 0:
        return
    emit_progress(
        progress_callback,
        "audiobook_voice_reference",
        f"Audio Frame preparing {total} reference voice task(s)",
        current=0,
        total=total,
    )
    completed = 0
    for request_payload in request_payloads:
        voice_references = request_payload.get("voice_references") or []
        reference_by_voice_id = {str(item.get("voice_id") or ""): item for item in voice_references}
        for task in request_payload.get("voice_reference_tasks") or []:
            api_base = reference_api_bases[completed % len(reference_api_bases)]
            response = AudioFrameClient(api_base).synthesize(
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
                timeout=timeout,
            )
            reference_path = Path(str(task.get("reference_audio") or "")).resolve()
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference_path.write_bytes(base64.b64decode(str(response.get("audio_base64") or "")))
            record = reference_by_voice_id.get(str(task.get("voice_id") or ""))
            if record is not None:
                record["generated"] = True
                record["duration_seconds"] = round(_wav_duration_seconds(reference_path), 3)
            completed += 1
            emit_progress(
                progress_callback,
                "audiobook_voice_reference",
                f"Audio Frame finished reference voice {completed}/{total}",
                current=completed,
                total=total,
            )


def _run_audio_frame_batch_service(
    request_payloads: list[dict],
    runtime: dict,
    progress_callback=None,
) -> None:
    request_payloads = [payload for payload in request_payloads if payload]
    if not request_payloads:
        return

    _run_audio_frame_reference_tasks(request_payloads, runtime, progress_callback=progress_callback)

    slots = _audio_frame_endpoint_slots(runtime)
    slot_by_id = {slot.slot_id: slot for slot in slots}
    timeout = int(runtime.get("timeout") or 0)
    configured_api_bases = _audio_frame_api_bases(runtime)
    configured_endpoints = _audio_frame_manifest_endpoints(runtime)
    jobs: list[AudioFrameSegmentJob] = []
    generated_paths: list[list[Path | None]] = []
    manifest_segments: list[list[dict | None]] = []
    chapter_completed = [0] * len(request_payloads)
    used_slots: list[set[str]] = [set() for _ in request_payloads]

    for payload_index, request_payload in enumerate(request_payloads):
        project_path = Path(request_payload["project_path"])
        segments_dir = Path(request_payload["segments_dir"])
        segments_dir.mkdir(parents=True, exist_ok=True)
        segments = [dict(item) for item in request_payload.get("segments") or []]
        generated_paths.append([None] * len(segments))
        manifest_segments.append([None] * len(segments))
        for position, item in enumerate(segments):
            jobs.append(
                AudioFrameSegmentJob(
                    payload_index=payload_index,
                    position=position,
                    item=item,
                    text_chars=_audio_frame_text_chars(item.get("text")),
                )
            )

    if jobs:
        emit_progress(
            progress_callback,
            "audiobook_audio_frame",
            f"Audio Frame queued {len(jobs)} segment task(s) across {len(slots)} endpoint slot(s)",
            current=0,
            total=len(jobs),
        )

    save_lock = Lock()
    retry_lock = Lock()
    unhealthy_lock = Lock()
    unhealthy_slot_ids: set[str] = set()
    retry_counts: dict[tuple[int, int], int] = {}
    retry_errors: dict[tuple[int, int], list[str]] = {}
    batch_completed = 0
    max_retries = max(0, int(runtime.get("max_retries") or runtime.get("retries") or 2))

    def job_key(job: AudioFrameSegmentJob) -> tuple[int, int]:
        return (job.payload_index, job.position)

    def pending_unique(jobs_to_retry: list[AudioFrameSegmentJob]) -> list[AudioFrameSegmentJob]:
        seen: set[tuple[int, int]] = set()
        result: list[AudioFrameSegmentJob] = []
        with save_lock:
            for job in jobs_to_retry:
                key = job_key(job)
                if key in seen or generated_paths[job.payload_index][job.position] is not None:
                    continue
                seen.add(key)
                result.append(job)
        return result

    def synthesize_segment(slot: AudioFrameEndpointSlot, job: AudioFrameSegmentJob) -> tuple[AudioFrameSegmentJob, bytes]:
        request_payload = request_payloads[job.payload_index]
        voice = job.item.get("voice") or {}
        response = AudioFrameClient(slot.api_base).synthesize(
            text=str(job.item.get("text") or ""),
            control_instruction=str(voice.get("control_instruction") or ""),
            reference_audio=str(voice.get("reference_audio") or ""),
            prompt_text=str(voice.get("prompt_text") or ""),
            cfg_value=float(voice.get("cfg_value") or request_payload["runtime"].get("cfg_value") or 2.0),
            normalize=bool(request_payload["runtime"].get("normalize", True)),
            denoise=bool(request_payload["runtime"].get("denoise", False)),
            inference_timesteps=int(
                voice.get("inference_timesteps") or request_payload["runtime"].get("inference_timesteps") or 10
            ),
            timeout=timeout,
        )
        return job, base64.b64decode(str(response.get("audio_base64") or ""))

    def save_segment_result(slot: AudioFrameEndpointSlot, job: AudioFrameSegmentJob, audio_bytes: bytes) -> None:
        nonlocal batch_completed
        request_payload = request_payloads[job.payload_index]
        project_path = Path(request_payload["project_path"])
        segments_dir = Path(request_payload["segments_dir"])
        local_path = segments_dir / f"{job.item.get('segment_id') or f'segment_{job.position + 1:04d}'}.wav"
        local_path.write_bytes(audio_bytes)
        item = dict(job.item)
        item["audio_file"] = _relative_path(project_path, local_path)
        item["duration_seconds"] = round(_wav_duration_seconds(local_path), 3)
        item["audio_frame"] = {
            "api_base": slot.api_base,
            "kind": slot.kind,
            "slot": slot.slot_id,
            "text_chars": job.text_chars,
        }
        retry_count = retry_counts.get(job_key(job), 0)
        if retry_count:
            item["audio_frame"]["retry_count"] = retry_count
        with save_lock:
            generated_paths[job.payload_index][job.position] = local_path
            manifest_segments[job.payload_index][job.position] = item
            used_slots[job.payload_index].add(slot.slot_id)
            chapter_completed[job.payload_index] += 1
            batch_completed += 1
            emit_progress(
                progress_callback,
                "audiobook_audio_frame",
                (
                    f"Audio Frame finished {request_payload['chapter_slug']} "
                    f"segment {chapter_completed[job.payload_index]}/{len(generated_paths[job.payload_index])}"
                ),
                current=batch_completed,
                total=len(jobs),
            )

    def record_endpoint_failure(
        slot: AudioFrameEndpointSlot,
        job: AudioFrameSegmentJob,
        exc: Exception,
        retry_jobs: list[AudioFrameSegmentJob],
        remaining_jobs: list[AudioFrameSegmentJob],
    ) -> None:
        with unhealthy_lock:
            unhealthy_slot_ids.add(slot.slot_id)
        message = f"{slot.api_base} ({slot.kind}) failed: {exc}"
        with retry_lock:
            retry_counts[job_key(job)] = retry_counts.get(job_key(job), 0) + 1
            retry_errors.setdefault(job_key(job), []).append(message)
            retry_jobs.extend(remaining_jobs)
        emit_progress(
            progress_callback,
            "audiobook_audio_frame_retry",
            (
                f"Audio Frame endpoint {slot.api_base} failed; "
                f"requeued {len(remaining_jobs)} segment task(s)"
            ),
            current=batch_completed,
            total=len(jobs),
        )

    def run_slot(slot_id: str, slot_jobs: list[AudioFrameSegmentJob], retry_jobs: list[AudioFrameSegmentJob]) -> None:
        slot = slot_by_id[slot_id]
        for index, job in enumerate(slot_jobs):
            with unhealthy_lock:
                if slot_id in unhealthy_slot_ids:
                    with retry_lock:
                        retry_jobs.extend(slot_jobs[index:])
                    return
            try:
                completed_job, audio_bytes = synthesize_segment(slot, job)
            except Exception as exc:
                record_endpoint_failure(slot, job, exc, retry_jobs, slot_jobs[index:])
                return
            save_segment_result(slot, completed_job, audio_bytes)

    def run_assignment_round(assignments: dict[str, list[AudioFrameSegmentJob]]) -> list[AudioFrameSegmentJob]:
        retry_jobs: list[AudioFrameSegmentJob] = []
        if not assignments:
            return retry_jobs
        with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="audio-frame-slot") as executor:
            futures = [
                executor.submit(run_slot, slot_id, slot_jobs, retry_jobs)
                for slot_id, slot_jobs in assignments.items()
            ]
            for future in as_completed(futures):
                future.result()
        return pending_unique(retry_jobs)

    pending_retry_jobs = run_assignment_round(_assign_audio_frame_jobs(jobs, slots))
    retry_round = 0
    while pending_retry_jobs and retry_round < max_retries:
        retry_round += 1
        with unhealthy_lock:
            available_slots = [slot for slot in slots if slot.slot_id not in unhealthy_slot_ids]
        retry_assignments, unassigned_jobs = _assign_audio_frame_retry_jobs(pending_retry_jobs, available_slots)
        for job in unassigned_jobs:
            retry_errors.setdefault(job_key(job), []).append(
                "No healthy GPU endpoint remained and the segment is too long for CPU fallback."
            )
        if not retry_assignments:
            pending_retry_jobs = pending_unique(unassigned_jobs)
            break
        retry_task_count = sum(len(items) for items in retry_assignments.values())
        emit_progress(
            progress_callback,
            "audiobook_audio_frame_retry",
            f"Audio Frame retry round {retry_round}/{max_retries}: {retry_task_count} segment task(s)",
            current=batch_completed,
            total=len(jobs),
        )
        pending_retry_jobs = pending_unique(unassigned_jobs + run_assignment_round(retry_assignments))

    if pending_retry_jobs:
        samples = []
        for job in pending_retry_jobs[:5]:
            request_payload = request_payloads[job.payload_index]
            segment_id = job.item.get("segment_id") or f"segment_{job.position + 1:04d}"
            errors = "; ".join(retry_errors.get(job_key(job), [])[-2:])
            samples.append(f"{request_payload['chapter_slug']}:{segment_id} ({errors or 'not attempted'})")
        raise RuntimeError(
            "Audio Frame failed to generate "
            f"{len(pending_retry_jobs)} segment(s) after {max_retries} retry round(s): "
            + "; ".join(samples)
        )

    for payload_index, request_payload in enumerate(request_payloads):
        project_path = Path(request_payload["project_path"])
        combined_audio_path = Path(request_payload["combined_audio_path"])
        manifest_path = Path(request_payload["manifest_path"])
        paths = generated_paths[payload_index]
        if any(path is None for path in paths):
            raise RuntimeError(f"Audio Frame did not generate every segment for {request_payload['chapter_slug']}.")
        _combine_wav_files(
            [path for path in paths if path is not None],
            combined_audio_path,
            silence_ms=int(request_payload["runtime"].get("silence_ms") or DEFAULT_AUDIOBOOK_RUNTIME["silence_ms"]),
        )
        rendered_segments = [item for item in manifest_segments[payload_index] if item is not None]
        save_json(
            str(manifest_path),
            {
                "chapter_slug": request_payload["chapter_slug"],
                "chapter_file": request_payload["chapter_file"],
                "generated_at": utc_now(),
                "status": "succeeded",
                "combined_audio": _relative_path(project_path, combined_audio_path),
                "segment_count": len(rendered_segments),
                "segments": rendered_segments,
                "narrator_id": request_payload.get("narrator_id", ""),
                "generation_mode": request_payload.get("generation_mode", GENERATION_MODE_ADVANCED),
                "voice_references": request_payload.get("voice_references") or [],
                "split_config": request_payload.get("split_config", {}),
                "audio_frame": {
                    "api_base": runtime["api_base"],
                    "api_bases": configured_api_bases,
                    "workers": max(1, len(used_slots[payload_index])),
                    "scheduler": "batch_global_weighted",
                    "endpoints": configured_endpoints,
                    "failed_slots": [
                        {
                            "slot": slot.slot_id,
                            "api_base": slot.api_base,
                            "kind": slot.kind,
                        }
                        for slot in slots
                        if slot.slot_id in unhealthy_slot_ids
                    ],
                },
                "voxcpm_runtime": {
                    "clone_mode": request_payload["runtime"].get("clone_mode", CLONE_MODE_STYLE_CONTROL),
                },
            },
        )


def _run_audio_frame_service(request_payload: dict, runtime_overrides: dict | None, progress_callback=None) -> None:
    runtime = load_audio_frame_runtime(_audio_frame_overrides(runtime_overrides))
    _run_audio_frame_batch_service([request_payload], runtime, progress_callback=progress_callback)


def _prepare_audiobook_chapter(
    project_path: str | Path,
    chapter_ref: str | None = None,
    *,
    force: bool = False,
    narrator_preset: str = "",
    generation_mode: str = GENERATION_MODE_ADVANCED,
    llm_config: dict | None = None,
    runtime_overrides: dict | None = None,
    split_config: dict | None = None,
    progress_callback=None,
) -> PreparedAudiobookChapter:
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
            return PreparedAudiobookChapter(project_path, chapter_slug, manifest_path, None, reused_manifest=existing)

    resolved_generation_mode = normalize_generation_mode(generation_mode)
    voice_config = update_selected_narrator(project_path, narrator_preset) if narrator_preset else ensure_voice_config(project_path)
    if voice_config.get("generation_mode") != resolved_generation_mode:
        voice_config["generation_mode"] = resolved_generation_mode
        voice_config = save_voice_config(project_path, voice_config)
    project_data = load_project(str(project_path))
    split_config_resolved = dict(DEFAULT_SPLIT_CONFIG)
    split_config_resolved.update(split_config or {})
    chapter_text = chapter_file.read_text(encoding="utf-8")
    segments = parse_chapter_segments(
        chapter_text,
        project_data.get("characters") or {},
        split_config=split_config_resolved,
        llm_config=llm_config,
        progress_callback=progress_callback,
        project_path=project_path,
    )
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
        runtime,
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
    return PreparedAudiobookChapter(project_path, chapter_slug, manifest_path, request_payload, voice_config=voice_config)


def _finish_prepared_audiobook_chapter(
    prepared: PreparedAudiobookChapter,
    result: subprocess.CompletedProcess,
    *,
    progress_callback=None,
) -> dict:
    if prepared.reused_manifest is not None:
        return prepared.reused_manifest
    if prepared.request_payload is None or prepared.voice_config is None:
        raise RuntimeError(f"{prepared.chapter_slug} 缺少有声章节生成请求。")
    if result.returncode != 0:
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 有声章节生成失败，退出码 {result.returncode}。\n{detail}")
    if not prepared.manifest_path.exists():
        detail = "\n".join(item for item in (result.stdout.strip(), result.stderr.strip()) if item)
        raise RuntimeError(f"VoxCPM2 worker 未生成 manifest.json。\n{detail}")

    manifest = load_json(str(prepared.manifest_path))
    _apply_voice_reference_records(
        prepared.project_path,
        prepared.voice_config,
        manifest.get("voice_references") or [],
    )
    manifest["reused"] = False
    segment_total = int(manifest.get("segment_count") or len(manifest.get("segments") or []))
    emit_progress(
        progress_callback,
        "audiobook_done",
        f"{prepared.chapter_slug} 的有声章节已完成",
        current=segment_total,
        total=segment_total,
    )
    return manifest


def generate_audiobook_chapter(
    project_path: str | Path,
    chapter_ref: str | None = None,
    *,
    force: bool = False,
    narrator_preset: str = "",
    generation_mode: str = GENERATION_MODE_ADVANCED,
    llm_config: dict | None = None,
    runtime_overrides: dict | None = None,
    split_config: dict | None = None,
    progress_callback=None,
    lock_project: bool = True,
) -> dict:
    if lock_project:
        with acquire_project_audio_lock(str(project_path), owner="generate_audiobook_chapter"):
            return generate_audiobook_chapter(
                project_path,
                chapter_ref,
                force=force,
                narrator_preset=narrator_preset,
                generation_mode=generation_mode,
                llm_config=llm_config,
                runtime_overrides=runtime_overrides,
                split_config=split_config,
                progress_callback=progress_callback,
                lock_project=False,
            )

    prepared = _prepare_audiobook_chapter(
        project_path,
        chapter_ref,
        force=force,
        narrator_preset=narrator_preset,
        generation_mode=generation_mode,
        llm_config=llm_config,
        runtime_overrides=runtime_overrides,
        split_config=split_config,
        progress_callback=progress_callback,
    )
    if prepared.reused_manifest is not None:
        return prepared.reused_manifest
    request_payload = prepared.request_payload or {}
    segments = request_payload.get("segments") or []
    emit_progress(
        progress_callback,
        "audiobook_worker",
        f"正在调用 VoxCPM2 合成 {len(segments)} 个片段",
        current=0,
        total=len(segments),
    )
    if _audiobook_backend(runtime_overrides) == "audio_frame":
        _run_audio_frame_service(request_payload, runtime_overrides, progress_callback=progress_callback)
        result = subprocess.CompletedProcess(["audio_frame"], 0, stdout="ok", stderr="")
    else:
        request_path = Path(str(prepared.manifest_path.parent / "request.json"))
        result = _run_worker(request_path, request_payload["runtime"])
    return _finish_prepared_audiobook_chapter(prepared, result, progress_callback=progress_callback)


def generate_audiobook_chapters(
    project_path: str | Path,
    *,
    chapter_refs: list[str] | None = None,
    force: bool = False,
    narrator_preset: str = "",
    generation_mode: str = GENERATION_MODE_ADVANCED,
    llm_config: dict | None = None,
    runtime_overrides: dict | None = None,
    progress_callback=None,
    lock_project: bool = True,
) -> list[dict]:
    if lock_project:
        with acquire_project_audio_lock(str(project_path), owner="generate_audiobook_chapters"):
            return generate_audiobook_chapters(
                project_path,
                chapter_refs=chapter_refs,
                force=force,
                narrator_preset=narrator_preset,
                generation_mode=generation_mode,
                llm_config=llm_config,
                runtime_overrides=runtime_overrides,
                progress_callback=progress_callback,
                lock_project=False,
            )

    refs = chapter_refs or ["latest"]
    if _audiobook_backend(runtime_overrides) == "audio_frame":
        prepared_items: list[tuple[int, PreparedAudiobookChapter]] = []
        results: list[dict | None] = [None] * len(refs)
        for index, chapter_ref in enumerate(refs):
            emit_progress(
                progress_callback,
                "audiobook_batch",
                f"正在准备第 {index + 1}/{len(refs)} 个有声章节",
                current=index,
                total=len(refs),
            )
            prepared = _prepare_audiobook_chapter(
                project_path,
                chapter_ref,
                force=force,
                narrator_preset=narrator_preset,
                generation_mode=generation_mode,
                llm_config=llm_config,
                runtime_overrides=runtime_overrides,
                progress_callback=progress_callback,
            )
            if prepared.reused_manifest is not None:
                results[index] = prepared.reused_manifest
            else:
                prepared_items.append((index, prepared))

        if prepared_items:
            payloads = [prepared.request_payload for _, prepared in prepared_items if prepared.request_payload is not None]
            segment_total = sum(len(payload.get("segments") or []) for payload in payloads)
            emit_progress(
                progress_callback,
                "audiobook_worker",
                f"正在调用 Audio Frame 全局调度合成 {len(prepared_items)} 章 / {segment_total} 个片段",
                current=0,
                total=segment_total,
            )
            runtime = load_audio_frame_runtime(_audio_frame_overrides(runtime_overrides))
            _run_audio_frame_batch_service(payloads, runtime, progress_callback=progress_callback)
            result = subprocess.CompletedProcess(["audio_frame"], 0, stdout="ok", stderr="")
            for index, prepared in prepared_items:
                results[index] = _finish_prepared_audiobook_chapter(
                    prepared,
                    result,
                    progress_callback=progress_callback,
                )
        return [result for result in results if result is not None]

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
                llm_config=llm_config,
                runtime_overrides=runtime_overrides,
                progress_callback=progress_callback,
                lock_project=False,
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
