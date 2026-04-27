"""Worker process for VoxCPM2 audiobook synthesis.

This file intentionally imports audio and model dependencies only inside the
worker process. The main novel writer app can stay standard-library-only.
"""

from __future__ import annotations

import argparse
import json
import random
import traceback
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from voxcpm import VoxCPM


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def relative(project_path: str | Path, path: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(project_path).resolve())).replace("\\", "/")


def build_text_for_voice(text: str, voice: dict) -> str:
    control = str(voice.get("control_instruction") or "").strip()
    if control and not str(voice.get("prompt_text") or "").strip():
        return f"({control}){text}"
    return text


def apply_voice_seed(voice: dict) -> None:
    try:
        seed = int((voice or {}).get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def generate_segment(model: VoxCPM, segment: dict, runtime: dict) -> tuple[int, np.ndarray]:
    voice = segment.get("voice") or {}
    text = str(segment.get("text") or "").strip()
    if not text:
        raise ValueError("segment text is empty")

    reference_audio = str(voice.get("reference_audio") or "").strip() or None
    prompt_text = str(voice.get("prompt_text") or "").strip()
    final_text = build_text_for_voice(text, voice)
    kwargs = {
        "text": final_text,
        "reference_wav_path": reference_audio,
        "cfg_value": float(voice.get("cfg_value") or runtime.get("cfg_value") or 2.0),
        "inference_timesteps": int(voice.get("inference_timesteps") or runtime.get("inference_timesteps") or 10),
        "normalize": bool(runtime.get("normalize", True)),
        "denoise": bool(runtime.get("denoise", False)) and bool(reference_audio),
    }
    if reference_audio and prompt_text:
        kwargs["prompt_wav_path"] = reference_audio
        kwargs["prompt_text"] = prompt_text

    apply_voice_seed(voice)
    wav = model.generate(**kwargs)
    return int(model.tts_model.sample_rate), np.asarray(wav, dtype=np.float32)


def generate_voice_reference(model: VoxCPM, task: dict, runtime: dict) -> tuple[int, np.ndarray]:
    text = str(task.get("prompt_text") or "").strip()
    if not text:
        raise ValueError("voice reference prompt_text is empty")
    segment = {
        "text": text,
        "voice": {
            "control_instruction": str(task.get("control_instruction") or "").strip(),
            "reference_audio": "",
            "prompt_text": "",
            "cfg_value": task.get("cfg_value"),
            "inference_timesteps": task.get("inference_timesteps"),
            "seed": task.get("seed"),
        },
    }
    return generate_segment(model, segment, runtime)


def build_failure_manifest(request_data: dict, message: str) -> dict:
    return {
        "chapter_slug": request_data.get("chapter_slug", ""),
        "chapter_file": request_data.get("chapter_file", ""),
        "generated_at": utc_now(),
        "status": "failed",
        "error": message,
        "narrator_id": request_data.get("narrator_id", ""),
        "combined_audio": "",
        "segments": request_data.get("segments", []),
        "generation_mode": request_data.get("generation_mode", "advanced"),
        "voice_references": request_data.get("voice_references", []),
        "voxcpm_runtime": request_data.get("runtime", {}),
        "split_config": request_data.get("split_config", {}),
    }


def run_request(request_path: str | Path) -> dict:
    request_data = load_json(request_path)
    project_path = request_data["project_path"]
    segments_dir = Path(request_data["segments_dir"])
    combined_audio_path = Path(request_data["combined_audio_path"])
    manifest_path = Path(request_data["manifest_path"])
    runtime = request_data.get("runtime") or {}
    segments = request_data.get("segments") or []
    voice_references = [dict(item) for item in request_data.get("voice_references") or []]
    reference_by_voice_id = {str(item.get("voice_id") or ""): item for item in voice_references}
    reference_tasks = request_data.get("voice_reference_tasks") or []
    if not segments:
        raise RuntimeError("request contains no segments")

    segments_dir.mkdir(parents=True, exist_ok=True)
    model = VoxCPM.from_pretrained(
        runtime.get("model_id") or "openbmb/VoxCPM2",
        load_denoiser=bool(runtime.get("load_denoiser", False)),
        optimize=bool(runtime.get("optimize", True)),
        device=str(runtime.get("device") or "auto"),
    )

    rendered_segments = []
    audio_chunks = []
    sample_rate = 0
    silence_ms = max(0, int(runtime.get("silence_ms") or 260))
    started = time.time()

    for task in reference_tasks:
        reference_started = time.time()
        reference_path = Path(str(task.get("reference_audio") or "")).resolve()
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        if reference_path.exists():
            record = reference_by_voice_id.get(str(task.get("voice_id") or ""))
            if record is not None:
                record["generated"] = False
                record["duration_seconds"] = 0
            continue
        sample_rate, reference_audio = generate_voice_reference(model, task, runtime)
        sf.write(str(reference_path), reference_audio, sample_rate)
        record = reference_by_voice_id.get(str(task.get("voice_id") or ""))
        if record is not None:
            record["generated"] = True
            record["duration_seconds"] = round(float(len(reference_audio)) / float(sample_rate), 3) if sample_rate else 0
            record["elapsed_seconds"] = round(time.time() - reference_started, 3)

    for index, segment in enumerate(segments, start=1):
        segment_started = time.time()
        item = dict(segment)
        try:
            sample_rate, audio = generate_segment(model, item, runtime)
            file_name = f"segment_{index:04d}.wav"
            local_path = segments_dir / file_name
            sf.write(str(local_path), audio, sample_rate)
            item["audio_file"] = relative(project_path, local_path)
            item["duration_seconds"] = round(float(len(audio)) / float(sample_rate), 3) if sample_rate else 0
            item["elapsed_seconds"] = round(time.time() - segment_started, 3)
            item["status"] = "succeeded"
            item["error"] = ""
            rendered_segments.append(item)
            audio_chunks.append(audio)
        except Exception as exc:
            item["status"] = "failed"
            item["error"] = f"{exc}\n{traceback.format_exc()}"
            rendered_segments.append(item)
            manifest = build_failure_manifest(request_data, str(exc))
            manifest["segments"] = rendered_segments
            save_json(manifest_path, manifest)
            raise

    if not audio_chunks:
        raise RuntimeError("no audio chunks were generated")

    if sample_rate and silence_ms:
        silence = np.zeros(int(sample_rate * silence_ms / 1000), dtype=np.float32)
        merged = []
        for index, chunk in enumerate(audio_chunks):
            if index:
                merged.append(silence)
            merged.append(chunk.astype(np.float32))
        combined = np.concatenate(merged)
    else:
        combined = np.concatenate([chunk.astype(np.float32) for chunk in audio_chunks])

    combined_audio_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(combined_audio_path), combined, sample_rate)
    manifest = {
        "chapter_slug": request_data.get("chapter_slug", ""),
        "chapter_file": request_data.get("chapter_file", ""),
        "generated_at": utc_now(),
        "status": "succeeded",
        "error": "",
        "narrator_id": request_data.get("narrator_id", ""),
        "combined_audio": relative(project_path, combined_audio_path),
        "combined_duration_seconds": round(float(len(combined)) / float(sample_rate), 3) if sample_rate else 0,
        "sample_rate": sample_rate,
        "segment_count": len(rendered_segments),
        "segments": rendered_segments,
        "generation_mode": request_data.get("generation_mode", "advanced"),
        "voice_references": voice_references,
        "voxcpm_runtime": {
            "model_id": runtime.get("model_id") or "openbmb/VoxCPM2",
            "device": runtime.get("device") or "auto",
            "cfg_value": runtime.get("cfg_value", 2.0),
            "inference_timesteps": runtime.get("inference_timesteps", 10),
            "silence_ms": silence_ms,
        },
        "split_config": request_data.get("split_config", {}),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audiobook WAV files with VoxCPM2")
    parser.add_argument("--request", required=True, help="Path to audiobook worker request JSON")
    args = parser.parse_args()

    try:
        manifest = run_request(args.request)
    except Exception as exc:
        request_data = {}
        try:
            request_data = load_json(args.request)
            manifest_path = request_data.get("manifest_path")
            if manifest_path:
                save_json(manifest_path, build_failure_manifest(request_data, f"{exc}\n{traceback.format_exc()}"))
        except Exception:
            pass
        raise

    print(json.dumps({"manifest_path": load_json(args.request).get("manifest_path", ""), "status": manifest.get("status")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
