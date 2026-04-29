from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audiobook_manager
from audiobook_manager import (
    UploadedVoiceFile,
    ensure_voice_config,
    generate_audiobook_chapter,
    parse_chapter_segments,
    save_uploaded_voice_reference,
    split_text_for_tts,
)
from project_manager import create_state_snapshot, load_json, rollback_project, save_json
from tests.test_support import create_test_project


def _tiny_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 80)
    return buffer.getvalue()


class AudiobookManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_path = create_test_project(Path(self.temp_dir.name), project_id="audio")
        self.chapter_path = self.project_path / "chapters" / "chapter_0001.md"
        self.chapter_path.write_text(
            "林宇说：“我们先检查门。”\n\n"
            "“我去看控制板。”苏浅低声道。\n\n"
            "林宇心想，必须撑到天亮。",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)

    def test_parse_chapter_segments_assigns_dialogue_and_inner_voice(self) -> None:
        characters = load_json(str(self.project_path / "characters.json"))

        segments = parse_chapter_segments(self.chapter_path.read_text(encoding="utf-8"), characters)

        dialogue = [item for item in segments if item["type"] == "dialogue"]
        inner = [item for item in segments if item["type"] == "inner_monologue"]
        self.assertEqual(dialogue[0]["speaker"], "林宇")
        self.assertEqual(dialogue[0]["text"], "我们先检查门。")
        self.assertEqual(dialogue[1]["speaker"], "苏浅")
        self.assertEqual(dialogue[1]["text"], "我去看控制板。")
        self.assertEqual(inner[0]["speaker"], "林宇")

    def test_parse_chapter_segments_uses_llm_to_distinguish_quote_and_inner_voice(self) -> None:
        characters = load_json(str(self.project_path / "characters.json"))
        chapter_text = "林宇想起墙上的标语：“安全第一。”\n\n苏浅握紧工具，心里想，不能让他发现。"
        response_payload = {
            "segments": [
                {"id": "unit_0001", "type": "narration", "speaker": "旁白", "confidence": 0.96},
                {"id": "unit_0002", "type": "quoted_text", "speaker": "旁白", "confidence": 0.98},
                {
                    "id": "unit_0003",
                    "type": "inner_monologue",
                    "speaker": "苏浅",
                    "emotion": "紧张",
                    "tone": "压低声音",
                    "delivery": "停顿明显",
                    "performance_instruction": "紧张，压低声音，停顿明显",
                    "confidence": 0.97,
                },
            ]
        }
        llm_config = {
            "model_provider": "ollama",
            "model": "llama3.2",
            "api_base": "http://127.0.0.1:11434/v1",
        }

        with patch(
            "audiobook_manager.generate_text_with_metadata",
            return_value=(json.dumps(response_payload, ensure_ascii=False), {"usage": {"total_tokens": 12}}),
        ) as mocked_llm:
            segments = parse_chapter_segments(chapter_text, characters, llm_config=llm_config)

        prompt = mocked_llm.call_args.args[0]
        self.assertIn("quoted_text", prompt)
        self.assertIn("人物心理活动", prompt)
        quoted = [item for item in segments if item["type"] == "quoted_text"]
        inner = [item for item in segments if item["type"] == "inner_monologue"]
        self.assertEqual(quoted[0]["speaker"], "旁白")
        self.assertEqual(quoted[0]["text"], "安全第一。")
        self.assertEqual(inner[0]["speaker"], "苏浅")
        self.assertEqual(inner[0]["text"], "苏浅握紧工具，心里想，不能让他发现。")
        self.assertEqual(inner[0]["performance_instruction"], "紧张，压低声音，停顿明显")

    def test_split_text_for_tts_keeps_chunks_under_hard_limit(self) -> None:
        text = "。".join(["这是一段需要稳定切分的长句子" * 4 for _ in range(5)])

        chunks = split_text_for_tts(text, {"target_chars": 70, "max_chars": 120})

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))

    def test_voice_config_and_reference_upload_are_persisted(self) -> None:
        config = ensure_voice_config(self.project_path)
        self.assertIn("warm_female", [item["id"] for item in config["narrator_presets"]])
        self.assertIn("林宇", config["character_voices"])

        updated = save_uploaded_voice_reference(
            self.project_path,
            target="林宇",
            uploaded_file=UploadedVoiceFile(filename="linyu.wav", content=b"RIFF....WAVE"),
            prompt_text="参考文本",
        )

        voice = updated["character_voices"]["林宇"]
        self.assertTrue(voice["reference_audio"].startswith("audiobook/voice_refs/"))
        self.assertEqual(voice["prompt_text"], "参考文本")
        self.assertTrue((self.project_path / voice["reference_audio"]).exists())

    def test_generate_audiobook_chapter_builds_worker_request_and_reads_manifest(self) -> None:
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            combined = Path(payload["combined_audio_path"])
            combined.parent.mkdir(parents=True, exist_ok=True)
            combined.write_bytes(b"fake wav")
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            manifest = generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                runtime_overrides={"backend": "local_worker"},
            )

        self.assertEqual(manifest["chapter_slug"], "chapter_0001")
        self.assertEqual(captured["payload"]["chapter_slug"], "chapter_0001")
        self.assertGreaterEqual(len(captured["payload"]["segments"]), 3)
        self.assertIn("voice", captured["payload"]["segments"][0])
        self.assertEqual(captured["payload"]["generation_mode"], "advanced")
        self.assertGreaterEqual(len(captured["payload"]["voice_reference_tasks"]), 3)

    def test_simple_mode_uses_one_reference_voice_for_all_segments(self) -> None:
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "generation_mode": payload["generation_mode"],
                    "voice_references": payload["voice_references"],
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="simple",
                runtime_overrides={"backend": "local_worker"},
            )

        payload = captured["payload"]
        self.assertEqual(payload["generation_mode"], "simple")
        self.assertEqual(len(payload["voice_reference_tasks"]), 1)
        self.assertEqual(len({item["voice"]["voice_id"] for item in payload["segments"]}), 1)
        self.assertEqual(len({item["voice"]["reference_audio"] for item in payload["segments"]}), 1)
        self.assertTrue(all(item["voice"]["mode"] == "reference" for item in payload["segments"]))

    def test_advanced_mode_builds_distinct_reference_tasks_for_narrator_and_characters(self) -> None:
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "generation_mode": payload["generation_mode"],
                    "voice_references": payload["voice_references"],
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="advanced",
                runtime_overrides={"backend": "local_worker"},
            )

        tasks = captured["payload"]["voice_reference_tasks"]
        task_ids = {task["voice_id"] for task in tasks}
        self.assertIn("narrator:warm_female", task_ids)
        self.assertIn("character:林宇", task_ids)
        self.assertIn("character:苏浅", task_ids)
        linyu_task = next(task for task in tasks if task["voice_id"] == "character:林宇")
        self.assertIn("负责行动", linyu_task["control_instruction"])
        self.assertIn("黑发，沉稳", linyu_task["control_instruction"])
        self.assertIsInstance(linyu_task["seed"], int)

        voices_by_speaker: dict[str, set[tuple[str, str, int]]] = {}
        for item in captured["payload"]["segments"]:
            voice = item["voice"]
            voices_by_speaker.setdefault(item["speaker"], set()).add(
                (
                    voice["voice_id"],
                    voice["reference_audio"],
                    int(voice["seed"]),
                )
            )
        self.assertTrue(voices_by_speaker)
        self.assertTrue(all(len(voice_keys) == 1 for voice_keys in voices_by_speaker.values()))

    def test_uploaded_reference_is_not_replaced_by_auto_reference_task(self) -> None:
        save_uploaded_voice_reference(
            self.project_path,
            target="林宇",
            uploaded_file=UploadedVoiceFile(filename="linyu.wav", content=_tiny_wav_bytes()),
            prompt_text="上传参考文本",
        )
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "generation_mode": payload["generation_mode"],
                    "voice_references": payload["voice_references"],
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="advanced",
                runtime_overrides={"backend": "local_worker"},
            )

        self.assertNotIn("character:林宇", {task["voice_id"] for task in captured["payload"]["voice_reference_tasks"]})
        linyu_segments = [item for item in captured["payload"]["segments"] if item["speaker"] == "林宇"]
        self.assertTrue(linyu_segments)
        self.assertTrue(all(item["voice"]["reference_source"] == "uploaded" for item in linyu_segments))
        self.assertTrue(all(item["voice"]["clone_mode"] == "style_control" for item in linyu_segments))
        self.assertTrue(all(item["voice"]["prompt_text"] == "" for item in linyu_segments))
        self.assertTrue(all(item["voice"]["reference_prompt_text"] == "上传参考文本" for item in linyu_segments))

    def test_hifi_clone_mode_preserves_prompt_text_for_reference_continuation(self) -> None:
        save_uploaded_voice_reference(
            self.project_path,
            target="林宇",
            uploaded_file=UploadedVoiceFile(filename="linyu.wav", content=_tiny_wav_bytes()),
            prompt_text="上传参考文本",
        )
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "generation_mode": payload["generation_mode"],
                    "voice_references": payload["voice_references"],
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="advanced",
                runtime_overrides={"backend": "local_worker", "clone_mode": "hifi"},
            )

        linyu_segments = [item for item in captured["payload"]["segments"] if item["speaker"] == "林宇"]
        self.assertTrue(linyu_segments)
        self.assertTrue(all(item["voice"]["clone_mode"] == "hifi" for item in linyu_segments))
        self.assertTrue(all(item["voice"]["prompt_text"] == "上传参考文本" for item in linyu_segments))

    def test_llm_performance_instruction_is_applied_to_segment_voice(self) -> None:
        self.chapter_path.write_text("林宇说：“快走。”", encoding="utf-8")
        captured = {}
        llm_response = {
            "segments": [
                {"id": "unit_0001", "type": "narration", "speaker": "旁白"},
                {
                    "id": "unit_0002",
                    "type": "dialogue",
                    "speaker": "林宇",
                    "performance_instruction": "急促，压低声音",
                },
            ]
        }

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch(
            "audiobook_manager.generate_text_with_metadata",
            return_value=(json.dumps(llm_response, ensure_ascii=False), {"usage": {}}),
        ), patch("audiobook_manager._run_worker", side_effect=fake_run_worker):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                llm_config={
                    "model_provider": "ollama",
                    "model": "llama3.2",
                    "api_base": "http://127.0.0.1:11434/v1",
                },
                runtime_overrides={"backend": "local_worker"},
            )

        linyu_segment = next(item for item in captured["payload"]["segments"] if item["speaker"] == "林宇")
        self.assertEqual(linyu_segment["performance_instruction"], "急促，压低声音")
        self.assertEqual(linyu_segment["voice"]["prompt_text"], "")
        self.assertEqual(linyu_segment["voice"]["clone_mode"], "style_control")
        self.assertIn("本句表演：急促，压低声音", linyu_segment["voice"]["control_instruction"])

    def test_audio_frame_generates_reference_before_segments(self) -> None:
        calls = []

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                calls.append(kwargs)
                return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}

        with patch("audiobook_manager.AudioFrameClient", FakeAudioFrameClient):
            manifest = generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="simple",
                runtime_overrides={"backend": "audio_frame", "audio_frame_api_base": "http://127.0.0.1:9999"},
            )

        self.assertEqual(manifest["generation_mode"], "simple")
        self.assertGreater(len(calls), 1)
        self.assertEqual(calls[0]["reference_audio"], "")
        self.assertIn("参考音频", calls[0]["text"])
        self.assertTrue(calls[1]["reference_audio"])
        self.assertEqual(calls[1]["prompt_text"], "")
        self.assertTrue(calls[1]["control_instruction"])
        voices = load_json(str(self.project_path / "audiobook" / "voices.json"))
        self.assertEqual(voices["simple_voice"]["reference_source"], "auto")
        self.assertTrue((self.project_path / voices["simple_voice"]["reference_audio"]).exists())

    def test_audio_frame_parallel_runtime_round_robins_api_bases(self) -> None:
        calls = []

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                calls.append((self.api_base, kwargs))
                return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}

        with patch("audiobook_manager.AudioFrameClient", FakeAudioFrameClient):
            manifest = generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="simple",
                runtime_overrides={
                    "backend": "audio_frame",
                    "audio_frame_api_bases": "http://127.0.0.1:8810 http://127.0.0.1:8811 http://127.0.0.1:8812",
                    "audio_frame_workers": 3,
                },
            )

        used_bases = {api_base for api_base, kwargs in calls if kwargs["reference_audio"]}
        self.assertEqual(
            used_bases,
            {"http://127.0.0.1:8810", "http://127.0.0.1:8811", "http://127.0.0.1:8812"},
        )
        self.assertEqual(manifest["audio_frame"]["workers"], 3)
        self.assertEqual(len(manifest["audio_frame"]["api_bases"]), 3)

    def test_audio_frame_weighted_scheduler_routes_short_segments_to_cpu(self) -> None:
        calls = []
        planned_segments = [
            {"segment_id": "segment_0001", "index": 1, "type": "narration", "speaker": "旁白", "text": "短句。"},
            {
                "segment_id": "segment_0002",
                "index": 2,
                "type": "narration",
                "speaker": "旁白",
                "text": "这是一段明显更长的旁白内容，用来确认调度器会把主要工作继续交给 GPU 处理。",
            },
        ]

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                calls.append((self.api_base, kwargs))
                return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}

        with patch("audiobook_manager.parse_chapter_segments", return_value=planned_segments), patch(
            "audiobook_manager.AudioFrameClient", FakeAudioFrameClient
        ):
            manifest = generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="simple",
                runtime_overrides={
                    "backend": "audio_frame",
                    "audio_frame_endpoints": [
                        {"api_base": "http://gpu.local", "kind": "gpu", "capacity": 1, "speed": 1.0},
                        {
                            "api_base": "http://cpu.local",
                            "kind": "cpu",
                            "capacity": 2,
                            "max_chars": 24,
                            "speed": 1.0,
                        },
                    ],
                },
            )

        by_text = {item["text"]: item["audio_frame"]["kind"] for item in manifest["segments"]}
        self.assertEqual(by_text["短句。"], "cpu")
        self.assertEqual(by_text[planned_segments[1]["text"]], "gpu")
        segment_calls = [(base, kwargs) for base, kwargs in calls if kwargs["reference_audio"]]
        self.assertTrue(any(base == "http://cpu.local" for base, _ in segment_calls))
        self.assertFalse(any(base == "http://cpu.local" and len(kwargs["text"]) > 24 for base, kwargs in segment_calls))

    def test_audio_frame_weighted_scheduler_avoids_cpu_tail_drag(self) -> None:
        planned_segments = [
            {
                "segment_id": f"segment_{index:04d}",
                "index": index,
                "type": "narration",
                "speaker": "旁白",
                "text": "这是二十字左右短句",
            }
            for index in range(1, 4)
        ]

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}

        with patch("audiobook_manager.parse_chapter_segments", return_value=planned_segments), patch(
            "audiobook_manager.AudioFrameClient", FakeAudioFrameClient
        ):
            manifest = generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="simple",
                runtime_overrides={
                    "backend": "audio_frame",
                    "audio_frame_endpoints": [
                        {"api_base": "http://gpu.local", "kind": "gpu", "capacity": 1, "speed": 1.0},
                        {
                            "api_base": "http://cpu.local",
                            "kind": "cpu",
                            "capacity": 2,
                            "max_chars": 24,
                            "speed": 0.15,
                        },
                    ],
                },
            )

        self.assertEqual({item["audio_frame"]["kind"] for item in manifest["segments"]}, {"gpu"})

    def test_audio_frame_batch_scheduler_runs_segments_across_chapters(self) -> None:
        (self.project_path / "chapters" / "chapter_0002.md").write_text("第二章", encoding="utf-8")
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 2
        save_json(str(self.project_path / "project.json"), project)
        chapter_1_segments = [
            {
                "segment_id": "segment_0001",
                "index": 1,
                "type": "narration",
                "speaker": "旁白",
                "text": "chapter-one-long-" + "甲" * 60,
            },
            {
                "segment_id": "segment_0002",
                "index": 2,
                "type": "narration",
                "speaker": "旁白",
                "text": "chapter-one-short",
            },
        ]
        chapter_2_segments = [
            {
                "segment_id": "segment_0001",
                "index": 1,
                "type": "narration",
                "speaker": "旁白",
                "text": "chapter-two-long-" + "乙" * 60,
            },
            {
                "segment_id": "segment_0002",
                "index": 2,
                "type": "narration",
                "speaker": "旁白",
                "text": "chapter-two-short-a",
            },
            {
                "segment_id": "segment_0003",
                "index": 3,
                "type": "narration",
                "speaker": "旁白",
                "text": "chapter-two-short-b",
            },
        ]
        timings = []
        timing_lock = threading.Lock()

        def fake_parse(chapter_text, *_args, **_kwargs):
            return chapter_2_segments if "第二章" in chapter_text else chapter_1_segments

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                text = kwargs["text"]
                start = time.monotonic()
                if kwargs["reference_audio"] and text.startswith("chapter-one-long"):
                    time.sleep(0.05)
                elif kwargs["reference_audio"]:
                    time.sleep(0.01)
                end = time.monotonic()
                with timing_lock:
                    timings.append((text, start, end, bool(kwargs["reference_audio"])))
                return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}

        with patch("audiobook_manager.parse_chapter_segments", side_effect=fake_parse), patch(
            "audiobook_manager.AudioFrameClient", FakeAudioFrameClient
        ):
            manifests = audiobook_manager.generate_audiobook_chapters(
                self.project_path,
                chapter_refs=["chapter_0001", "chapter_0002"],
                force=True,
                generation_mode="simple",
                runtime_overrides={
                    "backend": "audio_frame",
                    "audio_frame_endpoints": [
                        {"api_base": "http://gpu-a.local", "kind": "gpu", "capacity": 1},
                        {"api_base": "http://gpu-b.local", "kind": "gpu", "capacity": 1},
                    ],
                },
            )

        segment_timings = [item for item in timings if item[3]]
        first_chapter_end = max(end for text, _start, end, _is_segment in segment_timings if text.startswith("chapter-one"))
        second_chapter_start = min(
            start for text, start, _end, _is_segment in segment_timings if text.startswith("chapter-two")
        )
        self.assertLess(second_chapter_start, first_chapter_end)
        self.assertEqual([manifest["chapter_slug"] for manifest in manifests], ["chapter_0001", "chapter_0002"])
        self.assertTrue(all(manifest["audio_frame"]["scheduler"] == "batch_global_weighted" for manifest in manifests))

    def test_audio_frame_parallel_runtime_serializes_auto_reference_generation(self) -> None:
        calls = []
        lock = threading.Lock()
        in_flight_references = 0
        max_in_flight_references = 0

        class FakeAudioFrameClient:
            def __init__(self, api_base: str) -> None:
                self.api_base = api_base

            def synthesize(self, **kwargs):
                nonlocal in_flight_references, max_in_flight_references
                is_auto_reference = not kwargs["reference_audio"]
                if is_auto_reference:
                    with lock:
                        in_flight_references += 1
                        max_in_flight_references = max(max_in_flight_references, in_flight_references)
                    time.sleep(0.02)
                try:
                    calls.append((self.api_base, kwargs))
                    return {"audio_base64": base64.b64encode(_tiny_wav_bytes()).decode("ascii")}
                finally:
                    if is_auto_reference:
                        with lock:
                            in_flight_references -= 1

        with patch("audiobook_manager.AudioFrameClient", FakeAudioFrameClient):
            generate_audiobook_chapter(
                self.project_path,
                "chapter_0001",
                force=True,
                generation_mode="advanced",
                runtime_overrides={
                    "backend": "audio_frame",
                    "audio_frame_api_bases": "http://127.0.0.1:8810 http://127.0.0.1:8811 http://127.0.0.1:8812",
                    "audio_frame_workers": 3,
                },
            )

        reference_calls = [kwargs for _, kwargs in calls if not kwargs["reference_audio"]]
        self.assertGreater(len(reference_calls), 1)
        self.assertEqual(max_in_flight_references, 1)

    def test_generate_audiobook_chapter_uses_external_service_config(self) -> None:
        service_config = Path(self.temp_dir.name) / "external_services.json"
        service_config.write_text(
            json.dumps(
                {
                    "voxcpm2": {
                        "root": "/srv/VoxCPM2",
                        "python": "/srv/VoxCPM2/.venv/bin/python",
                        "model_id": "/models/VoxCPM2",
                        "device": "cpu",
                        "clone_mode": "hifi",
                        "silence_ms": 120,
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        captured = {}

        def fake_run_worker(request_path: Path, runtime: dict) -> subprocess.CompletedProcess:
            payload = load_json(str(request_path))
            captured["payload"] = payload
            save_json(
                payload["manifest_path"],
                {
                    "chapter_slug": payload["chapter_slug"],
                    "generated_at": "2026-04-20T00:00:00+00:00",
                    "status": "succeeded",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "segment_count": len(payload["segments"]),
                    "segments": payload["segments"],
                },
            )
            return subprocess.CompletedProcess(["worker"], 0, stdout="ok", stderr="")

        with patch.dict(
            os.environ,
            {
                "NOVEL_EXTERNAL_SERVICES_CONFIG": str(service_config),
                "NOVEL_VOXCPM2_ROOT": "",
                "NOVEL_VOXCPM2_PYTHON": "",
                "NOVEL_VOXCPM2_MODEL_ID": "",
                "NOVEL_VOXCPM2_DEVICE": "",
            },
        ), patch(
            "audiobook_manager._run_worker", side_effect=fake_run_worker
        ):
            generate_audiobook_chapter(self.project_path, "chapter_0001", force=True)

        runtime = captured["payload"]["runtime"]
        self.assertEqual(runtime["voxcpm_root"], "/srv/VoxCPM2")
        self.assertEqual(runtime["voxcpm_python"], "/srv/VoxCPM2/.venv/bin/python")
        self.assertEqual(runtime["model_id"], "/models/VoxCPM2")
        self.assertEqual(runtime["device"], "cpu")
        self.assertEqual(runtime["clone_mode"], "hifi")
        self.assertEqual(runtime["silence_ms"], 120)

    def test_rollback_removes_future_audiobook_records(self) -> None:
        create_state_snapshot(str(self.project_path), chapter_count=1, note="test checkpoint")
        (self.project_path / "chapters" / "chapter_0002.md").write_text("第二章", encoding="utf-8")
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 2
        save_json(str(self.project_path / "project.json"), project)
        future_dir = self.project_path / "audiobook" / "chapter_0002"
        future_dir.mkdir(parents=True)
        (future_dir / "chapter_0002.wav").write_bytes(b"fake wav")

        result = rollback_project(str(self.project_path), 1)

        self.assertIn("audiobook/chapter_0002", result["removed"]["audiobook"])
        self.assertFalse(future_dir.exists())


if __name__ == "__main__":
    unittest.main()
