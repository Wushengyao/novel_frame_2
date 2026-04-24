from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
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
            manifest = generate_audiobook_chapter(self.project_path, "chapter_0001", force=True)

        self.assertEqual(manifest["chapter_slug"], "chapter_0001")
        self.assertEqual(captured["payload"]["chapter_slug"], "chapter_0001")
        self.assertGreaterEqual(len(captured["payload"]["segments"]), 3)
        self.assertIn("voice", captured["payload"]["segments"][0])

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
