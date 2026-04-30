from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from polish_manager import run_chapter_polish
from progression_manager import save_progression_session
from project_manager import load_json, save_json
from tests.test_support import create_test_project, read_json, runtime_config


class PolishManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_path = create_test_project(Path(self.temp_dir.name), project_id="polish")
        self.chapter_path = self.project_path / "chapters" / "chapter_0001.md"
        self.original_text = (
            "林宇把最后一只储物箱推到门后，金属箱底擦过地面，留下低而哑的响声。\n\n"
            "苏浅蹲在控制板旁，指尖飞快划过裂开的屏幕。她没有抬头，只低声提醒他门锁还差一次校验。"
        )
        self.chapter_path.write_text(self.original_text + "\n", encoding="utf-8")
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)

    def test_run_chapter_polish_overwrites_chapter_and_writes_backup(self) -> None:
        save_progression_session(
            str(self.project_path),
            {
                "session_id": "session_polish",
                "created_at": "2026-04-20T00:00:00+00:00",
                "project_chapter_count": 1,
                "target_chapter_number": 2,
                "planning_mode": "chapter",
                "runtime_overrides": {},
                "recommended_option_id": "option_1",
                "options": [],
                "status": "pending",
            },
        )
        captured: dict[str, object] = {}
        polished = (
            "林宇把最后一只储物箱稳稳推到门后，金属箱底擦过地面，拖出一声低哑的回响。\n\n"
            "苏浅蹲在控制板旁，指尖掠过裂开的屏幕，语气比警报声还镇定：门锁还差最后一次校验。"
        )

        def fake_generate(prompt: str, config: dict, log_context=None, system_prompt: str = "", response_format: str = ""):
            captured["prompt"] = prompt
            captured["config"] = config
            captured["log_context"] = log_context
            captured["system_prompt"] = system_prompt
            return polished, {"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

        with patch("polish_manager.generate_text_with_metadata", side_effect=fake_generate):
            result = run_chapter_polish(
                str(self.project_path),
                runtime_config("chapter"),
                "chapter_0001",
                preset_ids=["details", "cheerful", "unknown"],
                custom_request="多一点轻松互怼",
            )

        self.assertEqual(self.chapter_path.read_text(encoding="utf-8").strip(), polished)
        self.assertIn("细节增强", str(captured["prompt"]))
        self.assertIn("更欢乐", str(captured["prompt"]))
        self.assertIn("多一点轻松互怼", str(captured["prompt"]))
        self.assertIn("润色", str(captured["system_prompt"]))
        self.assertEqual(captured["log_context"]["phase"], "polish")
        self.assertEqual(result["staled_progression_sessions"], 1)

        backup_path = Path(result["backup_path"])
        metadata_path = Path(result["metadata_path"])
        self.assertTrue(backup_path.exists())
        self.assertTrue(metadata_path.exists())
        self.assertEqual(backup_path.read_text(encoding="utf-8").strip(), self.original_text)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["preset_ids"], ["details", "cheerful"])
        self.assertEqual(metadata["custom_request"], "多一点轻松互怼")

        saved_session = read_json(self.project_path / "progression_sessions" / "progression_session_polish.json")
        self.assertEqual(saved_session["status"], "stale")

    def test_run_chapter_polish_rejects_invalid_or_missing_chapter(self) -> None:
        with self.assertRaises(ValueError):
            run_chapter_polish(str(self.project_path), runtime_config("chapter"), "../chapter_0001")

        with self.assertRaises(FileNotFoundError):
            run_chapter_polish(str(self.project_path), runtime_config("chapter"), "chapter_9999")

    def test_run_chapter_polish_keeps_original_when_model_returns_empty_text(self) -> None:
        with patch("polish_manager.generate_text_with_metadata", return_value=("", {"usage": {}})):
            with self.assertRaises(ValueError):
                run_chapter_polish(
                    str(self.project_path),
                    runtime_config("chapter"),
                    "chapter_0001",
                    preset_ids=[],
                    custom_request="",
                )

        self.assertEqual(self.chapter_path.read_text(encoding="utf-8").strip(), self.original_text)
        backup_root = self.project_path / "polish_backups" / "chapter_0001"
        self.assertFalse(backup_root.exists())

    def test_run_chapter_polish_keeps_original_when_response_is_truncated(self) -> None:
        with patch(
            "polish_manager.generate_text_with_metadata",
            return_value=("半截润色正文", {"usage": {"total_tokens": 4000}, "finish_reason": "length", "truncated": True}),
        ):
            with self.assertRaises(RuntimeError):
                run_chapter_polish(
                    str(self.project_path),
                    runtime_config("chapter"),
                    "chapter_0001",
                    preset_ids=[],
                    custom_request="",
                )

        self.assertEqual(self.chapter_path.read_text(encoding="utf-8").strip(), self.original_text)
        backup_root = self.project_path / "polish_backups" / "chapter_0001"
        self.assertFalse(backup_root.exists())


if __name__ == "__main__":
    unittest.main()
