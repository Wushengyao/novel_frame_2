from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from progression_manager import (
    CUSTOM_PROGRESSION_OPTION_ID,
    ensure_fresh_progression_session,
    normalize_progression_options_response,
    resolve_progression_selection,
    save_progression_session,
    validate_option_count,
)

from tests.test_support import create_test_project, read_json


class ProgressionManagerTests(unittest.TestCase):
    def test_validate_option_count_rejects_unsupported_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_option_count(2)

    def test_normalize_progression_options_requires_exactly_one_recommended(self) -> None:
        payload = {
            "options": [
                {
                    "option_id": "option_1",
                    "title": "A",
                    "summary": "A",
                    "key_events": ["1", "2"],
                    "writer_guidance": "A",
                    "recommended": False,
                },
                {
                    "option_id": "option_2",
                    "title": "B",
                    "summary": "B",
                    "key_events": ["1", "2"],
                    "writer_guidance": "B",
                    "recommended": False,
                },
                {
                    "option_id": "option_3",
                    "title": "C",
                    "summary": "C",
                    "key_events": ["1", "2"],
                    "writer_guidance": "C",
                    "recommended": False,
                },
            ]
        }

        with self.assertRaises(ValueError):
            normalize_progression_options_response(payload, 3)

    def test_resolve_progression_selection_requires_custom_idea_for_blank_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp))
            session = {
                "session_id": "session_custom",
                "created_at": "2026-04-20T00:00:00+00:00",
                "project_chapter_count": 0,
                "target_chapter_number": 1,
                "planning_mode": "chapter",
                "source_user_request": "",
                "runtime_overrides": {},
                "recommended_option_id": "option_1",
                "options": [
                    {
                        "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                        "title": "空白自定义项",
                        "summary": "由用户自己定义",
                        "key_events": ["用户定义本章主要推进", "保持与当前状态一致"],
                        "writer_guidance": "请以用户随后填写的创意为准。",
                        "recommended": False,
                        "custom": True,
                    }
                ],
                "status": "pending",
                "selected_option_id": "",
                "selection_feedback": "",
            }
            save_progression_session(str(project_path), session)

            with self.assertRaises(ValueError):
                resolve_progression_selection(
                    str(project_path),
                    "session_custom",
                    CUSTOM_PROGRESSION_OPTION_ID,
                    selection_feedback="",
                )

    def test_stale_session_is_marked_when_chapter_count_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp))
            session = {
                "session_id": "session_1",
                "created_at": "2026-04-20T00:00:00+00:00",
                "project_chapter_count": 0,
                "target_chapter_number": 1,
                "planning_mode": "chapter",
                "source_user_request": "",
                "runtime_overrides": {},
                "recommended_option_id": "option_1",
                "options": [],
                "status": "pending",
                "selected_option_id": "",
                "selection_feedback": "",
            }
            save_progression_session(str(project_path), session)

            project = read_json(project_path / "project.json")
            project["chapter_count"] = 1
            (project_path / "project.json").write_text(
                json.dumps(project, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            refreshed = ensure_fresh_progression_session(str(project_path), session)
            self.assertEqual(refreshed["status"], "stale")
            saved = read_json(project_path / "progression_sessions" / "progression_session_1.json")
            self.assertEqual(saved["status"], "stale")

    def test_resolve_progression_selection_persists_progression_task_without_touching_outline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp))
            original_outlines = read_json(project_path / "outlines.json")
            original_plot_state = read_json(project_path / "plot_state.json")
            session = {
                "session_id": "session_override",
                "created_at": "2026-04-20T00:00:00+00:00",
                "project_chapter_count": 0,
                "target_chapter_number": 1,
                "planning_mode": "chapter",
                "source_user_request": "先侦查",
                "runtime_overrides": {},
                "recommended_option_id": "option_1",
                "options": [
                    {
                        "option_id": "option_1",
                        "title": "主动侦查",
                        "summary": "先外出试探",
                        "key_events": ["试探通道", "收集情报"],
                        "writer_guidance": "让角色谨慎外出。",
                        "recommended": True,
                    }
                ],
                "status": "pending",
                "selected_option_id": "",
                "selection_feedback": "",
            }
            save_progression_session(str(project_path), session)

            selection = resolve_progression_selection(
                str(project_path),
                "session_override",
                "option_1",
                selection_feedback="增加一点角色互相试探的对话",
            )

            outlines = read_json(project_path / "outlines.json")
            self.assertEqual(outlines, original_outlines)
            self.assertEqual(selection["session"]["status"], "selected")
            task_card = read_json(project_path / "task_cards" / "chapter_0001.json")
            self.assertEqual(task_card["source"], "progression_selected")
            self.assertEqual(task_card["summary"], "先外出试探")
            self.assertEqual(task_card["goal"], "建立临时安全区")
            self.assertIn("用户补充细化", task_card["writer_guidance"])
            self.assertIn("既定目标", task_card["writer_guidance"])
            self.assertEqual(task_card["derived_from"]["option_id"], "option_1")
            self.assertEqual(task_card["derived_from"]["baseline_source"], "chapter_outline")
            plot_state = read_json(project_path / "plot_state.json")
            self.assertEqual(plot_state, original_plot_state)

    def test_resolve_progression_selection_persists_custom_progression_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp))
            session = {
                "session_id": "session_custom_apply",
                "created_at": "2026-04-20T00:00:00+00:00",
                "project_chapter_count": 0,
                "target_chapter_number": 1,
                "planning_mode": "chapter",
                "source_user_request": "",
                "runtime_overrides": {},
                "recommended_option_id": "option_1",
                "options": [
                    {
                        "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                        "title": "空白自定义项",
                        "summary": "由用户自己定义",
                        "key_events": ["用户定义本章主要推进", "保持与当前状态一致"],
                        "writer_guidance": "请以用户随后填写的创意为准。",
                        "recommended": False,
                        "custom": True,
                    }
                ],
                "status": "pending",
                "selected_option_id": "",
                "selection_feedback": "",
            }
            save_progression_session(str(project_path), session)

            resolve_progression_selection(
                str(project_path),
                "session_custom_apply",
                CUSTOM_PROGRESSION_OPTION_ID,
                selection_feedback="这一章我想看他们冒险去主控区边缘搜集药品，并在途中爆发一次激烈争执。",
            )

            task_card = read_json(project_path / "task_cards" / "chapter_0001.json")
            self.assertEqual(task_card["source"], "progression_selected")
            self.assertEqual(task_card["derived_from"]["option_id"], CUSTOM_PROGRESSION_OPTION_ID)
            self.assertIn("主控区边缘搜集药品", task_card["summary"])
            self.assertIn("用户自定义创意", task_card["writer_guidance"])


if __name__ == "__main__":
    unittest.main()
