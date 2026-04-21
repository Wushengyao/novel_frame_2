from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import run_next_chapter_from_progression
from progression_manager import generate_progression_options

from tests.test_support import create_test_project, read_json, runtime_config


class GuidedFlowTests(unittest.TestCase):
    def test_guided_flow_generates_session_and_writes_next_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="flow")
            options_payload = {
                "recommended_option_id": "option_2",
                "options": [
                    {
                        "option_id": "option_1",
                        "title": "加固据点",
                        "summary": "先留在隔离区内强化防线。",
                        "why_now": "外部环境仍不稳定。",
                        "key_events": ["检查门锁", "分配值守"],
                        "writer_guidance": "让三人先做防御准备。",
                        "chapter_outline": {
                            "title": "加固据点",
                            "summary": "三人内部整备。",
                            "goal": "稳住据点",
                            "key_events": ["检查门锁", "分配值守"],
                        },
                        "recommended": False,
                    },
                    {
                        "option_id": "option_2",
                        "title": "试探外部走廊",
                        "summary": "谨慎离开隔离区进行首次短程侦查。",
                        "why_now": "必须尽快知道外部风险和可用资源。",
                        "key_events": ["规划撤退路线", "短暂外出试探"],
                        "writer_guidance": "让三人完成一次短程试探，注意紧张感与配合。",
                        "chapter_outline": {
                            "title": "试探外部走廊",
                            "summary": "三人完成第一次短程侦查。",
                            "goal": "获取第一手外部情报",
                            "key_events": ["规划撤退路线", "短暂外出试探"],
                        },
                        "recommended": True,
                    },
                    {
                        "option_id": "option_3",
                        "title": "修复设备",
                        "summary": "先修复传感设备。",
                        "why_now": "没有设备支持很难长期生存。",
                        "key_events": ["拆开设备", "测试线路"],
                        "writer_guidance": "把重点放在技术协作上。",
                        "chapter_outline": {
                            "title": "修复设备",
                            "summary": "团队尝试修复设备。",
                            "goal": "恢复基础监测能力",
                            "key_events": ["拆开设备", "测试线路"],
                        },
                        "recommended": False,
                    },
                    {
                        "option_id": "option_4",
                        "title": "内部磨合",
                        "summary": "先强化信任。",
                        "why_now": "没有默契很难执行外出行动。",
                        "key_events": ["明确分工", "彼此试探"],
                        "writer_guidance": "增加人物互动与心理描写。",
                        "chapter_outline": {
                            "title": "内部磨合",
                            "summary": "三人在隔离区内重新确认分工。",
                            "goal": "建立更稳定的合作关系",
                            "key_events": ["明确分工", "彼此试探"],
                        },
                        "recommended": False,
                    },
                ],
            }
            summary_payload = {
                "recent_events": ["三人完成了第一次短程侦查。"],
                "open_threads": ["走廊尽头的异常信号来源未明"],
                "foreshadowing": ["异常信号可能连接主控区"],
                "character_updates": ["林宇开始主动承担风险"],
                "next_chapter_goal": "决定是否进一步深入走廊",
            }

            with patch(
                "progression_manager.generate_text_with_metadata",
                return_value=(json.dumps(options_payload, ensure_ascii=False), {"usage": {}}),
            ):
                session = generate_progression_options(
                    str(project_path),
                    runtime_config("chapter"),
                    user_request="我想看一次更谨慎的外出试探",
                    option_count=4,
                )

            with patch(
                "app.generate_text_with_metadata",
                return_value=("走廊外一片死寂，三人在门后短暂停顿后，小心地推门而出。", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(summary_payload, ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter_from_progression(
                    str(project_path),
                    runtime_config("chapter"),
                    progression_session=session["session_id"],
                    progression_option=session["recommended_option_id"],
                    progression_feedback="增加一点角色之间试探性的对话",
                )

            self.assertTrue(Path(chapter_path).exists())
            project = read_json(project_path / "project.json")
            self.assertEqual(project["chapter_count"], 1)
            plot_state = read_json(project_path / "plot_state.json")
            self.assertEqual(plot_state["recent_events"], ["三人完成了第一次短程侦查。"])
            self.assertEqual(plot_state["next_chapter_goal"], "决定是否进一步深入走廊")
            outlines = read_json(project_path / "outlines.json")
            self.assertEqual(outlines["volumes"][0]["chapters"][0]["status"], "completed")
            self.assertEqual(outlines["volumes"][0]["chapters"][0]["goal"], "建立临时安全区")
            task_card = read_json(project_path / "task_cards" / "chapter_0001.json")
            self.assertEqual(task_card["source"], "progression_selected")
            self.assertTrue((project_path / "snapshots" / "chapter_0001").exists())

    def test_cli_rejects_guided_batch_count_greater_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="cli")
            config_path = Path(tmp) / "runtime.json"
            config_path.write_text(json.dumps(runtime_config("chapter"), ensure_ascii=False, indent=2), encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(Path(__file__).resolve().parents[1] / "app.py"),
                    "next",
                    "--project",
                    str(project_path),
                    "--config",
                    str(config_path),
                    "--count",
                    "2",
                    "--progression-session",
                    "session_x",
                    "--progression-option",
                    "option_1",
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("guided progression only supports --count 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
