from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_workflows import init_project_agentic
from tests.test_support import read_json


class AgentWorkflowTests(unittest.TestCase):
    def test_agentic_init_writes_project_artifacts_snapshot_and_agent_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config_path = base / "config.json"
            config = {
                "project_id": "agentic_init",
                "project_name": "Agentic Init",
                "project_description": "Agentic init test.",
                "project_path": str(base / "novel_project_{project_id}"),
                "init_with_llm": True,
                "story_request": "空间站入侵后，三名幸存者建立避难据点。",
                "planning_mode": "chapter",
                "workflow_mode": "agentic",
                "model_provider": "openai_compatible",
                "model_name": "test-model",
                "api_base": "https://example.local/v1",
                "api_key": "test-key",
                "max_tokens": 4000,
                "timeout": 120,
            }
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            setup_payload = {
                "world": {
                    "title": "星环余烬",
                    "genre": "科幻生存",
                    "setting": "被异族占领的高级太空站",
                    "background": ["隔离区暂时安全。"],
                    "rules": ["安全门需要能源配额。"],
                },
                "characters": {
                    "protagonists": [
                        {"name": "林宇", "role": "男主", "description": "团队力量担当", "appearance": "黑发青年"},
                        {"name": "苏浅", "role": "女主", "description": "团队智力担当", "appearance": "短发少女"},
                    ],
                    "supporting": [],
                },
            }
            init_payload = {
                "world": {},
                "characters": {},
                "plot_state": {
                    "main_plot": "幸存者在太空站中建立据点并寻找出路",
                    "current_arc": "开篇阶段",
                    "active_characters": ["林宇", "苏浅"],
                    "current_location": "隔离区",
                    "current_time": "入侵后第48小时",
                    "next_chapter_goal": "建立临时安全区",
                },
                "style": {
                    "tone": "紧张中带温情",
                    "pov": "第三人称",
                    "requirements": ["重视协作细节"],
                },
            }
            volume_payload = {
                "volumes": [
                    {
                        "volume_number": 1,
                        "title": "隔离区",
                        "summary": "幸存者建立初始据点。",
                        "story_goal": "建立安全据点",
                        "planned_chapter_count": 1,
                    }
                ]
            }
            chapter_payload = {
                "chapters": [
                    {
                        "chapter_in_volume": 1,
                        "title": "死寂的隔离区",
                        "summary": "林宇和苏浅确认避难点安全。",
                        "goal": "建立临时安全区",
                        "key_events": ["封门", "分工"],
                    }
                ]
            }

            with patch(
                "project_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(setup_payload, ensure_ascii=False), {"usage": {}}),
                    (json.dumps(init_payload, ensure_ascii=False), {"usage": {}}),
                ],
            ), patch(
                "outline_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(volume_payload, ensure_ascii=False), {"usage": {}}),
                    (json.dumps(chapter_payload, ensure_ascii=False), {"usage": {}}),
                ],
            ):
                project_path = Path(init_project_agentic(str(config_path)))

            self.assertTrue((project_path / "project.json").exists())
            self.assertTrue((project_path / "world.json").exists())
            self.assertTrue((project_path / "characters.json").exists())
            self.assertTrue((project_path / "plot_state.json").exists())
            self.assertTrue((project_path / "style.json").exists())
            self.assertTrue((project_path / "outlines.json").exists())
            self.assertTrue((project_path / "snapshots" / "chapter_0000" / "snapshot.json").exists())

            project = read_json(project_path / "project.json")
            self.assertEqual(project["workflow_mode"], "agentic")
            self.assertEqual(project["llm_config"]["workflow_mode"], "agentic")
            self.assertEqual(project["llm_config"]["api_key"], "")

            run_files = sorted((project_path / "agent_runs").glob("agent_run_*.json"))
            self.assertEqual(len(run_files), 1)
            agent_run = read_json(run_files[0])
            self.assertEqual(agent_run["workflow"], "init_project")
            self.assertEqual(agent_run["status"], "succeeded")
            skill_ids = [event["skill_id"] for event in agent_run["skill_events"]]
            self.assertIn("init.generate_story_data", skill_ids)
            self.assertIn("init.write_project_files", skill_ids)
            self.assertIn("outline.regenerate_volume", skill_ids)
            self.assertIn("outline.regenerate_chapter", skill_ids)
            self.assertIn("snapshot.create_post", skill_ids)


if __name__ == "__main__":
    unittest.main()
