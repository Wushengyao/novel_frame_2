from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import run_next_chapter_from_progression, run_next_chapters
from progression_manager import CUSTOM_PROGRESSION_OPTION_ID, generate_progression_options

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
                        "plan_summary": "先留在隔离区内强化防线。",
                        "plan_steps": ["检查门锁", "分配值守"],
                        "plan_guidance": "让三人先做防御准备。",
                        "recommended": False,
                    },
                    {
                        "option_id": "option_2",
                        "title": "试探外部走廊",
                        "plan_summary": "谨慎离开隔离区进行首次短程侦查。",
                        "plan_steps": ["规划撤退路线", "短暂外出试探"],
                        "plan_guidance": "让三人完成一次短程试探，注意紧张感与配合。",
                        "recommended": True,
                    },
                    {
                        "option_id": "option_3",
                        "title": "修复设备",
                        "plan_summary": "先修复传感设备。",
                        "plan_steps": ["拆开设备", "测试线路"],
                        "plan_guidance": "把重点放在技术协作上。",
                        "recommended": False,
                    },
                    {
                        "option_id": "option_4",
                        "title": "内部磨合",
                        "plan_summary": "先强化信任。",
                        "plan_steps": ["明确分工", "彼此试探"],
                        "plan_guidance": "增加人物互动与心理描写。",
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
            ) as mocked_progression_generate:
                session = generate_progression_options(
                    str(project_path),
                    runtime_config("chapter"),
                    objective_override="建立临时安全区，并确认是否需要进一步深入走廊",
                    user_request="我想看一次更谨慎的外出试探",
                    option_count=4,
                )
            mocked_progression_generate.assert_called_once()
            _, progression_call_kwargs = mocked_progression_generate.call_args
            self.assertEqual(progression_call_kwargs["log_context"]["phase"], "outline")
            self.assertEqual(progression_call_kwargs["log_context"]["option_count"], 4)
            self.assertEqual(session["recommended_option_id"], "option_2")
            self.assertEqual(session["objective"], "建立临时安全区，并确认是否需要进一步深入走廊")
            self.assertTrue(any(option.get("custom") for option in session["options"]))
            self.assertEqual(session["options"][-1]["option_id"], CUSTOM_PROGRESSION_OPTION_ID)

            with patch(
                "app.generate_text_with_metadata",
                return_value=("走廊外一片死寂，三人在门后短暂停顿后，小心地推门而出。", {"usage": {}}),
            ) as mocked_writer_generate, patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(summary_payload, ensure_ascii=False), {"usage": {}}),
            ) as mocked_state_generate:
                chapter_path = run_next_chapter_from_progression(
                    str(project_path),
                    runtime_config("chapter"),
                    progression_session=session["session_id"],
                    progression_option=session["recommended_option_id"],
                    progression_feedback="增加一点角色之间试探性的对话",
                )
            mocked_writer_generate.assert_called_once()
            _, writer_call_kwargs = mocked_writer_generate.call_args
            self.assertEqual(writer_call_kwargs["log_context"]["phase"], "writer")
            self.assertEqual(writer_call_kwargs["log_context"]["source"], "run_next_chapter_from_progression")

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
            self.assertEqual(task_card["objective"], "建立临时安全区，并确认是否需要进一步深入走廊")
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

    def test_auto_continue_recommended_selects_recommended_plan_and_persists_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="auto_recommended")
            options_payload = {
                "recommended_option_id": "option_2",
                "options": [
                    {
                        "option_id": "option_1",
                        "title": "保守防守",
                        "plan_summary": "先稳住隔离区防线。",
                        "plan_steps": ["检查门锁", "重新分工"],
                        "plan_guidance": "让角色先稳住局面。",
                        "recommended": False,
                    },
                    {
                        "option_id": "option_2",
                        "title": "短程试探",
                        "plan_summary": "谨慎离开隔离区进行短程侦查。",
                        "plan_steps": ["规划路线", "短暂外出试探"],
                        "plan_guidance": "让三人完成一次短程试探。",
                        "recommended": True,
                    },
                    {
                        "option_id": "option_3",
                        "title": "修设备",
                        "plan_summary": "优先修复传感设备。",
                        "plan_steps": ["拆开设备", "测试线路"],
                        "plan_guidance": "把重点放在技术协作上。",
                        "recommended": False,
                    },
                    {
                        "option_id": "option_4",
                        "title": "内部磨合",
                        "plan_summary": "先强化信任。",
                        "plan_steps": ["明确分工", "彼此试探"],
                        "plan_guidance": "增加人物互动。",
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
            ) as mocked_progression_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("走廊外一片死寂，三人在门后短暂停顿后，小心地推门而出。", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(summary_payload, ensure_ascii=False), {"usage": {}}),
            ):
                chapter_paths = run_next_chapters(
                    str(project_path),
                    runtime_config("chapter"),
                    1,
                    user_request="我想看一次更谨慎的外出试探",
                    selection_mode="recommended",
                )

            self.assertEqual(len(chapter_paths), 1)
            mocked_progression_generate.assert_called_once()
            session_files = sorted((project_path / "progression_sessions").glob("progression_*.json"))
            self.assertEqual(len(session_files), 1)
            session = read_json(session_files[0])
            self.assertEqual(session["status"], "selected")
            self.assertEqual(session["selection_mode"], "recommended")
            self.assertEqual(session["selection_origin"], "auto")
            self.assertEqual(session["selected_option_id"], "option_2")
            self.assertEqual(session["auto_batch_request"], "我想看一次更谨慎的外出试探")
            self.assertTrue(session["selected_at"])
            task_card = read_json(project_path / "task_cards" / "chapter_0001.json")
            self.assertEqual(task_card["derived_from"]["option_id"], "option_2")

    def test_auto_continue_random_in_none_mode_regenerates_objective_each_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="auto_none", planning_mode="none")
            progress_prompts: list[str] = []
            objective_texts = iter(
                [
                    "先确认隔离区是否还能继续坚守，并决定要不要外出搜集物资",
                    "前往主控区边缘尝试获取医疗与供电线索",
                ]
            )
            option_payloads = iter(
                [
                    {
                        "recommended_option_id": "option_2",
                        "options": [
                            {
                                "option_id": "option_1",
                                "title": "加固防线",
                                "plan_summary": "先围绕隔离区内部完成防御准备。",
                                "plan_steps": ["清点物资", "检查门锁"],
                                "plan_guidance": "偏稳妥推进。",
                                "recommended": False,
                            },
                            {
                                "option_id": "option_2",
                                "title": "试探外出",
                                "plan_summary": "谨慎离开隔离区做短程试探。",
                                "plan_steps": ["规划路线", "短程侦查"],
                                "plan_guidance": "保持紧张感。",
                                "recommended": True,
                            },
                            {
                                "option_id": "option_3",
                                "title": "修设备",
                                "plan_summary": "优先修复广播与传感设备。",
                                "plan_steps": ["拆机", "测试"],
                                "plan_guidance": "突出协作。",
                                "recommended": False,
                            },
                            {
                                "option_id": "option_4",
                                "title": "内部磨合",
                                "plan_summary": "先处理队内分歧。",
                                "plan_steps": ["重新分工", "交换情报"],
                                "plan_guidance": "增加对话张力。",
                                "recommended": False,
                            },
                        ],
                    },
                    {
                        "recommended_option_id": "option_1",
                        "options": [
                            {
                                "option_id": "option_1",
                                "title": "边缘搜药",
                                "plan_summary": "前往主控区边缘搜索药品与电力线索。",
                                "plan_steps": ["绕开危险区域", "搜集药品"],
                                "plan_guidance": "突出风险与收获。",
                                "recommended": True,
                            },
                            {
                                "option_id": "option_2",
                                "title": "设备试探",
                                "plan_summary": "先试着恢复部分设备再外出。",
                                "plan_steps": ["恢复电路", "读取残余日志"],
                                "plan_guidance": "偏技术推进。",
                                "recommended": False,
                            },
                            {
                                "option_id": "option_3",
                                "title": "抓俘问话",
                                "plan_summary": "控制可疑幸存者并逼问情报。",
                                "plan_steps": ["布置陷阱", "审问对方"],
                                "plan_guidance": "强化压迫感。",
                                "recommended": False,
                            },
                            {
                                "option_id": "option_4",
                                "title": "诱敌试探",
                                "plan_summary": "用诱饵测试外部危险反应。",
                                "plan_steps": ["准备诱饵", "观察反馈"],
                                "plan_guidance": "保持悬念。",
                                "recommended": False,
                            },
                        ],
                    },
                ]
            )

            def fake_progression_generate(prompt, config, log_context=None):
                progress_prompts.append(prompt)
                if log_context and log_context.get("prompt_type") == "auto_objective":
                    return (json.dumps({"objective": next(objective_texts)}, ensure_ascii=False), {"usage": {}})
                return (json.dumps(next(option_payloads), ensure_ascii=False), {"usage": {}})

            summary_payloads = iter(
                [
                    {
                        "recent_events": ["三人短程外出后确认主控区方向可能有补给。"],
                        "open_threads": ["主控区边缘仍存在未知危险"],
                        "foreshadowing": ["主控区可能藏有医疗与供电线索"],
                        "character_updates": ["苏浅开始主动推动外出决策"],
                        "next_chapter_goal": "前往主控区边缘尝试获取医疗与供电线索",
                    },
                    {
                        "recent_events": ["三人带回药品并锁定新的供电线索。"],
                        "open_threads": ["主控区深处的威胁仍未查明"],
                        "foreshadowing": ["供电线索可能连接更深层区域"],
                        "character_updates": ["林宇决定继续深入调查"],
                        "next_chapter_goal": "决定是否进一步深入主控区",
                    },
                ]
            )

            with patch(
                "progression_manager.generate_text_with_metadata",
                side_effect=fake_progression_generate,
            ), patch(
                "progression_manager.random.choice",
                side_effect=lambda options: options[-1],
            ), patch(
                "app.generate_text_with_metadata",
                side_effect=[
                    ("第一章正文", {"usage": {}}),
                    ("第二章正文", {"usage": {}}),
                ],
            ), patch(
                "state_updater.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(next(summary_payloads), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(next(summary_payloads), ensure_ascii=False), {"usage": {}}),
                ],
            ):
                chapter_paths = run_next_chapters(
                    str(project_path),
                    runtime_config("none"),
                    2,
                    user_request="我想看他们逐步把目标转向主控区补给线索",
                    selection_mode="random",
                )

            self.assertEqual(len(chapter_paths), 2)
            self.assertEqual(len(progress_prompts), 4)
            self.assertIn("前往主控区边缘尝试获取医疗与供电线索", progress_prompts[2])
            session_files = sorted((project_path / "progression_sessions").glob("progression_*.json"))
            self.assertEqual(len(session_files), 2)
            sessions = sorted(
                (read_json(path) for path in session_files),
                key=lambda payload: payload["target_chapter_number"],
            )
            self.assertEqual([session["selection_mode"] for session in sessions], ["random", "random"])
            self.assertEqual([session["selection_origin"] for session in sessions], ["auto", "auto"])
            self.assertEqual([session["selected_option_id"] for session in sessions], ["option_4", "option_4"])
            self.assertTrue(all(session["selected_at"] for session in sessions))
            self.assertEqual(
                [session["objective"] for session in sessions],
                [
                    "先确认隔离区是否还能继续坚守，并决定要不要外出搜集物资",
                    "前往主控区边缘尝试获取医疗与供电线索",
                ],
            )
            first_task = read_json(project_path / "task_cards" / "chapter_0001.json")
            second_task = read_json(project_path / "task_cards" / "chapter_0002.json")
            self.assertEqual(first_task["objective"], "先确认隔离区是否还能继续坚守，并决定要不要外出搜集物资")
            self.assertEqual(second_task["objective"], "前往主控区边缘尝试获取医疗与供电线索")
            self.assertEqual(first_task["derived_from"]["option_id"], "option_4")
            self.assertEqual(second_task["derived_from"]["option_id"], "option_4")


if __name__ == "__main__":
    unittest.main()
