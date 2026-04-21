from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from context_builder import (
    WRITER_HARD_TOTAL_CHARS,
    _apply_writer_total_budget,
    build_batch_plan_context,
    build_chapter_outline_context,
    build_chapter_task_card,
    build_progression_context,
    build_writer_context,
    select_recent_scene_window,
)
from prompt_builder import (
    build_batch_chapter_plan_prompt,
    build_chapter_outline_prompt,
    build_progression_options_prompt,
    build_writer_prompt,
)
from project_manager import load_json, load_project, save_json
from state_updater import update_plot_state

from tests.test_support import create_test_project, runtime_config


class ContextBuilderTests(unittest.TestCase):
    def test_select_recent_scene_window_keeps_paragraph_boundaries(self) -> None:
        text = (
            "第一段写角色在隔离区里检查门锁和呼吸声，长度足够长。\n\n"
            "第二段写他们沿着走廊缓慢前进，互相确认撤退路线，长度也足够长。\n\n"
            "第三段写他们在门缝后观察异常信号，并准备进一步试探。"
        )

        window = select_recent_scene_window(text, min_chars=35, max_chars=90)

        self.assertTrue(window.startswith("第三段") or window.startswith("第二段"))
        self.assertIn("\n\n", window)
        self.assertNotIn("第一段写角色在隔离区里检查门锁和呼吸声", window)

    def test_writer_budget_trimming_keeps_high_priority_sections(self) -> None:
        sections = {
            "author_intent": "A" * 600,
            "chapter_task": "B" * 500,
            "live_state": "C" * 2000,
            "retrieved_memory": "D" * 2200,
            "recent_scene": "E" * 3600,
            "style_contract": "F" * 400,
            "static_world": "G" * 700,
            "static_characters": "H" * 900,
        }

        trimmed = _apply_writer_total_budget(sections)

        self.assertEqual(trimmed["author_intent"], sections["author_intent"])
        self.assertEqual(trimmed["chapter_task"], sections["chapter_task"])
        self.assertLess(len(trimmed["retrieved_memory"]), len(sections["retrieved_memory"]))
        self.assertLess(len(trimmed["recent_scene"]), len(sections["recent_scene"]))
        self.assertLessEqual(sum(len(value) for value in trimmed.values()), WRITER_HARD_TOTAL_CHARS)

    def test_update_plot_state_refreshes_live_state_and_generates_arc_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="summary")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 5
            save_json(str(project_path / "project.json"), project)

            for chapter_number in range(1, 5):
                save_json(
                    str(project_path / "summaries" / f"summary_{chapter_number:04d}.json"),
                    {
                        "chapter_summary": f"第{chapter_number}章推进了据点建设。",
                        "current_location": "旧隔离区",
                        "current_time": f"入侵后第{chapter_number}天",
                        "current_arc": "据点建设",
                        "recent_events": [f"事件{chapter_number}"],
                        "open_threads": ["旧门锁风险"],
                        "resolved_threads": [],
                        "foreshadowing": [f"伏笔{chapter_number}"],
                        "character_updates": [f"更新{chapter_number}"],
                        "active_characters": ["林宇", "苏浅"],
                        "retrieval_tags": ["据点", "门锁"],
                        "next_chapter_goal": "继续稳固据点",
                    },
                )

            summary_payload = {
                "chapter_summary": "三人完成主控区入口的临时封锁，并锁定异常信号来自更深层区域。",
                "current_location": "主控区外围维护走廊",
                "current_time": "入侵后第5天夜间",
                "current_arc": "主控区试探",
                "recent_events": ["三人完成临时封锁", "异常信号被重新定位"],
                "open_threads": ["旧门锁风险", "异常信号源尚未确认"],
                "resolved_threads": ["旧门锁风险"],
                "foreshadowing": ["异常信号可能连接主控区更深处"],
                "character_updates": ["林宇开始承担诱敌风险"],
                "active_characters": ["林宇", "苏浅", "叶宁"],
                "retrieval_tags": ["主控区", "异常信号", "封锁"],
                "next_chapter_goal": "进入更深层区域确认信号源",
            }

            with patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(summary_payload, ensure_ascii=False), {"usage": {}}),
            ):
                update_plot_state(str(project_path), "新章节正文", runtime_config("chapter"))

            plot_state = load_json(str(project_path / "plot_state.json"))
            self.assertEqual(plot_state["current_location"], "主控区外围维护走廊")
            self.assertEqual(plot_state["current_time"], "入侵后第5天夜间")
            self.assertEqual(plot_state["current_arc"], "主控区试探")
            self.assertNotIn("旧门锁风险", plot_state["open_threads"])
            self.assertIn("旧门锁风险", plot_state["resolved_threads"])

            summary_file = load_json(str(project_path / "summaries" / "summary_0005.json"))
            self.assertEqual(summary_file["chapter_summary"], summary_payload["chapter_summary"])

            arc_summary = load_json(str(project_path / "arc_summaries" / "arc_0001.json"))
            self.assertEqual(arc_summary["arc_index"], 1)
            self.assertEqual(arc_summary["current_arc"], "主控区试探")

    def test_volume_mode_persists_lightweight_task_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="volume", planning_mode="volume")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": {
                    "chapter_number": 1,
                    "chapter_in_volume": 1,
                    "title": "",
                    "summary": "先加固隔离区并评估外部噪音来源。",
                    "goal": "",
                    "key_events": [],
                },
            }

            task_card = build_chapter_task_card(
                str(project_path),
                project_data,
                next_context,
                planning_mode="volume",
                user_request="先强化门禁，再试探异常噪音",
            )

            self.assertEqual(task_card["source"], "volume_outline")
            saved = load_json(str(project_path / "task_cards" / "chapter_0001.json"))
            self.assertEqual(saved["goal"], task_card["goal"])

    def test_progression_selected_task_card_is_highest_priority_and_deduplicates_next_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="selected")
            save_json(
                str(project_path / "task_cards" / "chapter_0001.json"),
                {
                    "chapter_number": 1,
                    "planning_mode": "chapter",
                    "source": "progression_selected",
                    "title": "先试探外部",
                    "summary": "先离开隔离区做一次短程侦查。",
                    "goal": "完成一次谨慎的外出试探",
                    "key_events": ["规划撤退路线", "短程外出试探"],
                    "volume_title": "第一卷",
                    "volume_goal": "建立据点",
                    "writer_guidance": "保持紧张感与配合。",
                    "derived_from": {
                        "session_id": "session_a",
                        "option_id": "option_2",
                        "base_planning_mode": "chapter",
                        "baseline_source": "chapter_outline",
                    },
                },
            )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }

            writer_context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "最近正文",
                planning_mode="chapter",
            )
            progression_context = build_progression_context(
                str(project_path),
                project_data,
                next_context,
                "最近正文",
                user_request="希望这次把人物互动写得更细一点。",
                option_count=4,
                planning_mode="chapter",
            )
            prompt = build_writer_prompt(writer_context)

            self.assertEqual(writer_context["task_card"]["source"], "progression_selected")
            self.assertEqual(writer_context["task_card"]["goal"], "完成一次谨慎的外出试探")
            self.assertEqual(progression_context["task_card"]["source"], "progression_selected")
            self.assertEqual(progression_context["task_card"]["goal"], "完成一次谨慎的外出试探")
            self.assertNotIn("下一目标", writer_context["sections"]["live_state"])
            self.assertNotIn("建立临时安全区", prompt)
            self.assertIn("完成一次谨慎的外出试探", prompt)

    def test_writer_context_retrieves_saved_memory_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="memory")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 6
            save_json(str(project_path / "project.json"), project)

            for chapter_number, summary_text in {
                1: "旧通风口里藏着备用氧气瓶。",
                2: "三人第一次封锁走廊。",
                3: "异常信号曾短暂出现在主控区附近。",
                4: "他们决定暂时不深入。",
                5: "最近一次试探后气氛紧张。",
                6: "苏浅重新校准了扫描设备。",
            }.items():
                save_json(
                    str(project_path / "summaries" / f"summary_{chapter_number:04d}.json"),
                    {
                        "chapter_summary": summary_text,
                        "current_location": "隔离区",
                        "current_time": f"入侵后第{chapter_number}天",
                        "current_arc": "试探推进",
                        "recent_events": [summary_text],
                        "open_threads": ["异常信号源尚未确认"],
                        "resolved_threads": [],
                        "foreshadowing": [],
                        "character_updates": [],
                        "active_characters": ["林宇", "苏浅"],
                        "retrieval_tags": ["异常信号", "主控区", "氧气瓶"] if chapter_number in {1, 3} else ["试探"],
                        "next_chapter_goal": "继续试探",
                    },
                )

            save_json(
                str(project_path / "arc_summaries" / "arc_0001.json"),
                {
                    "arc_index": 1,
                    "chapter_range": [1, 5],
                    "summary": "异常信号多次在主控区附近出现，旧通风口还藏有备用氧气瓶。",
                    "current_arc": "试探推进",
                    "open_threads": ["异常信号源尚未确认"],
                    "resolved_threads": [],
                    "active_characters": ["林宇", "苏浅"],
                    "key_locations": ["主控区附近", "旧通风口"],
                    "retrieval_tags": ["异常信号", "主控区", "氧气瓶"],
                },
            )

            project_data = load_project(str(project_path))
            next_context = {
                "volume": {},
                "chapter": {
                    "chapter_number": 7,
                    "title": "再探主控区",
                    "summary": "再次追查主控区附近的异常信号。",
                    "goal": "确认信号源与备用物资是否存在联系。",
                    "key_events": ["进入主控区外围", "验证氧气瓶线索"],
                },
            }
            recent_text = "第一段。\n\n第二段他们重新检查扫描器。\n\n第三段他们决定去主控区外围。"

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                recent_text,
                user_request="优先追查主控区附近的异常信号",
                planning_mode="volume",
            )

            self.assertIn("主控区", context["sections"]["retrieved_memory"])
            self.assertTrue(
                "氧气瓶" in context["sections"]["retrieved_memory"]
                or "异常信号" in context["sections"]["retrieved_memory"]
            )

    def test_compact_prompts_stay_within_budget_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="budget")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 40
            save_json(str(project_path / "project.json"), project)

            project_data = load_project(str(project_path))
            previous_volumes = []
            for volume_number in range(1, 6):
                previous_volumes.append(
                    {
                        "volume_number": volume_number,
                        "title": f"第{volume_number}卷" + "扩写" * 20,
                        "summary": "本卷总结" * 50,
                        "story_goal": "阶段目标" * 40,
                        "planned_chapter_count": 10,
                        "chapters": [
                            {
                                "chapter_number": volume_number * 10 + chapter_index,
                                "chapter_in_volume": chapter_index,
                                "title": "章节标题" * 8,
                                "summary": "章节摘要" * 20,
                                "goal": "章节目标" * 16,
                                "key_events": ["事件A", "事件B", "事件C"],
                            }
                            for chapter_index in range(1, 11)
                        ],
                    }
                )

            completed_chapters = previous_volumes[-1]["chapters"][:8]
            outline_context = build_chapter_outline_context(
                str(project_path),
                project_data,
                previous_volumes[-1],
                previous_volumes[:-1],
                completed_chapters,
                "想让这一卷更强调主控区线索与人物默契升级。",
            )
            outline_prompt = build_chapter_outline_prompt(
                outline_context,
                previous_volumes[-1],
                previous_volumes[:-1],
                completed_chapters,
                "想让这一卷更强调主控区线索与人物默契升级。",
            )
            self.assertLessEqual(len(outline_prompt), 15000)

            progression_context = build_progression_context(
                str(project_path),
                project_data,
                {
                    "volume": previous_volumes[-1],
                    "chapter": {
                        "chapter_number": 41,
                        "title": "意外警报",
                        "summary": "主控区外围出现意外警报。",
                        "goal": "测试团队反应并继续追查信号源。",
                        "key_events": ["切断热源", "判断警报真假"],
                    },
                },
                "第一段。" + ("\n\n第二段扩写" * 200),
                user_request="先制造一场突发危机，再推进信号源线索。",
                option_count=4,
                planning_mode="volume",
            )
            progression_prompt = build_progression_options_prompt(
                progression_context,
                "最近正文",
                {},
                user_request="先制造一场突发危机，再推进信号源线索。",
                option_count=4,
                planning_mode="volume",
            )
            self.assertLessEqual(len(progression_prompt), 12000)

            batch_context = build_batch_plan_context(
                str(project_path),
                project_data,
                [
                    {
                        "chapter_number": 41,
                        "summary": "危机预警",
                        "goal": "建立新风险",
                        "key_events": ["A", "B"],
                    },
                    {
                        "chapter_number": 42,
                        "summary": "深入调查",
                        "goal": "追查信号",
                        "key_events": ["C", "D"],
                    },
                    {
                        "chapter_number": 43,
                        "summary": "阶段兑现",
                        "goal": "获得线索",
                        "key_events": ["E", "F"],
                    },
                ],
                "先制造一场突发危机，再推进信号源线索。",
            )
            batch_prompt = build_batch_chapter_plan_prompt(
                batch_context,
                [],
                "先制造一场突发危机，再推进信号源线索。",
            )
            self.assertLessEqual(len(batch_prompt), 10000)


if __name__ == "__main__":
    unittest.main()
