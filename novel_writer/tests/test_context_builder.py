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
    build_progression_selected_task_card,
    build_chapter_task_card,
    build_progression_context,
    build_summary_context,
    build_writer_context,
    select_recent_scene_window,
)
from prompt_builder import (
    build_auto_objective_prompt,
    build_batch_chapter_plan_prompt,
    build_chapter_outline_prompt,
    build_craft_brief_prompt,
    build_progression_options_prompt,
    build_quality_review_prompt,
    build_summary_prompt,
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

    def test_recent_craft_memory_and_brief_are_available_to_writer_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="craft_memory")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 3
            save_json(str(project_path / "project.json"), project)
            for chapter_number in range(1, 4):
                save_json(
                    str(project_path / "summaries" / f"summary_{chapter_number:04d}.json"),
                    {
                        "chapter_summary": f"第{chapter_number}章完成一次谨慎试探。",
                        "current_location": "隔离区",
                        "current_time": f"第{chapter_number}天",
                        "current_arc": "开篇阶段",
                        "recent_events": [f"第{chapter_number}章试探外部"],
                        "open_threads": [],
                        "resolved_threads": [],
                        "foreshadowing": [],
                        "character_updates": [],
                        "active_characters": ["林宇", "苏浅"],
                        "retrieval_tags": ["试探", "隔离区"],
                        "next_chapter_goal": "继续确认异常信号",
                        "craft_notes": {
                            "repeated_actions": ["三人在门后短暂停顿", "小心推门观察"],
                            "recurring_gestures": ["林宇按住门把手", "苏浅压低声音"],
                            "scene_type": "门口试探",
                            "emotional_beat": "紧张观察后松一口气",
                            "ending_pattern": "听到走廊尽头异常声响",
                            "notable_phrasing": ["走廊外一片死寂"],
                        },
                    },
                )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }
            craft_brief = {
                "chapter_hook": "开章让备用灯突然熄灭。",
                "context_bridge": "开场提醒读者三人仍被困在隔离区，备用电力正在下降。",
                "dramatic_question": "他们能否在不重复门口试探的情况下确认异常来源？",
                "conflict_pressure": "电量下降，外部噪音逼近。",
                "action_reasoning": "因为备用灯熄灭且噪音逼近，他们必须改用设备交叉验证异常来源。",
                "emotional_turn": "苏浅主动提出改变行动方式。",
                "scene_movement": ["熄灯", "手势分工", "设备交叉验证"],
                "sensory_palette": ["焦味", "冷光"],
                "fresh_interaction_patterns": ["用设备读数与手势配合推进"],
                "forbidden_repeats": ["不要再写三人在门后短暂停顿"],
                "success_criteria": ["不用推门观察完成异常确认", "结尾带回可验证的新线索"],
                "focus_notes": "让本章互动方式变化。",
            }

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "上一章正文",
                planning_mode="chapter",
                craft_brief=craft_brief,
            )
            prompt = build_writer_prompt(context)

            self.assertIn("三人在门后短暂停顿", context["sections"]["recent_craft_memory"])
            self.assertIn("禁用重复", context["sections"]["craft_brief"])
            self.assertIn("近期写法避让", prompt)
            self.assertIn("本章创作蓝图", prompt)
            self.assertIn("不要再写三人在门后短暂停顿", prompt)
            self.assertIn("读者入口/连续性桥", prompt)
            self.assertIn("行动理由", prompt)
            self.assertIn("验收标准", prompt)
            self.assertIn("不用推门观察完成异常确认", prompt)
            self.assertLessEqual(sum(len(value) for value in context["sections"].values()), WRITER_HARD_TOTAL_CHARS)

    def test_writer_context_exposes_creative_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="creative_contract")
            save_json(
                str(project_path / "author_intent.json"),
                {
                    "premise": "故事发生在一座全面停电的超级摩天楼中，男女主被困后合作生存。",
                    "long_arc": "从临时避难到长期建设据点，并逐步揭开怪物来源。",
                    "tone_contract": "第一人称宅男幽默 / 微恐、温馨、成人暧昧",
                    "narrative_engine": "封闭摩天楼内的合作生存、资源搜集、避难所建设和怪物试探。",
                    "relationship_engine": "男主用吐槽和黄段子缓解恐惧，女主从戒备到信任，在照料和并肩求生中升温。",
                    "voice_rules": ["第一人称宅男视角", "幽默吐槽", "黄段子只能成年人调侃"],
                    "scene_promises": ["摩天楼停电危机", "躲避怪物", "囤积物资", "改善避难所"],
                    "anti_flat_rules": ["不能只概括推进", "关键场景要有动作、感官、心理和对话交替"],
                    "must_haves": ["注重女主身心反应"],
                    "must_not_break": ["不能写露骨性行为"],
                    "creativity_guidance": "优先写出新鲜的场景调度和互动细节。",
                },
            )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "",
                planning_mode="chapter",
            )
            prompt = build_writer_prompt(context)

            self.assertIn("creative_contract", context["sections"])
            self.assertIn("关系引擎", context["sections"]["creative_contract"])
            self.assertIn("叙述声音", context["sections"]["creative_contract"])
            self.assertIn("场景承诺", context["sections"]["creative_contract"])
            self.assertIn("平淡规避", context["sections"]["creative_contract"])
            self.assertIn("创作风味契约", prompt)
            self.assertIn("成人暧昧只写成年人之间的张力", prompt)

    def test_craft_brief_and_quality_review_prompts_expose_json_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_prompts")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }
            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "",
                planning_mode="chapter",
            )

            craft_prompt = build_craft_brief_prompt(context)
            review_prompt = build_quality_review_prompt(context, "草稿正文", strict=True)

            for key in (
                "chapter_hook",
                "context_bridge",
                "dramatic_question",
                "conflict_pressure",
                "action_reasoning",
                "emotional_turn",
                "scene_movement",
                "sensory_palette",
                "fresh_interaction_patterns",
                "forbidden_repeats",
                "success_criteria",
            ):
                self.assertIn(key, craft_prompt)
            for key in (
                "task_completion",
                "reader_hook",
                "scene_freshness",
                "character_specificity",
                "motivation_causality",
                "repetition_risk",
                "continuity",
                "score_reasons",
                "blocking_issues",
                "nice_to_have",
                "rewrite_plan",
                "review_unavailable",
            ):
                self.assertIn(key, review_prompt)
            self.assertIn("高质量模式", review_prompt)

    def test_first_chapter_writer_prompt_requires_opening_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="first_chapter")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "",
                planning_mode="chapter",
            )
            prompt = build_writer_prompt(context)
            task_card = context["task_card"]

            self.assertIn("当前将要写的是第一章", prompt)
            self.assertIn("首章读者入口约束", prompt)
            self.assertIn("读者开卷导语（读者可见）", prompt)
            self.assertIn("读者没有看过设定文件", prompt)
            self.assertIn("正文仍必须独立可读", prompt)
            self.assertIn("前 600-1000 字必须把读者入口做成可读场景", prompt)
            self.assertIn("核心人物为何同场或彼此认识", prompt)
            self.assertIn("为什么现在必须行动", prompt)
            self.assertIn("先用动作间隙、感官、内心和对白完成开场桥", prompt)
            self.assertIn("opening_contract", context["sections"])
            self.assertIn("reader_setup", context["sections"])
            self.assertIn("读者还不知道设定文件里的前情", context["sections"]["opening_contract"])
            self.assertIn("读者开卷导语", context["sections"]["reader_setup"])
            self.assertIn("读者入口", task_card["plan_steps"][0])

    def test_first_chapter_progression_prompt_requires_reader_entry_plan_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="first_chapter_progression", planning_mode="none")
            save_json(
                str(project_path / "world.json"),
                {
                    "title": "逃兵之旅",
                    "genre": "奇幻冒险",
                    "setting": "灰烬平原前线",
                    "background": [
                        "战争起因：双方因领土争端开战。",
                        "开篇困境：两名主角被督战队识破假打，被押上处刑台，即将处决时炮击来临。",
                        "追捕压力：督战队会立刻追杀逃兵。",
                    ],
                    "rules": [],
                },
            )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": {},
                "chapter": {"chapter_number": 1, "title": "第一章任务"},
            }
            context = build_progression_context(
                str(project_path),
                project_data,
                next_context,
                "",
                user_request="",
                option_count=4,
                planning_mode="none",
            )
            prompt = build_progression_options_prompt(
                context,
                "",
                next_context,
                user_request="",
                option_count=4,
                planning_mode="none",
            )

            self.assertIn("首章读者入口约束", prompt)
            self.assertIn("读者开卷导语（读者可见）", prompt)
            self.assertIn("plan_steps` 第一项必须先设计“读者入口/开场桥”", prompt)
            self.assertIn("不要把导语当成正文已经完成的交代", prompt)
            self.assertIn("开篇困境", context["sections"]["static_world"])
            self.assertIn("reader_setup", context["sections"])
            self.assertIn("读者入口", context["task_card"]["plan_steps"][0])

    def test_non_first_prompts_skip_first_chapter_branch_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="non_first_branching", planning_mode="chapter")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 1
            save_json(str(project_path / "project.json"), project)
            (project_path / "chapters" / "chapter_0001.md").write_text("第一章正文。", encoding="utf-8")
            save_json(
                str(project_path / "summaries" / "summary_0001.json"),
                {
                    "chapter_summary": "三人建立临时安全区。",
                    "current_location": "空间站隔离区",
                    "current_time": "入侵后第48小时",
                    "current_arc": "开篇阶段",
                    "recent_events": ["建立临时安全区"],
                    "open_threads": [],
                    "resolved_threads": [],
                    "foreshadowing": [],
                    "continuity_anchors": ["隔离区门已封好"],
                    "causal_links": ["确认安全需求后封门"],
                    "character_updates": [],
                    "active_characters": ["林宇", "苏浅"],
                    "retrieval_tags": ["安全区"],
                    "next_chapter_goal": "尝试离开隔离区",
                },
            )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][1],
            }

            writer_context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "第一章正文。",
                planning_mode="chapter",
            )
            progression_context = build_progression_context(
                str(project_path),
                project_data,
                next_context,
                "第一章正文。",
                user_request="",
                option_count=4,
                planning_mode="chapter",
            )
            combined_prompt = "\n".join(
                [
                    build_writer_prompt(writer_context),
                    build_progression_options_prompt(
                        progression_context,
                        "第一章正文。",
                        next_context,
                        user_request="",
                        option_count=4,
                        planning_mode="chapter",
                    ),
                    build_auto_objective_prompt(
                        progression_context,
                        "第一章正文。",
                        next_context,
                        user_request="",
                        planning_mode="chapter",
                    ),
                ]
            )

            self.assertEqual(writer_context["sections"].get("opening_contract"), "")
            self.assertEqual(progression_context["sections"].get("opening_contract"), "")
            self.assertNotIn("首章读者入口约束", combined_prompt)
            self.assertNotIn("读者开卷导语（读者可见）", combined_prompt)
            self.assertNotIn("当前将要写的是第一章", combined_prompt)
            self.assertNotIn("如果提供了", combined_prompt)
            self.assertNotIn("如果这是第一章", combined_prompt)
            self.assertNotIn("读者尚未看见", combined_prompt)

    def test_first_chapter_selected_progression_task_card_prepends_reader_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="first_chapter_selected", planning_mode="none")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": {},
                "chapter": {"chapter_number": 1, "title": "第一章任务"},
            }
            baseline = {
                "chapter_number": 1,
                "planning_mode": "none",
                "source": "plot_state",
                "title": "逃出营地",
                "objective": "从炮击混乱中逃离处刑现场。",
                "plan_summary": "趁炮击逃离。",
                "plan_steps": ["炮击砸断栅栏", "两人冲向荒野"],
            }
            task_card = build_progression_selected_task_card(
                str(project_path),
                project_data,
                next_context,
                {
                    "title": "混乱中联手",
                    "plan_summary": "爆炸后两人立刻逃离。",
                    "plan_steps": ["炮击砸断栅栏", "两人冲向荒野"],
                    "plan_guidance": "写出混乱。",
                },
                baseline,
                session_id="session",
                option_id="option_1",
                planning_mode="none",
                baseline_source="plot_state",
                persist=False,
            )

            self.assertIn("读者入口", task_card["plan_steps"][0])
            self.assertIn("炮击砸断栅栏", task_card["plan_steps"][1])

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
                        "continuity_anchors": [f"锚点{chapter_number}"],
                        "causal_links": [f"因果{chapter_number}"],
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
                "continuity_anchors": ["主控区外围维护走廊已被临时封锁"],
                "causal_links": ["异常信号被重新定位，所以三人决定进入更深层区域确认源头"],
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
            self.assertIn("主控区外围维护走廊已被临时封锁", plot_state["continuity_anchors"])
            self.assertIn("异常信号被重新定位，所以三人决定进入更深层区域确认源头", plot_state["causal_links"])

            summary_file = load_json(str(project_path / "summaries" / "summary_0005.json"))
            self.assertEqual(summary_file["chapter_summary"], summary_payload["chapter_summary"])
            self.assertEqual(summary_file["continuity_anchors"], summary_payload["continuity_anchors"])
            self.assertEqual(summary_file["causal_links"], summary_payload["causal_links"])

            arc_summary = load_json(str(project_path / "arc_summaries" / "arc_0001.json"))
            self.assertEqual(arc_summary["arc_index"], 1)
            self.assertEqual(arc_summary["current_arc"], "主控区试探")
            self.assertIn("主控区外围维护走廊已被临时封锁", arc_summary["continuity_anchors"])
            self.assertIn("异常信号被重新定位，所以三人决定进入更深层区域确认源头", arc_summary["causal_links"])

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

    def test_freeform_task_card_synthesizes_distinct_summary_and_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="freeform", planning_mode="none")
            project_data = load_project(str(project_path))

            task_card = build_chapter_task_card(
                str(project_path),
                project_data,
                {"volume": {}, "chapter": {}},
                planning_mode="none",
            )

            self.assertEqual(task_card["goal"], "建立临时安全区")
            self.assertNotEqual(task_card["summary"], task_card["goal"])
            self.assertIn("建立临时安全区", task_card["summary"])

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
            prompt_without_reader_setup = prompt.replace(writer_context["sections"].get("reader_setup", ""), "")
            self.assertNotIn("建立临时安全区", prompt_without_reader_setup)
            self.assertIn("建立临时安全区", writer_context["sections"]["reader_setup"])
            self.assertIn("完成一次谨慎的外出试探", prompt)
            self.assertIn("新的可验证变化", prompt)

    def test_author_intent_block_is_rendered_as_compact_writer_facing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="author_intent_compact")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "最近正文",
                planning_mode="chapter",
            )
            author_intent = context["sections"]["author_intent"]

            self.assertIn("写作核心", author_intent)
            self.assertIn("优先强调", author_intent)
            self.assertNotIn("长期主线", author_intent)
            self.assertNotIn("不能破坏", author_intent)
            self.assertNotIn("从建立安全区逐步走向外部探索与长期生存", author_intent)

    def test_author_intent_summary_falls_back_when_premise_is_low_signal_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="author_intent_fallback")
            save_json(
                str(project_path / "author_intent.json"),
                {
                    "premise": "由模型根据需求自动生成设定的长篇小说项目。",
                    "long_arc": "男女主被困在摩天楼中合作求生、搜集物资并逐步建立安全区。",
                    "tone_contract": "紧张中带温情 / 第三人称",
                    "must_haves": ["维持生存压力"],
                    "must_not_break": ["人物不能 OOC"],
                    "creativity_guidance": "保持推进新鲜感。",
                },
            )
            project_data = load_project(str(project_path))
            next_context = {
                "volume": project_data["outlines"]["volumes"][0],
                "chapter": project_data["outlines"]["volumes"][0]["chapters"][0],
            }

            context = build_writer_context(
                str(project_path),
                project_data,
                next_context,
                "最近正文",
                planning_mode="chapter",
            )
            author_intent = context["sections"]["author_intent"]

            self.assertIn("摩天楼中合作求生", author_intent)
            self.assertNotIn("自动生成设定的长篇小说项目", author_intent)

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
                        "continuity_anchors": ["旧通风口备用氧气瓶仍未取走"] if chapter_number == 1 else [],
                        "causal_links": ["异常信号反复出现，所以三人必须验证主控区"] if chapter_number == 3 else [],
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
                    "continuity_anchors": ["旧通风口备用氧气瓶仍未取走"],
                    "causal_links": ["异常信号反复出现，所以三人必须验证主控区"],
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
            self.assertTrue(
                "旧通风口备用氧气瓶仍未取走" in context["sections"]["retrieved_memory"]
                or "异常信号反复出现" in context["sections"]["retrieved_memory"]
            )
            self.assertNotIn("最近一次试探后气氛紧张", context["sections"]["retrieved_memory"])
            self.assertIn("第三段他们决定去主控区外围。", context["sections"]["recent_scene"])

            progression_context = build_progression_context(
                str(project_path),
                project_data,
                next_context,
                recent_text,
                user_request="优先追查主控区附近的异常信号",
                option_count=4,
                planning_mode="volume",
            )
            progression_prompt = build_progression_options_prompt(
                progression_context,
                recent_text,
                next_context,
                user_request="优先追查主控区附近的异常信号",
                option_count=4,
                planning_mode="volume",
            )
            self.assertIn("更早相关记忆", progression_prompt)
            self.assertIn("氧气瓶", progression_context["sections"]["retrieved_memory"])
            self.assertIn("本章 objective", progression_context["sections"]["chapter_task"])
            self.assertNotIn("why_now", progression_prompt)
            self.assertNotIn("chapter_outline", progression_prompt)
            self.assertIn("plan_summary", progression_prompt)

    def test_single_progression_prompt_requests_unique_best_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="single_prompt")
            project_data = load_project(str(project_path))
            next_context = {
                "volume": {"volume_number": 1, "title": "第一卷", "story_goal": "建立据点"},
                "chapter": {
                    "chapter_number": 1,
                    "title": "死寂的隔离区",
                    "summary": "三人确认隔离区现状。",
                    "goal": "建立临时安全区",
                    "key_events": ["检查门锁", "分配值守"],
                },
            }
            recent_text = "这是开篇前状态。"

            progression_context = build_progression_context(
                str(project_path),
                project_data,
                next_context,
                recent_text,
                user_request="快速推进一次外部试探",
                option_count=1,
                planning_mode="chapter",
            )
            single_prompt = build_progression_options_prompt(
                progression_context,
                recent_text,
                next_context,
                user_request="快速推进一次外部试探",
                option_count=1,
                planning_mode="chapter",
            )
            multi_context = dict(progression_context)
            multi_context["sections"] = dict(progression_context["sections"])
            multi_context["sections"]["option_count"] = 4
            multi_prompt = build_progression_options_prompt(
                multi_context,
                recent_text,
                next_context,
                user_request="快速推进一次外部试探",
                option_count=4,
                planning_mode="chapter",
            )

            self.assertIn("唯一最优推进项", single_prompt)
            self.assertIn("恰好 1 个推进项", single_prompt)
            self.assertIn("recommended=true", single_prompt)
            self.assertNotIn("唯一最优推进项", multi_prompt)

    def test_summary_context_includes_completed_task_and_prompt_guides_next_goal_forward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="summary_task")
            project = load_json(str(project_path / "project.json"))
            project["chapter_count"] = 1
            save_json(str(project_path / "project.json"), project)
            save_json(
                str(project_path / "task_cards" / "chapter_0001.json"),
                {
                    "chapter_number": 1,
                    "planning_mode": "chapter",
                    "source": "chapter_outline",
                    "title": "死寂的隔离区",
                    "summary": "三人确认避难点安全并完成最初分工。",
                    "goal": "建立临时安全区",
                    "key_events": ["封门", "分工", "检查设备"],
                    "volume_title": "第一卷",
                    "volume_goal": "建立据点",
                    "writer_guidance": "",
                },
            )
            project_data = load_project(str(project_path))

            context = build_summary_context(str(project_path), project_data, "新章节正文")
            prompt = build_summary_prompt(context, "新章节正文")

            self.assertIn("本章写前任务卡", prompt)
            self.assertIn("本章 objective: 建立临时安全区", context["sections"]["completed_task"])
            self.assertIn("不要直接重复任务卡原句", prompt)
            self.assertIn("continuity_anchors", prompt)
            self.assertIn("causal_links", prompt)

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
