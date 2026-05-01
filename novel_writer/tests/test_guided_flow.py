from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from app import (
    _audiobook_segment_model_overrides_from_args,
    _expert_mode_overrides_from_args,
    _quality_model_overrides_from_args,
    main,
    run_next_chapter,
    run_next_chapter_from_progression,
    run_next_chapters,
)
from prompt_builder import build_system_prompt
from project_manager import ProjectWriteLockError, acquire_project_write_lock
from progression_manager import CUSTOM_PROGRESSION_OPTION_ID, generate_progression_options

from tests.test_support import create_test_project, read_json, runtime_config


class GuidedFlowTests(unittest.TestCase):
    def test_cli_quality_model_overrides_are_nested(self) -> None:
        overrides = _quality_model_overrides_from_args(
            Namespace(
                quality_provider="gemini",
                quality_model="gemini-2.5-pro",
                quality_api_base="",
                quality_temperature=0.4,
                quality_max_tokens=6000,
                quality_timeout=180,
            )
        )

        self.assertEqual(overrides["quality_model"]["model_provider"], "gemini")
        self.assertEqual(overrides["quality_model"]["model_name"], "gemini-2.5-pro")
        self.assertNotIn("temperature", overrides["quality_model"])
        self.assertEqual(overrides["quality_model"]["max_tokens"], "6000")
        self.assertEqual(overrides["quality_model"]["timeout"], "180")

    def test_cli_expert_mode_overrides_are_nested(self) -> None:
        overrides = _expert_mode_overrides_from_args(
            Namespace(
                expert_mode=True,
                no_expert_mode=False,
                expert_models_json=json.dumps(
                    [{"model_provider": "gemini", "model_name": "gemini-3.1-pro-preview"}],
                    ensure_ascii=False,
                ),
            )
        )

        self.assertTrue(overrides["expert_mode"]["enabled"])
        self.assertEqual(overrides["expert_mode"]["models"][0]["model_provider"], "gemini")
        self.assertEqual(overrides["expert_mode"]["models"][0]["model_name"], "gemini-3.1-pro-preview")

    def test_cli_audiobook_segment_model_overrides_are_nested(self) -> None:
        overrides = _audiobook_segment_model_overrides_from_args(
            Namespace(
                audiobook_segment_provider="gemini",
                audiobook_segment_model="gemini-2.5-flash",
                audiobook_segment_api_base="",
                audiobook_segment_max_tokens=3000,
                audiobook_segment_timeout=240,
            )
        )

        segment_model = overrides["audiobook_segment_model"]
        self.assertEqual(segment_model["model_provider"], "gemini")
        self.assertEqual(segment_model["model_name"], "gemini-2.5-flash")
        self.assertEqual(segment_model["max_tokens"], "3000")
        self.assertEqual(segment_model["timeout"], "240")

    def _summary_payload(self, next_goal: str = "继续推进下一步") -> dict:
        return {
            "chapter_summary": "本章完成了当前推进。",
            "recent_events": ["角色完成了当前推进。"],
            "open_threads": ["仍有未解风险"],
            "foreshadowing": ["后续风险会继续扩大"],
            "continuity_anchors": ["临时安全区仍是三人的主要避难点"],
            "causal_links": ["外部噪音逼近，促使林宇决定先确认信号源再扩大探索范围"],
            "character_updates": ["林宇承担了新的压力"],
            "active_characters": ["林宇", "苏浅"],
            "next_chapter_goal": next_goal,
            "craft_notes": {
                "repeated_actions": ["短暂停顿后观察环境"],
                "recurring_gestures": ["压低声音确认彼此状态"],
                "scene_type": "谨慎试探",
                "emotional_beat": "紧张后达成共识",
                "ending_pattern": "发现新的异常信号",
                "notable_phrasing": ["走廊外一片死寂"],
            },
        }

    def _craft_brief_payload(self) -> dict:
        return {
            "chapter_hook": "开章用异常冷光打破临时安全区的安稳。",
            "dramatic_question": "三人能否在不暴露位置的前提下确认信号源？",
            "conflict_pressure": "外部噪音逼近，内部设备电量不足。",
            "context_bridge": "开场补清三人暂居隔离区、设备电量不足和信号异常的处境。",
            "action_reasoning": "因为噪音正在接近且电量下降，三人必须在暴露前确认下一步路线。",
            "emotional_turn": "林宇从单独承担风险转为接受苏浅的共同判断。",
            "scene_movement": ["冷光异常", "低声争执", "无声协作", "带回一条新线索"],
            "sensory_palette": ["冷白灯", "金属焦味", "远处震动"],
            "fresh_interaction_patterns": ["用手势和设备读数交叉确认，而不是反复推门观察"],
            "forbidden_repeats": ["不要再写三人在门后短暂停顿后小心推门"],
            "success_criteria": [
                "开章用异常冷光形成具体压力。",
                "三人通过新的协作方式确认信号源。",
                "结尾留下新的线索或代价。",
            ],
            "focus_notes": "保持生存压力，同时让互动方式更有变化。",
        }

    def _review_payload(self, *, passed: bool, score: int = 8) -> dict:
        scores = {
            "task_completion": score,
            "reader_hook": score,
            "scene_freshness": score,
            "character_specificity": score,
            "motivation_causality": score,
            "repetition_risk": score,
            "continuity": score,
        }
        return {
            "scores": scores,
            "passed": passed,
            "strengths": ["任务清楚"],
            "issues": [] if passed else ["开章钩子弱，动作和上一章接近"],
            "revision_guidance": "" if passed else "换一个开场压力，减少推门和短暂停顿动作。",
            "repeat_examples": [] if passed else ["短暂停顿后小心推门"],
        }

    def test_run_next_chapters_rejects_when_project_write_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="app_lock")

            with acquire_project_write_lock(str(project_path), owner="test"):
                with self.assertRaises(ProjectWriteLockError):
                    run_next_chapters(
                        str(project_path),
                        runtime_config("chapter"),
                        1,
                    )

            project = read_json(project_path / "project.json")
            self.assertEqual(project["chapter_count"], 0)
            self.assertEqual(list((project_path / "chapters").glob("chapter_*.md")), [])

    def test_guided_continue_rejects_when_project_write_lock_is_held(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="guided_lock")

            with acquire_project_write_lock(str(project_path), owner="test"):
                with self.assertRaises(ProjectWriteLockError):
                    run_next_chapter_from_progression(
                        str(project_path),
                        runtime_config("chapter"),
                        progression_session="missing",
                        progression_option="option_1",
                    )

            self.assertEqual(list((project_path / "chapters").glob("chapter_*.md")), [])

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

    def test_light_quality_mode_only_writes_and_summarizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_light")

            with patch(
                "quality_manager.generate_text_with_metadata",
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("轻量模式正文", {"usage": {}}),
            ) as mocked_writer_generate, patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ) as mocked_state_generate:
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="light"),
                )

            mocked_quality_generate.assert_not_called()
            mocked_writer_generate.assert_called_once()
            self.assertEqual(mocked_writer_generate.call_args.kwargs["system_prompt"], build_system_prompt("writer"))
            mocked_state_generate.assert_called_once()
            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "轻量模式正文")
            self.assertEqual(list((project_path / "craft_briefs").glob("*.json")), [])
            self.assertEqual(list((project_path / "quality_reviews").glob("*.json")), [])

    def test_writer_retries_truncated_response_before_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="writer_truncated_retry")

            with patch(
                "app.generate_text_with_metadata",
                side_effect=[
                    ("半截正文", {"usage": {"total_tokens": 4000}, "finish_reason": "length", "truncated": True}),
                    ("完整正文", {"usage": {"total_tokens": 80}, "finish_reason": "stop", "truncated": False}),
                ],
            ) as mocked_writer_generate, patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="light"),
                )

            self.assertEqual(mocked_writer_generate.call_count, 2)
            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "完整正文")
            self.assertNotIn("半截正文", Path(chapter_path).read_text(encoding="utf-8"))

    def test_expert_mode_reviews_after_summary_without_changing_chapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(
                Path(tmp),
                project_id="expert_flow",
                expert_mode={"enabled": True, "models": [{"model_provider": "ollama", "model_name": "expert"}]},
            )
            expert_payload = {
                "schema_version": 1,
                "review_unavailable": False,
                "quality_summary": "正文可用，但任务压力偏弱。",
                "overall_score": 0.7,
                "confidence": 0.8,
                "root_causes": [
                    {
                        "category": "prompt",
                        "severity": "major",
                        "confidence": 0.8,
                        "issue": "任务卡没有强调失败代价。",
                        "evidence": "行动选择缺少压力。",
                        "trace_refs": [],
                        "recommended_change": "在 writer prompt 中加入失败代价。",
                    }
                ],
            }

            with patch(
                "app.generate_text_with_metadata",
                return_value=("专家模式正文", {"usage": {}}),
            ) as mocked_writer_generate, patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ) as mocked_state_generate, patch(
                "expert_review_manager.generate_text_with_metadata",
                return_value=(json.dumps(expert_payload, ensure_ascii=False), {"usage": {}}),
            ) as mocked_expert_generate:
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config(
                        "chapter",
                        writing_quality_mode="light",
                        expert_mode={"enabled": True, "models": [{"model_provider": "ollama", "model_name": "expert"}]},
                    ),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "专家模式正文")
            writer_context = mocked_writer_generate.call_args.kwargs["log_context"]
            summary_context = mocked_state_generate.call_args.kwargs["log_context"]
            expert_context = mocked_expert_generate.call_args.kwargs["log_context"]
            self.assertTrue(writer_context["workflow_id"])
            self.assertEqual(summary_context["workflow_id"], writer_context["workflow_id"])
            self.assertEqual(expert_context["workflow_id"], writer_context["workflow_id"])
            aggregate = read_json(project_path / "expert_reviews" / "chapter_0001" / "aggregate.json")
            self.assertEqual(aggregate["report_type"], "aggregate")
            self.assertEqual(aggregate["root_causes"][0]["category"], "prompt")

    def test_balanced_quality_mode_generates_craft_brief_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_balanced")

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._review_payload(passed=True, score=8), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("平衡模式正文", {"usage": {}}),
            ) as mocked_writer_generate, patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config(
                        "chapter",
                        writing_quality_mode="balanced",
                        quality_model={"model_name": "qwen2.5:14b", "temperature": 0.3},
                    ),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "平衡模式正文")
            mocked_writer_generate.assert_called_once()
            self.assertEqual(mocked_writer_generate.call_args.args[1]["model_name"], "llama3.2")
            self.assertEqual(
                [call.args[1]["model_name"] for call in mocked_quality_generate.call_args_list],
                ["qwen2.5:14b", "qwen2.5:14b"],
            )
            self.assertEqual(
                [call.args[1]["temperature"] for call in mocked_quality_generate.call_args_list],
                [0.3, 0.3],
            )
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(phases, ["craft_brief", "quality_review"])
            craft_brief = read_json(project_path / "craft_briefs" / "chapter_0001.json")
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            self.assertEqual(craft_brief["chapter_hook"], self._craft_brief_payload()["chapter_hook"])
            self.assertEqual(craft_brief["success_criteria"], self._craft_brief_payload()["success_criteria"])
            self.assertTrue(review["passed"])

    def test_balanced_quality_mode_rewrites_when_flatness_scores_are_low(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_balanced_flat")
            low_flatness_review = self._review_payload(passed=True, score=8)
            low_flatness_review["scores"]["scene_freshness"] = 6

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(low_flatness_review, ensure_ascii=False), {"usage": {}}),
                    ("重写后更鲜活的正文", {"usage": {}}),
                    (json.dumps(self._review_payload(passed=True, score=8), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("平淡但完成任务的原始正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="balanced", review_mode="auto"),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "重写后更鲜活的正文")
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(phases, ["craft_brief", "quality_review", "rewrite", "quality_review"])
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            self.assertTrue(review["passed"])
            self.assertTrue(review["needs_rewrite"])
            self.assertIn("scene_freshness", review["flatness_issues"])
            pre_rewrite = project_path / "quality_drafts" / "chapter_0001_before_rewrite_1.md"
            self.assertEqual(pre_rewrite.read_text(encoding="utf-8").strip(), "平淡但完成任务的原始正文")

    def test_high_quality_mode_rewrites_once_and_reviews_rewrite_when_auto_review_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_high")

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._review_payload(passed=False, score=4), ensure_ascii=False), {"usage": {}}),
                    ("重写后的正文", {"usage": {}}),
                    (json.dumps(self._review_payload(passed=True, score=8), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("原始正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config(
                        "chapter",
                        writing_quality_mode="high",
                        review_mode="auto",
                        quality_model={"model_provider": "ollama", "model_name": "qwen2.5:14b"},
                    ),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "重写后的正文")
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(
                [call.args[1]["model_name"] for call in mocked_quality_generate.call_args_list],
                ["qwen2.5:14b", "qwen2.5:14b", "qwen2.5:14b", "qwen2.5:14b"],
            )
            self.assertEqual(phases, ["craft_brief", "quality_review", "rewrite", "quality_review"])
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            second_review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_2.json")
            pre_rewrite = project_path / "quality_drafts" / "chapter_0001_before_rewrite_1.md"
            self.assertFalse(review["passed"])
            self.assertTrue(second_review["passed"])
            self.assertEqual(pre_rewrite.read_text(encoding="utf-8").strip(), "原始正文")

    def test_unavailable_quality_review_does_not_trigger_auto_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_unavailable")

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    RuntimeError("review timeout"),
                    RuntimeError("review retry timeout"),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("质检不可用时保留的原始正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="high", review_mode="auto"),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "质检不可用时保留的原始正文")
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(phases, ["craft_brief", "quality_review", "quality_review"])
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            self.assertFalse(review["passed"])
            self.assertTrue(review["review_unavailable"])
            self.assertEqual(review["scores"]["task_completion"], 0.0)

    def test_quality_review_parse_failure_retries_before_saving_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_retry")

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    (json.dumps({"issues": ["missing scores"]}, ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._review_payload(passed=True, score=8), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("解析重试后保留的正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="balanced", review_mode="auto"),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "解析重试后保留的正文")
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(phases, ["craft_brief", "quality_review", "quality_review"])
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            self.assertTrue(review["passed"])
            self.assertFalse(review["review_unavailable"])
            failed_files = list((project_path / "failed_llm_outputs").glob("*_quality_review.json"))
            self.assertEqual(len(failed_files), 1)
            self.assertIn("missing scores", read_json(failed_files[0])["response_text"])

    def test_manual_review_mode_saves_report_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="quality_manual")

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._craft_brief_payload(), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._review_payload(passed=False, score=4), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_quality_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("人工审稿保留的原始正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_path = run_next_chapter(
                    str(project_path),
                    runtime_config("chapter", writing_quality_mode="high", review_mode="manual"),
                )

            self.assertEqual(Path(chapter_path).read_text(encoding="utf-8").strip(), "人工审稿保留的原始正文")
            phases = [call.kwargs["log_context"]["phase"] for call in mocked_quality_generate.call_args_list]
            self.assertEqual(phases, ["craft_brief", "quality_review"])
            review = read_json(project_path / "quality_reviews" / "chapter_0001_attempt_1.json")
            self.assertFalse(review["passed"])

    def test_cli_rejects_guided_batch_count_greater_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="cli")
            config_path = Path(tmp) / "runtime.json"
            config_path.write_text(json.dumps(runtime_config("chapter"), ensure_ascii=False, indent=2), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
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

    def test_cli_next_config_restores_project_path_for_payload_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="cli_project_path")
            config_path = Path(tmp) / "runtime.json"
            config_payload = runtime_config("chapter")
            config_payload["log_llm_payload"] = True
            config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            captured: dict[str, object] = {}

            def fake_run_next_chapters(project: str, config: dict, count: int, **_kwargs) -> list[str]:
                captured["project"] = project
                captured["config"] = dict(config)
                captured["count"] = count
                return [str(Path(project) / "chapters" / "chapter_0001.md")]

            with patch.object(
                sys,
                "argv",
                [
                    "app.py",
                    "next",
                    "--project",
                    str(project_path),
                    "--config",
                    str(config_path),
                ],
            ), patch("app.run_next_chapters", side_effect=fake_run_next_chapters):
                main()

            self.assertEqual(captured["count"], 1)
            self.assertEqual(captured["config"]["project_path"], str(project_path.resolve()))
            self.assertTrue(captured["config"]["log_llm_payload"])

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

    def test_auto_continue_single_generates_one_plan_and_selects_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="auto_single")
            options_payload = {
                "options": [
                    {
                        "option_id": "option_1",
                        "title": "短程试探",
                        "plan_summary": "谨慎离开隔离区进行一次短程侦查。",
                        "plan_steps": ["规划路线", "短程侦查"],
                        "plan_guidance": "保持紧张感并推进外部风险认知。",
                        "recommended": False,
                    }
                ]
            }

            with patch(
                "progression_manager.generate_text_with_metadata",
                return_value=(json.dumps(options_payload, ensure_ascii=False), {"usage": {}}),
            ) as mocked_progression_generate, patch(
                "app.generate_text_with_metadata",
                return_value=("第一章正文", {"usage": {}}),
            ), patch(
                "state_updater.generate_text_with_metadata",
                return_value=(json.dumps(self._summary_payload(), ensure_ascii=False), {"usage": {}}),
            ):
                chapter_paths = run_next_chapters(
                    str(project_path),
                    runtime_config("chapter"),
                    1,
                    user_request="我想快速推进一次外部试探",
                    selection_mode="single",
                )

            self.assertEqual(len(chapter_paths), 1)
            mocked_progression_generate.assert_called_once()
            _, progression_call_kwargs = mocked_progression_generate.call_args
            self.assertEqual(progression_call_kwargs["log_context"]["option_count"], 1)
            session_files = sorted((project_path / "progression_sessions").glob("progression_*.json"))
            self.assertEqual(len(session_files), 1)
            session = read_json(session_files[0])
            self.assertEqual(session["option_count"], 1)
            self.assertEqual(session["selection_mode"], "single")
            self.assertEqual(session["selection_origin"], "auto")
            self.assertEqual(session["selected_option_id"], "option_1")
            self.assertEqual(session["recommended_option_id"], "option_1")
            task_card = read_json(project_path / "task_cards" / "chapter_0001.json")
            self.assertEqual(task_card["derived_from"]["option_id"], "option_1")
            self.assertEqual(task_card["plan_summary"], "谨慎离开隔离区进行一次短程侦查。")

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

            def fake_progression_generate(prompt, config, log_context=None, system_prompt: str = "", response_format: str = ""):
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
