from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_manager import (
    ProjectWriteLockError,
    _generate_initial_story_data,
    _build_persisted_llm_config,
    _prune_initial_supporting_characters,
    acquire_project_write_lock,
    rollback_project,
    save_json,
    update_project_stats,
)
from prompt_builder import build_init_prompt

from tests.test_support import create_test_project, read_json


class ProjectManagerTests(unittest.TestCase):
    def test_persisted_llm_config_keeps_quality_model_but_clears_api_key(self) -> None:
        persisted = _build_persisted_llm_config(
            {
                "model_provider": "ollama",
                "model_name": "llama3.2",
                "api_key": "main-key",
                "quality_model": {
                    "model_provider": "gemini",
                    "model_name": "gemini-2.5-pro",
                    "api_key": "quality-key",
                    "temperature": 0.4,
                },
            }
        )

        self.assertEqual(persisted["api_key"], "")
        self.assertEqual(persisted["quality_model"]["api_key"], "")
        self.assertEqual(persisted["quality_model"]["model_name"], "gemini-2.5-pro")
        self.assertEqual(persisted["quality_model"]["temperature"], 0.4)

    def test_project_write_lock_rejects_same_project_reentry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="lock_same")

            with acquire_project_write_lock(str(project_path), owner="outer") as lock:
                self.assertTrue(lock.lock_path.exists())
                lock_data = read_json(lock.lock_path)
                self.assertEqual(lock_data["owner"], "outer")
                with self.assertRaises(ProjectWriteLockError):
                    with acquire_project_write_lock(str(project_path), owner="inner"):
                        pass

            self.assertFalse(lock.lock_path.exists())

    def test_project_write_lock_allows_different_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_a = create_test_project(Path(tmp), project_id="lock_a")
            project_b = create_test_project(Path(tmp), project_id="lock_b")

            with acquire_project_write_lock(str(project_a), owner="a") as lock_a:
                with acquire_project_write_lock(str(project_b), owner="b") as lock_b:
                    self.assertTrue(lock_a.lock_path.exists())
                    self.assertTrue(lock_b.lock_path.exists())

            self.assertFalse(lock_a.lock_path.exists())
            self.assertFalse(lock_b.lock_path.exists())

    def test_project_write_lock_releases_after_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="lock_exception")

            with self.assertRaises(RuntimeError):
                with acquire_project_write_lock(str(project_path), owner="boom") as lock:
                    self.assertTrue(lock.lock_path.exists())
                    raise RuntimeError("boom")

            self.assertFalse((project_path / ".project_write.lock").exists())

    def test_update_project_stats_records_token_and_cost_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="cost_totals")
            metadata = {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 1000,
                    "cached_tokens": 400,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            }

            update_project_stats(str(project_path), phase="writer", success=True, usage=metadata["usage"], metadata=metadata)

            stats = read_json(project_path / "project.json")["stats"]
            expected_cost = (600 * 0.14 + 400 * 0.028 + 500 * 0.28) / 1_000_000
            self.assertEqual(stats["total"]["total_tokens"], 1500)
            self.assertEqual(stats["by_phase"]["writer"]["prompt_tokens"], 1000)
            self.assertEqual(stats["cost"]["currency"], "USD")
            self.assertEqual(stats["cost"]["priced_tokens"], 1500)
            self.assertEqual(stats["cost"]["unpriced_tokens"], 0)
            self.assertAlmostEqual(stats["cost"]["estimated_total_usd"], expected_cost)
            self.assertIn("deepseek:deepseek-v4-flash", stats["cost"]["by_model"])

    def test_update_project_stats_preserves_legacy_tokens_without_cost_backfill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="legacy_cost")
            project_file = project_path / "project.json"
            project = read_json(project_file)
            project["stats"] = {
                "total": {
                    "requests": 2,
                    "successes": 2,
                    "failures": 0,
                    "prompt_tokens": 3000,
                    "completion_tokens": 2000,
                    "total_tokens": 5000,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "thought_tokens": 0,
                },
                "by_phase": {},
            }
            save_json(str(project_file), project)
            metadata = {
                "provider": "ollama",
                "model": "llama3.2",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }

            update_project_stats(str(project_path), phase="writer", success=True, usage=metadata["usage"], metadata=metadata)

            stats = read_json(project_file)["stats"]
            legacy_tokens = stats["total"]["total_tokens"] - stats["cost"]["priced_tokens"] - stats["cost"]["unpriced_tokens"]
            self.assertEqual(stats["total"]["total_tokens"], 5150)
            self.assertEqual(stats["cost"]["priced_tokens"], 150)
            self.assertEqual(stats["cost"]["unpriced_tokens"], 0)
            self.assertEqual(legacy_tokens, 5000)

    def test_update_project_stats_tracks_unpriced_tokens_without_cost(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="unpriced_cost")
            metadata = {
                "provider": "doubao",
                "model": "doubao-seed-1-8-251228",
                "usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
            }

            update_project_stats(str(project_path), phase="outline", success=True, usage=metadata["usage"], metadata=metadata)

            stats = read_json(project_path / "project.json")["stats"]
            model_entry = stats["cost"]["by_model"]["doubao:doubao-seed-1-8-251228"]
            self.assertEqual(stats["cost"]["estimated_total_usd"], 0.0)
            self.assertEqual(stats["cost"]["priced_tokens"], 0)
            self.assertEqual(stats["cost"]["unpriced_tokens"], 280)
            self.assertEqual(model_entry["pricing_status"], "unpriced")

    def test_init_prompt_limits_supporting_characters_to_opening_cast(self) -> None:
        prompt = build_init_prompt(
            {
                "project_name": "Test Project",
                "project_description": "高层封闭空间求生",
                "story_request": "男女主被困大楼，在开篇建立临时安全区。",
                "world_seed": {},
                "characters_seed": {},
                "plot_state_seed": {},
                "style_seed": {},
            }
        )

        self.assertIn("`supporting` 只保留第一章到前几章就会实际出场", prompt)
        self.assertIn("不要为了“以后可能会用到”提前创建", prompt)
        self.assertIn('"supporting": []', prompt)

    def test_prune_initial_supporting_characters_keeps_only_opening_cast(self) -> None:
        characters = {
            "protagonists": [
                {"name": "江哲", "role": "主角", "description": "求生者", "appearance": "黑发青年"},
                {"name": "琉璃", "role": "女主", "description": "神秘少女", "appearance": "银发少女"},
            ],
            "supporting": [
                {"name": "王建国", "role": "邻居", "description": "老安保", "appearance": "高大硬朗"},
                {"name": "赵龙", "role": "打手", "description": "后续敌人", "appearance": "壮硕平头"},
            ],
        }
        plot_state = {
            "active_characters": ["江哲", "琉璃", "王建国"],
        }

        pruned = _prune_initial_supporting_characters(characters, plot_state, seeded_characters={})

        self.assertEqual([item["name"] for item in pruned["supporting"]], ["王建国"])

    def test_prune_initial_supporting_characters_preserves_seeded_supporting(self) -> None:
        characters = {
            "protagonists": [
                {"name": "江哲", "role": "主角", "description": "求生者", "appearance": "黑发青年"},
            ],
            "supporting": [
                {"name": "王建国", "role": "邻居", "description": "老安保", "appearance": "高大硬朗"},
            ],
        }
        plot_state = {
            "active_characters": ["江哲"],
        }
        seeded_characters = {
            "protagonists": [],
            "supporting": [
                {"name": "王建国", "role": "邻居", "description": "老安保", "appearance": "高大硬朗"},
            ],
        }

        pruned = _prune_initial_supporting_characters(
            characters,
            plot_state,
            seeded_characters=seeded_characters,
        )

        self.assertEqual([item["name"] for item in pruned["supporting"]], ["王建国"])

    def test_generate_initial_story_data_drops_unused_generated_supporting(self) -> None:
        config = {
            "init_with_llm": True,
            "project_name": "Test Project",
            "project_description": "高层封闭空间求生",
            "story_request": "男女主被困大楼，在开篇建立临时安全区。",
            "model_provider": "openai_compatible",
            "model": "test-model",
            "api_base": "https://example.local/v1",
            "api_key": "test-key",
        }
        payload = {
            "world": {
                "title": "测试世界",
                "genre": "末日求生",
                "setting": "被封锁的摩天楼",
                "background": [],
                "rules": [],
            },
            "characters": {
                "protagonists": [
                    {"name": "江哲", "role": "主角", "description": "求生者", "appearance": "黑发青年"},
                    {"name": "琉璃", "role": "女主", "description": "神秘少女", "appearance": "银发少女"},
                ],
                "supporting": [
                    {"name": "王建国", "role": "邻居", "description": "老安保", "appearance": "高大硬朗"},
                    {"name": "赵龙", "role": "后续反派", "description": "暂未登场", "appearance": "壮硕平头"},
                ],
            },
            "plot_state": {
                "main_plot": "在摩天楼中求生并寻找异变真相",
                "current_arc": "开篇阶段",
                "active_characters": ["江哲", "琉璃", "王建国"],
                "current_location": "高层避难层",
                "current_time": "异变第一夜",
                "next_chapter_goal": "建立临时安全区",
            },
            "style": {
                "tone": "紧张幽默",
                "pov": "第三人称",
                "requirements": ["保持生存压力"],
            },
        }

        with patch(
            "project_manager.generate_text_with_metadata",
            return_value=(json.dumps(payload, ensure_ascii=False), {"usage": {}}),
        ):
            data, meta = _generate_initial_story_data(config)

        self.assertTrue(meta["used_llm"])
        self.assertEqual([item["name"] for item in data["characters"]["supporting"]], ["王建国"])

    def test_rollback_removes_future_quality_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="rollback_quality")
            project = read_json(project_path / "project.json")
            project["chapter_count"] = 2
            save_json(str(project_path / "project.json"), project)
            for chapter_number in (1, 2):
                (project_path / "chapters" / f"chapter_{chapter_number:04d}.md").write_text(
                    f"第{chapter_number}章正文",
                    encoding="utf-8",
                )
                save_json(
                    str(project_path / "summaries" / f"summary_{chapter_number:04d}.json"),
                    {
                        "chapter_summary": f"第{chapter_number}章摘要",
                        "current_location": "隔离区",
                        "current_time": f"第{chapter_number}天",
                        "current_arc": "开篇阶段",
                        "recent_events": [f"事件{chapter_number}"],
                        "open_threads": [],
                        "resolved_threads": [],
                        "foreshadowing": [],
                        "character_updates": [],
                        "active_characters": ["林宇"],
                        "retrieval_tags": ["隔离区"],
                        "next_chapter_goal": "继续推进",
                    },
                )
                save_json(str(project_path / "task_cards" / f"chapter_{chapter_number:04d}.json"), {"chapter_number": chapter_number})
                save_json(str(project_path / "craft_briefs" / f"chapter_{chapter_number:04d}.json"), {"chapter_hook": "hook"})
                save_json(
                    str(project_path / "quality_reviews" / f"chapter_{chapter_number:04d}_attempt_1.json"),
                    {"passed": True},
                )
                (project_path / "quality_drafts" / f"chapter_{chapter_number:04d}_before_rewrite_1.md").write_text(
                    f"第{chapter_number}章重写前正文",
                    encoding="utf-8",
                )

            result = rollback_project(str(project_path), 1)

            self.assertIn("craft_briefs/chapter_0002.json", result["removed"]["craft_briefs"])
            self.assertIn("quality_reviews/chapter_0002_attempt_1.json", result["removed"]["quality_reviews"])
            self.assertIn("quality_drafts/chapter_0002_before_rewrite_1.md", result["removed"]["quality_drafts"])
            self.assertTrue((project_path / "craft_briefs" / "chapter_0001.json").exists())
            self.assertTrue((project_path / "quality_reviews" / "chapter_0001_attempt_1.json").exists())
            self.assertTrue((project_path / "quality_drafts" / "chapter_0001_before_rewrite_1.md").exists())
            self.assertFalse((project_path / "craft_briefs" / "chapter_0002.json").exists())
            self.assertFalse((project_path / "quality_reviews" / "chapter_0002_attempt_1.json").exists())
            self.assertFalse((project_path / "quality_drafts" / "chapter_0002_before_rewrite_1.md").exists())


if __name__ == "__main__":
    unittest.main()
