from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from project_manager import (
    PROJECT_EXPORT_MANIFEST_FILENAME,
    ProjectWriteLockError,
    _build_persisted_llm_config,
    _build_author_intent_from_project,
    _generate_initial_story_data,
    _prune_initial_supporting_characters,
    acquire_project_audio_lock,
    acquire_project_write_lock,
    export_project_archive,
    import_project_archive,
    project_audio_lock_is_active,
    rollback_project,
    save_json,
    update_project_stats,
)
from prompt_builder import build_init_prompt, build_story_setup_prompt

from tests.test_support import create_test_project, read_json


class ProjectManagerTests(unittest.TestCase):
    def test_author_intent_premise_prefers_story_request_over_meta_description(self) -> None:
        intent = _build_author_intent_from_project(
            {
                "story_request": "故事发生在一座设施齐全的超级摩天楼中，主角被困后合作生存。",
                "description": "用于分析 novel_frame_2 无大纲模式创建项目到首次续写的提示词链路。",
            },
            {"setting": "备用设定"},
            {"tone": "微恐、温馨", "pov": "第一人称", "requirements": ["注重生存细节"]},
            {"main_plot": "摩天楼停电后，主角们建立避难所并揭开怪物来源。"},
        )

        self.assertIn("超级摩天楼", intent["premise"])
        self.assertNotIn("提示词链路", intent["premise"])
        self.assertIn("注重生存细节", intent["must_haves"])
        self.assertTrue(intent["voice_rules"])
        self.assertTrue(intent["scene_promises"])
        self.assertTrue(intent["anti_flat_rules"])

    def test_author_intent_uses_valid_description_only_when_story_request_missing(self) -> None:
        intent = _build_author_intent_from_project(
            {
                "story_request": "",
                "description": "封闭地铁城里的长期求生与派系博弈。",
            },
            {"setting": ""},
            {"tone": "", "pov": "", "requirements": []},
            {"main_plot": ""},
        )

        self.assertIn("封闭地铁城", intent["premise"])

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
                "expert_mode": {
                    "enabled": True,
                    "models": [
                        {
                            "model_provider": "gemini",
                            "model_name": "gemini-3.1-pro-preview",
                            "api_key": "expert-key",
                        }
                    ],
                },
            }
        )

        self.assertEqual(persisted["api_key"], "")
        self.assertEqual(persisted["quality_model"]["api_key"], "")
        self.assertEqual(persisted["quality_model"]["model_name"], "gemini-2.5-pro")
        self.assertEqual(persisted["quality_model"]["temperature"], 0.4)
        self.assertTrue(persisted["expert_mode"]["enabled"])
        self.assertEqual(persisted["expert_mode"]["models"][0]["api_key"], "")

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

    def test_project_write_lock_recovers_stale_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="lock_stale")
            lock_path = project_path / ".project_write.lock"
            save_json(
                str(lock_path),
                {
                    "pid": 999999999,
                    "owner": "old-run",
                    "created_at": "2026-04-26T00:00:00+00:00",
                    "project_path": str(project_path),
                    "token": "old-token",
                },
            )

            with patch("project_manager._lock_owner_process_still_active", return_value=False):
                with acquire_project_write_lock(str(project_path), owner="new-run") as lock:
                    self.assertTrue(lock.lock_path.exists())
                    lock_data = read_json(lock.lock_path)
                    self.assertEqual(lock_data["owner"], "new-run")

            self.assertFalse(lock_path.exists())

    def test_project_audio_lock_can_overlap_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="lock_audio")

            with acquire_project_audio_lock(str(project_path), owner="audio") as audio_lock:
                self.assertTrue(audio_lock.lock_path.exists())
                self.assertTrue(project_audio_lock_is_active(project_path))
                with acquire_project_write_lock(str(project_path), owner="writer") as write_lock:
                    self.assertTrue(write_lock.lock_path.exists())

            self.assertFalse((project_path / ".project_audio.lock").exists())

    def test_rollback_rejects_active_audio_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="rollback_audio")

            with acquire_project_audio_lock(str(project_path), owner="audio"):
                with self.assertRaises(ProjectWriteLockError):
                    rollback_project(str(project_path), 0)

    def test_rollback_deletes_future_expert_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="rollback_expert")
            (project_path / "chapters" / "chapter_0001.md").write_text("第一章\n", encoding="utf-8")
            (project_path / "chapters" / "chapter_0002.md").write_text("第二章\n", encoding="utf-8")
            project = read_json(project_path / "project.json")
            project["chapter_count"] = 2
            save_json(str(project_path / "project.json"), project)
            save_json(str(project_path / "summaries" / "summary_0001.json"), {"chapter_summary": "第一章"})
            future_dir = project_path / "expert_reviews" / "chapter_0002"
            future_dir.mkdir(parents=True)
            save_json(str(future_dir / "aggregate.json"), {"ok": True})

            result = rollback_project(str(project_path), 1)

            self.assertFalse(future_dir.exists())
            self.assertIn("expert_reviews/chapter_0002", result["removed"]["expert_reviews"])

    def test_export_project_archive_contains_complete_project_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="export_full")
            (project_path / "chapters" / "chapter_0001.md").write_text("第一章正文\n", encoding="utf-8")
            image_dir = project_path / "illustrations" / "chapter_0001"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "image_0001.png").write_bytes(b"\x89PNG\r\n")
            logs_dir = project_path / "llm_logs"
            logs_dir.mkdir(exist_ok=True)
            (logs_dir / "llm_interactions.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
            save_json(
                str(project_path / ".project_write.lock"),
                {
                    "pid": 999999999,
                    "owner": "stale",
                    "created_at": "2026-04-26T00:00:00+00:00",
                    "project_path": str(project_path),
                    "token": "stale-token",
                },
            )

            archive_path = Path(tmp) / "export_full.zip"
            result = export_project_archive(str(project_path), str(archive_path))

            self.assertEqual(result["project_id"], "export_full")
            self.assertTrue(archive_path.exists())
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read(PROJECT_EXPORT_MANIFEST_FILENAME).decode("utf-8"))

            root = project_path.name
            self.assertEqual(manifest["format_version"], 1)
            self.assertTrue(manifest["complete_project"])
            self.assertIn(f"{root}/project.json", names)
            self.assertIn(f"{root}/chapters/chapter_0001.md", names)
            self.assertIn(f"{root}/illustrations/chapter_0001/image_0001.png", names)
            self.assertIn(f"{root}/llm_logs/llm_interactions.jsonl", names)
            self.assertNotIn(f"{root}/.project_write.lock", names)

    def test_export_project_archive_rejects_active_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="export_busy")

            with acquire_project_write_lock(str(project_path), owner="unit-test"):
                with self.assertRaises(ProjectWriteLockError):
                    export_project_archive(str(project_path), str(Path(tmp) / "busy.zip"))

    def test_import_project_archive_round_trips_and_updates_project_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_path = create_test_project(base / "source", project_id="roundtrip")
            (source_path / "chapters" / "chapter_0001.md").write_text("第一章正文\n", encoding="utf-8")
            archive_path = base / "roundtrip.zip"
            export_project_archive(str(source_path), str(archive_path))

            output_dir = base / "imported"
            result = import_project_archive(str(archive_path), str(output_dir))

            imported_path = Path(result["project_path"])
            imported_project = read_json(imported_path / "project.json")
            self.assertEqual(result["project_id"], "roundtrip")
            self.assertFalse(result["renamed"])
            self.assertEqual(imported_project["project_path"], str(imported_path))
            self.assertEqual(imported_project["project_id"], "roundtrip")
            self.assertTrue((imported_path / "chapters" / "chapter_0001.md").exists())
            self.assertTrue((imported_path / "progression_sessions").is_dir())

    def test_import_project_archive_renames_conflicting_project_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            project_path = create_test_project(output_dir, project_id="same")
            archive_path = Path(tmp) / "same.zip"
            export_project_archive(str(project_path), str(archive_path))

            result = import_project_archive(str(archive_path), str(output_dir))

            self.assertTrue(result["renamed"])
            self.assertEqual(result["source_project_id"], "same")
            self.assertNotEqual(result["project_id"], "same")
            self.assertTrue((output_dir / "novel_project_same").exists())
            self.assertTrue(Path(result["project_path"]).exists())
            imported_project = read_json(Path(result["project_path"]) / "project.json")
            self.assertEqual(imported_project["project_id"], result["project_id"])
            self.assertEqual(imported_project["project_path"], result["project_path"])

    def test_import_project_archive_sanitizes_unsafe_project_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project_path = create_test_project(base / "source", project_id="unsafe_source")
            project_file = project_path / "project.json"
            project = read_json(project_file)
            project["project_id"] = "bad/id"
            save_json(str(project_file), project)
            archive_path = base / "unsafe_id.zip"
            export_project_archive(str(project_path), str(archive_path))

            result = import_project_archive(str(archive_path), str(base / "output"))

            self.assertTrue(result["renamed"])
            self.assertEqual(result["source_project_id"], "bad/id")
            self.assertEqual(result["project_id"], "bad_id")
            self.assertNotIn("/", result["project_id"])
            imported_project = read_json(Path(result["project_path"]) / "project.json")
            self.assertEqual(imported_project["project_id"], "bad_id")
            self.assertEqual(imported_project["project_path"], result["project_path"])

    def test_import_project_archive_rejects_unsafe_or_malformed_archives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manifest = json.dumps(
                {
                    "format_version": 1,
                    "app": "novel_writer",
                    "project_id": "bad",
                    "project_dir_name": "novel_project_bad",
                },
                ensure_ascii=False,
            )

            missing_manifest = base / "missing_manifest.zip"
            with zipfile.ZipFile(missing_manifest, "w") as archive:
                archive.writestr("novel_project_bad/project.json", "{}")
            with self.assertRaises(ValueError):
                import_project_archive(str(missing_manifest), str(base / "out_missing"))

            incomplete_project = base / "incomplete_project.zip"
            with zipfile.ZipFile(incomplete_project, "w") as archive:
                archive.writestr(PROJECT_EXPORT_MANIFEST_FILENAME, manifest)
                archive.writestr("novel_project_bad/project.json", "{}")
            with self.assertRaisesRegex(ValueError, "world.json"):
                import_project_archive(str(incomplete_project), str(base / "out_incomplete"))

            zip_slip = base / "zip_slip.zip"
            with zipfile.ZipFile(zip_slip, "w") as archive:
                archive.writestr(PROJECT_EXPORT_MANIFEST_FILENAME, manifest)
                archive.writestr("novel_project_bad/../evil.txt", "evil")
                archive.writestr("novel_project_bad/project.json", "{}")
            with self.assertRaises(ValueError):
                import_project_archive(str(zip_slip), str(base / "out_slip"))

            multi_root = base / "multi_root.zip"
            with zipfile.ZipFile(multi_root, "w") as archive:
                archive.writestr(PROJECT_EXPORT_MANIFEST_FILENAME, manifest)
                archive.writestr("novel_project_a/project.json", "{}")
                archive.writestr("novel_project_b/project.json", "{}")
            with self.assertRaises(ValueError):
                import_project_archive(str(multi_root), str(base / "out_multi"))

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

    def test_story_setup_prompt_focuses_on_characters_and_background(self) -> None:
        prompt = build_story_setup_prompt(
            {
                "project_name": "Test Project",
                "project_description": "高层封闭空间求生",
                "story_request": "男女主被困大楼，在开篇建立临时安全区。",
                "world_seed": {},
                "characters_seed": {},
            }
        )

        self.assertIn("先根据用户输入的故事需求，具体化并创造人物和背景设定", prompt)
        self.assertIn("不是记录或复述用户需求", prompt)
        self.assertIn("主动补足用户没有细写但故事需要的内容", prompt)
        self.assertIn("这个阶段只做人设和背景设定", prompt)
        self.assertIn('"world"', prompt)
        self.assertIn('"characters"', prompt)

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

    def test_generate_initial_story_data_uses_story_setup_as_init_seed(self) -> None:
        config = {
            "init_with_llm": True,
            "project_name": "Test Project",
            "project_description": "",
            "story_request": "空间站入侵后，三名幸存者建立避难据点。",
            "model_provider": "openai_compatible",
            "model": "test-model",
            "api_base": "https://example.local/v1",
            "api_key": "test-key",
        }
        setup_payload = {
            "world": {
                "title": "星环余烬",
                "genre": "科幻生存",
                "setting": "被异族占领的高级太空站",
                "background": ["异族突袭后，隔离区成为少数安全地带。"],
                "rules": ["安全门需要能源配额。"],
            },
            "characters": {
                "protagonists": [
                    {"name": "林宇", "role": "男主", "description": "团队力量担当", "appearance": "黑发青年"},
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
                "active_characters": ["林宇"],
            },
            "style": {
                "tone": "温馨求生",
                "pov": "第三人称",
                "requirements": ["重视协作细节"],
            },
        }

        with patch(
            "project_manager.generate_text_with_metadata",
            side_effect=[
                (json.dumps(setup_payload, ensure_ascii=False), {"usage": {}}),
                (json.dumps(init_payload, ensure_ascii=False), {"usage": {}}),
            ],
        ) as mocked_generate:
            data, meta = _generate_initial_story_data(config)

        self.assertEqual(mocked_generate.call_count, 2)
        first_prompt = mocked_generate.call_args_list[0].args[0]
        second_prompt = mocked_generate.call_args_list[1].args[0]
        self.assertIn("具体化并创造人物和背景设定", first_prompt)
        self.assertIn("被异族占领的高级太空站", second_prompt)
        self.assertTrue(meta["used_story_setup_llm"])
        self.assertEqual(data["world"]["setting"], "被异族占领的高级太空站")
        self.assertEqual(data["characters"]["protagonists"][0]["name"], "林宇")
        self.assertEqual(data["story_setup"]["world"]["title"], "星环余烬")

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
