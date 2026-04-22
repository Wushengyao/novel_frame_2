from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from project_manager import _generate_initial_story_data, _prune_initial_supporting_characters
from prompt_builder import build_init_prompt


class ProjectManagerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
