from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_manager import save_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def create_test_project(base_dir: Path, *, project_id: str = "test", planning_mode: str = "chapter") -> Path:
    project_path = base_dir / f"novel_project_{project_id}"
    project_path.mkdir(parents=True, exist_ok=True)
    for name in ("chapters", "summaries", "arc_summaries", "task_cards", "illustrations", "audiobook", "snapshots"):
        (project_path / name).mkdir(exist_ok=True)

    save_json(
        str(project_path / "project.json"),
        {
            "project_id": project_id,
            "name": "Test Project",
            "description": "Test project for guided continuation.",
            "project_path": str(project_path),
            "story_request": "测试故事",
            "planning_mode": planning_mode,
            "created_at": "2026-04-20T00:00:00+00:00",
            "updated_at": "2026-04-20T00:00:00+00:00",
            "chapter_count": 0,
            "llm_config": {
                "model_provider": "ollama",
                "model": "llama3.2",
                "model_name": "llama3.2",
                "api_base": "http://127.0.0.1:11434/v1",
                "api_key": "",
                "temperature": 0.8,
                "max_tokens": 4000,
                "timeout": 900,
                "planning_mode": planning_mode,
            },
        },
    )
    save_json(
        str(project_path / "world.json"),
        {
            "title": "测试世界",
            "genre": "科幻",
            "setting": "测试空间站",
            "background": ["异族入侵后生存"],
            "rules": [],
        },
    )
    save_json(
        str(project_path / "characters.json"),
        {
            "protagonists": [
                {
                    "name": "林宇",
                    "role": "主角",
                    "description": "负责行动",
                    "appearance": "黑发，沉稳",
                },
                {
                    "name": "苏浅",
                    "role": "主角",
                    "description": "负责技术",
                    "appearance": "短发，冷静",
                },
            ],
            "supporting": [],
        },
    )
    save_json(
        str(project_path / "plot_state.json"),
        {
            "main_plot": "求生",
            "current_arc": "开篇阶段",
            "recent_events": [],
            "open_threads": [],
            "resolved_threads": [],
            "foreshadowing": [],
            "character_updates": [],
            "active_characters": ["林宇", "苏浅"],
            "current_location": "空间站隔离区",
            "current_time": "入侵后第48小时",
            "next_chapter_goal": "建立临时安全区",
        },
    )
    save_json(
        str(project_path / "style.json"),
        {
            "tone": "紧张中带温情",
            "pov": "第三人称",
            "requirements": ["延续人物状态"],
        },
    )
    save_json(
        str(project_path / "author_intent.json"),
        {
            "premise": "三人在被入侵后的空间站隔离区内求生。",
            "long_arc": "从建立安全区逐步走向外部探索与长期生存。",
            "tone_contract": "紧张中带温情 / 第三人称",
            "must_haves": ["人物状态延续", "维持生存压力"],
            "must_not_break": ["人物不能 OOC", "不能提前写完后续主线"],
            "creativity_guidance": "在保持一致性的前提下，让每章的推进方式有变化。",
        },
    )
    save_json(
        str(project_path / "outlines.json"),
        {
            "meta": {
                "chapter_outline_stale": False,
                "last_volume_outline_request": "",
                "last_chapter_outline_request": "",
                "updated_at": "2026-04-20T00:00:00+00:00",
            },
            "volumes": [
                {
                    "volume_number": 1,
                    "title": "第一卷",
                    "summary": "生存开局",
                    "story_goal": "建立据点",
                    "planned_chapter_count": 2,
                    "chapters": [
                        {
                            "chapter_number": 1,
                            "chapter_in_volume": 1,
                            "title": "死寂的隔离区",
                            "summary": "三人确认避难点安全。",
                            "goal": "建立临时安全区",
                            "key_events": ["封门", "分工", "检查设备"],
                            "status": "planned",
                        },
                        {
                            "chapter_number": 2,
                            "chapter_in_volume": 2,
                            "title": "试探前行",
                            "summary": "开始向外探索。",
                            "goal": "尝试离开隔离区",
                            "key_events": ["制定路线", "试探外部情况"],
                            "status": "planned",
                        },
                    ],
                }
            ],
        },
    )
    return project_path


def runtime_config(planning_mode: str = "chapter") -> dict:
    return {
        "model_provider": "ollama",
        "model": "llama3.2",
        "model_name": "llama3.2",
        "api_base": "http://127.0.0.1:11434/v1",
        "api_key": "",
        "temperature": 0.8,
        "max_tokens": 4000,
        "timeout": 900,
        "planning_mode": planning_mode,
    }
