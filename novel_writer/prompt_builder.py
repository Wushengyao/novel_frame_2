"""Prompt builders for novel generation and state updates."""

from __future__ import annotations

import json
from typing import Any


def _to_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_init_prompt(data: dict) -> str:
    project_name = data.get("project_name", "")
    project_description = data.get("project_description", "")
    story_request = data.get("story_request", "")
    world_seed = _to_block(data.get("world_seed", {}))
    characters_seed = _to_block(data.get("characters_seed", {}))
    plot_state_seed = _to_block(data.get("plot_state_seed", {}))
    style_seed = _to_block(data.get("style_seed", {}))

    return f"""你是一名长篇小说策划助手。请根据用户需求，为小说生成初始化设定。

【项目名】
{project_name}

【项目简介】
{project_description}

【用户需求】
{story_request}

【世界观种子】
{world_seed}

【人物种子】
{characters_seed}

【剧情状态种子】
{plot_state_seed}

【文风种子】
{style_seed}

要求：
1. 输出必须是合法 JSON
2. 设定要适合长篇连载，人物关系和剧情目标要能持续推进
3. 人物要鲜明稳定，避免脸谱化
4. plot_state 必须兼容以下结构
5. style 要明确语气、视角和写作要求
6. 如果种子设定为空，请根据用户需求自行完整设计

输出 JSON：
{{
  "world": {{
    "title": "",
    "genre": "",
    "setting": "",
    "background": [],
    "rules": []
  }},
  "characters": {{
    "protagonists": [],
    "supporting": []
  }},
  "plot_state": {{
    "main_plot": "",
    "recent_events": [],
    "open_threads": [],
    "foreshadowing": [],
    "character_updates": [],
    "current_location": "",
    "current_time": "",
    "next_chapter_goal": ""
  }},
  "style": {{
    "tone": "",
    "pov": "",
    "requirements": []
  }}
}}
"""


def build_writer_prompt(data: dict, recent_text: str, user_request: str = "") -> str:
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    next_chapter_goal = data.get("plot_state", {}).get("next_chapter_goal", "")
    user_request_block = user_request.strip() if user_request.strip() else "无。请在既有设定下自由发挥，自然推进剧情。"
    chapter_count = int(data.get("project", {}).get("chapter_count", 0) or 0)
    first_chapter_note = (
        "当前将要写的是第一章。因为正文尚未开始，recent_events、open_threads、foreshadowing、character_updates 此时应为空，"
        "不要把后续章节才会出现的事件总结提前写进当前状态。重点是完成开篇铺陈。"
        if chapter_count == 0
        else ""
    )

    return f"""你是一名长篇小说写作助手。请续写下一章。

【世界观】
{world}

【人物】
{characters}

【剧情状态】
{plot_state}

【最近正文】
{recent_text}

【本章目标】
{next_chapter_goal}

【开篇说明】
{first_chapter_note or "这不是第一章，请延续已有状态与正文。"}

【用户额外要求】
{user_request_block}

【文风】
{style}

要求：
1. 人物不能OOC
2. 不遗忘伏笔
3. 循序渐进推进剧情或场景，不能急于结束或完成
4. 输出纯正文
5. 不要输出章标题、序号、小标题、Markdown标题
6. 如果用户额外要求与既有设定不冲突，优先吸收；若有冲突，以既有设定一致性为先，并尽量柔和地兼容用户意图
"""


def build_summary_prompt(data: dict, new_text: str) -> str:
    plot_state = _to_block(data.get("plot_state", {}))
    characters = _to_block(data.get("characters", {}))

    return f"""请更新小说状态。

【已有状态】
{plot_state}

【人物】
{characters}

【新章节】
{new_text}

输出 JSON：
{{
  "recent_events": [],
  "open_threads": [],
  "foreshadowing": [],
  "character_updates": [],
  "next_chapter_goal": ""
}}
"""
