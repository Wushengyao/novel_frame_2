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
4. 每个角色对象都必须包含 `name`、`role`、`description`、`appearance` 四个字段
5. `description` 侧重人物身份、性格、能力、关系与叙事定位
6. `appearance` 单独描写人物外貌特征（包括人种，避免文生图模型的不确定性）、体态气质、发型发色、五官特点、常见衣着与穿搭风格
7. plot_state 必须兼容以下结构
8. style 要明确语气、视角和写作要求
9. 如果种子设定为空，请根据用户需求自行完整设计

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
    "protagonists": [
      {{
        "name": "",
        "role": "",
        "description": "",
        "appearance": ""
      }}
    ],
    "supporting": [
      {{
        "name": "",
        "role": "",
        "description": "",
        "appearance": ""
      }}
    ]
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


def build_volume_outline_prompt(data: dict, user_request: str = "") -> str:
    project = _to_block(data.get("project", {}))
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    story_request = data.get("project", {}).get("story_request", "") or data.get("story_request", "")
    completed_chapters = _to_block(data.get("completed_chapters", []))
    user_request_block = user_request.strip() or "无额外要求。请基于现有设定给出合理的长篇分卷规划。"

    return f"""你是一名长篇小说总策划助手。请先为这部小说设计分卷大纲。

【项目】
{project}

【用户需求】
{story_request}

【世界观】
{world}

【人物】
{characters}

【当前剧情状态】
{plot_state}

【文风】
{style}

【已完成章节（如有）】
{completed_chapters}

【用户对分卷的额外要求】
{user_request_block}

要求：
1. 输出必须是合法 JSON
2. 分卷规划要适合长篇连载，整体节奏要有递进，不要过早完结
3. 如果已有已完成章节，请把它们视为既成事实，不要推翻
4. 每卷都要有清晰的阶段目标、核心矛盾和推进重点
5. `planned_chapter_count` 请给出一个合理正整数，建议单卷 4 到 15 章
6. 卷与卷之间要有承接关系，既要延续主线，也要给人物关系升级留空间
7. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "volumes": [
    {{
      "volume_number": 1,
      "title": "",
      "summary": "",
      "story_goal": "",
      "planned_chapter_count": 8
    }}
  ]
}}
"""


def build_chapter_outline_prompt(
    data: dict,
    volume: dict,
    previous_volumes: list[dict] | None = None,
    completed_chapters: list[dict] | None = None,
    user_request: str = "",
) -> str:
    project = _to_block(data.get("project", {}))
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    story_request = data.get("project", {}).get("story_request", "") or data.get("story_request", "")
    volume_block = _to_block(volume)
    previous_volumes_block = _to_block(previous_volumes or [])
    completed_block = _to_block(completed_chapters or [])
    user_request_block = user_request.strip() or "无额外要求。请基于当前卷纲细化出稳定的分章推进。"

    return f"""你是一名长篇小说分章策划助手。请基于当前分卷大纲，为这一卷设计分章大纲。

【项目】
{project}

【用户需求】
{story_request}

【世界观】
{world}

【人物】
{characters}

【当前剧情状态】
{plot_state}

【文风】
{style}

【前序分卷大纲】
{previous_volumes_block}

【当前卷大纲】
{volume_block}

【本卷已完成章节（如有）】
{completed_block}

【用户对分章的额外要求】
{user_request_block}

要求：
1. 输出必须是合法 JSON
2. 本卷必须输出与 `planned_chapter_count` 一致数量的章节规划
3. 如果已有已完成章节，请把它们视为既成事实，并让剩余章节自然衔接
4. 每章都要有明确任务，不要让多章内容重复或空转
5. `summary` 侧重这一章会发生什么，`goal` 侧重写作时需要完成的叙事目标
6. `key_events` 请列出 2 到 5 个关键事件或推进点
7. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "volume_number": {int(volume.get("volume_number", 1) or 1)},
  "chapters": [
    {{
      "chapter_in_volume": 1,
      "title": "",
      "summary": "",
      "goal": "",
      "key_events": []
    }}
  ]
}}
"""


def build_writer_prompt(
    data: dict,
    recent_text: str,
    user_request: str = "",
    current_volume_outline: dict | None = None,
    chapter_outline: dict | None = None,
) -> str:
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    volume_outline = _to_block(current_volume_outline or {})
    chapter_outline_block = _to_block(chapter_outline or {})
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

【所属卷大纲】
{volume_outline}

【本章分章大纲】
{chapter_outline_block}

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
1. 循序渐进推进剧情或场景，不能急于结束或完成
2. 输出纯正文
3. 不要输出章标题、序号、小标题、Markdown标题
4. 如果用户额外要求与既有设定不冲突，优先吸收；若有冲突，以既有设定一致性为先，并尽量柔和地兼容用户意图
5. 本章必须完成分章大纲中的核心任务，同时与所属卷的阶段目标保持一致
6. 字数建议在 3000 字以上，5000 字以下，保持内容的丰富性和可读性
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


def build_illustration_prompt(data: dict, chapter_text: str, user_request: str = "") -> str:
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    project = _to_block(data.get("project", {}))
    excerpt = (chapter_text or "").strip()
    if len(excerpt) > 2600:
        excerpt = excerpt[:1300].rstrip() + "\n...\n" + excerpt[-1300:].lstrip()
    user_request_block = user_request.strip() or "无额外要求。"

    return f"""你是一名小说插画提示词助手。请基于已有设定和本章正文，为这一章设计一张单幅插图。

【项目】
{project}

【世界观】
{world}

【人物】
{characters}

【剧情状态】
{plot_state}

【文风】
{style}

【本章正文节选】
{excerpt}

【用户额外要求】
{user_request_block}

要求：
1. 只选择本章中最有画面感、最适合单张插图的一个瞬间
2. 保持人物外观、气质、场景与既有设定一致
3. 输出必须是合法 JSON
4. `positive_prompt` 必须直接写成适合文生图模型理解的中文提示词，不能写成几个抽象关键词
5. `positive_prompt` 需要明确包含：人物外貌、发型发色、衣着、动作、姿势、表情、视线或互动关系、场景环境、构图、景别、镜头角度、光线、材质/细节质感
6. 人物外貌必须优先参考 characters 中的 `appearance`，不得随意改脸、改体型、改服装风格
7. `characters` 字段中只保留“所选这个瞬间里实际出场、能被镜头看到”的角色，不要把本章其他未出现在该瞬间的人物塞进来
8. 每个角色都要包含 `name`、`appearance`、`outfit`、`action`、`expression`
9. `environment` 要写清前景/中景/远景或至少空间层次，以及关键道具、家具、自然或建筑环境
10. `composition` 要写清景别、机位、构图重点
11. `lighting` 要写清光源类型、冷暖关系、氛围
12. `negative_prompt` 需要简洁有效，避免文字、水印、多视角分镜、畸形肢体、低质量
13. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "scene_summary": "",
  "characters": [
    {{
      "name": "",
      "appearance": "",
      "outfit": "",
      "action": "",
      "expression": ""
    }}
  ],
  "environment": "",
  "composition": "",
  "lighting": "",
  "positive_prompt": "",
  "negative_prompt": ""
}}
"""
