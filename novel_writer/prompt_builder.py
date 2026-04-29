"""Prompt builders for novel generation and state updates."""

from __future__ import annotations

import json
from typing import Any


def _to_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _section_block(title: str, content: Any) -> str:
    text = str(content or "").strip()
    if not text:
        return ""
    return f"【{title}】\n{text}"


def _join_blocks(*blocks: str) -> str:
    return "\n\n".join(block for block in blocks if block)


_BASE_SYSTEM_PROMPT = (
    "你是小说创作工作流中的稳定执行代理。始终优先遵守用户消息里的任务、输出格式、"
    "既有设定和连续性事实；不要编造缺失的工程信息，不要输出与任务无关的解释。"
)


_SYSTEM_PROMPTS = {
    "planner": (
        "你负责长篇连载小说的规划。规划必须服务连续写作，尊重已完成章节，保持阶段目标清晰，"
        "避免空转、重复和过早完结。需要结构化输出时，只输出合法 JSON。"
    ),
    "writer": (
        "你负责撰写长篇连载小说正文。保持人物、地点、时间、伏笔、因果和动机线一致；"
        "只写当前章节，不提前完成后续章节核心情节；输出纯正文，不写说明、标题或 Markdown。"
    ),
    "craft_brief": (
        "你负责写前创作蓝图。只为当前章节提供可执行建议，强化开章钩子、行动理由、场景推进、"
        "人物互动和重复规避；需要结构化输出时，只输出合法 JSON。"
    ),
    "quality_review": (
        "你负责章节质检。严格检查任务完成、吸引力、场景新鲜度、人物具体性、动机因果、连续性和重复风险；"
        "证据要具体，修订建议要可执行；只输出合法 JSON。"
    ),
    "rewrite": (
        "你负责按审稿意见改写章节。保留正确完成的剧情事实和连续性，优先修复硬伤、低分项和重复写法；"
        "只输出改写后的完整正文，不写改稿说明。"
    ),
    "summary": (
        "你负责维护小说 live state。只记录新章节确实发生并会影响后续的事实、限制、线索、关系变化和写法记忆；"
        "需要结构化输出时，只输出合法 JSON。"
    ),
    "polish": (
        "你负责润色已完成章节。保留核心剧情事实、事件顺序、人物决定和后续承接点；"
        "可增强表达、节奏、细节和对话，但不得新增会改变后续状态的重大事实。"
    ),
    "illustration": (
        "你负责把小说章节转化为单幅插图提示词。严格保持人物外貌、场景事实和章节瞬间一致；"
        "提示词要具体可画，需要结构化输出时，只输出合法 JSON。"
    ),
}


def build_system_prompt(stage: str) -> str:
    stage_key = str(stage or "").strip().lower()
    stage_prompt = _SYSTEM_PROMPTS.get(stage_key, _SYSTEM_PROMPTS["planner"])
    return f"{_BASE_SYSTEM_PROMPT}\n{stage_prompt}"


def _writer_volume_outline_block(volume: dict | None, chapter_outline: dict | None) -> str:
    source = volume if isinstance(volume, dict) else {}
    chapter = chapter_outline if isinstance(chapter_outline, dict) else {}
    compact = {
        "volume_number": source.get("volume_number", ""),
        "title": source.get("title", ""),
        "summary": source.get("summary", ""),
        "story_goal": source.get("story_goal", ""),
        "planned_chapter_count": source.get("planned_chapter_count", 0),
        "current_chapter_number": chapter.get("chapter_number", ""),
        "current_chapter_title": chapter.get("title", ""),
    }
    return _to_block(compact)


def _writer_plot_state_block(plot_state: dict | None, chapter_outline: dict | None) -> str:
    state = dict(plot_state) if isinstance(plot_state, dict) else {}
    if chapter_outline:
        state.pop("next_chapter_goal", None)
    return _to_block(state)


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
4. `protagonists` 只放核心常驻人物；`supporting` 只保留第一章到前几章就会实际出场、并对开篇推进不可缺少的必要配角，没有就返回空数组
5. 不要为了“以后可能会用到”提前创建导师、邻居、系统 AI、反派手下等暂时不会出场的配角档案
6. 每个角色对象都必须包含 `name`、`role`、`description`、`appearance` 四个字段
7. `description` 侧重人物身份、性格、能力、关系与叙事定位
8. `appearance` 单独描写人物外貌特征（包括人种，避免文生图模型的不确定性）、体态气质、发型发色、五官特点、常见衣着与穿搭风格
9. `plot_state.active_characters` 在初始化阶段只填写第一章开场就会实际出场的人物，通常 1 到 3 人，不要把未来角色写进去
10. plot_state 必须兼容以下结构
11. style 要明确语气、视角和写作要求
12. 如果种子设定为空，请根据用户需求自行完整设计

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
    "supporting": []
  }},
  "plot_state": {{
    "main_plot": "",
    "current_arc": "",
    "recent_events": [],
    "open_threads": [],
    "resolved_threads": [],
    "foreshadowing": [],
    "continuity_anchors": [],
    "causal_links": [],
    "character_updates": [],
    "active_characters": [],
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


def build_story_setup_prompt(data: dict) -> str:
    project_name = data.get("project_name", "")
    project_description = data.get("project_description", "")
    story_request = data.get("story_request", "")
    world_seed = _to_block(data.get("world_seed", {}))
    characters_seed = _to_block(data.get("characters_seed", {}))

    return f"""你是一名长篇小说前期设定策划。请在项目正式初始化和大纲规划之前，先根据用户输入的故事需求，具体化并创造人物和背景设定。

【项目名】
{project_name}

【项目简介】
{project_description}

【用户故事需求】
{story_request}

【已有世界观种子】
{world_seed}

【已有人物种子】
{characters_seed}

要求：
1. 输出必须是合法 JSON，不要输出 Markdown 或解释
2. 这个阶段不是记录或复述用户需求，而是把需求扩写成可直接支撑后续写作的原创设定
3. 必须主动补足用户没有细写但故事需要的内容：人物姓名、身份来历、关系张力、能力边界、背景机制、资源限制、开篇困境
4. 可以在不违背用户需求的前提下创造具体细节；如果用户需求含糊，要选择一个清晰可写的版本落地
5. 这个阶段只做人设和背景设定，不要写章节大纲，不要提前安排具体章节事件
6. `world.background` 要把故事前情、当前危机、社会/科技/魔法/组织背景、资源约束和开篇处境整理清楚，避免只把用户原话拆成条目
7. `world.rules` 只写会影响后续连续性的硬设定、限制和禁忌
8. `protagonists` 只放核心常驻人物；`supporting` 只保留开篇前几章实际需要出场的必要配角，没有就返回空数组
9. 不要为了“以后可能会用到”提前创建暂时不会出场的导师、邻居、系统 AI、反派手下等配角档案
10. 每个角色对象都必须包含 `name`、`role`、`description`、`appearance` 四个字段
11. `description` 侧重身份、性格、能力、关系、欲望/弱点和叙事功能，不要只列用户已经说过的标签
12. `appearance` 单独描写人物外貌特征（包括人种，避免文生图模型的不确定性）、体态气质、发型发色、五官特点、常见衣着与穿搭风格
13. 人物设定必须直接服务用户需求中的关系、题材、冲突和长期写作方向

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
    "supporting": []
  }}
}}
"""


def build_volume_outline_prompt(data: dict, user_request: str = "") -> str:
    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        return f"""你是一名长篇小说总策划助手。请先为这部小说设计分卷大纲。

【作者意图】
{sections.get("author_intent", "")}

【世界观速览】
{sections.get("world", "")}

【角色速览】
{sections.get("characters", "")}

【当前 live state】
{sections.get("live_state", "")}

【文风契约】
{sections.get("style_contract", "")}

【已完成章节（如有）】
{sections.get("completed_chapters", "[]")}

【用户对分卷的额外要求】
{sections.get("user_request", user_request.strip() or "无额外要求。请基于现有设定给出合理的长篇分卷规划。")}

要求：
1. 输出必须是合法 JSON
2. 分卷规划要适合长篇连载，整体节奏要有递进，不要过早完结
3. 已完成章节是既成事实，不要推翻
4. 每卷都要有明确阶段目标、核心矛盾和推进重点
5. `planned_chapter_count` 给出合理正整数，建议单卷 4 到 15 章
6. 卷与卷之间必须承接自然，既延续主线，也给人物关系升级留空间
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
    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        current_volume = sections.get("current_volume", _to_block(volume))
        return f"""你是一名长篇小说分章策划助手。请基于当前分卷大纲，为这一卷设计分章大纲。

【作者意图】
{sections.get("author_intent", "")}

【世界观速览】
{sections.get("world", "")}

【角色速览】
{sections.get("characters", "")}

【当前 live state】
{sections.get("live_state", "")}

【文风契约】
{sections.get("style_contract", "")}

【前序分卷摘要】
{sections.get("previous_volumes", "[]")}

【当前卷大纲】
{current_volume}

【本卷已完成章节（如有）】
{sections.get("completed_chapters", "[]")}

【用户对分章的额外要求】
{sections.get("user_request", user_request.strip() or "无额外要求。请基于当前卷纲细化出稳定的分章推进。")}

要求：
1. 输出必须是合法 JSON
2. 本卷必须输出与 `planned_chapter_count` 一致数量的章节规划
3. 已完成章节是既成事实，剩余章节要自然衔接
4. 每章都要有明确任务，不要让多章内容重复或空转
5. `summary` 写这一章会发生什么，`goal` 写这一章必须完成的叙事目标
6. `key_events` 列出 2 到 5 个关键事件或推进点
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


def build_batch_chapter_plan_prompt(
    data: dict,
    upcoming_chapters: list[dict],
    user_request: str,
) -> str:
    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        return f"""你是长篇连载小说的续写规划助手。请根据用户现在想看的情节，为接下来几章做一次短期续写编排。

【作者意图】
{sections.get("author_intent", "")}

【世界观速览】
{sections.get("static_world", "")}

【角色速览】
{sections.get("static_characters", "")}

【当前 live state】
{sections.get("live_state", "")}

【检索到的相关记忆】
{sections.get("retrieved_memory", "无")}

【最近场景与近两章摘要】
{sections.get("recent_scene", "")}

【文风契约】
{sections.get("style_contract", "")}

【接下来待写的章节框架】
{sections.get("upcoming_chapters", _to_block(upcoming_chapters))}

【用户这次想看的内容】
{sections.get("user_request", user_request.strip())}

要求：
1. 输出必须是合法 JSON
2. 把用户想看的内容合理分配到接下来这些章节，而不是每章重复同一句要求
3. 不同章节可分别承担铺垫、受阻、推进、兑现或余波
4. 规划必须尊重既有顺序、当前状态与已有任务，尽量做细化和重心调整
5. 每章都要给出清晰且彼此不同的 `writer_guidance`
6. `key_events` 列出 2 到 5 个关键推进点
7. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "",
      "summary": "",
      "goal": "",
      "key_events": [],
      "request_focus": "",
      "request_role": "",
      "writer_guidance": ""
    }}
  ]
}}
"""

    project = _to_block(data.get("project", {}))
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    story_request = data.get("project", {}).get("story_request", "") or data.get("story_request", "")
    chapters_block = _to_block(upcoming_chapters)

    return f"""你是长篇连载小说的续写规划助手。请根据用户现在想看的情节，为接下来几章做一次“短期续写编排”。
【项目】{project}

【用户最初的故事需求】{story_request}

【世界观】{world}

【人物】{characters}

【当前剧情状态】{plot_state}

【文风】{style}

【接下来待写的章节框架】{chapters_block}

【用户这次想看的内容】{user_request.strip()}

要求：
1. 输出必须是合法 JSON
2. 目标是把“用户这次想看的内容”合理分配到接下来这些章节里，而不是每一章都机械重复同一句要求
3. 有些章节可以负责铺垫、准备、试探、受阻、推进、阶段性完成或余波，不要求每章都直接完成用户想看的核心场景
4. 规划必须尊重现有章节顺序、当前剧情状态和原本分章目标，尽量做“细化与重心调整”，不要彻底推翻既有大纲
5. 如果用户要求天然只适合其中一两章，请让其他章节承担前置准备或后续影响，避免重复
6. 每章都要给出清晰且彼此不同的 `writer_guidance`
7. `key_events` 请列出 2 到 5 个关键推进点
8. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "chapters": [
    {{
      "chapter_number": 1,
      "title": "",
      "summary": "",
      "goal": "",
      "key_events": [],
      "request_focus": "",
      "request_role": "",
      "writer_guidance": ""
    }}
  ]
}}
"""


def build_auto_objective_prompt(
    data: dict,
    recent_text: str,
    next_context: dict,
    *,
    user_request: str = "",
    planning_mode: str = "",
) -> str:
    project = _to_block(data.get("project", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    current_volume = _to_block(next_context.get("volume", {}))
    current_chapter = _to_block(next_context.get("chapter", {}))
    user_request_block = user_request.strip() or "无额外要求。请仅基于当前状态提炼下一章 objective。"

    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        prompt_body = _join_blocks(
            "你是长篇连载小说的章节规划助手。请只为下一章提炼一个清晰的 objective。",
            _section_block("作者意图", sections.get("author_intent", "")),
            _section_block("当前任务卡基线", sections.get("chapter_task", "")),
            _section_block("世界观速览", sections.get("static_world", "")),
            _section_block("角色速览", sections.get("static_characters", "")),
            _section_block("当前 live state", sections.get("live_state", "")),
            _section_block("更早相关记忆", sections.get("retrieved_memory", "")),
            _section_block("最近场景", sections.get("recent_scene", "")),
            _section_block("补充写作约束", sections.get("style_contract", "")),
            _section_block("当前 planning mode", sections.get("planning_mode", planning_mode)),
            _section_block("用户这次想看的方向", sections.get("user_request", user_request_block)),
        )
        return f"""{prompt_body}

要求：
1. 输出必须是合法 JSON
2. 只生成“下一章”的 objective，不要把两三章后的目标提前写进来
3. objective 必须是本章要完成的叙事任务，不要写成执行 plan、情绪要求或文风要求
4. objective 要尊重当前 live state、最近场景、相关记忆和基线任务卡；若用户有额外意图，请自然吸收进本章任务
5. objective 应当具体、可执行、可验证，长度尽量控制在 1 句话内
6. 不要输出解释，不要输出 Markdown

输出 JSON：
{{"objective":""}}
"""

    return f"""你是长篇连载小说的章节规划助手。请只为下一章提炼一个清晰的 objective。

【项目】{project}

【当前剧情状态】{plot_state}

【当前 planning mode】{planning_mode}

【所属卷与待写章节上下文】
{{
  "volume": {current_volume},
  "chapter": {current_chapter}
}}

【最近正文】
{recent_text}

【用户这次想看的方向】
{user_request_block}

要求：
1. 输出必须是合法 JSON
2. 只生成“下一章”的 objective，不要把两三章后的目标提前写进来
3. objective 必须是本章要完成的叙事任务，不要写成执行 plan、情绪要求或文风要求
4. objective 要尊重当前状态和最近正文；若用户有额外意图，请自然吸收进本章任务
5. objective 应当具体、可执行、可验证，长度尽量控制在 1 句话内
6. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "objective": ""
}}
"""


def build_progression_options_prompt(
    data: dict,
    recent_text: str,
    next_context: dict,
    *,
    user_request: str = "",
    option_count: int = 4,
    planning_mode: str = "",
) -> str:
    project = _to_block(data.get("project", {}))
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    story_request = data.get("project", {}).get("story_request", "") or data.get("story_request", "")
    current_volume = _to_block(next_context.get("volume", {}))
    current_chapter = _to_block(next_context.get("chapter", {}))
    user_request_block = user_request.strip() or "无额外要求。请仅基于当前状态给出下一章推进选项。"

    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        option_total = int(sections.get("option_count", option_count) or option_count)
        if option_total == 1:
            opening = "你是长篇连载小说的剧情推进顾问。请只为下一章设计一个唯一最优推进项，供自动续写直接执行。"
            count_rule = "必须返回恰好 1 个推进项；它应是当前信息下质量最高、最稳妥且最有推进力的方案"
            recommendation_rule = "这个唯一推进项必须 `recommended=true`，`recommended_option_id` 必须与它一致"
            objective_rule = "推进项要尊重既有设定、当前状态、最近场景和下一章任务卡；把任务卡里的 `objective` 当作硬约束，设计一个可直接执行的 plan"
            scope_rule = "推进项只明确这一章的切入角度、推进顺序和强调重点，不要另起一个与当前任务卡冲突的新目标"
        else:
            opening = "你是长篇连载小说的剧情推进顾问。请只为下一章设计若干互斥且都合理的推进方案。"
            count_rule = f"必须返回恰好 {option_total} 个互斥选项，且只能有一个 `recommended=true`"
            recommendation_rule = "`recommended_option_id` 必须与唯一的推荐项一致"
            objective_rule = "方案要尊重既有设定、当前状态、最近场景和下一章任务卡；先把任务卡里的 `objective` 当作硬约束，再基于它设计不同 plan"
            scope_rule = "选项只改变这一章的切入角度、推进顺序和强调重点，不要另起一个与当前任务卡冲突的新目标"
        prompt_body = _join_blocks(
            opening,
            _section_block("作者意图", sections.get("author_intent", "")),
            _section_block("下一章任务卡", sections.get("chapter_task", "")),
            _section_block("世界观速览", sections.get("static_world", "")),
            _section_block("角色速览", sections.get("static_characters", "")),
            _section_block("当前 live state", sections.get("live_state", "")),
            _section_block("更早相关记忆", sections.get("retrieved_memory", "")),
            _section_block("最近场景", sections.get("recent_scene", "")),
            _section_block("补充写作约束", sections.get("style_contract", "")),
            _section_block("当前 planning mode", sections.get("planning_mode", planning_mode)),
            _section_block("用户这次想看的方向", sections.get("user_request", user_request_block)),
        )
        return f"""{prompt_body}

要求：
1. 输出必须是合法 JSON
2. 只针对下一章给方案，不要把两三章后的核心剧情提前塞进来
3. {count_rule}
4. {recommendation_rule}
5. {objective_rule}
6. {scope_rule}
7. 每个选项都必须包含 `option_id`、`title`、`plan_summary`、`plan_steps`、`plan_guidance`、`recommended`
8. `plan_steps` 给 2 到 5 个条目，写本章真正会发生的推进节点
9. `plan_summary` 要描述这一章会怎么推进；`plan_guidance` 只补充写法与强调点，不要偷偷改写章节主目标
10. 不要输出解释，不要输出 Markdown

输出 JSON 骨架：
{{"recommended_option_id":"option_1","options":[{{"option_id":"option_1","title":"","plan_summary":"","plan_steps":["",""],"plan_guidance":"","recommended":true}}]}}
"""

    option_total = int(option_count or 4)
    if option_total == 1:
        opening = "你是长篇连载小说的剧情推进顾问。请为“下一章”设计一个唯一最优推进项，供自动续写直接执行。"
        count_rule = "必须返回恰好 1 个推进项；它应是当前信息下质量最高、最稳妥且最有推进力的方案"
        recommendation_rule = "这个唯一推进项必须 `recommended=true`，并且 `recommended_option_id` 必须与该推进项一致"
        objective_rule = "推进项要尊重既有设定、最近正文、当前剧情状态和下一章上下文，不要推翻已有章纲；把当前任务卡里的 `objective` 当作硬约束，设计一个可直接执行的 plan"
        scope_rule = "推进项只明确这一章怎么推进，不要另起一个与当前任务卡或章纲冲突的新目标"
    else:
        opening = "你是长篇连载小说的剧情推进顾问。请为“下一章”设计若干互斥但都合理的推进方案，供用户二选一或多选一中的单选。"
        count_rule = f"必须返回恰好 {option_total} 个互斥选项，每个选项都应代表这一章的不同重心或不同推进路径"
        recommendation_rule = "只能有一个选项 `recommended=true`，并且 `recommended_option_id` 必须与该选项一致"
        objective_rule = "方案要尊重既有设定、最近正文、当前剧情状态和下一章上下文，不要推翻已有章纲；先把当前任务卡里的 `objective` 当作硬约束，再基于它设计不同 plan"
        scope_rule = "选项只改变这一章怎么推进，不要另起一个与当前任务卡或章纲冲突的新目标"

    return f"""{opening}

【项目】{project}

【用户最初的故事需求】{story_request}

【世界观】{world}

【人物】{characters}

【当前剧情状态】{plot_state}

【当前 planning mode】{planning_mode}

【所属卷与待写章节上下文】
{{
  "volume": {current_volume},
  "chapter": {current_chapter}
}}

【最近正文】
{recent_text}

【用户这次想看的方向】
{user_request_block}

【需要给出的选项数】
{option_total}

要求：
1. 输出必须是合法 JSON
2. 必须只针对“下一章”给方案，不要把两三章后的核心剧情提前塞进来
3. {count_rule}
4. {objective_rule}
5. {scope_rule}
6. 每个选项都必须包含：
   - `option_id`
   - `title`
   - `plan_summary`
   - `plan_steps`
   - `plan_guidance`
   - `recommended`
7. `plan_summary` 描述本章会怎么推进；`plan_guidance` 只补充写法、氛围和强调点，不要改写章节主目标
8. `plan_steps` 要给出 2 到 5 个条目，写本章会实际发生的关键推进
9. {recommendation_rule}
10. 不要输出解释，不要输出 Markdown

输出 JSON：
{{
  "recommended_option_id": "option_1",
  "options": [
    {{
      "option_id": "option_1",
      "title": "",
      "plan_summary": "",
      "plan_steps": [],
      "plan_guidance": "",
      "recommended": true
    }}
  ]
}}
"""


def build_craft_brief_prompt(data: dict) -> str:
    sections = data.get("sections", {}) if isinstance(data, dict) else {}
    prompt_body = _join_blocks(
        "你是一名长篇小说创作总监。请在正式写正文前，为下一章设计一份短小但可执行的创作蓝图。",
        _section_block("作者意图", sections.get("author_intent", "")),
        _section_block("下一章任务卡", sections.get("chapter_task", "")),
        _section_block("当前 live state", sections.get("live_state", "")),
        _section_block("世界", sections.get("static_world", "")),
        _section_block("角色", sections.get("static_characters", "")),
        _section_block("更早相关记忆", sections.get("retrieved_memory", "")),
        _section_block("近期写法避让", sections.get("recent_craft_memory", "")),
        _section_block("最近场景", sections.get("recent_scene", "")),
        _section_block("补充写作约束", sections.get("style_contract", "")),
    )
    return f"""{prompt_body}

要求：
1. 输出必须是合法 JSON
2. 只设计当前这一章，不要改变任务卡的核心目标
3. 蓝图要帮助正文更吸引人：开章钩子、戏剧问题、压力来源、人物选择、情绪转折都要具体
4. `context_bridge` 写清本章开场需要补给读者的处境、人物入口或连续性锚点
5. `action_reasoning` 写清人物采取关键行动的直接原因、压力与选择依据
6. `forbidden_repeats` 必须列出需要避开的上一章表层动作、姿态、句式或结尾套路
7. `fresh_interaction_patterns` 要给出新的互动方式，不要只写“更细腻”“更紧张”这类抽象要求
8. `success_criteria` 给出 2 到 5 条本章必须兑现的可检查目标，用于写后质检
9. 不要输出解释，不要输出 Markdown

输出 JSON 骨架：
{{"chapter_hook":"","context_bridge":"","dramatic_question":"","conflict_pressure":"","action_reasoning":"","emotional_turn":"","scene_movement":[],"sensory_palette":[],"fresh_interaction_patterns":[],"forbidden_repeats":[],"success_criteria":[],"focus_notes":""}}
"""


def build_writer_prompt(
    data: dict,
    recent_text: str = "",
    user_request: str = "",
    current_volume_outline: dict | None = None,
    chapter_outline: dict | None = None,
) -> str:
    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        chapter_count = int(data.get("chapter_count", 0) or 0)
        opening_note = (
            "当前将要写的是第一章。开章可以有钩子，但必须在前 800-1200 字内自然交代读者入口："
            "故事发生的基本处境、地点/时间/关键世界规则、至少核心人物的姓名/关系/当前状态，"
            "以及他们为什么现在必须行动。背景和人物要嵌入观察、对话和选择，不要直接跳到任务事件，"
            "也不要写成设定说明书；不要把后续章节才会发生的大事件提前写出。"
            if chapter_count == 0
            else "这不是第一章，请延续已有状态、记忆与场景动势。"
        )
        prompt_body = _join_blocks(
            "你是一名长篇小说写作助手。请续写下一章。",
            _section_block("作者意图", sections.get("author_intent", "")),
            _section_block("下一章任务卡", sections.get("chapter_task", "")),
            _section_block("当前 live state", sections.get("live_state", "")),
            _section_block("世界", sections.get("static_world", "")),
            _section_block("角色", sections.get("static_characters", "")),
            _section_block("更早相关记忆", sections.get("retrieved_memory", "")),
            _section_block("近期写法避让", sections.get("recent_craft_memory", "")),
            _section_block("本章创作蓝图", sections.get("craft_brief", "")),
            _section_block("最近场景", sections.get("recent_scene", "")),
            _section_block("补充写作约束", sections.get("style_contract", "")),
            _section_block("开篇说明", opening_note),
        )
        return f"""{prompt_body}

要求：
1. 人物、地点、时间、未解线程、连续性锚点、因果/动机线与已写正文必须一致；不要遗忘伏笔，也不要把已解决的事情重新写回未解决
2. 每个关键行动都要能看出“为什么现在做、人物想达成什么、受到什么压力或限制、选择造成什么结果”
3. 本章必须产生至少一项新的可验证变化；若沿用同一地点、目标或冲突，也要写出新的信息、代价、决定或结果
4. 严格只写当前这一章。任务卡是当前章节任务的最高优先级来源，不要提前完成下一章或后续章节的大事件
5. 如果提供了“近期写法避让”和“本章创作蓝图”，必须避开其中列出的表层重复；同类动作只有在产生新功能、新代价或新关系变化时才能使用
6. 开章要尽快给读者一个具体钩子；场景推进要有压力、选择、结果，不要只在同一种姿态和情绪里反复停留
7. 结尾可留下明确悬念或过渡，但不要把后续章核心情节直接写完
8. 输出纯正文，不要章标题、序号、小标题、Markdown 标题
9. 字数建议在 3000 字以上、5000 字以下，保持内容丰富且可读
"""

    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _writer_plot_state_block(data.get("plot_state", {}), chapter_outline)
    style = _to_block(data.get("style", {}))
    volume_outline = _writer_volume_outline_block(current_volume_outline, chapter_outline)
    chapter_outline_block = _to_block(chapter_outline or {})
    next_chapter_goal = data.get("plot_state", {}).get("next_chapter_goal", "")
    has_chapter_outline = bool(chapter_outline)
    fallback_goal_block = next_chapter_goal.strip() if not has_chapter_outline else ""
    user_request_block = user_request.strip() if user_request.strip() else "无。请在既有设定下自由发挥，自然推进剧情。"
    chapter_count = int(data.get("project", {}).get("chapter_count", 0) or 0)
    first_chapter_note = (
        "当前将要写的是第一章。因为正文尚未开始，recent_events、open_threads、foreshadowing、character_updates 此时应为空，"
        "不要把后续章节才会出现的事件总结提前写进当前状态。开章可以有钩子，但必须在前 800-1200 字内自然交代读者入口："
        "故事发生的基本处境、地点/时间/关键世界规则、至少核心人物的姓名/关系/当前状态，以及他们为什么现在必须行动。"
        "背景和人物要嵌入观察、对话和选择，不要直接跳到任务事件，也不要写成设定说明书。"
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

【本章目标补充说明】
{fallback_goal_block or "无额外补充；请以本章分章大纲为准。"}

【开篇说明】
{first_chapter_note or "这不是第一章，请延续已有状态与正文。"}

【用户额外要求】
{user_request_block}

【文风】
{style}

要求：
1. 人物不能 OOC
2. 不遗忘伏笔、连续性锚点和因果/动机线
3. 循序渐进推进剧情或场景，不能急于结束或完成；关键行动必须有当前原因、人物目标、压力限制和可见结果
4. 输出纯正文
5. 不要输出章标题、序号、小标题、Markdown 标题
6. 如果用户额外要求与既有设定不冲突，优先吸收；若有冲突，以既有设定一致性为先，并尽量柔和地兼容用户意图
7. 如果提供了“本章分章大纲”，它就是当前章节任务的最高优先级来源；不要再自行改写成另一个目标
8. 本章必须完成分章大纲中的核心任务，同时与所属卷的阶段目标保持一致
9. 严格只写当前这一章，不要提前完成下一章或后续章节的大事件
10. 如果本章结尾需要承接后续内容，可以留下明确悬念或过渡，但不要把后续章的核心情节直接写完
11. 第一章必须先建立背景、人物和行动理由，再推进任务事件；后续章节必须承接已保存的状态与记忆
12. 字数建议在 3000 字以上、5000 字以下，保持内容的丰富性和可读性
"""


def build_quality_review_prompt(data: dict, draft_text: str, *, strict: bool = False) -> str:
    sections = data.get("sections", {}) if isinstance(data, dict) else {}
    threshold_note = "高质量模式：评分要严格，重复动作、无效拖延、弱钩子都应指出。" if strict else "平衡模式：重点识别明显问题。"
    prompt_body = _join_blocks(
        "你是一名长篇小说责任编辑。请审阅这章草稿是否完成任务、是否吸引人、是否重复上一章写法。",
        _section_block("审稿模式", threshold_note),
        _section_block("作者意图", sections.get("author_intent", "")),
        _section_block("下一章任务卡", sections.get("chapter_task", "")),
        _section_block("当前 live state", sections.get("live_state", "")),
        _section_block("近期写法避让", sections.get("recent_craft_memory", "")),
        _section_block("本章创作蓝图", sections.get("craft_brief", "")),
        _section_block("最近场景", sections.get("recent_scene", "")),
        _section_block("待审草稿", draft_text),
    )
    return f"""{prompt_body}

要求：
1. 输出必须是合法 JSON
2. 七个分项分数都用 0 到 10，分数越高越好
3. `repetition_risk` 的高分代表重复风险低、写法新鲜；低分代表动作/句式/场景结构复用明显
4. `motivation_causality` 检查关键行动是否有明确动因、压力、选择和结果
5. 重点检查“本章创作蓝图”里的验收标准是否兑现；未兑现的必须写入 `blocking_issues` 或 `issues`
6. `passed` 表示是否可以作为最终章节保存；有 `severity="blocker"` 的问题时必须为 false
7. `score_reasons` 为低分或关键分项给出一句具体理由
8. `blocking_issues` 只放会阻止保存的硬伤，每项包含 `category`、`severity`、`issue`、`evidence`、`fix`
9. `nice_to_have` 放不阻止保存但值得优化的问题
10. `rewrite_plan` 给出可直接交给改稿模型执行的 2 到 6 步修订方案
11. `revision_guidance` 必须具体指出需要如何改，不要泛泛而谈
12. `review_unavailable` 正常审稿时必须为 false
13. 不要输出解释，不要输出 Markdown

输出 JSON 骨架：
{{"schema_version":2,"scores":{{"task_completion":0,"reader_hook":0,"scene_freshness":0,"character_specificity":0,"motivation_causality":0,"repetition_risk":0,"continuity":0}},"score_reasons":{{"task_completion":"","reader_hook":"","scene_freshness":"","character_specificity":"","motivation_causality":"","repetition_risk":"","continuity":""}},"passed":false,"review_unavailable":false,"strengths":[],"issues":[],"blocking_issues":[{{"category":"","severity":"blocker","issue":"","evidence":"","fix":""}}],"nice_to_have":[],"revision_guidance":"","rewrite_plan":[],"repeat_examples":[]}}
"""


def build_rewrite_prompt(data: dict, draft_text: str, review_report: dict) -> str:
    sections = data.get("sections", {}) if isinstance(data, dict) else {}
    review_block = _to_block(review_report)
    prompt_body = _join_blocks(
        "你是一名长篇小说改稿助手。请根据审稿意见重写当前章节，并只输出重写后的完整正文。",
        _section_block("作者意图", sections.get("author_intent", "")),
        _section_block("下一章任务卡", sections.get("chapter_task", "")),
        _section_block("当前 live state", sections.get("live_state", "")),
        _section_block("近期写法避让", sections.get("recent_craft_memory", "")),
        _section_block("本章创作蓝图", sections.get("craft_brief", "")),
        _section_block("最近场景", sections.get("recent_scene", "")),
        _section_block("审稿报告", review_block),
        _section_block("原草稿", draft_text),
    )
    return f"""{prompt_body}

要求：
1. 保留已经正确完成的剧情目标和连续性
2. 优先修复审稿报告中的 `blocking_issues`、低分项和 `rewrite_plan`
3. 尤其注意重复动作、弱钩子、场景空转、人物反应泛化、关键行动缺少原因
4. 不要写改稿说明，不要输出 JSON，不要输出 Markdown 标题
5. 输出纯正文，字数建议仍在 3000 字以上、5000 字以下
"""


def build_summary_prompt(data: dict, new_text: str) -> str:
    if isinstance(data, dict) and isinstance(data.get("sections"), dict):
        sections = data["sections"]
        prompt_body = _join_blocks(
            "请基于新章节更新小说 live state。",
            _section_block("已有 live state", sections.get("live_state", "")),
            _section_block("角色速览", sections.get("static_characters", "")),
            _section_block("本章写前任务卡", sections.get("completed_task", "无")),
            _section_block("新章节", sections.get("chapter_text", new_text)),
        )
        return f"""{prompt_body}

要求：
1. 输出必须是合法 JSON
2. `current_location`、`current_time`、`current_arc` 以本章结束时的状态为准
3. `open_threads` 只保留仍未解决的问题，已解决的写入 `resolved_threads`
4. `active_characters` 只保留本章真正参与推进的角色
5. `chapter_summary` 用 1 到 2 句话概括本章核心推进
6. `next_chapter_goal` 写本章结束后最该继续推进的一步；如果本章主任务已完成，不要直接重复任务卡原句
7. `continuity_anchors` 记录后续必须记住且不能随意改写的事实、限制、资源状态、关系变化或承诺
8. `causal_links` 记录本章形成的行动原因链：因为什么压力/信息，谁决定做什么，导致什么新局面
9. `retrieval_tags` 给出便于后续检索的简短标签
10. `craft_notes` 记录本章已经用过的写法，供下一章避让重复；包括 repeated_actions、recurring_gestures、scene_type、emotional_beat、ending_pattern、notable_phrasing
11. 不要输出解释，不要输出 Markdown

输出 JSON 骨架：
{{"chapter_summary":"","current_location":"","current_time":"","current_arc":"","recent_events":[],"open_threads":[],"resolved_threads":[],"foreshadowing":[],"continuity_anchors":[],"causal_links":[],"character_updates":[],"active_characters":[],"retrieval_tags":[],"next_chapter_goal":"","craft_notes":{{"repeated_actions":[],"recurring_gestures":[],"scene_type":"","emotional_beat":"","ending_pattern":"","notable_phrasing":[]}}}}
"""

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
  "resolved_threads": [],
  "foreshadowing": [],
  "continuity_anchors": [],
  "causal_links": [],
  "character_updates": [],
  "active_characters": [],
  "retrieval_tags": [],
  "next_chapter_goal": "",
  "craft_notes": {{
    "repeated_actions": [],
    "recurring_gestures": [],
    "scene_type": "",
    "emotional_beat": "",
    "ending_pattern": "",
    "notable_phrasing": []
  }}
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


def build_chapter_polish_prompt(
    data: dict,
    chapter_text: str,
    *,
    polish_directions: list[str] | None = None,
    custom_request: str = "",
) -> str:
    project = _to_block(data.get("project", {}))
    world = _to_block(data.get("world", {}))
    characters = _to_block(data.get("characters", {}))
    plot_state = _to_block(data.get("plot_state", {}))
    style = _to_block(data.get("style", {}))
    author_intent = _to_block(data.get("author_intent", {}))
    directions = polish_directions or ["基础润色"]
    directions_block = "\n".join(f"- {item}" for item in directions if str(item or "").strip()) or "- 基础润色"
    custom_request_block = custom_request.strip() or "无额外自定义要求。"

    return f"""你是一名长篇小说章节润色助手。请对用户提供的已完成章节进行润色，并只输出润色后的完整章节正文。

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

【作者意图】
{author_intent}

【润色方向】
{directions_block}

【用户自定义润色要求】
{custom_request_block}

【原章节正文】
{chapter_text.strip()}

要求：
1. 只输出润色后的完整章节正文，不要输出解释、标题说明、Markdown 代码块或修改清单
2. 保留原章节的核心剧情事实、事件顺序、人物决定、信息结论和后续承接点
3. 允许轻微补充过渡、动作、表情、环境、内心和对白，让桥段更顺、更有画面感
4. 不得新增会影响后续剧情状态的事实、线索、设定、角色关系转折或重大道具
5. 保持人物称谓、视角、基调、世界观规则和既有设定一致，不要让角色 OOC
6. 如果要求“更长”，应优先扩写描写、节奏和互动，不要靠重复原句凑长度
7. 如果要求“更欢乐”，应增加轻松互动或语言节奏，但不要破坏当前危机和人物处境
8. 输出必须是正文文本本身，不能包裹 JSON，不能使用 Markdown fenced code block
"""
