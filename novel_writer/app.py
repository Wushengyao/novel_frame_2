"""CLI entry point for the structured-memory novel writer MVP."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_client import generate_text_with_metadata
from prompt_builder import build_writer_prompt
from project_manager import (
    get_last_chapter_text,
    init_project,
    load_json,
    load_project,
    normalize_chapter_text,
    save_chapter,
    update_project_stats,
)
from state_updater import update_plot_state


def run_next_chapter(project_path: str, config: dict, user_request: str = ""):
    project_data = load_project(project_path)
    last_chapter = get_last_chapter_text(project_path)
    recent_text = last_chapter[-1000:] if last_chapter else "这是小说的开篇章节，请自然展开故事。"

    prompt = build_writer_prompt(project_data, recent_text, user_request=user_request)
    try:
        response_text, metadata = generate_text_with_metadata(prompt, config)
    except Exception:
        update_project_stats(project_path, phase="writer", success=False, usage=None)
        raise

    update_project_stats(
        project_path,
        phase="writer",
        success=True,
        usage=metadata.get("usage"),
    )
    chapter_text = normalize_chapter_text(response_text)
    chapter_path = save_chapter(project_path, chapter_text)
    update_plot_state(project_path, chapter_text, config)
    return chapter_path


def run_next_chapters(project_path: str, config: dict, count: int, user_request: str = "") -> list[str]:
    if count < 1:
        raise ValueError("count must be at least 1.")

    chapter_paths = []
    for _ in range(count):
        chapter_paths.append(run_next_chapter(project_path, config, user_request=user_request))
    return chapter_paths


def _print_status(project_path: str) -> None:
    project_data = load_project(project_path)
    project = project_data["project"]
    plot_state = project_data["plot_state"]
    llm_config = project.get("llm_config", {})
    stats = project.get("stats") or {}
    total_stats = stats.get("total") or {}
    last_goal = plot_state.get("next_chapter_goal", "")
    print(f"项目ID: {project.get('project_id', 'legacy_project')}")
    print(f"项目名称: {project.get('name', '')}")
    print(f"章节数量: {project.get('chapter_count', 0)}")
    print(f"更新时间: {project.get('updated_at', '')}")
    print(f"模型后端: {llm_config.get('model_provider', '')}")
    print(f"模型名称: {llm_config.get('model_name') or llm_config.get('model', '')}")
    print(
        "请求统计: "
        f"{total_stats.get('requests', 0)} 次"
        f"（成功 {total_stats.get('successes', 0)} / 失败 {total_stats.get('failures', 0)}）"
    )
    print(
        "Token统计: "
        f"prompt={total_stats.get('prompt_tokens', 0)}, "
        f"completion={total_stats.get('completion_tokens', 0)}, "
        f"total={total_stats.get('total_tokens', 0)}"
    )
    extra_usage = []
    if total_stats.get("cached_tokens", 0):
        extra_usage.append(f"cached={total_stats.get('cached_tokens', 0)}")
    if total_stats.get("reasoning_tokens", 0):
        extra_usage.append(f"reasoning={total_stats.get('reasoning_tokens', 0)}")
    if total_stats.get("thought_tokens", 0):
        extra_usage.append(f"thoughts={total_stats.get('thought_tokens', 0)}")
    if extra_usage:
        print("附加Token统计: " + ", ".join(extra_usage))
    print(f"当前地点: {plot_state.get('current_location', '')}")
    print(f"当前时间: {plot_state.get('current_time', '')}")
    print(f"下章目标: {last_goal}")
    open_threads = plot_state.get("open_threads", [])
    if open_threads:
        print("未解线索:")
        for item in open_threads:
            print(f"- {item}")


def _load_runtime_config(project_path: str) -> dict:
    project = load_project(project_path)["project"]
    return project.get("llm_config", {})


def _extract_llm_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    raw = load_json(str(config_file))
    return {
        "model_provider": raw.get("model_provider", "openai_compatible"),
        "model": raw.get("model") or raw.get("model_name", ""),
        "model_name": raw.get("model_name") or raw.get("model", ""),
        "api_base": raw.get("api_base", ""),
        "api_key": raw.get("api_key", ""),
        "temperature": raw.get("temperature", 0.8),
        "max_tokens": raw.get("max_tokens", 4000),
        "timeout": raw.get("timeout", 120),
        "thinking_level": raw.get("thinking_level"),
        "thinking_budget": raw.get("thinking_budget"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel writer MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize a novel project")
    init_parser.add_argument("--config", required=True, help="Path to config.json")

    next_parser = subparsers.add_parser("next", help="Generate the next chapter")
    next_parser.add_argument("--project", required=True, help="Path to novel_project")
    next_parser.add_argument(
        "--config",
        help="Optional config.json to override the project's saved LLM settings",
    )
    next_parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Generate multiple chapters sequentially in one run",
    )
    next_parser.add_argument(
        "--user-request",
        default="",
        help="Optional user preference for this batch of chapters, such as desired scenes or plot direction",
    )

    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project", required=True, help="Path to novel_project")

    args = parser.parse_args()

    if args.command == "init":
        project_path = init_project(args.config)
        print(f"项目已初始化: {project_path}")
    elif args.command == "next":
        if args.count < 1:
            parser.error("--count must be at least 1")
        config = (
            _extract_llm_config(args.config)
            if args.config
            else _load_runtime_config(args.project)
        )
        chapter_paths = run_next_chapters(
            args.project,
            config,
            args.count,
            user_request=args.user_request,
        )
        print(f"本次共生成章节数: {len(chapter_paths)}")
        for chapter_path in chapter_paths:
            print(f"新章节已保存: {chapter_path}")
    elif args.command == "status":
        _print_status(args.project)


if __name__ == "__main__":
    main()
