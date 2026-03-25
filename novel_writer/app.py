"""CLI entry point for the structured-memory novel writer MVP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from illustration_manager import illustrate_chapters, illustrate_project_assets
from llm_client import generate_text_with_metadata
from outline_manager import (
    ensure_project_outlines,
    find_next_chapter_context,
    get_outline_status,
    regenerate_chapter_outline,
    regenerate_volume_outline,
    sync_outline_progress,
)
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


def _add_illustration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--comfyui-api-base",
        help="ComfyUI API base URL, default http://127.0.0.1:8188",
    )
    parser.add_argument(
        "--comfyui-root",
        help="Optional ComfyUI root directory used for auto-detecting checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint name for CheckpointLoaderSimple, e.g. illusious/illustrij_v21.safetensors",
    )
    parser.add_argument(
        "--workflow-template",
        help="Optional ComfyUI workflow template JSON path",
    )
    parser.add_argument("--width", type=int, help="Illustration width")
    parser.add_argument("--height", type=int, help="Illustration height")
    parser.add_argument("--steps", type=int, help="Sampling steps")
    parser.add_argument("--cfg", type=float, help="CFG scale")
    parser.add_argument("--sampler-name", help="Sampler name, e.g. euler")
    parser.add_argument("--scheduler", help="Scheduler name, e.g. normal")
    parser.add_argument("--seed", type=int, help="Optional fixed illustration seed")


def _extract_illustration_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "comfyui_api_base": getattr(args, "comfyui_api_base", None),
        "comfyui_root": getattr(args, "comfyui_root", None),
        "checkpoint": getattr(args, "checkpoint", None),
        "workflow_template": getattr(args, "workflow_template", None),
        "width": getattr(args, "width", None),
        "height": getattr(args, "height", None),
        "steps": getattr(args, "steps", None),
        "cfg": getattr(args, "cfg", None),
        "sampler_name": getattr(args, "sampler_name", None),
        "scheduler": getattr(args, "scheduler", None),
        "seed": getattr(args, "seed", None),
    }
    return {key: value for key, value in mapping.items() if value not in (None, "")}


def run_next_chapter(project_path: str, config: dict, user_request: str = ""):
    outlines = ensure_project_outlines(project_path, config)
    project_data = load_project(project_path)
    next_context = find_next_chapter_context(outlines, int(project_data["project"].get("chapter_count", 0) or 0))
    if next_context is None:
        raise ValueError("当前没有可用的下一章分章大纲，请先重新生成分章大纲。")
    last_chapter = get_last_chapter_text(project_path)
    recent_text = last_chapter[-3000:] if last_chapter else "这是小说的开篇章节，请自然展开故事。"

    prompt = build_writer_prompt(
        project_data,
        recent_text,
        user_request=user_request,
        current_volume_outline=next_context["volume"],
        chapter_outline=next_context["chapter"],
    )
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
    sync_outline_progress(project_path)
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
    outline_status = get_outline_status(project_path)
    if outline_status.get("has_outlines"):
        print(f"分卷数量: {outline_status.get('volume_count', 0)}")
        if outline_status.get("chapter_outline_stale"):
            print("分章大纲状态: 已过期，请先重新生成")
        next_context = outline_status.get("next_context")
        if next_context:
            volume = next_context.get("volume") or {}
            chapter = next_context.get("chapter") or {}
            print(
                "下一章对应大纲: "
                f"第{volume.get('volume_number', '?')}卷《{volume.get('title', '')}》"
                f" / 第{chapter.get('chapter_number', '?')}章《{chapter.get('title', '')}》"
            )
            print(f"下一章章纲摘要: {chapter.get('summary', '')}")
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
    next_parser.add_argument(
        "--illustrate",
        action="store_true",
        help="Generate ComfyUI illustrations for the newly written chapters",
    )
    next_parser.add_argument(
        "--illustration-request",
        default="",
        help="Optional extra art direction for ComfyUI illustrations",
    )
    next_parser.add_argument(
        "--force-illustration",
        action="store_true",
        help="Regenerate illustrations even if the chapter already has images",
    )
    _add_illustration_arguments(next_parser)

    illustrate_parser = subparsers.add_parser("illustrate", help="Generate ComfyUI illustrations for chapters")
    illustrate_parser.add_argument("--project", required=True, help="Path to novel_project")
    illustrate_parser.add_argument(
        "--chapter",
        default="latest",
        help="Chapter slug/path to illustrate, default latest. Ignored when --all is set",
    )
    illustrate_parser.add_argument(
        "--all",
        action="store_true",
        help="Generate illustrations for all chapters in the project",
    )
    illustrate_parser.add_argument(
        "--config",
        help="Optional config.json used to provide LLM credentials for prompt refinement",
    )
    illustrate_parser.add_argument(
        "--user-request",
        default="",
        help="Optional extra art direction for the illustration prompt",
    )
    illustrate_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate illustrations even if metadata already exists",
    )
    _add_illustration_arguments(illustrate_parser)

    assets_parser = subparsers.add_parser("illustrate-assets", help="Generate ComfyUI cover and character portraits")
    assets_parser.add_argument("--project", required=True, help="Path to novel_project")
    assets_parser.add_argument(
        "--user-request",
        default="",
        help="Optional extra art direction for the cover and character portraits",
    )
    assets_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cover and character portraits even if metadata already exists",
    )
    _add_illustration_arguments(assets_parser)

    status_parser = subparsers.add_parser("status", help="Show project status")
    status_parser.add_argument("--project", required=True, help="Path to novel_project")

    outline_parser = subparsers.add_parser("outline", help="Generate or regenerate volume/chapter outlines")
    outline_parser.add_argument("--project", required=True, help="Path to novel_project")
    outline_parser.add_argument(
        "--config",
        help="Optional config.json to override the project's saved LLM settings",
    )
    outline_parser.add_argument(
        "--stage",
        choices=("volumes", "chapters", "all"),
        default="all",
        help="Which outline stage to regenerate",
    )
    outline_parser.add_argument(
        "--volume",
        type=int,
        help="Optional volume number for chapter outline regeneration",
    )
    outline_parser.add_argument(
        "--user-request",
        default="",
        help="Optional extra plot requirements for outline regeneration",
    )

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
        if args.illustrate:
            illustration_results = illustrate_chapters(
                args.project,
                chapter_refs=chapter_paths,
                llm_config=config,
                user_request=args.illustration_request,
                force=args.force_illustration,
                overrides=_extract_illustration_overrides(args),
            )
            for result in illustration_results:
                state = "复用现有插图" if result.get("reused") else "已生成插图"
                print(f"{state}: {result.get('chapter_slug', '')}")
                for image in result.get("images", []):
                    print(f"- {image.get('relative_path', '')}")
    elif args.command == "illustrate":
        config = _extract_llm_config(args.config) if args.config else None
        chapter_refs = [args.chapter]
        if args.all:
            chapters_dir = Path(args.project) / "chapters"
            chapter_refs = [str(path) for path in sorted(chapters_dir.glob("chapter_*.md"))]
        results = illustrate_chapters(
            args.project,
            chapter_refs=chapter_refs,
            llm_config=config,
            user_request=args.user_request,
            force=args.force,
            overrides=_extract_illustration_overrides(args),
        )
        print(f"本次处理插图章节数: {len(results)}")
        for result in results:
            state = "复用现有插图" if result.get("reused") else "已生成插图"
            print(f"{state}: {result.get('chapter_slug', '')}")
            for image in result.get("images", []):
                print(f"- {image.get('relative_path', '')}")
    elif args.command == "illustrate-assets":
        result = illustrate_project_assets(
            args.project,
            user_request=args.user_request,
            force=args.force,
            overrides=_extract_illustration_overrides(args),
        )
        cover = result.get("cover") or {}
        cover_state = "复用现有封面" if cover.get("reused") else "已生成封面"
        print(f"{cover_state}: {cover.get('asset_slug', 'cover')}")
        for image in cover.get("images", []):
            print(f"- {image.get('relative_path', '')}")

        portraits = result.get("portraits") or []
        print(f"本次处理人物立绘数: {len(portraits)}")
        for portrait in portraits:
            portrait_state = "复用现有人物立绘" if portrait.get("reused") else "已生成人物立绘"
            print(f"{portrait_state}: {portrait.get('character_name', portrait.get('asset_slug', ''))}")
            for image in portrait.get("images", []):
                print(f"- {image.get('relative_path', '')}")
    elif args.command == "status":
        _print_status(args.project)
    elif args.command == "outline":
        config = (
            _extract_llm_config(args.config)
            if args.config
            else _load_runtime_config(args.project)
        )
        if args.stage in {"volumes", "all"}:
            outlines = regenerate_volume_outline(
                args.project,
                config,
                user_request=args.user_request,
            )
            print(f"已生成分卷大纲，卷数: {len(outlines.get('volumes', []))}")
        if args.stage in {"chapters", "all"}:
            outlines = regenerate_chapter_outline(
                args.project,
                config,
                volume_number=args.volume if args.stage == "chapters" else None,
                user_request=args.user_request,
            )
            target = f"第 {args.volume} 卷" if args.volume else "全部卷"
            print(f"已生成分章大纲，范围: {target}")
            next_context = find_next_chapter_context(
                outlines,
                int(load_project(args.project)["project"].get("chapter_count", 0) or 0),
            )
            if next_context:
                volume = next_context["volume"]
                chapter = next_context["chapter"]
                print(
                    f"下一章已对齐到第{volume.get('volume_number', '?')}卷《{volume.get('title', '')}》"
                    f" / 第{chapter.get('chapter_number', '?')}章《{chapter.get('title', '')}》"
                )


if __name__ == "__main__":
    main()
