"""Basic web UI for browsing and continuing novel projects."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import urllib.parse
from datetime import datetime, timezone
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app import run_next_chapters
from project_manager import init_project, load_json, load_project


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
API_KEYS_PATH = BASE_DIR / "api_keys.sh"
PROJECT_DIR_PATTERN = re.compile(r"^novel_project_")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_api_keys() -> dict[str, str]:
    env_keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GROK_API_KEY": os.environ.get("GROK_API_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
    }
    if all(env_keys.values()) or not API_KEYS_PATH.exists():
        return env_keys

    content = API_KEYS_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r'export\s+([A-Z0-9_]+)=("([^"]*)"|\'([^\']*)\')')
    for match in pattern.finditer(content):
        key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(4) or ""
        if key in env_keys and not env_keys[key]:
            env_keys[key] = value
    return env_keys


def _api_key_for_provider(provider: str, api_keys: dict[str, str]) -> str:
    mapping = {
        "gemini": api_keys.get("GEMINI_API_KEY", ""),
        "grok": api_keys.get("GROK_API_KEY", ""),
        "deepseek": api_keys.get("DEEPSEEK_API_KEY", ""),
    }
    return mapping.get(provider, "")


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "gemini": "gemini-3.1-flash-lite-preview",
        "grok": "grok-4.20-beta-latest-non-reasoning",
        "deepseek": "deepseek-chat",
    }
    return defaults.get(provider, "")


def _default_thinking_level(provider: str) -> str:
    return "medium" if provider == "gemini" else ""


def _list_projects() -> list[dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    projects = []
    for path in sorted(OUTPUT_DIR.iterdir(), reverse=True):
        if not path.is_dir() or not PROJECT_DIR_PATTERN.match(path.name):
            continue
        project_file = path / "project.json"
        if not project_file.exists():
            continue
        try:
            project = load_json(str(project_file))
        except Exception:
            continue
        projects.append(
            {
                "dir_name": path.name,
                "path": path,
                "project_id": project.get("project_id", path.name),
                "name": project.get("name", path.name),
                "description": project.get("description", ""),
                "chapter_count": project.get("chapter_count", 0),
                "updated_at": project.get("updated_at", ""),
                "provider": (project.get("llm_config") or {}).get("model_provider", ""),
            }
        )
    return projects


def _find_project(project_id: str) -> Path | None:
    for project in _list_projects():
        if project["project_id"] == project_id or project["dir_name"] == project_id:
            return project["path"]
    return None


def _build_runtime_config(project_path: Path, overrides: dict[str, str], api_keys: dict[str, str]) -> dict:
    project = load_json(str(project_path / "project.json"))
    saved = project.get("llm_config", {})
    provider = (overrides.get("provider") or saved.get("model_provider") or "gemini").strip().lower()
    if provider not in {"gemini", "grok", "deepseek", "openai_compatible"}:
        provider = "gemini"

    runtime = {
        "model_provider": provider,
        "model_name": (overrides.get("model_name") or saved.get("model_name") or saved.get("model") or "").strip(),
        "model": (overrides.get("model_name") or saved.get("model") or saved.get("model_name") or "").strip(),
        "api_base": (overrides.get("api_base") or saved.get("api_base") or "").strip(),
        "api_key": _api_key_for_provider(provider, api_keys) or overrides.get("api_key", ""),
        "temperature": float(overrides.get("temperature") or saved.get("temperature", 0.8)),
        "max_tokens": int(overrides.get("max_tokens") or saved.get("max_tokens", 4000)),
        "timeout": int(overrides.get("timeout") or saved.get("timeout", 120)),
    }

    thinking_level = (overrides.get("thinking_level") or saved.get("thinking_level") or "").strip()
    if thinking_level:
        runtime["thinking_level"] = thinking_level
    elif provider == "gemini":
        runtime["thinking_level"] = _default_thinking_level(provider)

    if not runtime["model_name"]:
        runtime["model_name"] = _default_model_for_provider(provider)
        runtime["model"] = runtime["model_name"]

    if not runtime["api_key"] and provider in {"gemini", "grok", "deepseek"}:
        raise RuntimeError(f"provider={provider} 缺少 API key，请先填写 api_keys.sh")
    return runtime


def _create_project(form: dict[str, str], api_keys: dict[str, str]) -> str:
    provider = (form.get("provider") or "gemini").strip().lower()
    if provider not in {"gemini", "grok", "deepseek"}:
        raise RuntimeError(f"不支持的 provider: {provider}")

    api_key = _api_key_for_provider(provider, api_keys)
    if not api_key:
        raise RuntimeError(f"provider={provider} 缺少 API key，请先填写 api_keys.sh")

    config = {
        "project_name": (form.get("project_name") or "Novel Project").strip(),
        "project_description": (form.get("project_description") or "").strip(),
        "project_path": str(OUTPUT_DIR / "novel_project_{project_id}"),
        "init_with_llm": True,
        "story_request": (form.get("story_request") or "").strip(),
        "model_provider": provider,
        "model_name": (form.get("model_name") or _default_model_for_provider(provider)).strip(),
        "api_base": (form.get("api_base") or "").strip(),
        "api_key": api_key,
        "temperature": float(form.get("temperature") or 0.9),
        "max_tokens": int(form.get("max_tokens") or 4000),
        "timeout": int(form.get("timeout") or 120),
    }
    thinking_level = (form.get("thinking_level") or _default_thinking_level(provider)).strip()
    if thinking_level:
        config["thinking_level"] = thinking_level

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(config, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name

    try:
        return init_project(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _read_chapters(project_path: Path) -> list[dict]:
    chapters_dir = project_path / "chapters"
    chapters = []
    for chapter_file in sorted(chapters_dir.glob("chapter_*.md")):
        text = chapter_file.read_text(encoding="utf-8")
        chapters.append(
            {
                "name": chapter_file.name,
                "slug": chapter_file.stem,
                "text": text,
            }
        )
    return chapters


def _render_page(title: str, body: str, notice: str = "", error: str = "") -> str:
    flash = ""
    if notice:
        flash += f'<div class="flash notice">{escape(notice)}</div>'
    if error:
        flash += f'<div class="flash error">{escape(error)}</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: rgba(255, 251, 244, 0.9);
      --ink: #1d1a16;
      --muted: #6f6254;
      --accent: #b44f2f;
      --accent-dark: #7f331c;
      --line: #d9cdbf;
      --shadow: 0 18px 45px rgba(75, 46, 24, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Noto Serif SC", "Songti SC", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(234, 191, 138, 0.28), transparent 28%),
        linear-gradient(180deg, #f2ece1 0%, #f7f4ee 40%, #efe6d7 100%);
      min-height: 100vh;
    }}
    a {{ color: var(--accent-dark); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .shell {{
      width: min(1160px, calc(100% - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }}
    .topbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      margin-bottom: 22px;
    }}
    .brand {{
      font-size: clamp(28px, 4vw, 42px);
      letter-spacing: 0.04em;
      margin: 0;
    }}
    .sub {{
      color: var(--muted);
      margin: 4px 0 0;
      font-size: 15px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 20px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid rgba(124, 91, 62, 0.15);
      box-shadow: var(--shadow);
      border-radius: 22px;
      padding: 20px;
      backdrop-filter: blur(10px);
    }}
    h2, h3 {{ margin-top: 0; }}
    .flash {{
      margin: 0 0 18px;
      padding: 14px 16px;
      border-radius: 14px;
      border: 1px solid transparent;
    }}
    .notice {{
      background: rgba(120, 164, 112, 0.14);
      border-color: rgba(120, 164, 112, 0.28);
    }}
    .error {{
      background: rgba(185, 72, 72, 0.12);
      border-color: rgba(185, 72, 72, 0.28);
    }}
    .project-card {{
      padding: 14px 0;
      border-top: 1px solid var(--line);
    }}
    .project-card:first-of-type {{ border-top: 0; padding-top: 0; }}
    .meta {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 6px;
    }}
    .pill {{
      display: inline-block;
      border: 1px solid rgba(180, 79, 47, 0.28);
      border-radius: 999px;
      padding: 3px 10px;
      font-size: 12px;
      color: var(--accent-dark);
      margin-right: 6px;
      margin-bottom: 6px;
    }}
    form {{
      display: grid;
      gap: 12px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 14px;
      color: var(--muted);
    }}
    input, textarea, select, button {{
      font: inherit;
    }}
    input, textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 11px 12px;
      background: rgba(255,255,255,0.88);
      color: var(--ink);
    }}
    textarea {{
      min-height: 110px;
      resize: vertical;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: linear-gradient(135deg, var(--accent) 0%, #cf7c52 100%);
      color: #fff9f3;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{
      filter: brightness(0.97);
    }}
    .chapter-list a {{
      display: block;
      padding: 10px 12px;
      border-radius: 12px;
      margin-bottom: 8px;
      background: rgba(255,255,255,0.5);
    }}
    .chapter-view {{
      white-space: pre-wrap;
      line-height: 1.9;
      font-size: 17px;
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .hero {{
      margin-bottom: 18px;
      padding: 18px 20px;
      border-radius: 20px;
      background: linear-gradient(135deg, rgba(255, 242, 224, 0.92), rgba(250, 235, 215, 0.78));
      border: 1px solid rgba(180, 79, 47, 0.16);
    }}
    .hero h2 {{
      margin-bottom: 8px;
      font-size: clamp(24px, 3vw, 34px);
    }}
    .stack > * + * {{ margin-top: 18px; }}
    @media (max-width: 920px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .two-col {{ grid-template-columns: 1fr; }}
      .shell {{ width: min(100% - 20px, 1160px); }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <div class="topbar">
      <div>
        <h1 class="brand">Novel Writer Web UI</h1>
        <p class="sub">浏览项目、在线阅读章节、直接续写。</p>
      </div>
      <div><a href="/projects">项目列表</a></div>
    </div>
    {flash}
    {body}
  </div>
</body>
</html>
"""


class NovelWriterHandler(BaseHTTPRequestHandler):
    server_version = "NovelWriterWebUI/0.1"

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        notice = params.get("notice", [""])[0]
        error = params.get("error", [""])[0]

        if parsed.path in {"/", "/projects"}:
            self._handle_projects(notice=notice, error=error)
            return

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 2 and parts[0] == "project":
            self._handle_project(parts[1], notice=notice, error=error)
            return
        if len(parts) == 4 and parts[0] == "project" and parts[2] == "chapter":
            self._handle_chapter(parts[1], parts[3], notice=notice, error=error)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "页面不存在")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        form = self._read_form()

        if parsed.path == "/projects/create":
            self._handle_create_project(form)
            return

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "continue":
            self._handle_continue(parts[1], form)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "页面不存在")

    def _read_form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length).decode("utf-8")
        parsed = urllib.parse.parse_qs(raw, keep_blank_values=True)
        return {key: values[0] for key, values in parsed.items()}

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _write_html(self, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_projects(self, notice: str = "", error: str = "") -> None:
        projects = _list_projects()
        cards = []
        for item in projects:
            cards.append(
                f"""
                <div class="project-card">
                  <div><a href="/project/{escape(item['project_id'])}"><strong>{escape(item['name'])}</strong></a></div>
                  <div class="meta">{escape(item['description'] or '暂无简介')}</div>
                  <div class="meta">
                    <span class="pill">{escape(item['provider'] or 'unknown')}</span>
                    <span class="pill">{item['chapter_count']} 章</span>
                    <span class="pill">{escape(item['updated_at'] or '')}</span>
                  </div>
                </div>
                """
            )
        project_html = "".join(cards) or "<p>当前还没有项目，先在左侧创建一个新项目吧。</p>"

        body = f"""
        <div class="grid">
          <section class="panel">
            <h2>新建项目</h2>
            <form method="post" action="/projects/create">
              <div class="two-col">
                <label>模型后端
                  <select name="provider">
                    <option value="gemini">gemini</option>
                    <option value="grok">grok</option>
                    <option value="deepseek">deepseek</option>
                  </select>
                </label>
                <label>模型名（可选）
                  <input type="text" name="model_name" placeholder="留空则使用默认模型">
                </label>
              </div>
              <label>项目名
                <input type="text" name="project_name" value="雪封穹顶">
              </label>
              <label>项目简介
                <input type="text" name="project_description" value="由模型根据需求自动生成设定的长篇小说项目。">
              </label>
              <label>故事需求
                <textarea name="story_request" placeholder="把你想写的题材、角色、世界观、节奏偏好写在这里"></textarea>
              </label>
              <div class="two-col">
                <label>Temperature
                  <input type="number" step="0.1" name="temperature" value="0.9">
                </label>
                <label>Max Tokens
                  <input type="number" name="max_tokens" value="4000">
                </label>
              </div>
              <div class="two-col">
                <label>Timeout
                  <input type="number" name="timeout" value="120">
                </label>
                <label>Thinking Level
                  <input type="text" name="thinking_level" placeholder="Gemini 可填 medium/high">
                </label>
              </div>
              <label>API Base（可选）
                <input type="text" name="api_base" placeholder="如需自定义接口地址可填写">
              </label>
              <button type="submit">创建项目</button>
            </form>
          </section>
          <section class="panel">
            <div class="hero">
              <h2>项目书架</h2>
              <p class="sub">这里会列出 `output/` 目录中的全部小说项目。点击即可阅读和续写。</p>
            </div>
            {project_html}
          </section>
        </div>
        """
        self._write_html(_render_page("项目列表", body, notice=notice, error=error))

    def _handle_project(self, project_id: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        data = load_project(str(project_path))
        project = data["project"]
        plot_state = data["plot_state"]
        chapters = _read_chapters(project_path)
        stats = (project.get("stats") or {}).get("total", {})

        chapter_links = "".join(
            f'<a href="/project/{escape(project_id)}/chapter/{escape(chapter["slug"])}">{escape(chapter["name"])}</a>'
            for chapter in chapters
        ) or "<p>还没有章节。</p>"

        body = f"""
        <div class="grid">
          <aside class="stack">
            <section class="panel">
              <h2>{escape(project.get("name", project_id))}</h2>
              <p class="meta">{escape(project.get("description", ""))}</p>
              <p class="meta"><span class="pill">{escape((project.get("llm_config") or {}).get("model_provider", ""))}</span><span class="pill">{project.get("chapter_count", 0)} 章</span></p>
              <p><strong>下章目标：</strong>{escape(plot_state.get("next_chapter_goal", "") or "暂无")}</p>
              <p><strong>当前地点：</strong>{escape(plot_state.get("current_location", "") or "未知")}</p>
              <p><strong>当前时间：</strong>{escape(plot_state.get("current_time", "") or "未知")}</p>
              <p><strong>请求：</strong>{stats.get("requests", 0)} 次</p>
              <p><strong>Token：</strong>{stats.get("total_tokens", 0)}</p>
            </section>
            <section class="panel">
              <h3>续写</h3>
              <form method="post" action="/project/{escape(project_id)}/continue">
                <div class="two-col">
                  <label>续写章节数
                    <input type="number" name="count" value="1" min="1" max="20">
                  </label>
                  <label>临时后端覆盖
                    <select name="provider">
                      <option value="">沿用项目设置</option>
                      <option value="gemini">gemini</option>
                      <option value="grok">grok</option>
                      <option value="deepseek">deepseek</option>
                    </select>
                  </label>
                </div>
                <label>想看的内容 / 情节走向
                  <textarea name="user_request" placeholder="例如：先推进食堂据点建设，再增加一点轻松互怼的互动。"></textarea>
                </label>
                <div class="two-col">
                  <label>模型名（可选）
                    <input type="text" name="model_name" placeholder="留空则沿用项目设置">
                  </label>
                  <label>Thinking Level（可选）
                    <input type="text" name="thinking_level" placeholder="如 medium / high">
                  </label>
                </div>
                <div class="two-col">
                  <label>Temperature
                    <input type="number" step="0.1" name="temperature" placeholder="沿用项目设置">
                  </label>
                  <label>Max Tokens
                    <input type="number" name="max_tokens" placeholder="沿用项目设置">
                  </label>
                </div>
                <div class="two-col">
                  <label>Timeout
                    <input type="number" name="timeout" placeholder="沿用项目设置">
                  </label>
                  <label>API Base（可选）
                    <input type="text" name="api_base" placeholder="留空则沿用项目设置">
                  </label>
                </div>
                <button type="submit">开始续写</button>
              </form>
            </section>
            <section class="panel">
              <h3>章节目录</h3>
              <div class="chapter-list">{chapter_links}</div>
            </section>
          </aside>
          <main class="stack">
            <section class="panel">
              <h2>剧情状态</h2>
              <div class="chapter-view">{escape(json.dumps(plot_state, ensure_ascii=False, indent=2))}</div>
            </section>
            <section class="panel">
              <h2>最近一章</h2>
              <div class="chapter-view">{escape(chapters[-1]["text"]) if chapters else "还没有正文。"}</div>
            </section>
          </main>
        </div>
        """
        self._write_html(_render_page(project.get("name", project_id), body, notice=notice, error=error))

    def _handle_chapter(self, project_id: str, chapter_slug: str, notice: str = "", error: str = "") -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        chapter_file = project_path / "chapters" / f"{chapter_slug}.md"
        if not chapter_file.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "章节不存在")
            return

        project = load_json(str(project_path / "project.json"))
        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}">返回项目</a>
            <h2>{escape(chapter_file.name)}</h2>
            <div class="chapter-view">{escape(chapter_file.read_text(encoding="utf-8"))}</div>
          </section>
        </div>
        """
        self._write_html(_render_page(f"{project.get('name', project_id)} - {chapter_file.name}", body, notice=notice, error=error))

    def _handle_create_project(self, form: dict[str, str]) -> None:
        api_keys = _load_api_keys()
        try:
            if not (form.get("story_request") or "").strip():
                raise RuntimeError("故事需求不能为空。")
            project_path = _create_project(form, api_keys)
            project_id = load_json(str(Path(project_path) / "project.json")).get("project_id", Path(project_path).name)
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote("项目创建成功。")
            )
        except Exception as exc:
            self._redirect("/projects?error=" + urllib.parse.quote(str(exc)))

    def _handle_continue(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            count = int(form.get("count") or "1")
            if count < 1:
                raise RuntimeError("续写章节数必须至少为 1。")
            runtime_config = _build_runtime_config(project_path, form, api_keys)
            run_next_chapters(
                str(project_path),
                runtime_config,
                count,
                user_request=(form.get("user_request") or "").strip(),
            )
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote(f"续写完成，共生成 {count} 章。")
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic web UI for Novel Writer")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host, use 0.0.0.0 for remote access")
    parser.add_argument("--port", type=int, default=8008, help="Bind port")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), NovelWriterHandler)
    print(f"[{_utc_now()}] Web UI listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n[{_utc_now()}] Web UI stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
