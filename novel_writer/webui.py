"""Basic web UI for browsing and continuing novel projects."""

from __future__ import annotations

import argparse
import json
import mimetypes
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
from illustration_manager import get_illustration_record, illustrate_chapters, list_illustration_records
from project_manager import init_project, load_json, load_project


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
API_KEYS_PATH = BASE_DIR / "api_keys.sh"
PROJECT_DIR_PATTERN = re.compile(r"^novel_project_")
MOJIBAKE_HINT_CHARS = set("闆皝绌归《鍙鍦鏄鐨勪簡鍚庡墠闂閿璇浠绗锛銆鈥€")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _looks_like_mojibake(text: str) -> bool:
  if not text:
    return False
  hint_count = sum(1 for ch in text if ch in MOJIBAKE_HINT_CHARS)
  return hint_count >= max(1, len(text) // 4)


def _mojibake_score(text: str) -> int:
  return sum(1 for ch in text if ch in MOJIBAKE_HINT_CHARS) + text.count("�") * 2


def _repair_display_text(text: str) -> str:
  if not isinstance(text, str) or not _looks_like_mojibake(text):
    return text
  best = text
  best_score = _mojibake_score(text)
  for encoding in ("gb18030", "gbk", "cp1252", "latin1"):
    try:
      repaired = text.encode(encoding).decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
      continue
    if not repaired:
      continue
    repaired_score = _mojibake_score(repaired)
    if repaired_score < best_score:
      best = repaired
      best_score = repaired_score
  return best


def _load_api_keys() -> dict[str, str]:
    env_keys = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GROK_API_KEY": os.environ.get("GROK_API_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        "DOUBAO_API_KEY": os.environ.get("DOUBAO_API_KEY", ""),
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
        "doubao": api_keys.get("DOUBAO_API_KEY", ""),
        "ollama": os.environ.get("OLLAMA_API_KEY", ""),
    }
    return mapping.get(provider, "")


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "gemini": "gemini-3.1-flash-lite-preview",
        "grok": "grok-4.20-beta-latest-non-reasoning",
        "deepseek": "deepseek-chat",
        "doubao": "doubao-seed-1-8-251228",
        "ollama": "llama3.2",
    }
    return defaults.get(provider, "")


def _default_api_base_for_provider(provider: str) -> str:
    defaults = {
        "doubao": "https://ark.cn-beijing.volces.com/api/v3",
        "ollama": "http://127.0.0.1:11434/v1",
    }
    return defaults.get(provider, "")


def _default_thinking_level(provider: str) -> str:
    return "medium" if provider == "gemini" else ""


def _provider_requires_api_key(provider: str) -> bool:
    return provider in {"gemini", "grok", "deepseek", "doubao"}


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
            "name": _repair_display_text(project.get("name", path.name)),
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
    saved_provider = (saved.get("model_provider") or "gemini").strip().lower()
    provider = (overrides.get("provider") or saved_provider or "gemini").strip().lower()
    if provider not in {"gemini", "grok", "deepseek", "doubao", "openai_compatible", "ollama"}:
        provider = "gemini"

    override_model_name = (overrides.get("model_name") or "").strip()
    saved_model_name = (saved.get("model_name") or saved.get("model") or "").strip()
    override_api_base = (overrides.get("api_base") or "").strip()
    saved_api_base = (saved.get("api_base") or "").strip()

    runtime = {
        "model_provider": provider,
        "model_name": override_model_name
        or (_default_model_for_provider(provider) if provider != saved_provider else saved_model_name)
        or _default_model_for_provider(provider),
        "model": override_model_name
        or (_default_model_for_provider(provider) if provider != saved_provider else saved_model_name)
        or _default_model_for_provider(provider),
        "api_base": override_api_base
        or (
            _default_api_base_for_provider(provider)
            if provider != saved_provider
            else (saved_api_base or _default_api_base_for_provider(provider))
        ),
        "api_key": _api_key_for_provider(provider, api_keys) or overrides.get("api_key", ""),
        "temperature": float(overrides.get("temperature") or saved.get("temperature", 0.8)),
        "max_tokens": int(overrides.get("max_tokens") or saved.get("max_tokens", 4000)),
        "timeout": int(overrides.get("timeout") or saved.get("timeout", 120)),
    }

    thinking_level = (overrides.get("thinking_level") or "").strip()
    if not thinking_level and provider == saved_provider:
        thinking_level = (saved.get("thinking_level") or "").strip()
    if thinking_level:
        runtime["thinking_level"] = thinking_level
    elif provider == "gemini":
        runtime["thinking_level"] = _default_thinking_level(provider)

    if not runtime["model_name"]:
        runtime["model_name"] = _default_model_for_provider(provider)
        runtime["model"] = runtime["model_name"]

    if not runtime["api_key"] and _provider_requires_api_key(provider):
        raise RuntimeError(f"provider={provider} 缺少 API key，请先填写 api_keys.sh")
    return runtime


def _create_project(form: dict[str, str], api_keys: dict[str, str]) -> str:
    provider = (form.get("provider") or "gemini").strip().lower()
    if provider not in {"gemini", "grok", "deepseek", "doubao", "ollama"}:
        raise RuntimeError(f"不支持的 provider: {provider}")

    api_key = _api_key_for_provider(provider, api_keys)
    if not api_key and _provider_requires_api_key(provider):
        raise RuntimeError(f"provider={provider} 缺少 API key，请先填写 api_keys.sh")

    config = {
        "project_name": (form.get("project_name") or "Novel Project").strip(),
        "project_description": (form.get("project_description") or "").strip(),
        "project_path": str(OUTPUT_DIR / "novel_project_{project_id}"),
        "init_with_llm": True,
        "story_request": (form.get("story_request") or "").strip(),
        "model_provider": provider,
        "model_name": (form.get("model_name") or _default_model_for_provider(provider)).strip(),
        "api_base": (form.get("api_base") or _default_api_base_for_provider(provider)).strip(),
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


def _illustration_overrides_from_form(form: dict[str, str]) -> dict:
    mapping = {
        "checkpoint": (form.get("checkpoint") or "").strip(),
        "width": (form.get("width") or "").strip(),
        "height": (form.get("height") or "").strip(),
        "steps": (form.get("steps") or "").strip(),
        "cfg": (form.get("cfg") or "").strip(),
        "comfyui_api_base": (form.get("comfyui_api_base") or "").strip(),
    }
    return {key: value for key, value in mapping.items() if value}


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
    .chapter-nav {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 20px;
      padding-top: 18px;
      border-top: 1px solid var(--line);
    }}
    .chapter-nav-link, .chapter-nav-disabled {{
      display: inline-flex;
      align-items: center;
      min-width: 160px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(124, 91, 62, 0.16);
    }}
    .chapter-nav-link.next, .chapter-nav-disabled.next {{
      margin-left: auto;
      justify-content: flex-end;
      text-align: right;
    }}
    .chapter-nav-disabled {{
      color: var(--muted);
      background: rgba(255,255,255,0.38);
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
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .thumb {{
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 16px;
      border: 1px solid rgba(124, 91, 62, 0.15);
      background: rgba(255,255,255,0.56);
    }}
    .thumb img {{
      display: block;
      width: 100%;
      aspect-ratio: 3 / 4;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid rgba(124, 91, 62, 0.12);
      background: rgba(255,255,255,0.8);
    }}
    .muted {{ color: var(--muted); font-size: 14px; }}
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
        if len(parts) == 5 and parts[0] == "project" and parts[2] == "illustration-file":
            self._handle_illustration_file(parts[1], parts[3], parts[4])
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
        if len(parts) == 3 and parts[0] == "project" and parts[2] == "illustrate":
            self._handle_illustrate(parts[1], form)
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

    def _write_file(self, file_path: Path) -> None:
        data = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
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
                    <option value="doubao">doubao</option>
                    <option value="ollama">ollama</option>
                  </select>
                </label>
                <label>模型名（可选）
                  <input type="text" name="model_name" placeholder="留空则使用对应后端默认模型 / Model ID">
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
        project_name = _repair_display_text(project.get("name", project_id))
        plot_state = data["plot_state"]
        chapters = _read_chapters(project_path)
        stats = (project.get("stats") or {}).get("total", {})
        illustration_records = list_illustration_records(str(project_path))

        chapter_links = "".join(
            f"<a href=\"/project/{escape(project_id)}/chapter/{escape(chapter['slug'])}\">{escape(chapter['name'])}</a>"
            for chapter in chapters
        ) or "<p>还没有章节。</p>"

        chapter_options = ['<option value="latest">最新章节</option>'] + [
            f"<option value=\"{escape(chapter['slug'])}\">{escape(chapter['name'])}</option>"
            for chapter in chapters
        ]

        illustration_cards = []
        for record in illustration_records[:6]:
            chapter_slug = str(record.get("chapter_slug", ""))
            images = record.get("images") or []
            if not chapter_slug or not images:
                continue
            image = images[0]
            image_url = (
                f"/project/{urllib.parse.quote(project_id)}/illustration-file/"
                f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(image.get('file_name', ''))}"
            )
            illustration_cards.append(
                f"""
                <div class="thumb">
                  <a href="/project/{escape(project_id)}/chapter/{escape(chapter_slug)}"><img src="{image_url}" alt="{escape(chapter_slug)}"></a>
                  <div><strong>{escape(chapter_slug)}</strong></div>
                  <div class="muted">{escape(record.get('scene_summary', '') or '已生成插图')}</div>
                </div>
                """
            )
        illustration_gallery = "".join(illustration_cards) or "<p>当前还没有章节插图。</p>"

        body = f"""
        <div class="grid">
          <aside class="stack">
            <section class="panel">
              <h2>{escape(project_name)}</h2>
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
                      <option value="doubao">doubao</option>
                      <option value="ollama">ollama</option>
                    </select>
                  </label>
                </div>
                <label>想看的内容 / 情节走向
                  <textarea name="user_request" placeholder="例如：先推进食堂据点建设，再增加一点轻松互怼的互动。"></textarea>
                </label>
                <div class="two-col">
                  <label>模型名（可选）
                    <input type="text" name="model_name" placeholder="留空则沿用项目设置；切换后端时自动改为该后端默认模型 / Model ID">
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
                <label><input type="checkbox" name="illustrate_generated" value="1"> 续写完成后立即调用 ComfyUI 生成插图</label>
                <label>插图额外要求（可选）
                  <input type="text" name="illustration_request" placeholder="例如：突出雪夜窗景与室内暖光反差。">
                </label>
                <button type="submit">开始续写</button>
              </form>
            </section>
            <section class="panel">
              <h3>生成插图</h3>
              <form method="post" action="/project/{escape(project_id)}/illustrate">
                <label>目标章节
                  <select name="chapter_slug">
                    {''.join(chapter_options)}
                  </select>
                </label>
                <label>插图要求（可选）
                  <textarea name="user_request" placeholder="例如：更强调角色站位、情绪与镜头感。"></textarea>
                </label>
                <div class="two-col">
                  <label>Checkpoint（可选）
                    <input type="text" name="checkpoint" placeholder="illusious/illustrij_v21.safetensors">
                  </label>
                  <label>ComfyUI API（可选）
                    <input type="text" name="comfyui_api_base" placeholder="http://127.0.0.1:8188">
                  </label>
                </div>
                <div class="two-col">
                  <label>宽度
                    <input type="number" name="width" placeholder="832">
                  </label>
                  <label>高度
                    <input type="number" name="height" placeholder="1216">
                  </label>
                </div>
                <div class="two-col">
                  <label>Steps
                    <input type="number" name="steps" placeholder="28">
                  </label>
                  <label>CFG
                    <input type="number" step="0.1" name="cfg" placeholder="6.5">
                  </label>
                </div>
                <label><input type="checkbox" name="force" value="1"> 强制重绘</label>
                <button type="submit">为章节生成插图</button>
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
            <section class="panel">
              <h2>最近插图</h2>
              <div class="gallery">{illustration_gallery}</div>
            </section>
          </main>
        </div>
        """
        self._write_html(_render_page(project_name, body, notice=notice, error=error))

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
        project_name = _repair_display_text(project.get("name", project_id))
        chapters = _read_chapters(project_path)
        current_index = next((idx for idx, chapter in enumerate(chapters) if chapter["slug"] == chapter_slug), -1)
        previous_chapter = chapters[current_index - 1] if current_index > 0 else None
        next_chapter = chapters[current_index + 1] if 0 <= current_index < len(chapters) - 1 else None
        previous_link = (
            f'<a class="chapter-nav-link prev" href="/project/{escape(project_id)}/chapter/{escape(previous_chapter["slug"])}">← 上一章：{escape(previous_chapter["name"])}</a>'
            if previous_chapter
            else '<span class="chapter-nav-disabled prev">← 已是第一章</span>'
        )
        next_link = (
            f'<a class="chapter-nav-link next" href="/project/{escape(project_id)}/chapter/{escape(next_chapter["slug"])}">下一章：{escape(next_chapter["name"])} →</a>'
            if next_chapter
            else '<span class="chapter-nav-disabled next">已是最后一章 →</span>'
        )
        illustration_record = get_illustration_record(str(project_path), chapter_slug)
        illustration_gallery = "<p>当前还没有本章插图。</p>"
        if illustration_record and illustration_record.get("images"):
            cards = []
            for image in illustration_record.get("images", []):
                image_url = (
                    f"/project/{urllib.parse.quote(project_id)}/illustration-file/"
                    f"{urllib.parse.quote(chapter_slug)}/{urllib.parse.quote(image.get('file_name', ''))}"
                )
                cards.append(
                    f"""
                    <div class="thumb">
                      <a href="{image_url}"><img src="{image_url}" alt="{escape(chapter_slug)}"></a>
                      <div class="muted">{escape(illustration_record.get('scene_summary', '') or '章节插图')}</div>
                    </div>
                    """
                )
            illustration_gallery = "".join(cards)

        body = f"""
        <div class="stack">
          <section class="panel">
            <a href="/project/{escape(project_id)}">返回项目</a>
            <h2>{escape(chapter_file.name)}</h2>
            <div class="chapter-view">{escape(chapter_file.read_text(encoding="utf-8"))}</div>
            <div class="chapter-nav">
              {previous_link}
              {next_link}
            </div>
          </section>
          <section class="panel">
            <h2>本章插图</h2>
            <div class="gallery">{illustration_gallery}</div>
            <form method="post" action="/project/{escape(project_id)}/illustrate">
              <input type="hidden" name="chapter_slug" value="{escape(chapter_slug)}">
              <label>插图要求（可选）
                <input type="text" name="user_request" placeholder="例如：强调人物表情与空间纵深。">
              </label>
              <div class="two-col">
                <label>Checkpoint（可选）
                  <input type="text" name="checkpoint" placeholder="illusious/illustrij_v21.safetensors">
                </label>
                <label>ComfyUI API（可选）
                  <input type="text" name="comfyui_api_base" placeholder="http://127.0.0.1:8188">
                </label>
              </div>
              <label><input type="checkbox" name="force" value="1"> 强制重绘</label>
              <button type="submit">为本章生成插图</button>
            </form>
          </section>
        </div>
        """
        self._write_html(_render_page(f"{project_name} - {chapter_file.name}", body, notice=notice, error=error))

    def _handle_illustration_file(self, project_id: str, chapter_slug: str, file_name: str) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        illustrations_root = (project_path / "illustrations").resolve()
        file_path = (illustrations_root / chapter_slug / file_name).resolve()
        if not file_path.exists() or illustrations_root not in file_path.parents:
            self.send_error(HTTPStatus.NOT_FOUND, "插图不存在")
            return
        self._write_file(file_path)

    def _handle_illustrate(self, project_id: str, form: dict[str, str]) -> None:
        project_path = _find_project(project_id)
        if project_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "项目不存在")
            return

        api_keys = _load_api_keys()
        try:
            llm_runtime_config = None
            try:
                llm_runtime_config = _build_runtime_config(project_path, {}, api_keys)
            except Exception:
                llm_runtime_config = None
            chapter_slug = (form.get("chapter_slug") or "latest").strip() or "latest"
            results = illustrate_chapters(
                str(project_path),
                chapter_refs=[chapter_slug],
                llm_config=llm_runtime_config,
                user_request=(form.get("user_request") or "").strip(),
                force=bool(form.get("force")),
                overrides=_illustration_overrides_from_form(form),
            )
            result = results[0]
            chapter_target = result.get("chapter_slug", chapter_slug)
            state = "已复用现有插图。" if result.get("reused") else "插图生成完成。"
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "/chapter/"
                + urllib.parse.quote(chapter_target)
                + "?notice="
                + urllib.parse.quote(state)
            )
        except Exception as exc:
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?error="
                + urllib.parse.quote(str(exc))
            )

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
            chapter_paths = run_next_chapters(
                str(project_path),
                runtime_config,
                count,
                user_request=(form.get("user_request") or "").strip(),
            )
            notice = f"续写完成，共生成 {count} 章。"
            if form.get("illustrate_generated"):
                illustration_results = illustrate_chapters(
                    str(project_path),
                    chapter_refs=chapter_paths,
                    llm_config=runtime_config,
                    user_request=(form.get("illustration_request") or "").strip(),
                    overrides=None,
                )
                new_count = sum(0 if item.get("reused") else 1 for item in illustration_results)
                notice += f" 插图处理完成（新生成 {new_count} 章插图）。"
            self._redirect(
                "/project/"
                + urllib.parse.quote(project_id)
                + "?notice="
                + urllib.parse.quote(notice)
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
