from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import webui
from project_manager import load_json, save_json
from web_auth import WebAuthSettings
from webui import NovelWriterHandler, ThreadingHTTPServer
from progression_manager import CUSTOM_PROGRESSION_OPTION_ID
from version import DISPLAY_VERSION

from tests.test_support import create_test_project, runtime_config


class WebUiGuidedFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_path = create_test_project(self.output_dir, project_id="web")
        self.original_output_dir = webui.OUTPUT_DIR
        self.original_registry = webui.JOB_REGISTRY
        webui._LOGIN_ATTEMPT_GUARDS.clear()
        webui.OUTPUT_DIR = self.output_dir
        webui.JOB_REGISTRY = webui.BackgroundJobRegistry()
        self.auth_settings_patch = patch("webui._auth_settings", return_value=self._make_auth_settings(enabled=False))
        self.auth_settings_patch.start()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), NovelWriterHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        self.addCleanup(self._cleanup_server)

    def _cleanup_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.auth_settings_patch.stop()
        webui.OUTPUT_DIR = self.original_output_dir
        webui.JOB_REGISTRY = self.original_registry

    def _make_auth_settings(self, *, enabled: bool, **overrides) -> WebAuthSettings:
        data = {
            "enabled": enabled,
            "username": "admin",
            "password": "letmein-test",
            "secret_key": "unit-test-secret",
            "cookie_name": "novel_writer_webui_session",
            "cookie_secure": False,
            "session_max_age_seconds": 3600,
            "login_max_attempts": 5,
            "login_window_seconds": 300,
            "login_lockout_seconds": 900,
            "config_path": "/tmp/test-webui-auth.env",
        }
        data.update(overrides)
        return WebAuthSettings(**data)

    def _post(self, path: str, body: str, *, headers: dict[str, str] | None = None) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        request_headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if headers:
            request_headers.update(headers)
        conn.request(
            "POST",
            path,
            body=body,
            headers=request_headers,
        )
        response = conn.getresponse()
        response.body = response.read().decode("utf-8", errors="replace")
        conn.close()
        return response

    def _post_multipart(
        self,
        path: str,
        fields: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> http.client.HTTPResponse:
        boundary = "----novel-writer-test-boundary"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            chunks.append(str(value).encode("utf-8"))
            chunks.append(b"\r\n")
        for name, (filename, content, content_type) in files.items():
            chunks.append(f"--{boundary}\r\n".encode("utf-8"))
            chunks.append(
                (
                    f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                    f"Content-Type: {content_type}\r\n\r\n"
                ).encode("utf-8")
            )
            chunks.append(content)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(chunks)
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Content-Length": str(len(body))},
        )
        response = conn.getresponse()
        response.body = response.read().decode("utf-8", errors="replace")
        conn.close()
        return response

    def _get(self, path: str, *, headers: dict[str, str] | None = None) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        conn.request("GET", path, headers=headers or {})
        response = conn.getresponse()
        response.body = response.read().decode("utf-8", errors="replace")
        conn.close()
        return response

    def _wait_for_job_status(self, job_id: str, statuses: set[str] | None = None) -> dict:
        target_statuses = statuses or {"succeeded", "failed"}
        for _ in range(30):
            job = webui.JOB_REGISTRY.get(job_id)
            if job and job.get("status") in target_statuses:
                return job
            time.sleep(0.1)
        self.fail(f"job {job_id} did not reach one of {sorted(target_statuses)}")

    def test_project_pages_show_token_cost_statistics(self) -> None:
        project_file = self.project_path / "project.json"
        project = load_json(str(project_file))
        project["stats"] = {
            "total": {
                "requests": 3,
                "successes": 2,
                "failures": 1,
                "prompt_tokens": 1200,
                "completion_tokens": 800,
                "total_tokens": 2500,
                "cached_tokens": 300,
                "reasoning_tokens": 20,
                "thought_tokens": 10,
            },
            "by_phase": {
                "writer": {
                    "requests": 1,
                    "successes": 1,
                    "failures": 0,
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "cached_tokens": 300,
                    "reasoning_tokens": 0,
                    "thought_tokens": 0,
                }
            },
            "cost": {
                "currency": "USD",
                "estimated_total_usd": 0.000123,
                "priced_tokens": 1500,
                "unpriced_tokens": 500,
                "started_at": "2026-04-26T00:00:00+00:00",
                "by_phase": {
                    "writer": {
                        "requests": 1,
                        "estimated_usd": 0.000123,
                        "priced_tokens": 1500,
                        "unpriced_tokens": 0,
                    }
                },
                "by_model": {
                    "deepseek:deepseek-v4-flash": {
                        "provider": "deepseek",
                        "model": "deepseek-v4-flash",
                        "requests": 1,
                        "total_tokens": 1500,
                        "estimated_usd": 0.000123,
                        "priced_tokens": 1500,
                        "unpriced_tokens": 0,
                        "pricing_status": "priced",
                        "source": {"name": "DeepSeek Models & Pricing"},
                    }
                },
            },
        }
        save_json(str(project_file), project)

        projects_page = self._get("/projects")
        project_page = self._get("/project/web")

        self.assertEqual(projects_page.status, 200)
        self.assertEqual(project_page.status, 200)
        self.assertIn("估算费用：$0.0001", projects_page.body)
        self.assertIn("未定价：500 tokens", projects_page.body)
        self.assertIn("历史未估价：500 tokens", projects_page.body)
        self.assertIn("Token / 费用统计", project_page.body)
        self.assertIn("deepseek-v4-flash", project_page.body)
        self.assertIn("DeepSeek Models &amp; Pricing", project_page.body)

    def test_progression_options_endpoint_saves_session_and_project_page_reflects_it(self) -> None:
        session_payload = {
            "session_id": "session_generated",
            "created_at": "2026-04-20T00:00:00+00:00",
            "project_chapter_count": 0,
            "target_chapter_number": 1,
            "planning_mode": "chapter",
            "source_user_request": "先看试探",
            "runtime_overrides": {},
            "recommended_option_id": "option_1",
            "objective": "建立临时安全区，并确认是否需要外出搜集物资",
            "options": [
                {
                    "option_id": "option_1",
                    "title": "先探查走廊",
                    "plan_summary": "三人离开隔离区短程试探。",
                    "plan_steps": ["规划路线", "短程离开"],
                    "plan_guidance": "保持谨慎与紧张感。",
                    "recommended": True,
                },
                {
                    "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                    "title": "空白自定义项",
                    "plan_summary": "不采用现有候选方案，改由你自己定义。",
                    "plan_steps": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "plan_guidance": "请把用户填写的创意作为本章执行 plan。",
                    "recommended": False,
                    "custom": True,
                },
            ],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }
        called = {}

        def fake_generate(*args, **kwargs):
            called["kwargs"] = kwargs
            (self.project_path / "progression_sessions").mkdir(exist_ok=True)
            (self.project_path / "progression_sessions" / "progression_session_generated.json").write_text(
                json.dumps(session_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return session_payload

        with patch("webui.generate_progression_options", side_effect=fake_generate):
            response = self._post(
                f"/project/web/progression-options",
                "option_count=4&planning_mode=chapter&objective=%E5%BB%BA%E7%AB%8B%E4%B8%B4%E6%97%B6%E5%AE%89%E5%85%A8%E5%8C%BA%EF%BC%8C%E5%B9%B6%E7%A1%AE%E8%AE%A4%E6%98%AF%E5%90%A6%E9%9C%80%E8%A6%81%E5%A4%96%E5%87%BA%E6%90%9C%E9%9B%86%E7%89%A9%E8%B5%84&user_request=%E5%85%88%E7%9C%8B%E8%AF%95%E6%8E%A2",
            )

        self.assertEqual(response.status, 303)
        self.assertIn("/project/web", response.getheader("Location"))
        self.assertEqual(
            called["kwargs"]["objective_override"],
            "建立临时安全区，并确认是否需要外出搜集物资",
        )
        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        progression_jobs = [job for job in jobs if job.get("kind") == "progression_options"]
        self.assertEqual(len(progression_jobs), 1)
        self._wait_for_job_status(progression_jobs[0]["id"])

        page = self._get("/project/web")
        self.assertEqual(page.status, 200)
        self.assertIn("先探查走廊", page.body)
        self.assertIn("空白自定义项", page.body)
        self.assertIn("project-layout", page.body)
        self.assertIn("有效当前章任务卡", page.body)
        self.assertIn("本章 objective（可修改）", page.body)
        self.assertIn("本组 plan 基于 objective", page.body)
        self.assertIn("选 plan 策略", page.body)
        self.assertIn("卷目标", page.body)
        self.assertNotIn("为什么现在", page.body)
        self.assertNotIn("本章纲要", page.body)

    def test_continue_guided_endpoint_creates_background_job(self) -> None:
        session = {
            "session_id": "session_web",
            "created_at": "2026-04-20T00:00:00+00:00",
            "project_chapter_count": 0,
            "target_chapter_number": 1,
            "planning_mode": "chapter",
            "source_user_request": "先探查",
            "runtime_overrides": {},
            "recommended_option_id": "option_1",
            "options": [
                {
                    "option_id": "option_1",
                    "title": "先探查走廊",
                    "plan_summary": "短程试探",
                    "plan_steps": ["规划路线", "短程离开"],
                    "plan_guidance": "保持谨慎。",
                    "recommended": True,
                },
                {
                    "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                    "title": "空白自定义项",
                    "plan_summary": "由用户自己定义",
                    "plan_steps": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "plan_guidance": "请把用户填写的创意作为本章执行 plan。",
                    "recommended": False,
                    "custom": True,
                },
            ],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }
        (self.project_path / "progression_sessions").mkdir(exist_ok=True)
        (self.project_path / "progression_sessions" / "progression_session_web.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with patch("webui.run_next_chapter_from_progression", return_value=str(self.project_path / "chapters" / "chapter_0001.md")):
            response = self._post(
                "/project/web/continue-guided",
                "progression_session=session_web&progression_option=option_1&progression_feedback=%E5%A4%9A%E4%B8%80%E7%82%B9%E8%AF%95%E6%8E%A2",
            )

        self.assertEqual(response.status, 303)
        location = response.getheader("Location")
        self.assertTrue(location.startswith("/job/"))
        job_id = location.rsplit("/", 1)[-1]

        self._wait_for_job_status(job_id)

        page = self._get(location)
        self.assertEqual(page.status, 200)
        self.assertIn("任务状态", page.body)

    def test_job_page_renders_existing_events_on_first_load(self) -> None:
        job = webui.JOB_REGISTRY.create_job(
            kind="continue",
            title="后台续写",
            project_id="web",
            project_path=str(self.project_path.resolve()),
        )
        webui.JOB_REGISTRY.mark_running(job["id"], "后台任务已启动")
        webui.JOB_REGISTRY.progress(job["id"], {"stage": "writer", "message": "正在写第 1 章"})

        page = self._get(f"/job/{job['id']}")
        self.assertEqual(page.status, 200)
        self.assertIn("任务日志", page.body)
        self.assertIn("任务已加入队列", page.body)
        self.assertIn("后台任务已启动", page.body)
        self.assertIn("正在写第 1 章", page.body)

    def test_continue_guided_rejects_blank_custom_option_without_user_idea(self) -> None:
        session = {
            "session_id": "session_web_custom",
            "created_at": "2026-04-20T00:00:00+00:00",
            "project_chapter_count": 0,
            "target_chapter_number": 1,
            "planning_mode": "chapter",
            "source_user_request": "",
            "runtime_overrides": {},
            "recommended_option_id": "option_1",
            "options": [
                {
                    "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                    "title": "空白自定义项",
                    "plan_summary": "由用户自己定义",
                    "plan_steps": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "plan_guidance": "请把用户填写的创意作为本章执行 plan。",
                    "recommended": False,
                    "custom": True,
                }
            ],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }
        (self.project_path / "progression_sessions").mkdir(exist_ok=True)
        (self.project_path / "progression_sessions" / "progression_session_web_custom.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        response = self._post(
            "/project/web/continue-guided",
            f"progression_session=session_web_custom&progression_option={CUSTOM_PROGRESSION_OPTION_ID}&progression_feedback=",
        )

        self.assertEqual(response.status, 303)
        self.assertIn("/project/web?error=", response.getheader("Location"))

    def test_pages_render_model_preset_controls(self) -> None:
        with patch(
            "webui._get_repo_admin_info",
            return_value={
                "repo_root": "/repo",
                "service_name": "novel-writer-webui.service",
                "service_scope": "user",
                "git_available": True,
                "systemd_run_available": True,
                "systemctl_available": True,
                "branch": "master",
                "commit": "abc1234",
                "upstream": "origin/master",
                "dirty": False,
                "error": "",
            },
        ), patch("webui._read_admin_action_status", return_value={}):
            projects_page = self._get("/projects")
        self.assertEqual(projects_page.status, 200)
        self.assertIn('name="model_preset"', projects_page.body)
        self.assertIn("使用 gemini 默认模型", projects_page.body)
        self.assertIn("维护操作", projects_page.body)
        self.assertIn("/admin/restart", projects_page.body)
        self.assertIn("/admin/update", projects_page.body)
        self.assertIn(f"版本 {DISPLAY_VERSION}", projects_page.body)
        self.assertIn("服务作用域", projects_page.body)
        self.assertIn("user", projects_page.body)

        project_page = self._get("/project/web")
        self.assertEqual(project_page.status, 200)
        self.assertIn('name="model_preset"', project_page.body)
        self.assertIn("沿用项目当前模型（llama3.2）", project_page.body)
        self.assertIn(f"版本 {DISPLAY_VERSION}", project_page.body)

    def test_protected_pages_redirect_to_login_when_auth_enabled(self) -> None:
        with patch("webui._auth_settings", return_value=self._make_auth_settings(enabled=True)):
            response = self._get("/projects")

        self.assertEqual(response.status, 303)
        self.assertIn("/login?next=", response.getheader("Location"))

    def test_login_submit_sets_cookie_and_allows_followup_access(self) -> None:
        settings = self._make_auth_settings(enabled=True)
        with patch("webui._auth_settings", return_value=settings):
            login_page = self._get("/login")
            self.assertEqual(login_page.status, 200)
            self.assertIn("登录后继续", login_page.body)

            response = self._post(
                "/login",
                "username=admin&password=letmein-test&next=%2Fprojects",
            )

            self.assertEqual(response.status, 303)
            self.assertEqual(response.getheader("Location"), "/projects")
            cookie_header = response.getheader("Set-Cookie")
            self.assertIn(settings.cookie_name, cookie_header)

            cookie_value = cookie_header.split(";", 1)[0]
            projects_page = self._get("/projects", headers={"Cookie": cookie_value})

        self.assertEqual(projects_page.status, 200)
        self.assertIn("项目书架", projects_page.body)
        self.assertIn("返回首页", projects_page.body)
        self.assertIn("退出登录", projects_page.body)

    def test_login_lockout_returns_retry_after(self) -> None:
        settings = self._make_auth_settings(
            enabled=True,
            login_max_attempts=1,
            login_window_seconds=60,
            login_lockout_seconds=30,
        )
        with patch("webui._auth_settings", return_value=settings):
            first = self._post(
                "/login",
                "username=admin&password=wrong&next=%2Fprojects",
            )
            second = self._post(
                "/login",
                "username=admin&password=wrong&next=%2Fprojects",
            )

        self.assertEqual(first.status, 401)
        self.assertEqual(second.status, 429)
        retry_after = int(second.getheader("Retry-After") or "0")
        self.assertGreaterEqual(retry_after, 1)
        self.assertLessEqual(retry_after, 30)

    def test_admin_restart_endpoint_launches_task(self) -> None:
        with patch("webui._launch_admin_task", return_value="novel-writer-admin-restart-test") as mocked_launch:
            response = self._post("/admin/restart", "")

        self.assertEqual(response.status, 303)
        self.assertIn("/projects?notice=", response.getheader("Location"))
        mocked_launch.assert_called_once_with("restart")

    def test_admin_restart_requires_login_when_auth_enabled(self) -> None:
        settings = self._make_auth_settings(enabled=True)
        with patch("webui._auth_settings", return_value=settings):
            response = self._post("/admin/restart", "")
            self.assertEqual(response.status, 303)
            self.assertIn("/login?next=", response.getheader("Location"))

            login_response = self._post(
                "/login",
                "username=admin&password=letmein-test&next=%2Fprojects",
            )
            cookie_value = login_response.getheader("Set-Cookie").split(";", 1)[0]

            with patch("webui._launch_admin_task", return_value="novel-writer-admin-restart-test") as mocked_launch:
                restart_response = self._post(
                    "/admin/restart",
                    "",
                    headers={"Cookie": cookie_value},
                )

        self.assertEqual(restart_response.status, 303)
        self.assertIn("/projects?notice=", restart_response.getheader("Location"))
        mocked_launch.assert_called_once_with("restart")

    def test_admin_update_endpoint_rejects_dirty_repo(self) -> None:
        with patch(
            "webui._get_repo_admin_info",
            return_value={
                "repo_root": "/repo",
                "service_name": "novel-writer-webui.service",
                "service_scope": "user",
                "git_available": True,
                "systemd_run_available": True,
                "systemctl_available": True,
                "branch": "master",
                "commit": "abc1234",
                "upstream": "origin/master",
                "dirty": True,
                "error": "",
            },
        ):
            response = self._post("/admin/update", "")

        self.assertEqual(response.status, 303)
        self.assertIn("/projects?error=", response.getheader("Location"))

    def test_admin_update_endpoint_launches_task_when_repo_clean(self) -> None:
        with patch(
            "webui._get_repo_admin_info",
            return_value={
                "repo_root": "/repo",
                "service_name": "novel-writer-webui.service",
                "service_scope": "user",
                "git_available": True,
                "systemd_run_available": True,
                "systemctl_available": True,
                "branch": "master",
                "commit": "abc1234",
                "upstream": "origin/master",
                "dirty": False,
                "error": "",
            },
        ), patch("webui._launch_admin_task", return_value="novel-writer-admin-update-test") as mocked_launch:
            response = self._post("/admin/update", "")

        self.assertEqual(response.status, 303)
        self.assertIn("/projects?notice=", response.getheader("Location"))
        mocked_launch.assert_called_once_with("update")

    def test_systemd_user_command_env_sets_runtime_bus_defaults(self) -> None:
        env = webui._systemd_user_command_env()
        runtime_dir = f"/run/user/{os.getuid()}"

        self.assertEqual(env["XDG_RUNTIME_DIR"], runtime_dir)
        self.assertEqual(env["DBUS_SESSION_BUS_ADDRESS"], f"unix:path={runtime_dir}/bus")

    def test_resolve_service_scope_falls_back_to_system_when_user_unit_missing(self) -> None:
        with patch("webui._systemctl_query_scope", side_effect=lambda service_name, scope: scope == "system"):
            self.assertEqual(webui._resolve_service_scope("novel-writer-webui.service"), "system")

    def test_launch_admin_task_uses_systemd_user_env(self) -> None:
        with patch("webui._resolve_service_scope", return_value="user"), patch("webui._write_admin_action_status"), patch(
            "webui._run_checked_command",
            return_value="queued",
        ) as mocked_run, patch("webui.shutil.which", return_value="/usr/bin/systemd-run"):
            unit_name = webui._launch_admin_task("restart")

        self.assertTrue(unit_name.startswith("novel-writer-admin-restart-"))
        _, kwargs = mocked_run.call_args
        self.assertEqual(kwargs["cwd"], str(webui.BASE_DIR))
        self.assertEqual(kwargs["env"]["XDG_RUNTIME_DIR"], f"/run/user/{os.getuid()}")
        self.assertEqual(
            kwargs["env"]["DBUS_SESSION_BUS_ADDRESS"],
            f"unix:path=/run/user/{os.getuid()}/bus",
        )

    def test_launch_admin_task_uses_system_scope_without_user_flag(self) -> None:
        with patch("webui._resolve_service_scope", return_value="system"), patch("webui._write_admin_action_status"), patch(
            "webui._run_checked_command",
            return_value="queued",
        ) as mocked_run, patch("webui.shutil.which", return_value="/usr/bin/systemd-run"):
            unit_name = webui._launch_admin_task("restart")

        self.assertTrue(unit_name.startswith("novel-writer-admin-restart-"))
        command, kwargs = mocked_run.call_args
        self.assertNotIn("--user", command[0])
        self.assertIsNone(kwargs["env"])

    def test_run_admin_task_restart_uses_system_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = str(Path(temp_dir) / "admin_status.json")
            with patch("webui._run_checked_command", return_value="ok") as mocked_run:
                code = webui._run_admin_task(
                    "restart",
                    repo_root=temp_dir,
                    service_name="novel-writer-webui.service",
                    service_scope="system",
                    status_path=status_path,
                )

        self.assertEqual(code, 0)
        calls = mocked_run.call_args_list
        self.assertEqual(calls[0].args[0], ["systemctl", "restart", "novel-writer-webui.service"])
        self.assertIsNone(calls[0].kwargs["env"])

    def test_model_preset_submission_resolves_without_manual_model_name(self) -> None:
        overrides = webui._runtime_overrides_from_form(
            {
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
            }
        )
        self.assertEqual(overrides["model_name"], "qwen2.5:14b")

        overrides = webui._runtime_overrides_from_form(
            {
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
                "model_name_custom": "my-local-model",
            }
        )
        self.assertEqual(overrides["model_name"], "my-local-model")

        captured: dict[str, dict] = {}

        def fake_init_project(config_path: str, progress_callback=None) -> str:
            captured["config"] = json.loads(Path(config_path).read_text(encoding="utf-8"))
            return str(self.project_path)

        with patch("webui.init_project", side_effect=fake_init_project):
            webui._create_project(
                {
                    "provider": "ollama",
                    "model_preset": "qwen2.5:14b",
                    "story_request": "测试故事",
                },
                {"OLLAMA_API_KEY": ""},
            )

        self.assertEqual(captured["config"]["model_name"], "qwen2.5:14b")
        self.assertEqual(captured["config"]["writing_quality_mode"], "balanced")
        self.assertEqual(captured["config"]["review_mode"], "auto")

        with patch("webui.init_project", side_effect=fake_init_project):
            webui._create_project(
                {
                    "provider": "ollama",
                    "model_preset": "qwen2.5:14b",
                    "story_request": "测试故事",
                    "writing_quality_mode": "high",
                    "review_mode": "manual",
                },
                {"OLLAMA_API_KEY": ""},
            )

        self.assertEqual(captured["config"]["writing_quality_mode"], "high")
        self.assertEqual(captured["config"]["review_mode"], "manual")

        with patch("webui.init_project", side_effect=fake_init_project):
            webui._create_project(
                {
                    "provider": "ollama",
                    "model_preset": "qwen2.5:14b",
                    "story_request": "娴嬭瘯鏁呬簨",
                    "quality_provider": "gemini",
                    "quality_model_name": "gemini-2.5-pro",
                },
                {"OLLAMA_API_KEY": "", "GEMINI_API_KEY": "gemini-key"},
            )

        self.assertEqual(captured["config"]["quality_model"]["model_provider"], "gemini")
        self.assertEqual(captured["config"]["quality_model"]["model_name"], "gemini-2.5-pro")
        self.assertEqual(captured["config"]["quality_model"]["api_key"], "gemini-key")
        self.assertNotIn("temperature", captured["config"]["quality_model"])

    def test_runtime_overrides_include_quality_and_review_modes(self) -> None:
        overrides = webui._runtime_overrides_from_form(
            {
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
                "writing_quality_mode": "high",
                "review_mode": "manual",
            }
        )

        self.assertEqual(overrides["writing_quality_mode"], "high")
        self.assertEqual(overrides["review_mode"], "manual")

    def test_runtime_config_defaults_and_overrides_quality_modes(self) -> None:
        project = load_json(str(self.project_path / "project.json"))
        project["llm_config"].pop("writing_quality_mode", None)
        project["llm_config"].pop("review_mode", None)
        save_json(str(self.project_path / "project.json"), project)

        config = webui._build_runtime_config(self.project_path, {}, {"OLLAMA_API_KEY": ""})
        self.assertEqual(config["writing_quality_mode"], "balanced")
        self.assertEqual(config["review_mode"], "auto")

        config = webui._build_runtime_config(
            self.project_path,
            {"writing_quality_mode": "high", "review_mode": "manual"},
            {"OLLAMA_API_KEY": ""},
        )
        self.assertEqual(config["writing_quality_mode"], "high")
        self.assertEqual(config["review_mode"], "manual")

    def test_runtime_config_resolves_quality_model_overrides(self) -> None:
        project = load_json(str(self.project_path / "project.json"))
        project["llm_config"]["quality_model"] = {
            "model_provider": "gemini",
            "model_name": "gemini-2.5-pro",
            "api_key": "",
        }
        save_json(str(self.project_path / "project.json"), project)

        config = webui._build_runtime_config(
            self.project_path,
            {"quality_model": {"model_name": "gemini-2.5-flash", "temperature": "0.2"}},
            {"OLLAMA_API_KEY": "", "GEMINI_API_KEY": "gemini-key"},
        )

        self.assertEqual(config["model_name"], "llama3.2")
        self.assertEqual(config["quality_model"]["model_provider"], "gemini")
        self.assertEqual(config["quality_model"]["model_name"], "gemini-2.5-flash")
        self.assertEqual(config["quality_model"]["api_key"], "gemini-key")
        self.assertEqual(config["quality_model"]["temperature"], "0.2")

    def test_runtime_overrides_include_quality_model_fields(self) -> None:
        overrides = webui._runtime_overrides_from_form(
            {
                "quality_provider": "gemini",
                "quality_model_name": "gemini-2.5-pro",
                "temperature": "1.7",
            }
        )

        self.assertEqual(overrides["quality_model"]["model_provider"], "gemini")
        self.assertEqual(overrides["quality_model"]["model_name"], "gemini-2.5-pro")
        self.assertNotIn("temperature", overrides)
        self.assertNotIn("temperature", overrides["quality_model"])

    def test_runtime_overrides_includes_log_llm_payload_when_checked(self) -> None:
        overrides = webui._runtime_overrides_from_form(
            {
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
                "log_llm_payload": "1",
            }
        )
        self.assertEqual(overrides["log_llm_payload"], "1")

        overrides = webui._runtime_overrides_from_form(
            {
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
            }
        )
        self.assertNotIn("log_llm_payload", overrides)

    def test_continue_async_starts_followup_progression_job(self) -> None:
        session_payload = {
            "session_id": "session_auto",
            "created_at": "2026-04-20T00:00:00+00:00",
            "project_chapter_count": 1,
            "target_chapter_number": 2,
            "planning_mode": "chapter",
            "source_user_request": "",
            "runtime_overrides": {},
            "recommended_option_id": "option_1",
            "options": [],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }

        with patch("webui.run_next_chapters", return_value=[str(self.project_path / "chapters" / "chapter_0001.md")]) as mocked_run_next_chapters, patch(
            "webui.generate_progression_options",
            return_value=session_payload,
        ):
            response = self._post(
                "/project/web/continue",
                "count=1&selection_mode=random&user_request=%E7%BB%A7%E7%BB%AD%E6%8E%A8%E8%BF%9B",
            )

        self.assertEqual(response.status, 303)
        job_id = response.getheader("Location").rsplit("/", 1)[-1]
        self._wait_for_job_status(job_id)
        _, run_kwargs = mocked_run_next_chapters.call_args
        self.assertEqual(run_kwargs["selection_mode"], "random")

        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        auto_jobs = [job for job in jobs if job.get("kind") == "progression_options_auto"]
        self.assertEqual(len(auto_jobs), 1)
        auto_job = self._wait_for_job_status(auto_jobs[0]["id"])
        self.assertFalse(auto_job.get("blocks_project", True))

    def test_continue_async_rejects_same_project_when_blocking_job_active(self) -> None:
        webui.JOB_REGISTRY.create_job(
            kind="continue",
            title="existing",
            project_id="web",
            project_path=str(self.project_path.resolve()),
        )

        with patch("webui.run_next_chapters") as mocked_run_next_chapters:
            response = self._post(
                "/project/web/continue",
                "count=1&selection_mode=recommended",
            )

        self.assertEqual(response.status, 303)
        self.assertIn("/project/web?error=", response.getheader("Location"))
        mocked_run_next_chapters.assert_not_called()
        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        self.assertEqual([job["kind"] for job in jobs].count("continue"), 1)

    def test_continue_async_allows_different_project_when_other_project_active(self) -> None:
        other_path = create_test_project(self.output_dir, project_id="web_b")
        webui.JOB_REGISTRY.create_job(
            kind="continue",
            title="existing",
            project_id="web",
            project_path=str(self.project_path.resolve()),
        )

        with patch(
            "webui.run_next_chapters",
            return_value=[str(other_path / "chapters" / "chapter_0001.md")],
        ) as mocked_run_next_chapters, patch("webui._enqueue_progression_job", return_value=None):
            response = self._post(
                "/project/web_b/continue",
                "count=1&selection_mode=recommended",
            )

        self.assertEqual(response.status, 303)
        location = response.getheader("Location")
        self.assertTrue(location.startswith("/job/"))
        job = self._wait_for_job_status(location.rsplit("/", 1)[-1])
        self.assertEqual(job["project_id"], "web_b")
        mocked_run_next_chapters.assert_called_once()

    def test_create_project_async_starts_followup_progression_job(self) -> None:
        session_payload = {
            "session_id": "session_project_bootstrap",
            "created_at": "2026-04-20T00:00:00+00:00",
            "project_chapter_count": 0,
            "target_chapter_number": 1,
            "planning_mode": "chapter",
            "source_user_request": "",
            "runtime_overrides": {},
            "recommended_option_id": "option_1",
            "options": [],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }

        with patch("webui._create_project", return_value=str(self.project_path)), patch(
            "webui.generate_progression_options",
            return_value=session_payload,
        ):
            response = self._post(
                "/projects/create",
                "provider=ollama&project_name=%E6%B5%8B%E8%AF%95&story_request=%E5%BC%80%E7%AF%87",
            )

        self.assertEqual(response.status, 303)
        job_id = response.getheader("Location").rsplit("/", 1)[-1]
        self._wait_for_job_status(job_id)

        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        auto_jobs = [job for job in jobs if job.get("kind") == "progression_options_auto"]
        self.assertEqual(len(auto_jobs), 1)
        self._wait_for_job_status(auto_jobs[0]["id"])

    def test_project_page_keeps_forms_available_during_non_blocking_progression_job(self) -> None:
        job = webui.JOB_REGISTRY.create_job(
            kind="progression_options_auto",
            title="后台生成推进选项",
            project_id="web",
            project_path=str(self.project_path.resolve()),
            blocks_project=False,
        )
        webui.JOB_REGISTRY.mark_running(job["id"], "正在生成下一章推进选项")

        page = self._get("/project/web")
        self.assertEqual(page.status, 200)
        self.assertIn("下一章推进选项正在后台生成", page.body)
        self.assertNotIn("为避免并发写入冲突", page.body)

    def test_chapter_page_shows_polish_form(self) -> None:
        (self.project_path / "chapters" / "chapter_0001.md").write_text(
            "林宇推上储物箱。\n\n苏浅检查控制板。",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)

        page = self._get("/project/web/chapter/chapter_0001")

        self.assertEqual(page.status, 200)
        self.assertIn("章节润色", page.body)
        self.assertIn("细节增强", page.body)
        self.assertIn("更欢乐", page.body)
        self.assertIn("自定义润色要求", page.body)
        self.assertNotIn("Planning Mode", page.body)
        self.assertIn("质量优化", page.body)
        self.assertIn("暂无质量报告或自动重写记录", page.body)

    def test_chapter_page_links_quality_reports_and_pre_rewrite_text(self) -> None:
        (self.project_path / "chapters" / "chapter_0001.md").write_text(
            "重写后的正文",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)
        save_json(
            str(self.project_path / "quality_reviews" / "chapter_0001_attempt_1.json"),
            {
                "schema_version": 2,
                "passed": False,
                "average_score": 4.0,
                "scores": {"task_completion": 4, "reader_hook": 4},
                "blocking_issues": [
                    {
                        "category": "task_completion",
                        "severity": "blocker",
                        "issue": "没有完成信号确认。",
                        "evidence": "全文停留在讨论。",
                        "fix": "补写确认行动。",
                    }
                ],
                "issues": ["开章钩子弱"],
                "revision_guidance": "换一个开场压力。",
                "rewrite_plan": ["补写确认动作", "收束到新线索"],
            },
        )
        (self.project_path / "quality_drafts" / "chapter_0001_before_rewrite_1.md").write_text(
            "重写前正文",
            encoding="utf-8",
        )

        page = self._get("/project/web/chapter/chapter_0001")

        self.assertEqual(page.status, 200)
        self.assertIn("已迭代优化次数", page.body)
        self.assertIn("查看质量报告", page.body)
        self.assertIn("查看重写前文本", page.body)

        report_page = self._get("/project/web/chapter/chapter_0001/quality-report")
        self.assertEqual(report_page.status, 200)
        self.assertIn("Attempt 1", report_page.body)
        self.assertIn("未通过", report_page.body)
        self.assertIn("没有完成信号确认", report_page.body)
        self.assertIn("原始 JSON", report_page.body)

        draft_page = self._get("/project/web/chapter/chapter_0001/pre-rewrite")
        self.assertEqual(draft_page.status, 200)
        self.assertIn("重写前文本 1", draft_page.body)
        self.assertIn("重写前正文", draft_page.body)

    def test_chapter_page_shows_audiobook_player_when_manifest_exists(self) -> None:
        (self.project_path / "chapters" / "chapter_0001.md").write_text(
            "林宇推上储物箱。\n\n苏浅检查控制板。",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)
        audio_dir = self.project_path / "audiobook" / "chapter_0001"
        audio_dir.mkdir(parents=True)
        (audio_dir / "chapter_0001.wav").write_bytes(b"fake wav")
        save_json(
            str(audio_dir / "manifest.json"),
            {
                "chapter_slug": "chapter_0001",
                "generated_at": "2026-04-20T00:00:00+00:00",
                "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                "segment_count": 2,
                "segments": [],
            },
        )

        page = self._get("/project/web/chapter/chapter_0001")

        self.assertEqual(page.status, 200)
        self.assertIn("本章有声小说", page.body)
        self.assertIn("/project/web/audiobook-file/chapter_0001/chapter_0001.wav", page.body)

    def test_audiobook_endpoint_accepts_reference_upload_and_creates_job(self) -> None:
        (self.project_path / "chapters" / "chapter_0001.md").write_text(
            "林宇说：“我们先检查门。”",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)

        with patch(
            "webui.generate_audiobook_chapters",
            return_value=[
                {
                    "chapter_slug": "chapter_0001",
                    "combined_audio": "audiobook/chapter_0001/chapter_0001.wav",
                    "reused": False,
                }
            ],
        ) as mocked_generate:
            response = self._post_multipart(
                "/project/web/audiobook",
                {
                    "chapter_slug": "chapter_0001",
                    "narrator_preset": "calm_male",
                    "character_voice_name": "林宇",
                    "character_prompt_text": "这是参考文本",
                    "force": "1",
                },
                {
                    "character_reference_audio": ("linyu.wav", b"RIFF....WAVE", "audio/wav"),
                },
            )
            self.assertEqual(response.status, 303)
            location = response.getheader("Location")
            self.assertTrue(location.startswith("/job/"))
            job_id = location.rsplit("/", 1)[-1]
            job = self._wait_for_job_status(job_id)

        self.assertEqual(job["kind"], "audiobook")

        _, kwargs = mocked_generate.call_args
        self.assertEqual(kwargs["chapter_refs"], ["chapter_0001"])
        self.assertEqual(kwargs["narrator_preset"], "calm_male")
        self.assertTrue(kwargs["force"])

        voices = load_json(str(self.project_path / "audiobook" / "voices.json"))
        self.assertTrue(voices["character_voices"]["林宇"]["reference_audio"].startswith("audiobook/voice_refs/"))
        self.assertEqual(voices["character_voices"]["林宇"]["prompt_text"], "这是参考文本")

    def test_polish_chapter_endpoint_creates_background_job_with_runtime_overrides(self) -> None:
        (self.project_path / "chapters" / "chapter_0001.md").write_text(
            "林宇推上储物箱。\n\n苏浅检查控制板。",
            encoding="utf-8",
        )
        project = load_json(str(self.project_path / "project.json"))
        project["chapter_count"] = 1
        save_json(str(self.project_path / "project.json"), project)

        body = urllib.parse.urlencode(
            {
                "polish_preset_details": "1",
                "polish_preset_longer": "1",
                "polish_custom_request": "多一点轻松互怼",
                "provider": "ollama",
                "model_preset": "qwen2.5:14b",
            }
        )

        with patch(
            "webui.run_chapter_polish",
            return_value={
                "chapter_slug": "chapter_0001",
                "backup_path": str(self.project_path / "polish_backups" / "chapter_0001" / "backup.md"),
                "staled_progression_sessions": 0,
            },
        ) as mocked_polish:
            response = self._post("/project/web/chapter/chapter_0001/polish", body)
            self.assertEqual(response.status, 303)
            location = response.getheader("Location")
            self.assertTrue(location.startswith("/job/"))
            job_id = location.rsplit("/", 1)[-1]
            job = self._wait_for_job_status(job_id)

        self.assertEqual(job["kind"], "polish_chapter")
        polish_args = mocked_polish.call_args.args
        polish_kwargs = mocked_polish.call_args.kwargs
        self.assertEqual(polish_args[0], str(self.project_path))
        self.assertEqual(polish_args[2], "chapter_0001")
        self.assertEqual(polish_args[1]["model_name"], "qwen2.5:14b")
        self.assertEqual(polish_kwargs["preset_ids"], ["details", "longer"])
        self.assertEqual(polish_kwargs["custom_request"], "多一点轻松互怼")


if __name__ == "__main__":
    unittest.main()
