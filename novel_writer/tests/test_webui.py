from __future__ import annotations

import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import webui
from web_auth import WebAuthSettings
from webui import NovelWriterHandler, ThreadingHTTPServer
from progression_manager import CUSTOM_PROGRESSION_OPTION_ID

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
            "options": [
                {
                    "option_id": "option_1",
                    "title": "先探查走廊",
                    "summary": "三人离开隔离区短程试探。",
                    "why_now": "外部信息缺口太大。",
                    "key_events": ["规划路线", "短程离开"],
                    "writer_guidance": "保持谨慎与紧张感。",
                    "chapter_outline": {
                        "title": "探查走廊",
                        "summary": "完成第一次外出试探。",
                        "goal": "获取走廊情报",
                        "key_events": ["规划路线", "短程离开"],
                    },
                    "recommended": True,
                },
                {
                    "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                    "title": "空白自定义项",
                    "summary": "不采用现有候选方案，改由你自己定义。",
                    "why_now": "用户已有更明确的章节灵感。",
                    "key_events": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "writer_guidance": "请把用户填写的创意作为本章主任务。",
                    "chapter_outline": {
                        "title": "由你填写",
                        "summary": "由你填写这一章想看的情节。",
                        "goal": "由你填写当前章目标。",
                        "key_events": ["由你填写", "系统衔接"],
                    },
                    "recommended": False,
                    "custom": True,
                },
            ],
            "status": "pending",
            "selected_option_id": "",
            "selection_feedback": "",
        }

        def fake_generate(*args, **kwargs):
            (self.project_path / "progression_sessions").mkdir(exist_ok=True)
            (self.project_path / "progression_sessions" / "progression_session_generated.json").write_text(
                json.dumps(session_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return session_payload

        with patch("webui.generate_progression_options", side_effect=fake_generate):
            response = self._post(
                f"/project/web/progression-options",
                "option_count=4&planning_mode=chapter&user_request=%E5%85%88%E7%9C%8B%E8%AF%95%E6%8E%A2",
            )

        self.assertEqual(response.status, 303)
        self.assertIn("/project/web", response.getheader("Location"))
        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        progression_jobs = [job for job in jobs if job.get("kind") == "progression_options"]
        self.assertEqual(len(progression_jobs), 1)
        self._wait_for_job_status(progression_jobs[0]["id"])

        page = self._get("/project/web")
        self.assertEqual(page.status, 200)
        self.assertIn("先探查走廊", page.body)
        self.assertIn("空白自定义项", page.body)
        self.assertIn("project-layout", page.body)
        self.assertIn("当前章任务", page.body)
        self.assertIn("卷目标", page.body)

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
                    "summary": "短程试探",
                    "why_now": "获取情报",
                    "key_events": ["规划路线", "短程离开"],
                    "writer_guidance": "保持谨慎。",
                    "chapter_outline": {
                        "title": "探查走廊",
                        "summary": "完成第一次外出试探。",
                        "goal": "获取走廊情报",
                        "key_events": ["规划路线", "短程离开"],
                    },
                    "recommended": True,
                },
                {
                    "option_id": CUSTOM_PROGRESSION_OPTION_ID,
                    "title": "空白自定义项",
                    "summary": "由用户自己定义",
                    "why_now": "用户已有更明确的章节灵感。",
                    "key_events": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "writer_guidance": "请把用户填写的创意作为本章主任务。",
                    "chapter_outline": {
                        "title": "由你填写",
                        "summary": "由你填写",
                        "goal": "由你填写",
                        "key_events": ["由你填写", "系统衔接"],
                    },
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
                    "summary": "由用户自己定义",
                    "why_now": "用户已有更明确的章节灵感。",
                    "key_events": ["用户自定义本章推进", "系统保持状态与卷目标一致"],
                    "writer_guidance": "请把用户填写的创意作为本章主任务。",
                    "chapter_outline": {
                        "title": "由你填写",
                        "summary": "由你填写",
                        "goal": "由你填写",
                        "key_events": ["由你填写", "系统衔接"],
                    },
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

        project_page = self._get("/project/web")
        self.assertEqual(project_page.status, 200)
        self.assertIn('name="model_preset"', project_page.body)
        self.assertIn("沿用项目当前模型（llama3.2）", project_page.body)

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

        with patch("webui.run_next_chapters", return_value=[str(self.project_path / "chapters" / "chapter_0001.md")]), patch(
            "webui.generate_progression_options",
            return_value=session_payload,
        ):
            response = self._post(
                "/project/web/continue",
                "count=1&user_request=%E7%BB%A7%E7%BB%AD%E6%8E%A8%E8%BF%9B",
            )

        self.assertEqual(response.status, 303)
        job_id = response.getheader("Location").rsplit("/", 1)[-1]
        self._wait_for_job_status(job_id)

        jobs = webui.JOB_REGISTRY.list_jobs(project_id="web", active_only=False, limit=8)
        auto_jobs = [job for job in jobs if job.get("kind") == "progression_options_auto"]
        self.assertEqual(len(auto_jobs), 1)
        auto_job = self._wait_for_job_status(auto_jobs[0]["id"])
        self.assertFalse(auto_job.get("blocks_project", True))

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


if __name__ == "__main__":
    unittest.main()
