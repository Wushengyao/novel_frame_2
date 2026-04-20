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
from webui import NovelWriterHandler, ThreadingHTTPServer

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
        webui.OUTPUT_DIR = self.output_dir
        webui.JOB_REGISTRY = webui.BackgroundJobRegistry()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), NovelWriterHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        self.addCleanup(self._cleanup_server)

    def _cleanup_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        webui.OUTPUT_DIR = self.original_output_dir
        webui.JOB_REGISTRY = self.original_registry

    def _post(self, path: str, body: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        conn.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response = conn.getresponse()
        response.body = response.read().decode("utf-8", errors="replace")
        conn.close()
        return response

    def _get(self, path: str) -> http.client.HTTPResponse:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        conn.request("GET", path)
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
                }
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
        self.assertIn("project-layout", page.body)

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
                }
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

    def test_pages_render_model_preset_controls(self) -> None:
        projects_page = self._get("/projects")
        self.assertEqual(projects_page.status, 200)
        self.assertIn('name="model_preset"', projects_page.body)
        self.assertIn("使用 gemini 默认模型", projects_page.body)

        project_page = self._get("/project/web")
        self.assertEqual(project_page.status, 200)
        self.assertIn('name="model_preset"', project_page.body)
        self.assertIn("沿用项目当前模型（llama3.2）", project_page.body)

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
