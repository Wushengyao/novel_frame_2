from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from expert_review_manager import (
    list_expert_review_artifacts,
    normalize_expert_review,
    run_expert_review_for_chapter,
)
from project_manager import save_chapter, save_json
from tests.test_support import create_test_project, read_json, runtime_config


class ExpertReviewManagerTests(unittest.TestCase):
    def _expert_payload(self, category: str = "prompt") -> dict:
        return {
            "schema_version": 1,
            "review_unavailable": False,
            "quality_summary": "章节完成了任务，但动作动因偏弱。",
            "overall_score": 0.62,
            "confidence": 0.8,
            "root_causes": [
                {
                    "category": category,
                    "severity": "major",
                    "confidence": 0.76,
                    "issue": "任务卡对行动压力描述不足。",
                    "evidence": "章节里关键行动缺少可见压力。",
                    "trace_refs": ["req_writer"],
                    "recommended_change": "在写作提示词里补充当前压力和失败代价。",
                }
            ],
            "recommended_actions": ["补强任务卡压力字段"],
            "trace_refs": ["req_writer"],
        }

    def test_normalize_expert_review_restricts_categories(self) -> None:
        report = normalize_expert_review(
            {
                "root_causes": [
                    {
                        "category": "unknown",
                        "severity": "severe",
                        "confidence": 88,
                        "issue": "泛化问题",
                    }
                ]
            }
        )

        self.assertEqual(report["root_causes"][0]["category"], "other")
        self.assertEqual(report["root_causes"][0]["severity"], "major")
        self.assertEqual(report["root_causes"][0]["confidence"], 0.88)

    def test_multi_model_reports_and_aggregate_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="expert_multi")
            chapter_path = Path(save_chapter(str(project_path), "第一章正文。"))
            save_json(str(project_path / "summaries" / "summary_0001.json"), {"chapter_summary": "完成据点封门。"})
            (project_path / "llm_logs").mkdir(exist_ok=True)
            (project_path / "llm_logs" / "llm_interactions.jsonl").write_text(
                json.dumps(
                    {
                        "request_id": "req_writer",
                        "phase": "writer",
                        "status": "succeeded",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "request": {"messages": [{"role": "user", "content": "写正文"}]},
                        "response_text": chapter_path.read_text(encoding="utf-8"),
                        "log_context": {"workflow_id": "wf_1", "target_chapter_number": 1},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            config = runtime_config(
                "chapter",
                expert_mode={
                    "enabled": True,
                    "models": [
                        {"model_provider": "ollama", "model_name": "expert-a"},
                        {"model_provider": "ollama", "model_name": "expert-b"},
                    ],
                },
            )

            with patch(
                "expert_review_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps(self._expert_payload("prompt"), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._expert_payload("workflow"), ensure_ascii=False), {"usage": {}}),
                    (json.dumps(self._expert_payload("workflow"), ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_generate:
                artifacts = run_expert_review_for_chapter(str(project_path), 1, "wf_1", config)

            self.assertEqual(mocked_generate.call_count, 3)
            self.assertEqual(len(artifacts["model_reports"]), 2)
            self.assertTrue((project_path / "expert_reviews" / "chapter_0001" / "aggregate.json").exists())
            aggregate = read_json(project_path / "expert_reviews" / "chapter_0001" / "aggregate.json")
            self.assertEqual(aggregate["report_type"], "aggregate")
            self.assertEqual(aggregate["root_causes"][0]["category"], "workflow")

    def test_expert_failure_saves_unavailable_report_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="expert_fail")
            save_chapter(str(project_path), "第一章正文。")
            config = runtime_config(
                "chapter",
                expert_mode={"enabled": True, "models": [{"model_provider": "ollama", "model_name": "expert"}]},
            )

            with patch("expert_review_manager.generate_text_with_metadata", side_effect=RuntimeError("timeout")):
                artifacts = run_expert_review_for_chapter(str(project_path), 1, "wf_fail", config)

        self.assertTrue(artifacts["aggregate"]["review_unavailable"])
        self.assertEqual(artifacts["aggregate"]["root_causes"][0]["category"], "logging")

    def test_list_expert_review_artifacts_handles_missing_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="expert_empty")
            artifacts = list_expert_review_artifacts(str(project_path), 1)

        self.assertEqual(artifacts["model_reports"], [])
        self.assertEqual(artifacts["aggregate"], {})


if __name__ == "__main__":
    unittest.main()
