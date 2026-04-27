from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common_utils import extract_json_object
from project_manager import load_json
from project_manager import save_json
from quality_manager import (
    list_quality_artifacts,
    normalize_craft_brief,
    normalize_quality_review,
    normalize_rewrite_response_text,
    quality_review_passed,
    rewrite_chapter_draft,
    save_pre_rewrite_draft,
)
from tests.test_support import create_test_project


class QualityManagerTests(unittest.TestCase):
    def _passing_scores(self) -> dict:
        return {
            "task_completion": 9,
            "reader_hook": 8,
            "scene_freshness": 8,
            "character_specificity": 8,
            "motivation_causality": 9,
            "repetition_risk": 8,
            "continuity": 9,
        }

    def test_craft_brief_normalizes_success_criteria(self) -> None:
        brief = normalize_craft_brief(
            {
                "chapter_hook": "用异常信号开章。",
                "success_criteria": [
                    "兑现异常信号造成的压力。",
                    "兑现异常信号造成的压力。",
                    "让角色做出有代价的选择。",
                    "结尾留下新线索。",
                    "避免重复上一章推门动作。",
                    "多余项会被裁掉。",
                ],
            }
        )

        self.assertEqual(
            brief["success_criteria"],
            [
                "兑现异常信号造成的压力。",
                "让角色做出有代价的选择。",
                "结尾留下新线索。",
                "避免重复上一章推门动作。",
                "多余项会被裁掉。",
            ],
        )

    def test_quality_review_v2_blocker_forces_failure(self) -> None:
        review = normalize_quality_review(
            {
                "scores": self._passing_scores(),
                "passed": True,
                "blocking_issues": [
                    {
                        "category": "task_completion",
                        "severity": "blocker",
                        "issue": "没有完成任务卡中的信号确认。",
                        "evidence": "全文只在原地讨论。",
                        "fix": "补写一次实际确认行动和结果。",
                    }
                ],
            }
        )

        self.assertFalse(review["passed"])
        self.assertFalse(quality_review_passed(review))
        self.assertEqual(review["schema_version"], 2)
        self.assertEqual(review["blocking_issues"][0]["severity"], "blocker")

    def test_quality_review_legacy_payload_gets_v2_defaults(self) -> None:
        review = normalize_quality_review(
            {
                "scores": self._passing_scores(),
                "passed": True,
                "strengths": ["任务清楚"],
                "issues": [],
                "revision_guidance": "",
                "repeat_examples": [],
            }
        )

        self.assertTrue(review["passed"])
        self.assertTrue(quality_review_passed(review))
        self.assertEqual(review["schema_version"], 2)
        self.assertFalse(review["review_unavailable"])
        self.assertEqual(review["score_reasons"], {})
        self.assertEqual(review["blocking_issues"], [])
        self.assertEqual(review["rewrite_plan"], [])

    def test_unavailable_review_never_passes(self) -> None:
        review = normalize_quality_review(None, fallback_passed=False, review_unavailable=True)

        self.assertFalse(review["passed"])
        self.assertFalse(quality_review_passed(review))
        self.assertTrue(review["review_unavailable"])
        self.assertEqual(review["scores"]["task_completion"], 0.0)

    def test_missing_review_payload_does_not_pass_by_default(self) -> None:
        review = normalize_quality_review(None)

        self.assertFalse(review["passed"])
        self.assertFalse(quality_review_passed(review))
        self.assertFalse(review["review_unavailable"])

    def test_quality_artifacts_list_reports_and_pre_rewrite_drafts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp)
            (project_path / "quality_reviews").mkdir()
            save_json(
                str(project_path / "quality_reviews" / "chapter_0001_attempt_1.json"),
                {"passed": False, "average_score": 4.0},
            )
            save_pre_rewrite_draft(str(project_path), 1, 1, "重写前正文")

            artifacts = list_quality_artifacts(str(project_path), 1)

            self.assertEqual(artifacts["rewrite_count"], 1)
            self.assertEqual(artifacts["reports"][0]["attempt"], 1)
            self.assertEqual(artifacts["reports"][0]["report"]["average_score"], 4.0)
            self.assertEqual(artifacts["pre_rewrite_drafts"][0]["rewrite_attempt"], 1)

    def test_extract_json_object_handles_preface_and_code_fence(self) -> None:
        payload = extract_json_object(
            "模型说明：\n```json\n{\"passed\": true, \"scores\": {}}\n```\n后续说明",
            "parse failed",
        )

        self.assertTrue(payload["passed"])

    def test_rewrite_response_accepts_accidental_json_payload(self) -> None:
        text = normalize_rewrite_response_text(
            "```json\n{\"chapter_text\": \"第1章\\n\\n重写后的正文。\"}\n```"
        )

        self.assertEqual(text, "第1章\n\n重写后的正文。")

    def test_rewrite_retries_when_json_payload_has_no_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="rewrite_retry")
            config = load_json(str(project_path / "project.json"))["llm_config"]
            prompt_context = {
                "task_card": {"chapter_number": 1},
                "sections": {},
                "section_chars": {},
            }

            with patch(
                "quality_manager.generate_text_with_metadata",
                side_effect=[
                    (json.dumps({"note": "只有说明，没有正文"}, ensure_ascii=False), {"usage": {}}),
                    (json.dumps({"rewritten_text": "重写后的正文。"}, ensure_ascii=False), {"usage": {}}),
                ],
            ) as mocked_generate:
                text = rewrite_chapter_draft(
                    str(project_path),
                    prompt_context,
                    "原草稿。",
                    {"rewrite_plan": ["补强动因"]},
                    config,
                    log_context={"phase": "writer"},
                )

        self.assertEqual(text, "重写后的正文。")
        self.assertEqual(mocked_generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
