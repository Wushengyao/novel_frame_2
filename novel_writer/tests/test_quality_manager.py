from __future__ import annotations

import unittest

from quality_manager import normalize_craft_brief, normalize_quality_review, quality_review_passed


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


if __name__ == "__main__":
    unittest.main()
