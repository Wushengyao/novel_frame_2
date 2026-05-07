from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_runtime import AgentRun, SkillRegistry, SkillSpec, WorkflowAgent


class AgentRuntimeTests(unittest.TestCase):
    def test_skill_registry_rejects_duplicate_and_missing_metadata(self) -> None:
        registry = SkillRegistry()
        registry.register(SkillSpec("demo.skill", "Demo skill"))

        with self.assertRaisesRegex(ValueError, "duplicate skill id"):
            registry.register(SkillSpec("demo.skill", "Duplicate"))

        with self.assertRaisesRegex(ValueError, "description is required"):
            registry.register(SkillSpec("demo.missing_description", ""))

        with self.assertRaisesRegex(ValueError, "skill id is required"):
            registry.register(SkillSpec("", "Missing id"))

    def test_agent_run_persists_skill_events_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = AgentRun(tmp, "unit_workflow", workflow_id="unit")
            run.append_skill_event({"skill_id": "demo.skill", "status": "succeeded"})
            run.add_artifacts({"project": "project.json"})
            run.finish_failure(RuntimeError("boom"))

            payload = json.loads((Path(tmp) / "agent_runs" / "agent_run_unit.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["workflow"], "unit_workflow")
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["error"], "boom")
            self.assertEqual(payload["artifacts"]["project"], "project.json")
            self.assertEqual(payload["skill_events"][0]["skill_id"], "demo.skill")

    def test_workflow_agent_emits_skill_events_to_progress_callback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events: list[dict] = []
            agent = WorkflowAgent(tmp, "chapter_workflow", workflow_id="chapter", progress_callback=events.append)

            result = agent.run_skill(
                "chapter.prepare_context",
                lambda: "context",
                inputs={"planning_mode": "chapter"},
                artifacts={"task_card": "task_cards/chapter_0001.json"},
                message="Context prepared",
            )
            agent.finish_success(message="done")

            self.assertEqual(result.value, "context")
            stages = [event.get("stage") for event in events]
            self.assertIn("agent_skill_start", stages)
            self.assertIn("agent_skill_done", stages)
            payload = json.loads((Path(tmp) / "agent_runs" / "agent_run_chapter.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "succeeded")
            self.assertEqual(payload["skill_events"][0]["skill_id"], "chapter.prepare_context")
            self.assertEqual(payload["skill_events"][0]["artifacts"]["task_card"], "task_cards/chapter_0001.json")


if __name__ == "__main__":
    unittest.main()
