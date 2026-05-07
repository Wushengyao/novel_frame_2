from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from runtime_config import build_runtime_config, sanitize_runtime_overrides
from tests.test_support import create_test_project
from workflow_modes import normalize_workflow_mode


class WorkflowModeTests(unittest.TestCase):
    def test_normalize_workflow_mode_defaults_invalid_values_to_classic(self) -> None:
        self.assertEqual(normalize_workflow_mode("agentic"), "agentic")
        self.assertEqual(normalize_workflow_mode(" classic "), "classic")
        self.assertEqual(normalize_workflow_mode(""), "classic")
        self.assertEqual(normalize_workflow_mode("surprise"), "classic")

    def test_runtime_overrides_sanitize_workflow_mode(self) -> None:
        self.assertEqual(sanitize_runtime_overrides({"workflow_mode": "agentic"})["workflow_mode"], "agentic")
        self.assertEqual(sanitize_runtime_overrides({"workflow_mode": "bad"})["workflow_mode"], "classic")

    def test_build_runtime_config_uses_project_workflow_mode_and_runtime_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = create_test_project(Path(tmp), project_id="workflow", workflow_mode="agentic")

            saved_config = build_runtime_config(str(project_path), {}, {})
            self.assertEqual(saved_config["workflow_mode"], "agentic")

            overridden_config = build_runtime_config(
                str(project_path),
                {"workflow_mode": "classic"},
                {},
            )
            self.assertEqual(overridden_config["workflow_mode"], "classic")


if __name__ == "__main__":
    unittest.main()
