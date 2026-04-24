from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import illustration_manager
from project_manager import load_json, save_json
from tests.test_support import create_test_project


class ExternalServicesConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project_path = create_test_project(Path(self.temp_dir.name), project_id="services")

    def test_illustration_runtime_uses_external_services_config(self) -> None:
        project_file = self.project_path / "project.json"
        project = load_json(str(project_file))
        project["illustration_config"] = {
            "comfyui_api_base": "http://old-host:8188",
            "checkpoint": "old/checkpoint.safetensors",
            "width": 640,
            "height": 640,
        }
        save_json(str(project_file), project)

        service_config = Path(self.temp_dir.name) / "external_services.json"
        service_config.write_text(
            json.dumps(
                {
                    "comfyui": {
                        "api_base": "10.0.0.8:8188",
                        "checkpoint": "new/checkpoint.safetensors",
                        "width": 1024,
                        "height": 768,
                        "steps": 12,
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {
                "NOVEL_EXTERNAL_SERVICES_CONFIG": str(service_config),
                "NOVEL_COMFYUI_API_BASE": "",
                "NOVEL_COMFYUI_ROOT": "",
                "NOVEL_COMFYUI_WORKFLOW_TEMPLATE": "",
                "NOVEL_COMFYUI_CHECKPOINT": "",
                "NOVEL_COMFYUI_WIDTH": "",
                "NOVEL_COMFYUI_HEIGHT": "",
                "NOVEL_COMFYUI_STEPS": "",
            },
        ):
            runtime = illustration_manager._build_runtime_config(str(self.project_path))

        self.assertEqual(runtime["comfyui_api_base"], "http://10.0.0.8:8188")
        self.assertEqual(runtime["checkpoint"], "new/checkpoint.safetensors")
        self.assertEqual(runtime["width"], 1024)
        self.assertEqual(runtime["height"], 768)
        self.assertEqual(runtime["steps"], 12)


if __name__ == "__main__":
    unittest.main()
