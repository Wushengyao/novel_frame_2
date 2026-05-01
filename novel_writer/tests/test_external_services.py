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
from external_services import ImageFrameClient, load_audio_frame_runtime, load_image_frame_runtime
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

    def test_image_frame_runtime_uses_image_frame_provider_ids(self) -> None:
        self.assertEqual(load_image_frame_runtime({"provider": "google"})["provider"], "google_ai")
        self.assertEqual(load_image_frame_runtime({"provider": "gemini"})["provider"], "google_ai")
        self.assertEqual(load_image_frame_runtime({"provider": "openai"})["provider"], "openai")
        self.assertEqual(load_image_frame_runtime({"provider": "xai"})["provider"], "xai")

    def test_illustration_runtime_uses_saved_image_frame_config(self) -> None:
        project_file = self.project_path / "project.json"
        project = load_json(str(project_file))
        project["illustration_config"] = {
            "backend": "image_frame",
            "image_frame_api_base": "http://127.0.0.1:8010",
            "image_frame_provider": "openai",
            "image_frame_model": "gpt-image-1.5",
            "image_frame_aspect_ratio": "16:9",
            "image_frame_google_image_size": "2K",
            "image_frame_num_outputs": 2,
            "image_frame_quality": "high",
            "image_frame_timeout": 900,
        }
        save_json(str(project_file), project)

        with patch.dict(
            os.environ,
            {
                "NOVEL_EXTERNAL_SERVICES_CONFIG": str(Path(self.temp_dir.name) / "missing.json"),
                "NOVEL_IMAGE_FRAME_API_BASE": "",
                "NOVEL_IMAGE_FRAME_PROVIDER": "",
                "NOVEL_IMAGE_FRAME_MODEL": "",
                "NOVEL_IMAGE_FRAME_SIZE": "",
                "NOVEL_IMAGE_FRAME_ASPECT_RATIO": "",
                "NOVEL_IMAGE_FRAME_GOOGLE_IMAGE_SIZE": "",
                "NOVEL_IMAGE_FRAME_QUALITY": "",
                "NOVEL_IMAGE_FRAME_BACKGROUND": "",
                "NOVEL_IMAGE_FRAME_MODERATION": "",
                "NOVEL_IMAGE_FRAME_NUM_OUTPUTS": "",
                "NOVEL_IMAGE_FRAME_TIMEOUT": "",
                "NOVEL_IMAGE_FRAME_POLL_INTERVAL": "",
                "NOVEL_IMAGE_FRAME_AUTH_USERNAME": "",
                "NOVEL_IMAGE_FRAME_AUTH_PASSWORD": "",
            },
        ):
            runtime = illustration_manager._build_runtime_config(str(self.project_path))
            overridden = illustration_manager._build_runtime_config(
                str(self.project_path),
                {"image_frame_provider": "xai", "image_frame_model": "grok-2-image"},
            )

        self.assertEqual(runtime["backend"], "image_frame")
        self.assertEqual(runtime["image_frame_provider"], "openai")
        self.assertEqual(runtime["image_frame_model"], "gpt-image-1.5")
        self.assertEqual(runtime["image_frame_aspect_ratio"], "16:9")
        self.assertEqual(runtime["image_frame_google_image_size"], "2K")
        self.assertEqual(runtime["image_frame_num_outputs"], 2)
        self.assertEqual(runtime["image_frame_quality"], "high")
        self.assertEqual(runtime["image_frame_timeout"], 900)
        self.assertEqual(overridden["image_frame_provider"], "xai")
        self.assertEqual(overridden["image_frame_model"], "grok-2-image")

    def test_image_frame_client_sends_seed_only_when_configured(self) -> None:
        runtime = load_image_frame_runtime({"provider": "openai", "model": "gpt-image-1.5"})
        client = ImageFrameClient("http://127.0.0.1:8010")

        with patch.object(client, "request_multipart", return_value={"id": "task-1"}) as mocked_request:
            client.create_text_to_image_task(runtime, "prompt")
        fields = mocked_request.call_args.kwargs["fields"]
        self.assertEqual(fields["mode"], "text_to_image")
        self.assertEqual(fields["provider"], "openai")
        self.assertNotIn("seed", fields)

        runtime["seed"] = 123
        with patch.object(client, "request_multipart", return_value={"id": "task-2"}) as mocked_request:
            client.create_text_to_image_task(runtime, "prompt")
        fields = mocked_request.call_args.kwargs["fields"]
        self.assertEqual(fields["seed"], "123")

    def test_audio_frame_runtime_supports_parallel_api_bases(self) -> None:
        service_config = Path(self.temp_dir.name) / "external_services.json"
        service_config.write_text(
            json.dumps(
                {
                    "audio_frame": {
                        "api_bases": [
                            "127.0.0.1:8810",
                            "http://127.0.0.1:8811",
                            "http://127.0.0.1:8811",
                        ],
                        "workers": 3,
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
                "NOVEL_AUDIO_FRAME_API_BASE": "",
                "NOVEL_AUDIO_FRAME_API_BASES": "",
                "NOVEL_AUDIO_FRAME_WORKERS": "",
            },
        ):
            runtime = load_audio_frame_runtime()

        self.assertEqual(runtime["api_base"], "http://127.0.0.1:8810")
        self.assertEqual(runtime["api_bases"], ["http://127.0.0.1:8810", "http://127.0.0.1:8811"])
        self.assertEqual(runtime["workers"], 3)

        with patch.dict(
            os.environ,
            {
                "NOVEL_EXTERNAL_SERVICES_CONFIG": str(service_config),
                "NOVEL_AUDIO_FRAME_API_BASE": "",
                "NOVEL_AUDIO_FRAME_API_BASES": "",
                "NOVEL_AUDIO_FRAME_WORKERS": "",
            },
        ):
            override_runtime = load_audio_frame_runtime({"api_base": "127.0.0.1:8899"})

        self.assertEqual(override_runtime["api_bases"], ["http://127.0.0.1:8899"])

    def test_audio_frame_runtime_supports_weighted_endpoints(self) -> None:
        service_config = Path(self.temp_dir.name) / "external_services.json"
        service_config.write_text(
            json.dumps(
                {
                    "audio_frame": {
                        "endpoints": [
                            {"api_base": "127.0.0.1:8808", "kind": "gpu", "capacity": 1},
                            {
                                "api_base": "127.0.0.1:8812",
                                "kind": "cpu",
                                "capacity": 4,
                                "max_chars": 24,
                                "speed": 0.15,
                            },
                        ],
                        "timeout": 30,
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
                "NOVEL_AUDIO_FRAME_API_BASE": "",
                "NOVEL_AUDIO_FRAME_API_BASES": "",
                "NOVEL_AUDIO_FRAME_ENDPOINTS": "",
                "NOVEL_AUDIO_FRAME_WORKERS": "",
            },
        ):
            runtime = load_audio_frame_runtime()

        self.assertEqual(runtime["api_base"], "http://127.0.0.1:8808")
        self.assertEqual(runtime["api_bases"], ["http://127.0.0.1:8808", "http://127.0.0.1:8812"])
        self.assertEqual(runtime["workers"], 5)
        self.assertEqual(runtime["timeout"], 30)
        self.assertEqual(runtime["endpoints"][1]["kind"], "cpu")
        self.assertEqual(runtime["endpoints"][1]["capacity"], 4)
        self.assertEqual(runtime["endpoints"][1]["max_chars"], 24)


if __name__ == "__main__":
    unittest.main()
