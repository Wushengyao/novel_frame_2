from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_client import generate_text_with_metadata


class LLMClientTests(unittest.TestCase):
    def test_generate_text_with_metadata_logs_prompt_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project"
            project_path.mkdir()
            config = {
                "project_path": str(project_path.resolve()),
                "model_provider": "openai_compatible",
                "model": "llama3.2",
                "api_key": "secret-key",
                "api_base": "https://example.local/v1",
                "temperature": 0.8,
                "max_tokens": 4000,
                "timeout": 120,
                "log_llm_payload": "1",
            }
            response_payload = {
                "choices": [{"message": {"content": "这是模型正文。"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20},
            }

            with patch(
                "llm_client._request_json",
                return_value=(response_payload, 2),
            ) as mocked_request:
                generate_text_with_metadata(
                    "请基于以下上下文写一段正文。",
                    config,
                    log_context={"phase": "writer", "project_id": "unit-test", "user_request": "续写紧张氛围"},
                    system_prompt="你是稳定的小说写作助手。",
                )

            request_body = mocked_request.call_args.args[2]
            self.assertEqual(request_body["messages"][0], {"role": "system", "content": "你是稳定的小说写作助手。"})
            self.assertEqual(request_body["messages"][1]["content"], "请基于以下上下文写一段正文。")

            log_file = project_path / "llm_logs" / "llm_interactions.jsonl"
            self.assertTrue(log_file.exists())
            line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
            entry = json.loads(line)
            self.assertEqual(entry["phase"], "writer")
            self.assertEqual(
                entry["request"]["messages"][1]["content"],
                "请基于以下上下文写一段正文。",
            )
            self.assertEqual(entry["request"]["messages"][0]["role"], "system")
            self.assertEqual(entry["response_text"], "这是模型正文。")
            self.assertEqual(entry["attempts"], 2)
            self.assertEqual(entry["config"]["api_key"], "***")
            self.assertEqual(entry["log_context"]["project_id"], "unit-test")

    def test_openai_compatible_omits_system_message_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_path = Path(tmp) / "project"
            project_path.mkdir()
            config = {
                "project_path": str(project_path.resolve()),
                "model_provider": "openai_compatible",
                "model": "llama3.2",
                "api_key": "secret-key",
                "api_base": "https://example.local/v1",
                "temperature": 0.8,
                "max_tokens": 4000,
                "timeout": 120,
                "log_llm_payload": "",
            }
            response_payload = {
                "choices": [{"message": {"content": "这是模型正文。"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 11, "total_tokens": 20},
            }

            with patch(
                "llm_client._request_json",
                return_value=(response_payload, 1),
            ) as mocked_request:
                generate_text_with_metadata("请写一段正文。", config)

            request_body = mocked_request.call_args.args[2]
            self.assertEqual(request_body["messages"], [{"role": "user", "content": "请写一段正文。"}])
            self.assertFalse((project_path / "llm_logs").exists())

    def test_gemini_uses_system_instruction_and_explicit_json_response_format(self) -> None:
        config = {
            "model_provider": "gemini",
            "model": "gemini-test",
            "api_key": "secret-key",
            "api_base": "https://generativelanguage.googleapis.com/v1beta",
            "temperature": 0.8,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "candidates": [{"content": {"parts": [{"text": "{\"ok\": true}"}]}}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "请返回结构化结果。",
                config,
                system_prompt="你是稳定的结构化写作助手。",
                response_format="json",
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(
            request_body["systemInstruction"],
            {"parts": [{"text": "你是稳定的结构化写作助手。"}]},
        )
        self.assertEqual(request_body["contents"], [{"role": "user", "parts": [{"text": "请返回结构化结果。"}]}])
        self.assertEqual(request_body["generationConfig"]["responseMimeType"], "application/json")


if __name__ == "__main__":
    unittest.main()
