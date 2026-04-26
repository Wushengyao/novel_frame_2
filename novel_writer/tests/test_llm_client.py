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
    def test_deepseek_v4_flash_writer_disables_thinking_and_tunes_temperature(self) -> None:
        config = {
            "model_provider": "deepseek",
            "model": "deepseek-v4-flash",
            "api_key": "secret-key",
            "temperature": 0.9,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "choices": [{"message": {"content": "chapter text"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            text, metadata = generate_text_with_metadata(
                "write a chapter",
                config,
                log_context={"phase": "writer"},
            )

        endpoint, headers, request_body, _timeout = mocked_request.call_args.args
        self.assertEqual(endpoint, "https://api.deepseek.com/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer secret-key")
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["temperature"], 1.3)
        self.assertNotIn("reasoning_effort", request_body)
        self.assertEqual(text, "chapter text")
        self.assertEqual(metadata["provider"], "deepseek")

    def test_deepseek_v4_pro_quality_review_enables_max_reasoning_json_mode(self) -> None:
        config = {
            "model_provider": "deepseek",
            "model": "deepseek-v4-pro",
            "api_key": "secret-key",
            "temperature": 0.8,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "choices": [{"message": {"content": "{\"passed\": true}"}}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            _text, metadata = generate_text_with_metadata(
                "review chapter",
                config,
                log_context={"phase": "quality_review"},
                response_format="json",
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(request_body["thinking"], {"type": "enabled"})
        self.assertEqual(request_body["reasoning_effort"], "max")
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", request_body)
        self.assertEqual(metadata["usage"]["reasoning_tokens"], 2)

    def test_deepseek_v4_preserves_custom_creative_temperature(self) -> None:
        config = {
            "model_provider": "deepseek",
            "model": "deepseek-v4-flash",
            "api_key": "secret-key",
            "temperature": 1.6,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "choices": [{"message": {"content": "chapter text"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "write a chapter",
                config,
                log_context={"phase": "writer"},
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(request_body["thinking"], {"type": "disabled"})
        self.assertEqual(request_body["temperature"], 1.6)
    def test_gemini_3_flash_writer_uses_minimal_thinking_and_creative_temperature(self) -> None:
        config = {
            "model_provider": "gemini",
            "model": "gemini-3.1-flash-lite-preview",
            "api_key": "secret-key",
            "api_base": "https://generativelanguage.googleapis.com/v1beta",
            "temperature": 0.9,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "candidates": [{"content": {"parts": [{"text": "chapter text"}]}}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "write a chapter",
                config,
                log_context={"phase": "writer"},
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(request_body["generationConfig"]["temperature"], 1.0)
        self.assertEqual(
            request_body["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "minimal"},
        )

    def test_gemini_25_pro_json_task_uses_dynamic_thinking_and_low_temperature(self) -> None:
        config = {
            "model_provider": "gemini",
            "model": "gemini-2.5-pro",
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
                "return JSON",
                config,
                log_context={"phase": "outline"},
                response_format="json",
            )

        generation_config = mocked_request.call_args.args[2]["generationConfig"]
        self.assertEqual(generation_config["temperature"], 0.2)
        self.assertEqual(generation_config["responseMimeType"], "application/json")
        self.assertEqual(generation_config["thinkingConfig"], {"thinkingBudget": -1})

    def test_grok_json_task_uses_response_format_and_drops_unsupported_reasoning_effort(self) -> None:
        config = {
            "model_provider": "grok",
            "model": "grok-4.20-beta-latest-non-reasoning",
            "api_key": "secret-key",
            "temperature": 0.9,
            "max_tokens": 4000,
            "timeout": 120,
            "request_options": {"reasoning_effort": "high"},
        }
        response_payload = {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "return JSON",
                config,
                log_context={"phase": "outline"},
                response_format="json",
            )

        request_body = mocked_request.call_args.args[2]
        timeout = mocked_request.call_args.args[3]
        self.assertEqual(request_body["temperature"], 0.2)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertNotIn("reasoning_effort", request_body)
        self.assertEqual(timeout, 120)

    def test_doubao_seed_writer_disables_thinking_and_tunes_temperature(self) -> None:
        config = {
            "model_provider": "doubao",
            "model": "doubao-seed-1-8-251228",
            "api_key": "secret-key",
            "temperature": 1.0,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "choices": [{"message": {"content": "chapter text"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "write a chapter",
                config,
                log_context={"phase": "writer"},
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(request_body["temperature"], 0.9)
        self.assertEqual(request_body["thinking"], {"type": "disabled"})

    def test_ollama_json_task_uses_low_temperature_and_json_response_format(self) -> None:
        config = {
            "model_provider": "ollama",
            "model": "llama3.2",
            "api_key": "",
            "temperature": 0.9,
            "max_tokens": 4000,
            "timeout": 120,
        }
        response_payload = {
            "choices": [{"message": {"content": "{\"ok\": true}"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

        with patch(
            "llm_client._request_json",
            return_value=(response_payload, 1),
        ) as mocked_request:
            generate_text_with_metadata(
                "return JSON",
                config,
                log_context={"phase": "summary"},
                response_format="json",
            )

        request_body = mocked_request.call_args.args[2]
        self.assertEqual(request_body["temperature"], 0.2)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
