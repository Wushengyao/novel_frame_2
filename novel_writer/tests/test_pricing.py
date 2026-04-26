from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing import estimate_llm_cost


class PricingTests(unittest.TestCase):
    def test_deepseek_flash_splits_cached_input_and_output(self) -> None:
        estimate = estimate_llm_cost(
            "deepseek",
            "deepseek-v4-flash",
            {
                "prompt_tokens": 1000,
                "cached_tokens": 400,
                "completion_tokens": 500,
                "total_tokens": 1500,
            },
        )

        expected = (600 * 0.14 + 400 * 0.028 + 500 * 0.28) / 1_000_000
        self.assertEqual(estimate["pricing_status"], "priced")
        self.assertEqual(estimate["priced_tokens"], 1500)
        self.assertAlmostEqual(estimate["estimated_cost_usd"], expected)

    def test_deepseek_pro_switches_from_discount_to_standard_price(self) -> None:
        usage = {
            "prompt_tokens": 1_000_000,
            "cached_tokens": 0,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        }

        discounted = estimate_llm_cost(
            "deepseek",
            "deepseek-v4-pro",
            usage,
            now=datetime(2026, 5, 5, 15, 58, tzinfo=timezone.utc),
        )
        standard = estimate_llm_cost(
            "deepseek",
            "deepseek-v4-pro",
            usage,
            now=datetime(2026, 5, 5, 16, 0, tzinfo=timezone.utc),
        )

        self.assertAlmostEqual(discounted["estimated_cost_usd"], 0.435 + 0.87)
        self.assertAlmostEqual(standard["estimated_cost_usd"], 1.74 + 3.48)

    def test_gemini_pro_uses_long_context_rates_above_200k_prompt_tokens(self) -> None:
        short_context = estimate_llm_cost(
            "gemini",
            "gemini-2.5-pro",
            {"prompt_tokens": 200_000, "completion_tokens": 10_000, "total_tokens": 210_000},
        )
        long_context = estimate_llm_cost(
            "gemini",
            "gemini-2.5-pro",
            {"prompt_tokens": 200_001, "completion_tokens": 10_000, "total_tokens": 210_001},
        )

        self.assertAlmostEqual(short_context["estimated_cost_usd"], (200_000 * 1.25 + 10_000 * 10.0) / 1_000_000)
        self.assertAlmostEqual(long_context["estimated_cost_usd"], (200_001 * 2.50 + 10_000 * 15.0) / 1_000_000)

    def test_gemini_31_pro_alias_uses_preview_tiered_rates(self) -> None:
        estimate = estimate_llm_cost(
            "gemini",
            "gemini-3.1-pro",
            {
                "prompt_tokens": 250_000,
                "cached_tokens": 50_000,
                "completion_tokens": 25_000,
            },
        )

        expected = (200_000 * 4.0 + 50_000 * 0.40 + 25_000 * 18.0) / 1_000_000
        self.assertEqual(estimate["pricing_status"], "priced")
        self.assertEqual(estimate["model"], "gemini-3.1-pro-preview")
        self.assertAlmostEqual(estimate["estimated_cost_usd"], expected)

    def test_xai_grok_420_uses_long_context_rates_above_200k_prompt_tokens(self) -> None:
        estimate = estimate_llm_cost(
            "grok",
            "grok-4.20-beta-latest-non-reasoning",
            {"prompt_tokens": 250_000, "cached_tokens": 50_000, "completion_tokens": 25_000},
        )

        expected = (200_000 * 4.0 + 50_000 * 0.40 + 25_000 * 12.0) / 1_000_000
        self.assertEqual(estimate["pricing_status"], "priced")
        self.assertAlmostEqual(estimate["estimated_cost_usd"], expected)

    def test_local_backends_are_zero_cost_and_doubao_is_unpriced(self) -> None:
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        local = estimate_llm_cost("ollama", "llama3.2", usage)
        llama_cpp = estimate_llm_cost("llama_cpp", "local-model", usage)
        unpriced = estimate_llm_cost("doubao", "doubao-seed-1-8-251228", usage)

        self.assertEqual(local["pricing_status"], "local")
        self.assertEqual(local["estimated_cost_usd"], 0.0)
        self.assertEqual(local["priced_tokens"], 150)
        self.assertEqual(llama_cpp["pricing_status"], "local")
        self.assertEqual(llama_cpp["estimated_cost_usd"], 0.0)
        self.assertEqual(llama_cpp["priced_tokens"], 150)
        self.assertEqual(unpriced["pricing_status"], "unpriced")
        self.assertEqual(unpriced["unpriced_tokens"], 150)
        self.assertEqual(unpriced["estimated_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
