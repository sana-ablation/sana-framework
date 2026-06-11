"""Unit tests for model pricing lookup behavior."""

import unittest
from types import SimpleNamespace

from sana_evaluation.helper.result import AgentResult


class _DummyMetrics:
    def __init__(self, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> None:
        self.accumulated_usage = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
        }
        if cached_input_tokens:
            self.accumulated_usage["cacheReadInputTokens"] = cached_input_tokens
        self.cycle_count = 0
        self.cycle_durations = []
        self.tool_metrics = {}


class _DummyToolMetric:
    def __init__(self, call_count: int) -> None:
        self.call_count = call_count
        self.success_count = call_count
        self.total_time = float(call_count)


def _build_result(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    *,
    cached_input_tokens: int = 0,
) -> AgentResult:
    return AgentResult(
        answer="",
        model="",
        model_name=model_name,
        metrics=_DummyMetrics(input_tokens, output_tokens, cached_input_tokens=cached_input_tokens),
    )


class PricingLookupTests(unittest.TestCase):
    def test_openai_prefixed_model_name_has_nonzero_cost(self) -> None:
        result = _build_result("openai/gpt-5.2", input_tokens=23616, output_tokens=285)
        self.assertGreater(result.cost_usd, 0.0)

    def test_openai_prefixed_and_unprefixed_keys_match_cost(self) -> None:
        prefixed = _build_result("openai/gpt-5.2", input_tokens=23616, output_tokens=285)
        unprefixed = _build_result("gpt-5.2", input_tokens=23616, output_tokens=285)
        self.assertAlmostEqual(prefixed.cost_usd, unprefixed.cost_usd, places=12)

    def test_new_openai_model_uses_registered_pricing(self) -> None:
        result = _build_result(
            "openai/gpt-5.4-nano",
            input_tokens=10_000,
            output_tokens=500,
            cached_input_tokens=8_000,
        )
        expected = ((2_000 * 0.20) + (8_000 * 0.02) + (500 * 1.25)) / 1_000_000
        self.assertAlmostEqual(result.cost_usd, expected, places=12)

    def test_gpt_5_4_uses_registered_pricing(self) -> None:
        result = _build_result(
            "openai/gpt-5.4",
            input_tokens=10_000,
            output_tokens=500,
            cached_input_tokens=8_000,
        )
        expected = ((2_000 * 2.50) + (8_000 * 0.25) + (500 * 15.00)) / 1_000_000
        self.assertAlmostEqual(result.cost_usd, expected, places=12)

    def test_gpt_5_nano_uses_registered_pricing(self) -> None:
        result = _build_result(
            "openai/gpt-5-nano",
            input_tokens=10_000,
            output_tokens=500,
            cached_input_tokens=8_000,
        )
        expected = ((2_000 * 0.05) + (8_000 * 0.005) + (500 * 0.40)) / 1_000_000
        self.assertAlmostEqual(result.cost_usd, expected, places=12)

    def test_bedrock_key_is_unchanged(self) -> None:
        result = _build_result("bedrock/claude-haiku-4.5", input_tokens=1000, output_tokens=500)
        # 1.00 * 1000/1e6 + 5.00 * 500/1e6
        expected = 0.0035
        self.assertAlmostEqual(result.cost_usd, expected, places=12)

    def test_openai_cached_input_tokens_use_discounted_rate(self) -> None:
        result = _build_result(
            "openai/gpt-5.2",
            input_tokens=10_000,
            output_tokens=500,
            cached_input_tokens=8_000,
        )
        expected = ((2_000 * 1.75) + (8_000 * 0.175) + (500 * 14.0)) / 1_000_000
        self.assertAlmostEqual(result.cost_usd, expected, places=12)
        self.assertEqual(result.cached_input_tokens, 8_000)
        self.assertEqual(result.uncached_input_tokens, 2_000)

    def test_unknown_model_returns_zero_and_logs_warning(self) -> None:
        result = _build_result("unknown/provider-model", input_tokens=1000, output_tokens=1000)
        with self.assertLogs("sana_evaluation.helper.result", level="WARNING") as captured:
            cost = result.cost_usd
        self.assertEqual(cost, 0.0)
        self.assertTrue(any("No pricing configured for model_name=unknown/provider-model" in m for m in captured.output))

    def test_api_tool_calls_include_current_data_search_and_ideal_tools(self) -> None:
        result = AgentResult(
            answer="",
            model="",
            metrics=SimpleNamespace(
                accumulated_usage={},
                cycle_count=0,
                cycle_durations=[],
                tool_metrics={
                    "peek_file": _DummyToolMetric(1),
                    "peek_multiple": _DummyToolMetric(2),
                    "read_file": _DummyToolMetric(3),
                    "grep_file": _DummyToolMetric(4),
                    "query_file": _DummyToolMetric(5),
                    "search_value": _DummyToolMetric(6),
                    "search_schema": _DummyToolMetric(7),
                    "search_reranked": _DummyToolMetric(8),
                    "search_ideal": _DummyToolMetric(9),
                    "execute_ideal": _DummyToolMetric(10),
                    "plan": _DummyToolMetric(100),
                    "plan_ideal": _DummyToolMetric(100),
                    "skills": _DummyToolMetric(100),
                    "submit_answer": _DummyToolMetric(100),
                },
            ),
        )

        self.assertEqual(result.get_api_tool_calls(), 55)


if __name__ == "__main__":
    unittest.main()
