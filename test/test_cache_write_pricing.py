"""Cache-write tokens are priced separately from ordinary input.

They arrive inside prompt_tokens_details alongside cached_tokens and were being
discarded, so they fell through to the uncached-input rate. On gpt-5.6-sol a
write is $5.00/MTok against $4.00 input, so discarding them undercounts.
"""
from types import SimpleNamespace

import pytest

from sana_evaluation.models import MODEL_PRICING
from sana_evaluation.runner.record import AgentResult
from sana_evaluation.llm.openai_cached_model import (
    extract_chat_cache_write_tokens,
    extract_chat_cached_input_tokens,
)


def _usage(**details):
    return SimpleNamespace(prompt_tokens_details=SimpleNamespace(**details))


class TestExtraction:
    def test_reads_and_writes_are_pulled_separately(self):
        u = _usage(cached_tokens=900, cache_write_tokens=100)
        assert extract_chat_cached_input_tokens(u) == 900
        assert extract_chat_cache_write_tokens(u) == 100

    def test_absent_write_field_is_none_not_zero(self):
        # None means "not reported"; zero would claim nothing was written.
        assert extract_chat_cache_write_tokens(_usage(cached_tokens=5)) is None


def _result(model, **usage):
    return AgentResult(answer="", model=model, model_name=model,
                       metrics=SimpleNamespace(accumulated_usage=usage))


class TestPricing:
    def test_writes_are_billed_at_the_write_rate(self):
        r = _result("gpt-5.6-sol", inputTokens=1000, cacheReadInputTokens=600,
                    cacheWriteInputTokens=200, outputTokens=100)
        # 200 uncached @4 + 600 read @0.4 + 200 write @5 + 100 out @20
        expected = (4.0 * 200 + 0.4 * 600 + 5.0 * 200 + 20.0 * 100) / 1e6
        assert round(r.get_price(), 10) == round(expected, 10)

    def test_writes_do_not_double_count_against_uncached(self):
        r = _result("gpt-5.6-sol", inputTokens=1000, cacheReadInputTokens=600,
                    cacheWriteInputTokens=200, outputTokens=0)
        assert r.uncached_input_tokens == 200

    def test_a_model_without_a_write_rate_bills_writes_as_input(self):
        # Every model behaved this way before the field existed; that must hold.
        r = _result("gpt-5-mini", inputTokens=1000, cacheReadInputTokens=0,
                    cacheWriteInputTokens=300, outputTokens=0)
        p = MODEL_PRICING["gpt-5-mini"]
        assert "cache_write_input" not in p
        assert round(r.get_price(), 10) == round(p["input"] * 1000 / 1e6, 10)

    def test_absent_write_tokens_leave_pricing_unchanged(self):
        r = _result("gpt-5.6-sol", inputTokens=1000, cacheReadInputTokens=600,
                    outputTokens=100)
        expected = (4.0 * 400 + 0.4 * 600 + 20.0 * 100) / 1e6
        assert round(r.get_price(), 10) == round(expected, 10)

    def test_sol_rates_match_what_was_published(self):
        p = MODEL_PRICING["gpt-5.6-sol"]
        assert (p["input"], p["cache_read_input"], p["cache_write_input"], p["output"]) \
            == (4.00, 0.40, 5.00, 20.00)
