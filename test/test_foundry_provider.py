"""Claude on Microsoft Foundry: factory wiring, request params, usage accounting.

The cache-token tests are the load-bearing ones. Strands' stock Anthropic
adapter reports only input/output/total; on Fable 5.1 a cached input token is
$0.25/MTok against $10.00 uncached, so losing that field would understate the
cached share by 40x in the opposite direction -- every cached token would be
billed as if it were fresh.
"""
import os

import pytest

from sana_evaluation.config import AgentConfig
from sana_evaluation.helper.constants import MODEL_PRICING, MODEL_REGISTRY
from sana_evaluation.llm.anthropic_foundry_model import AnthropicFoundryModel
from sana_evaluation.llm.llm_factory import build_model


@pytest.fixture(autouse=True)
def _foundry_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "test-resource")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_BASE_URL", raising=False)
    monkeypatch.delenv("SANA_CLAUDE_EFFORT", raising=False)


def _build(**kw):
    return build_model(AgentConfig(provider="foundry", model_id="claude-fable-5-1", **kw))


def test_factory_returns_the_foundry_model():
    assert isinstance(_build(), AnthropicFoundryModel)


def test_registry_points_foundry_at_the_bare_model_id():
    # Bedrock prefixes with "anthropic."; Foundry does not.
    assert MODEL_REGISTRY["foundry/claude-fable-5-1"] == ("foundry", "claude-fable-5-1")


def test_pricing_is_present_for_both_key_forms():
    for key in ("claude-fable-5-1", "foundry/claude-fable-5-1"):
        p = MODEL_PRICING[key]
        assert (p["input"], p["cache_read_input"], p["output"]) == (10.00, 0.25, 50.00)


# --- request parameters -------------------------------------------------

def test_default_temperature_is_not_sent():
    # Fable 5.1 rejects sampling parameters with a 400.
    assert "temperature" not in (_build().get_config().get("params") or {})


def test_explicit_temperature_is_sent():
    assert _build(temperature=0.7).get_config()["params"]["temperature"] == 0.7


def test_effort_defaults_to_medium():
    assert _build().get_config()["params"]["output_config"] == {"effort": "medium"}


def test_effort_env_override(monkeypatch):
    monkeypatch.setenv("SANA_CLAUDE_EFFORT", "low")
    assert _build().get_config()["params"]["output_config"] == {"effort": "low"}


def test_explicit_output_config_wins_over_the_default():
    m = _build(extra_model_kwargs={"params": {"output_config": {"effort": "high"}}})
    assert m.get_config()["params"]["output_config"] == {"effort": "high"}


def test_no_thinking_block_is_sent():
    # Thinking is always on for this model and any explicit config returns 400.
    assert "thinking" not in (_build().get_config().get("params") or {})


def test_max_tokens_reaches_the_model_config():
    assert _build(max_tokens=32000).get_config()["max_tokens"] == 32000


def test_conflicting_resource_and_base_url_raise_an_actionable_error(monkeypatch):
    # The SDK re-reads whichever argument is None from the environment, so this
    # cannot be resolved by preferring one; the builder must say what to unset.
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://example.invalid")
    with pytest.raises(ValueError, match="ANTHROPIC_FOUNDRY_RESOURCE"):
        _build()


def test_base_url_alone_is_enough(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_BASE_URL", "https://example.invalid/anthropic/")
    assert isinstance(_build(), AnthropicFoundryModel)


def test_missing_credentials_fail_loudly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_API_KEY", raising=False)
    with pytest.raises(Exception):
        _build()


# --- usage accounting ---------------------------------------------------

def _meta(usage):
    return AnthropicFoundryModel(
        model_id="claude-fable-5-1", max_tokens=1024,
    ).format_chunk({"type": "metadata", "usage": usage})["metadata"]["usage"]


def test_cache_read_tokens_are_preserved():
    u = _meta({"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 900})
    assert u["cacheReadInputTokens"] == 900
    assert u["inputTokens"] == 1000 and u["outputTokens"] == 50


def test_totals_still_match_the_stock_adapter():
    u = _meta({"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 900})
    assert u["totalTokens"] == 1050


def test_absent_cache_field_is_omitted_not_zeroed():
    # A zero would claim "nothing was cached"; absent means "not reported".
    assert "cacheReadInputTokens" not in _meta({"input_tokens": 10, "output_tokens": 2})


def test_cost_uses_the_cache_rate_for_cached_tokens():
    from sana_evaluation.helper.result import MODEL_PRICING as P
    p = P["claude-fable-5-1"]
    cost = (p["input"] * 100 / 1e6
            + p["cache_read_input"] * 900 / 1e6
            + p["output"] * 50 / 1e6)
    # 100 uncached @ $10, 900 cached @ $0.25, 50 out @ $50
    assert round(cost, 8) == round(0.001 + 0.000225 + 0.0025, 8)


def test_non_metadata_chunks_fall_through_to_the_parent():
    m = AnthropicFoundryModel(model_id="claude-fable-5-1", max_tokens=1024)
    assert m.format_chunk({"type": "message_stop", "message": {"stop_reason": "end_turn"}}) == {
        "messageStop": {"stopReason": "end_turn"}
    }
