"""Claude on Microsoft Foundry (Azure), with cache-token usage preserved.

Strands ships no Foundry provider, so this subclasses its Anthropic model and
swaps the transport. Two departures from the stock adapter:

  * the client is ``AsyncAnthropicFoundry`` rather than ``AsyncAnthropic``;
  * ``format_chunk`` restores ``cacheReadInputTokens``.

The second matters more than it looks. The stock adapter reports only
input/output/total, exactly as the OpenAI adapters did before
``openai_cached_model`` restored their cache counts. On Claude Fable 5.1 a
cached input token costs $0.25/MTok against $10.00/MTok uncached -- a factor of
40 -- so an eval that drops the cached share overstates its own spend by nearly
that much on a workload with a high hit rate. This repo's runs sit around 90%.
"""

from __future__ import annotations

from typing import Any

import anthropic
from strands.models._validation import validate_config_keys
from strands.models.anthropic import AnthropicModel
from strands.types.streaming import StreamEvent


class AnthropicFoundryModel(AnthropicModel):
    """Anthropic Messages API over a Microsoft Foundry deployment."""

    def __init__(self, *, client_args: dict[str, Any] | None = None, **model_config: Any) -> None:
        # Not super().__init__: that constructs AsyncAnthropic, which rejects
        # Foundry's client arguments (resource, azure_ad_token_provider).
        validate_config_keys(model_config, self.AnthropicConfig)
        self.config = AnthropicModel.AnthropicConfig(**model_config)
        self.client = anthropic.AsyncAnthropicFoundry(**(client_args or {}))

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        if event.get("type") != "metadata":
            return super().format_chunk(event)

        usage = event["usage"]
        payload: dict[str, Any] = {
            "inputTokens": usage["input_tokens"],
            "outputTokens": usage["output_tokens"],
            "totalTokens": usage["input_tokens"] + usage["output_tokens"],
        }
        # Anthropic reports cache reads and cache writes separately. Only the
        # read is priced differently enough to matter here; the write is billed
        # at a premium over the uncached rate and is already inside
        # input_tokens, so it needs no separate field.
        cached = usage.get("cache_read_input_tokens")
        if cached is not None:
            payload["cacheReadInputTokens"] = int(cached)

        return {"metadata": {"usage": payload, "metrics": {"latencyMs": 0}}}
