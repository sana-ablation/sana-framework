"""Repo-local OpenAI model wrappers that preserve cached-token usage details.

Strands' OpenAI providers already support optional cache token fields in the
internal ``Usage`` shape, but the stock OpenAI adapters currently flatten the
raw usage payloads and drop cache-read counts. These wrappers restore that
detail so repo-local cost accounting can price cached prompt tokens correctly.
"""

from __future__ import annotations

from typing import Any

from strands.models.openai import OpenAIModel
from strands.models.openai_responses import OpenAIResponsesModel
from strands.types.streaming import StreamEvent


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_chat_cached_input_tokens(usage: Any) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    return _coerce_int(getattr(details, "cached_tokens", None))


def extract_responses_cached_input_tokens(usage: Any) -> int | None:
    details = getattr(usage, "input_tokens_details", None)
    return _coerce_int(getattr(details, "cached_tokens", None))


class OpenAICachedUsageModel(OpenAIModel):
    """OpenAI Chat Completions provider that keeps cached input token counts."""

    def format_chunk(self, event: dict[str, Any], **kwargs: Any) -> StreamEvent:
        if event.get("chunk_type") != "metadata":
            return super().format_chunk(event, **kwargs)

        usage_payload = {
            "inputTokens": event["data"].prompt_tokens,
            "outputTokens": event["data"].completion_tokens,
            "totalTokens": event["data"].total_tokens,
        }
        cached_tokens = extract_chat_cached_input_tokens(event["data"])
        if cached_tokens is not None:
            usage_payload["cacheReadInputTokens"] = cached_tokens

        return {
            "metadata": {
                "usage": usage_payload,
                "metrics": {
                    "latencyMs": 0,  # TODO
                },
            }
        }


class OpenAIResponsesCachedUsageModel(OpenAIResponsesModel):
    """Responses API provider that keeps cached input token counts."""

    def _format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        if event.get("chunk_type") != "metadata":
            return super()._format_chunk(event)

        usage_payload = {
            "inputTokens": getattr(event["data"], "input_tokens", 0),
            "outputTokens": getattr(event["data"], "output_tokens", 0),
            "totalTokens": getattr(event["data"], "total_tokens", 0),
        }
        cached_tokens = extract_responses_cached_input_tokens(event["data"])
        if cached_tokens is not None:
            usage_payload["cacheReadInputTokens"] = cached_tokens

        return {
            "metadata": {
                "usage": usage_payload,
                "metrics": {
                    "latencyMs": 0,  # TODO
                },
            }
        }
