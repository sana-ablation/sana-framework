"""Conversation-manager helpers for the Strands evaluation harness."""

from __future__ import annotations

from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

from sana_evaluation.config import RunConfig

TECHNICAL_SUMMARIZATION_PROMPT = """
You are summarizing a tool-using data-analysis conversation.

Create a concise bullet-point summary that:
- Preserves datasets inspected, file paths, SQL queries, and important tool findings
- Preserves verified facts, failed approaches, and important constraints
- Preserves the best current answer candidate and remaining uncertainty
- Omits conversational filler and meta commentary
- Uses precise technical language

Format as bullet points only.
""".strip()


def build_conversation_manager(run_config: RunConfig):
    strategy = (run_config.conversation_manager_strategy or "summarizing").strip().lower()
    if strategy == "sliding_window":
        return SlidingWindowConversationManager(
            window_size=run_config.sliding_window_k,
            per_turn=True,
        )
    if strategy != "summarizing":
        raise ValueError(f"Unsupported conversation_manager_strategy: {run_config.conversation_manager_strategy}")
    return SummarizingConversationManager(
        summary_ratio=run_config.summary_ratio,
        preserve_recent_messages=run_config.preserve_recent_messages,
        summarization_system_prompt=TECHNICAL_SUMMARIZATION_PROMPT,
    )
