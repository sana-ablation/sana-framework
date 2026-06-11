"""Ideal planning helpers and tool surface.

In ideal management mode, the gold reasoning chain is preloaded into the
system prompt before the first model turn. The `plan_ideal` tool then saves
the agent-written execution plan so it persists across turns.
"""

from __future__ import annotations

from typing import Any, Iterable

from strands import tool
from strands.types.tools import ToolContext

_BASE_PROMPT: str = ""


def format_reasoning_chain(reasoning_chain: Any) -> str:
    """Render task reasoning_chain into stable prompt text."""
    if reasoning_chain is None:
        return ""
    if isinstance(reasoning_chain, str):
        text = reasoning_chain.strip()
        return text
    if isinstance(reasoning_chain, Iterable):
        lines = [str(item).strip() for item in reasoning_chain if str(item).strip()]
        return "\n".join(lines)
    return str(reasoning_chain).strip()


def inject_reasoning_chain_prompt(
    system_prompt: str,
    reasoning_chain: Any,
    *,
    task_id: str = "",
) -> str:
    """Append the gold reasoning chain as planning context for ideal mode."""
    _ = task_id  # Keep a stable section heading regardless of task id.
    chain_text = format_reasoning_chain(reasoning_chain)
    if not chain_text:
        return system_prompt

    section = (
        "\n\n## GOLD REASONING CHAIN\n"
        "Treat this as the canonical planning context for the task. "
        "Base any ideal-mode execution plan on this chain, keep its order intact, "
        "and keep the saved plan brief and action-oriented.\n"
        f"{chain_text}"
    )
    return system_prompt.rstrip() + section


@tool(context=True)
def plan_ideal(plan_text: str, tool_context: ToolContext) -> str:
    """Save the current ideal execution plan so it persists across turns."""
    global _BASE_PROMPT

    agent = tool_context.agent
    current = agent.system_prompt
    if not _BASE_PROMPT or "\n\n## IDEAL EXECUTION PLAN\n" not in current:
        _BASE_PROMPT = current
    agent.system_prompt = _BASE_PROMPT + "\n\n## IDEAL EXECUTION PLAN\n" + str(plan_text).strip()
    return "Ideal execution plan recorded."


__all__ = [
    "plan_ideal",
    "format_reasoning_chain",
    "inject_reasoning_chain_prompt",
]
