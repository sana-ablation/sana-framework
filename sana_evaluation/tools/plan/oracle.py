"""Ideal planning helpers and tool surface.

In ideal management mode, the gold reasoning chain is preloaded into the
system prompt before the first model turn. The `plan_ideal` tool then records
the agent-written execution plan so it persists across turns.

The saved plan is deliberately NOT written into ``agent.system_prompt``.
Strands re-reads the system prompt on every event-loop iteration, so mutating
it mid-run invalidates the provider's cached prefix from token 0 for every
remaining turn. Returning the plan as a tool result appends to the end of the
message list instead, which leaves the cacheable prefix intact.

``inject_reasoning_chain_prompt`` still composes prompt text, but it is applied
once during bundle construction (into ``ModeBundle.task_trailer``) before the
first model call, not mid-run.
"""

from __future__ import annotations

from typing import Any, Iterable

from strands import tool

# Heading kept identical to the previous system-prompt section so prompts,
# traces, and analysis that grep for it keep working.
IDEAL_PLAN_SECTION_HEADING = "## IDEAL EXECUTION PLAN"


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


@tool
def plan_ideal(plan_text: str) -> str:
    """Save the current ideal execution plan so it persists across turns."""
    body = str(plan_text or "").strip()
    if not body:
        return "Ideal execution plan not recorded: plan_text was empty."
    return f"Ideal execution plan recorded.\n\n{IDEAL_PLAN_SECTION_HEADING}\n{body}"


__all__ = [
    "plan_ideal",
    "format_reasoning_chain",
    "inject_reasoning_chain_prompt",
    "IDEAL_PLAN_SECTION_HEADING",
]
