"""
Planning tool for standard profile mode.

plan — record a research plan. The agent writes the plan itself; this tool
       echoes it back as the tool result so it persists in the message
       history. Call again to update the plan.

The plan is deliberately NOT written into ``agent.system_prompt``. Strands
re-reads the system prompt on every event-loop iteration, so mutating it
mid-run invalidates the provider's cached prefix from token 0 for every
remaining turn. Returning the plan as a tool result appends to the end of the
message list instead, which leaves the cacheable prefix intact.
"""
from strands import tool

# Heading kept identical to the previous system-prompt section so prompts,
# traces, and analysis that grep for it keep working.
PLAN_SECTION_HEADING = "## CURRENT PLAN"


@tool
def plan(plan_text: str) -> str:
    """Save your research plan so it persists across turns. Call again to update.

    Args:
        plan_text: Your written plan.
    """
    body = (plan_text or "").strip()
    if not body:
        return "Plan not recorded: plan_text was empty."
    return f"Plan recorded.\n\n{PLAN_SECTION_HEADING}\n{body}"


__all__ = ["plan", "PLAN_SECTION_HEADING"]
