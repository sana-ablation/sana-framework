"""
Planning tool for standard profile mode.

plan — save a research plan to the agent's working context so it persists
       across turns. The agent writes the plan itself; this tool just injects
       it into agent.system_prompt. Call again to update the plan.
"""
from strands import tool
from strands.types.tools import ToolContext

from sana_evaluation.helper.prompt_sections import upsert_prompt_section


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool(context=True)
def plan(plan_text: str, tool_context: ToolContext) -> str:
    """Save your research plan so it persists across turns. Call again to update.

    Args:
        plan_text: Your written plan.
    """
    agent = tool_context.agent
    agent.system_prompt = upsert_prompt_section(
        agent.system_prompt,
        "CURRENT PLAN",
        plan_text,
    )
    return "Plan recorded."


__all__ = ["plan"]
