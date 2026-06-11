"""Plugin to enforce a hard cap on search tool calls."""

from __future__ import annotations

from typing import Iterable, Optional, Set

from strands.hooks import AfterToolCallEvent, AgentInitializedEvent
from strands.plugins import hook
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler, ToolSteeringAction
from sana_evaluation.tools.agent_tools import get_submitted_answer

_DEFAULT_SEARCH_TOOLS = {
    "search_value",
    "search_schema",
    "search_reranked",
    "search_prefix",
    "search_ideal",
}


class SearchCallBudgetHandler(SteeringHandler):
    """Block additional search calls after a fixed search budget is consumed."""

    name = "search-call-budget"

    def __init__(
        self,
        max_search_calls: int,
        search_tools: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self._max = int(max_search_calls)
        self._count = 0
        self._search_tools: Set[str] = set(search_tools or _DEFAULT_SEARCH_TOOLS)

    @hook
    def on_agent_initialized(self, event: AgentInitializedEvent) -> None:
        self._count = 0

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "")
        if tool_name in self._search_tools:
            self._count += 1

    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        if get_submitted_answer() is not None:
            return Proceed(reason="answer already submitted; no further steering")

        if self._max <= 0:
            return Proceed(reason="search call budget disabled")

        tool_name = tool_use.get("name", "")
        if tool_name == "submit_answer":
            return Proceed(reason="submit_answer is always allowed")

        if tool_name not in self._search_tools:
            return Proceed(reason="non-search tool is allowed")

        if self._count < self._max:
            return Proceed(reason=f"search calls remaining: {self._max - self._count}")

        return Guide(
            reason=(
                f"Search call budget exhausted ({self._count}/{self._max}). "
                "Do not call any search tools again. "
                "Use non-search tools to finish (list/read/query/grep/download/execute_code), "
                "then call submit_answer."
            )
        )
