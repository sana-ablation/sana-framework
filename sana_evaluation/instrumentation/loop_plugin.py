import logging
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler, ToolSteeringAction
from sana_evaluation.tools.agent_tools import get_submitted_answer

logger = logging.getLogger(__name__)

class CategoryStagnationHandler(SteeringHandler):
    name = "category-stagnation"

    def __init__(self, max_consecutive_category: int = 4):
        super().__init__()
        self.max_consecutive = max_consecutive_category
        self.last_category = None
        self.consecutive_count = 0
        
        # Track total times we had to intervene in this run
        self.total_loop_interventions = 0 

        # Define your tool buckets and per-category thresholds.
        # execute/inspect tasks naturally chain (query multiple datasets in a row),
        # so they get a higher threshold than search.
        self.tool_categories = {
            "search": {
                "search",
                "search_keyword",
                "search_prefix",
                "search_value",
                "search_schema",
                "search_reranked",
                "search_ideal",
            },
            "inspect": {"list_files", "peek_file", "peek_multiple", "read_file"},
            "execute": {
                "execute_code",
                "execute_ideal",
                "query_file",
                "query_ideal",
                "parse_xml_records",
                "grep_file",
            },
        }
        self.category_thresholds = {
            "search": max_consecutive_category,
            "inspect": max_consecutive_category,
            "execute": max_consecutive_category,
        }

    def _get_category(self, tool_name: str) -> str:
        for category, tools in self.tool_categories.items():
            if tool_name in tools:
                return category
        return "other"

    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        tool_name = tool_use.get("name")

        if get_submitted_answer() is not None:
            return Proceed(reason="answer already submitted; no further steering")
        
        if tool_name in ["submit_answer", "plan", "plan_ideal", "download"]:
            self.last_category = "terminal_or_plan"
            self.consecutive_count = 0
            return Proceed(reason="Allowed tool")

        current_category = self._get_category(tool_name)

        if current_category == self.last_category:
            self.consecutive_count += 1
        else:
            self.last_category = current_category
            self.consecutive_count = 1

        threshold = self.category_thresholds.get(current_category, self.max_consecutive)
        if self.consecutive_count > threshold:
            display_count = self.consecutive_count
            # Increment the counter and log to the console/file
            self.total_loop_interventions += 1
            logger.info(
                f"[Stagnation] Nudging agent: '{current_category}' called {display_count} times "
                f"(intervention #{self.total_loop_interventions})"
            )
            
            self.consecutive_count = 0 
            
            return Guide(
                reason=(
                    f"Note: you have used '{current_category}' tools {display_count} times in a row. "
                    "If you are making steady progress, continue as planned — just retry the tool call you were about to make. "
                    "If you feel stuck, use the active planning tool to reassess. It does not count toward the tool-call limit."
                )
            )

        return Proceed(reason="No categorical stagnation detected")
