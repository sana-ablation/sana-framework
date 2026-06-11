import asyncio
import unittest

from sana_evaluation.instrumentation.loop_plugin import CategoryStagnationHandler
from sana_evaluation.tools import agent_tools


class TestCategoryStagnationHandler(unittest.TestCase):
    def test_guide_reports_true_consecutive_count_before_reset(self):
        handler = CategoryStagnationHandler(max_consecutive_category=2)

        asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "execute_code"}))
        asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "query_ideal"}))
        action = asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "execute_ideal"}))

        self.assertEqual(action.type, "guide")
        self.assertIn("used 'execute' tools 3 times in a row", action.reason)
        self.assertEqual(handler.consecutive_count, 0)

    def test_stagnation_guide_uses_generic_planning_tool_name(self):
        handler = CategoryStagnationHandler(max_consecutive_category=1)

        asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "execute_code"}))
        action = asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "query_ideal"}))

        self.assertEqual(action.type, "guide")
        self.assertIn("planning tool", action.reason)
        self.assertNotIn("`plan` tool", action.reason)

    def test_submitted_answer_short_circuits_stagnation_steering(self):
        handler = CategoryStagnationHandler(max_consecutive_category=1)

        agent_tools.clear_submitted_answer()
        try:
            agent_tools.submit_answer("[42]", "done")
            action = asyncio.run(handler.steer_before_tool(agent=None, tool_use={"name": "execute_code"}))
        finally:
            agent_tools.clear_submitted_answer()

        self.assertEqual(action.type, "proceed")
        self.assertIn("already submitted", action.reason)


if __name__ == "__main__":
    unittest.main()
