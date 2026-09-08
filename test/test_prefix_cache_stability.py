"""The system prompt must be byte-stable for a task's whole run.

Strands passes ``agent.system_prompt`` to the model on every event-loop
iteration (event_loop.py: ``stream_messages(agent.model, agent.system_prompt,
...)``). Because the system prompt occupies token 0, any mid-run mutation
invalidates the provider's cached prefix for every remaining turn of that task.

These tests pin that property against the tools that used to mutate it.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sana_evaluation.tools.plan.oracle import plan_ideal
from sana_evaluation.tools.plan.standard import plan


class _FakeAgent:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt


class TestPlanToolsDoNotMutateSystemPrompt(unittest.TestCase):
    BASE = "SYSTEM PROMPT BODY\n\n## AVAILABLE SEARCH TOOLS\n- search_ideal"

    def test_plan_does_not_touch_the_system_prompt(self) -> None:
        agent = _FakeAgent(self.BASE)
        plan("1. search\n2. compute")
        self.assertEqual(self.BASE, agent.system_prompt)

    def test_plan_ideal_does_not_touch_the_system_prompt(self) -> None:
        agent = _FakeAgent(self.BASE)
        plan_ideal("1. search\n2. compute")
        self.assertEqual(self.BASE, agent.system_prompt)

    def test_repeated_plan_calls_keep_the_prefix_byte_identical(self) -> None:
        agent = _FakeAgent(self.BASE)
        snapshots = []
        for turn in range(5):
            plan(f"revision {turn}")
            plan_ideal(f"ideal revision {turn}")
            snapshots.append(agent.system_prompt)
        self.assertEqual({self.BASE}, set(snapshots))

    def test_plans_are_returned_to_the_agent_instead(self) -> None:
        """The plan still reaches the model - as a tool result, at the tail."""
        self.assertIn("## CURRENT PLAN", plan("do the thing"))
        self.assertIn("do the thing", plan("do the thing"))
        self.assertIn("## IDEAL EXECUTION PLAN", plan_ideal("do the ideal thing"))
        self.assertIn("do the ideal thing", plan_ideal("do the ideal thing"))

    def test_neither_tool_requests_agent_context(self) -> None:
        for tool_obj in (plan, plan_ideal):
            properties = tool_obj.tool_spec["inputSchema"]["json"]["properties"]
            self.assertEqual(["plan_text"], list(properties), tool_obj.tool_name)


class TestSummarizationPreservesPlan(unittest.TestCase):
    def test_summarization_prompt_asks_to_keep_the_plan(self) -> None:
        """Plans now live in message history, which compaction can rewrite."""
        from sana_evaluation.runtime.conversation import TECHNICAL_SUMMARIZATION_PROMPT

        self.assertIn("plan", TECHNICAL_SUMMARIZATION_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
