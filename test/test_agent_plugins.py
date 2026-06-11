import asyncio
import time
import unittest
from types import SimpleNamespace

from strands.hooks.events import (
    AfterModelCallEvent,
    AgentInitializedEvent,
    BeforeModelCallEvent,
)
from sana_evaluation.config import AgentConfig
from sana_evaluation.instrumentation.agent_plugins import LoggingPlugin, ToolLimitSteeringHandler
from sana_evaluation.instrumentation.search_call_budget_plugin import SearchCallBudgetHandler
from sana_evaluation.tools import agent_tools, agent_tools_v2


class FakeModel:
    def __init__(self, config):
        self._config = config

    def get_config(self):
        return self._config

    def update_config(self, **model_config):
        self._config.update(model_config)


class TestToolLimitSteeringHandler(unittest.TestCase):
    def _make_agent(self, max_tokens=8096):
        return SimpleNamespace(
            model=FakeModel(
                {
                    "model_id": "gpt-5.2",
                    "params": {"max_completion_tokens": max_tokens},
                }
            )
        )

    def test_max_tokens_retry_guidance_lowers_retry_cap(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=9999)
        handler._start_time = time.time()
        agent = self._make_agent()

        action = asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "too long"}]},
                stop_reason="max_tokens",
            )
        )

        self.assertEqual(action.type, "guide")
        self.assertIn("hit the max token limit", action.reason)
        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 4096)

    def test_repeated_max_tokens_forces_submit_only_cap(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=9999, max_model_guides=2)
        handler._start_time = time.time()
        agent = self._make_agent()

        asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "too long"}]},
                stop_reason="max_tokens",
            )
        )
        action = asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "still too long"}]},
                stop_reason="max_tokens",
            )
        )

        self.assertEqual(action.type, "guide")
        self.assertIn("Call submit_answer NOW", action.reason)
        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 2048)

    def test_retry_cap_never_increases_a_smaller_user_cap(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=9999)
        handler._start_time = time.time()
        agent = self._make_agent(max_tokens=2048)

        asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "too long"}]},
                stop_reason="max_tokens",
            )
        )

        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 2048)

    def test_timeout_plain_text_turn_switches_to_submit_only(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=1)
        handler._start_time = time.time() - 5
        agent = self._make_agent()

        action = asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "final answer in prose"}]},
                stop_reason="end_turn",
            )
        )

        self.assertEqual(action.type, "guide")
        self.assertIn("Timeout reached", action.reason)
        self.assertIn("submit_answer NOW", action.reason)
        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 2048)

    def test_cancelled_stop_after_submission_does_not_reenter_guidance(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=1)
        handler._start_time = time.time() - 5
        agent = self._make_agent()

        agent_tools.clear_submitted_answer()
        try:
            agent_tools.submit_answer("[Ward 5]", "done")
            action = asyncio.run(
                handler.steer_after_model(
                    agent=agent,
                    message={"role": "assistant", "content": [{"text": "Cancelled by user"}]},
                    stop_reason="cancelled",
                )
            )
        finally:
            agent_tools.clear_submitted_answer()

        self.assertEqual(action.type, "proceed")
        self.assertIn("already submitted", action.reason)
        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 8096)

    def test_cancelled_stop_without_submission_does_not_reenter_guidance(self):
        handler = ToolLimitSteeringHandler(timeout_seconds=1)
        handler._start_time = time.time() - 5
        agent = self._make_agent()

        action = asyncio.run(
            handler.steer_after_model(
                agent=agent,
                message={"role": "assistant", "content": [{"text": "Cancelled by user"}]},
                stop_reason="cancelled",
            )
        )

        self.assertEqual(action.type, "proceed")
        self.assertIn("agent cancelled", action.reason)
        self.assertEqual(agent.model.get_config()["params"]["max_completion_tokens"], 8096)


class TestSearchCallBudgetHandler(unittest.TestCase):
    def test_submitted_answer_short_circuits_search_budget(self):
        handler = SearchCallBudgetHandler(max_search_calls=1)

        agent_tools.clear_submitted_answer()
        try:
            agent_tools.submit_answer("[42]", "done")
            action = asyncio.run(
                handler.steer_before_tool(
                    agent=None,
                    tool_use={"name": "search_ideal"},
                )
            )
        finally:
            agent_tools.clear_submitted_answer()

        self.assertEqual(action.type, "proceed")
        self.assertIn("already submitted", action.reason)

    def test_search_call_budget_still_counts_search_when_tool_limit_excludes_it(self):
        tool_limit = ToolLimitSteeringHandler(
            max_tool_calls=1,
            timeout_seconds=9999,
            excluded_tools=("search_ideal",),
        )
        search_budget = SearchCallBudgetHandler(max_search_calls=1, search_tools=("search_ideal",))
        agent = SimpleNamespace()
        tool_limit.on_agent_initialized(AgentInitializedEvent(agent=agent))
        search_budget.on_agent_initialized(AgentInitializedEvent(agent=agent))

        event = SimpleNamespace(tool_use={"name": "search_ideal"})
        tool_limit.on_after_tool(event)
        search_budget.on_after_tool(event)

        tool_limit_action = asyncio.run(
            tool_limit.steer_before_tool(agent=agent, tool_use={"name": "peek_file"})
        )
        search_budget_action = asyncio.run(
            search_budget.steer_before_tool(agent=agent, tool_use={"name": "search_ideal"})
        )

        self.assertEqual(tool_limit_action.type, "proceed")
        self.assertEqual(search_budget_action.type, "guide")
        self.assertIn("Search call budget exhausted", search_budget_action.reason)


class TestTokenBudgetDefaults(unittest.TestCase):
    def test_agent_config_default_max_tokens_is_8096(self):
        self.assertEqual(AgentConfig().max_tokens, 8096)

    def test_tool_result_caps_are_reduced(self):
        self.assertEqual(agent_tools._TOOL_RESULT_CHAR_CAP, 6_000)
        self.assertEqual(agent_tools_v2._TOOL_RESULT_CHAR_CAP, 6_000)


class TestLoggingPlugin(unittest.TestCase):
    def test_logs_final_text_response_from_after_model(self):
        plugin = LoggingPlugin()
        agent = SimpleNamespace()
        plugin.on_agent_initialized(AgentInitializedEvent(agent=agent))
        plugin.on_before_model(BeforeModelCallEvent(agent=agent, invocation_state={}))

        event = AfterModelCallEvent(
            agent=agent,
            invocation_state={},
            stop_response=AfterModelCallEvent.ModelStopResponse(
                message={"role": "assistant", "content": [{"text": "final answer text"}]},
                stop_reason="end_turn",
            ),
        )

        with self.assertLogs("sana_evaluation.instrumentation.agent_plugins", level="DEBUG") as logs:
            plugin.on_after_model(event)

        output = "\n".join(logs.output)
        self.assertIn("Model response #1 [stop_reason=end_turn]", output)
        self.assertIn("[role=assistant block=1 text] final answer text", output)

    def test_logs_final_tool_use_response_from_after_model(self):
        plugin = LoggingPlugin()
        agent = SimpleNamespace()
        plugin.on_agent_initialized(AgentInitializedEvent(agent=agent))
        plugin.on_before_model(BeforeModelCallEvent(agent=agent, invocation_state={}))

        event = AfterModelCallEvent(
            agent=agent,
            invocation_state={},
            stop_response=AfterModelCallEvent.ModelStopResponse(
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "name": "search_ideal",
                                "toolUseId": "tool-123",
                                "input": {"query": "district 3 school"},
                            }
                        }
                    ],
                },
                stop_reason="tool_use",
            ),
        )

        with self.assertLogs("sana_evaluation.instrumentation.agent_plugins", level="DEBUG") as logs:
            plugin.on_after_model(event)

        output = "\n".join(logs.output)
        self.assertIn("Model response #1 [stop_reason=tool_use]", output)
        self.assertIn("[role=assistant block=1 tool_use] search_ideal(", output)
        self.assertIn("\"query\": \"district 3 school\"", output)
        self.assertIn("[toolUseId=tool-123]", output)


if __name__ == "__main__":
    unittest.main()
