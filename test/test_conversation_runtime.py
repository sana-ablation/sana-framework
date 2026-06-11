import time
import unittest
from types import SimpleNamespace

from strands.agent.conversation_manager import (
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

from sana_evaluation.config import RunConfig
from sana_evaluation.helper.agent_runtime import invoke_with_watchdog
from sana_evaluation.helper.conversation import (
    TECHNICAL_SUMMARIZATION_PROMPT,
    build_conversation_manager,
)


class SlowAgent:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __call__(self, prompt):
        time.sleep(0.1)
        stop_reason = "cancelled" if self.cancelled else "end_turn"
        return SimpleNamespace(stop_reason=stop_reason)


class CancelIgnoringAgent:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __call__(self, prompt):
        time.sleep(0.1)
        return SimpleNamespace(stop_reason="end_turn")


class FastAgent:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def __call__(self, prompt):
        return SimpleNamespace(stop_reason="end_turn")


class TestConversationManagerBuilder(unittest.TestCase):
    def test_summarizing_is_default_and_uses_requested_settings(self):
        manager = build_conversation_manager(RunConfig())

        self.assertIsInstance(manager, SummarizingConversationManager)
        self.assertEqual(manager.summary_ratio, 0.4)
        self.assertEqual(manager.preserve_recent_messages, 12)
        self.assertEqual(manager.summarization_system_prompt, TECHNICAL_SUMMARIZATION_PROMPT)

    def test_sliding_window_strategy_still_supported(self):
        manager = build_conversation_manager(
            RunConfig(conversation_manager_strategy="sliding_window", sliding_window_k=17)
        )

        self.assertIsInstance(manager, SlidingWindowConversationManager)
        self.assertEqual(manager.window_size, 17)
        self.assertTrue(manager.per_turn)


class TestWatchdogInvocation(unittest.TestCase):
    def test_watchdog_cancels_runaway_invocation(self):
        agent = SlowAgent()
        outcome = invoke_with_watchdog(
            agent,
            "question",
            hard_deadline=time.time() + 0.02,
            timeout_seconds=1,
            submit_grace_seconds=1,
        )
        self.assertTrue(outcome.timed_out)
        self.assertIn("Hard timeout reached", outcome.timeout_reason or "")
        self.assertTrue(agent.cancelled)

    def test_watchdog_reports_timeout_even_if_agent_ignores_cancel_stop_reason(self):
        agent = CancelIgnoringAgent()
        outcome = invoke_with_watchdog(
            agent,
            "question",
            hard_deadline=time.time() + 0.02,
            timeout_seconds=1,
            submit_grace_seconds=1,
        )

        self.assertTrue(outcome.timed_out)
        self.assertIn("Hard timeout reached", outcome.timeout_reason or "")
        self.assertTrue(agent.cancelled)
        self.assertEqual(outcome.response.stop_reason, "end_turn")

    def test_watchdog_allows_fast_invocation(self):
        agent = FastAgent()
        outcome = invoke_with_watchdog(
            agent,
            "question",
            hard_deadline=time.time() + 1.0,
            timeout_seconds=1,
            submit_grace_seconds=1,
        )
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.response.stop_reason, "end_turn")
        self.assertFalse(agent.cancelled)


if __name__ == "__main__":
    unittest.main()
