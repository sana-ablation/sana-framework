"""Runtime helpers shared by the Strands agent runners."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass
class InvocationOutcome:
    response: Any | None
    timed_out: bool = False
    timeout_reason: Optional[str] = None


def invoke_with_watchdog(
    agent: Any,
    prompt: str,
    *,
    hard_deadline: float,
    timeout_seconds: int,
    submit_grace_seconds: int,
) -> InvocationOutcome:
    """Invoke an agent with a hard watchdog cancel after timeout + grace."""
    remaining = hard_deadline - time.time()
    reason = (
        f"Hard timeout reached ({timeout_seconds}s timeout + "
        f"{submit_grace_seconds}s submit grace exhausted)."
    )

    if remaining <= 0:
        return InvocationOutcome(response=None, timed_out=True, timeout_reason=reason)

    triggered = False

    def _cancel() -> None:
        nonlocal triggered
        triggered = True
        logger.warning(reason + " Cancelling agent invocation.")
        agent.cancel()

    timer = threading.Timer(remaining, _cancel)
    timer.daemon = True
    timer.start()

    try:
        response = agent(prompt)
    except Exception as exc:
        if triggered:
            logger.warning("Agent raised after watchdog cancellation: %s", exc)
            return InvocationOutcome(response=None, timed_out=True, timeout_reason=reason)
        raise
    finally:
        timer.cancel()

    if triggered:
        return InvocationOutcome(response=response, timed_out=True, timeout_reason=reason)

    return InvocationOutcome(response=response, timed_out=False, timeout_reason=None)
