"""Shared agent plugins and callback utilities.

This module centralizes plugin implementations used by both
`agent.py` and `agent_with_mode.py` to avoid duplicated logic.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterable, List

from strands import Plugin
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    AgentInitializedEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.plugins import hook
from strands.vended_plugins.steering import Guide, ModelSteeringAction, Proceed, SteeringHandler, ToolSteeringAction

from sana_evaluation.tools.agent_tools import get_submitted_answer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback handler state
# ---------------------------------------------------------------------------

_turn_count = 0
_run_start_time = 0.0
_reasoning_buf: List[str] = []
_text_buf: List[str] = []


def _flush_buffers() -> None:
    global _reasoning_buf, _text_buf
    if _reasoning_buf:
        full_reasoning = "".join(_reasoning_buf)
        logger.debug("Reasoning: %s", full_reasoning)
        _reasoning_buf = []
    if _text_buf:
        full_text = "".join(_text_buf)
        logger.debug("LLM content: %s", full_text)
        _text_buf = []


def _json_dumps_safe(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _iter_message_log_lines(message: dict[str, Any]) -> List[str]:
    role = str(message.get("role", "?"))
    content = message.get("content", [])
    if not isinstance(content, list) or not content:
        return [f"[role={role}] <empty>"]

    lines: List[str] = []
    for idx, block in enumerate(content, start=1):
        if not isinstance(block, dict):
            lines.append(f"[role={role} block={idx} unknown] {block}")
            continue

        reasoning = block.get("reasoningContent")
        if isinstance(reasoning, dict):
            reasoning_text = reasoning.get("reasoningText")
            if isinstance(reasoning_text, dict) and reasoning_text.get("text"):
                lines.append(f"[role={role} block={idx} reasoning] {reasoning_text['text']}")
            redacted = reasoning.get("redactedContent")
            if redacted is not None:
                redacted_size = len(redacted) if isinstance(redacted, (bytes, bytearray)) else "?"
                lines.append(f"[role={role} block={idx} reasoning_redacted] bytes={redacted_size}")

        if "text" in block:
            lines.append(f"[role={role} block={idx} text] {block['text']}")

        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict):
            tool_name = tool_use.get("name", "?")
            tool_id = tool_use.get("toolUseId", "?")
            tool_input = _json_dumps_safe(tool_use.get("input", {}))
            lines.append(
                f"[role={role} block={idx} tool_use] {tool_name}({tool_input}) [toolUseId={tool_id}]"
            )

        tool_result = block.get("toolResult")
        if isinstance(tool_result, dict):
            status = tool_result.get("status", "?")
            tool_use_id = tool_result.get("toolUseId", "?")
            lines.append(
                f"[role={role} block={idx} tool_result] status={status} toolUseId={tool_use_id} "
                f"{_json_dumps_safe(tool_result)}"
            )

        if "citationsContent" in block:
            lines.append(f"[role={role} block={idx} citations] {_json_dumps_safe(block['citationsContent'])}")

        known_keys = {
            "reasoningContent",
            "text",
            "toolUse",
            "toolResult",
            "citationsContent",
            "cachePoint",
            "document",
            "guardContent",
            "image",
            "video",
        }
        unknown_payload = {key: value for key, value in block.items() if key not in known_keys}
        if unknown_payload:
            lines.append(f"[role={role} block={idx} extra] {_json_dumps_safe(unknown_payload)}")

    return lines or [f"[role={role}] <no-loggable-blocks>"]


def event_loop_tracker(**kwargs):
    """Track loop-turn progress and stream logs for reasoning/text chunks."""
    global _turn_count, _run_start_time, _reasoning_buf, _text_buf

    if kwargs.get("init_event_loop", False):
        _turn_count = 0
        _run_start_time = time.time()
        _reasoning_buf = []
        _text_buf = []
        logger.debug("Event loop initialized")
        return

    if kwargs.get("start_event_loop", False):
        _flush_buffers()
        _turn_count += 1
        elapsed = time.time() - _run_start_time
        logger.info("\n--- Turn %s (elapsed: %.1fs) ---", _turn_count, elapsed)
        return

    if kwargs.get("reasoning") and kwargs.get("reasoningText"):
        _reasoning_buf.append(kwargs["reasoningText"])
        return

    if "data" in kwargs and not kwargs.get("reasoning"):
        _text_buf.append(kwargs["data"])
        return

    raw_event = kwargs.get("event", {})
    if raw_event.get("contentBlockStop") is not None:
        _flush_buffers()
        return

    if kwargs.get("force_stop", False):
        _flush_buffers()
        logger.debug("Event loop force-stopped: %s", kwargs.get("force_stop_reason", "unknown"))


# ---------------------------------------------------------------------------
# Steering / plugins
# ---------------------------------------------------------------------------


class ToolLimitSteeringHandler(SteeringHandler):
    """Guide the agent to call submit_answer when tool limit or timeout is reached."""

    name = "tool-limit"

    def __init__(
        self,
        max_tool_calls: int = 30,
        timeout_seconds: int = 300,
        retry_max_tokens: int | None = 4096,
        submit_only_max_tokens: int | None = 2048,
        max_model_guides: int = 2,
        excluded_tools: Iterable[str] = ("skills", "plan"),
    ) -> None:
        super().__init__()
        self._max = max_tool_calls
        self._timeout = timeout_seconds
        self._retry_max_tokens = retry_max_tokens
        self._submit_only_max_tokens = submit_only_max_tokens
        self._max_model_guides = max_model_guides
        self._excluded_tools = frozenset(str(name) for name in excluded_tools)
        self._count = 0
        self._start_time = 0.0
        self._guided = False
        self._overflow = False
        self._model_guides_used = 0

    def signal_context_overflow(self) -> None:
        self._overflow = True

    @hook
    def on_agent_initialized(self, event: AgentInitializedEvent) -> None:
        self._count = 0
        self._start_time = time.time()
        self._guided = False
        self._overflow = False
        self._model_guides_used = 0

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        if event.tool_use.get("name") not in self._excluded_tools:
            self._count += 1
        self._model_guides_used = 0

    def _current_limit_reason(self) -> str | None:
        elapsed = time.time() - self._start_time
        if self._overflow:
            return "Context window overflow — a tool result was too large for context."
        if elapsed >= self._timeout:
            return f"Timeout reached ({elapsed:.1f}s elapsed)."
        if self._count >= self._max:
            return f"Tool limit reached ({self._count}/{self._max} calls used)."
        return None

    def _update_model_token_cap(self, agent: Any, max_tokens: int | None) -> bool:
        if max_tokens is None:
            return False

        model = getattr(agent, "model", None)
        if model is None or not hasattr(model, "get_config") or not hasattr(model, "update_config"):
            return False

        try:
            config = model.get_config()
        except Exception:
            return False

        if not isinstance(config, dict):
            return False

        params = config.get("params")
        if isinstance(params, dict):
            updated_params = dict(params)
            for key in ("max_completion_tokens", "max_output_tokens", "max_tokens"):
                if key in updated_params:
                    current_value = updated_params.get(key)
                    if isinstance(current_value, int):
                        max_tokens = min(current_value, max_tokens)
                    if current_value == max_tokens:
                        return True
                    updated_params[key] = max_tokens
                    model.update_config(params=updated_params)
                    logger.info("Adjusted model token cap to %s via params.%s", max_tokens, key)
                    return True

        for key in ("max_completion_tokens", "max_output_tokens", "max_tokens"):
            if key in config:
                current_value = config.get(key)
                if isinstance(current_value, int):
                    max_tokens = min(current_value, max_tokens)
                if current_value == max_tokens:
                    return True
                model.update_config(**{key: max_tokens})
                logger.info("Adjusted model token cap to %s via %s", max_tokens, key)
                return True

        return False

    async def steer_before_tool(self, *, agent, tool_use, **kwargs) -> ToolSteeringAction:
        if get_submitted_answer() is not None:
            return Proceed(reason="answer already submitted; no further steering")

        tool_name = tool_use.get("name")
        if tool_name == "submit_answer":
            return Proceed(reason="submit_answer is always allowed")
        if tool_name in self._excluded_tools:
            return Proceed(reason="excluded administrative tool")

        reason = self._current_limit_reason()
        if reason is None:
            return Proceed(reason="within limits")

        if self._guided:
            logger.warning("Hard-stopping after second limit trigger: %s", reason)
            return Guide(
                reason=(
                    f"{reason} You have already been warned. "
                    "Call submit_answer NOW with your best current answer. No other tools are permitted."
                )
            )

        self._guided = True
        return Guide(
            reason=(
                f"{reason} You must stop using other tools and immediately call submit_answer "
                "with your best current answer and reasoning. "
                "Do not call any other tool before submit_answer."
            )
        )

    async def steer_after_model(self, *, agent, message, stop_reason, **kwargs) -> ModelSteeringAction:
        if get_submitted_answer() is not None:
            self._model_guides_used = 0
            return Proceed(reason="answer already submitted")

        if stop_reason == "cancelled":
            self._model_guides_used = 0
            return Proceed(reason="agent cancelled")

        if stop_reason == "tool_use":
            self._model_guides_used = 0
            return Proceed(reason="model produced a tool call")

        reason = self._current_limit_reason()
        if reason is not None:
            self._update_model_token_cap(agent, self._submit_only_max_tokens)
            self._model_guides_used = 0
            return Guide(
                reason=(
                    f"{reason} Do not continue with plain-text analysis. "
                    "Call submit_answer NOW with your best current answer. "
                    "Keep the reasoning concise, but it may be longer if needed. "
                    "Do not produce any other response."
                )
            )

        if stop_reason != "max_tokens":
            self._model_guides_used = 0
            return Proceed(reason=f"model stopped with {stop_reason}")

        self._model_guides_used += 1
        if self._model_guides_used >= self._max_model_guides:
            self._update_model_token_cap(agent, self._submit_only_max_tokens)
            return Guide(
                reason=(
                    "Your recent responses keep hitting the max token limit and are being discarded. "
                    "Stop writing long prose. Call submit_answer NOW with your best current answer "
                    "and concise reasoning. The answer may still be long if required."
                )
            )

        self._update_model_token_cap(agent, self._retry_max_tokens)
        return Guide(
            reason=(
                "Your last response hit the max token limit and was discarded. "
                "Be concise. Either make exactly one tool call, or if you can finish now, "
                "call submit_answer. Do not write a long free-text analysis."
            )
        )


class SubmitAnswerPlugin(Plugin):
    """Stop the agent loop immediately after submit_answer is executed."""

    name = "submit-answer"

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        tool_name = getattr(event, "tool_use", {}).get("name", "")
        if tool_name == "submit_answer" or get_submitted_answer() is not None:
            logger.info("Answer submitted! Triggering native agent cancellation.")
            event.agent.cancel()


class LoggingPlugin(Plugin):
    """Log tool calls and normalized tool results."""

    name = "logging"

    def __init__(self) -> None:
        super().__init__()
        self._model_call_count = 0

    @hook
    def on_agent_initialized(self, event: AgentInitializedEvent) -> None:
        self._model_call_count = 0

    @hook
    def on_before_model(self, event: BeforeModelCallEvent) -> None:
        self._model_call_count += 1

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        tool_name = event.tool_use.get("name", "?")
        tool_input = event.tool_use.get("input", {})
        try:
            args_str = json.dumps(tool_input, ensure_ascii=False)
        except Exception:
            args_str = str(tool_input)
        logger.info("Executing: %s(%s)", tool_name, args_str)

    @hook
    def on_after_tool(self, event: AfterToolCallEvent) -> None:
        result = event.result
        content = result.get("content", [])
        parts = []
        is_logical_error = False

        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
                if not is_logical_error:
                    try:
                        parsed = json.loads(block["text"])
                        if isinstance(parsed, dict) and ("error" in parsed or parsed.get("success") is False):
                            is_logical_error = True
                    except (json.JSONDecodeError, TypeError):
                        pass

        if is_logical_error:
            event.result["status"] = "error"

        result_str = " ".join(parts) if parts else str(result)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "..."

        if is_logical_error:
            logger.warning("Tool logical error (status=error): %s", result_str)
        else:
            logger.debug("Tool result: %s", result_str)

    @hook
    def on_after_model(self, event: AfterModelCallEvent) -> None:
        if event.exception is not None:
            logger.warning(
                "Model response #%s failed: %s: %s",
                self._model_call_count,
                type(event.exception).__name__,
                event.exception,
            )
            return

        stop_response = event.stop_response
        if stop_response is None:
            logger.debug("Model response #%s: <missing stop response>", self._model_call_count)
            return

        logger.debug(
            "Model response #%s [stop_reason=%s]",
            self._model_call_count,
            stop_response.stop_reason,
        )
        for line in _iter_message_log_lines(stop_response.message):
            logger.debug("Model response #%s %s", self._model_call_count, line)


class TelemetryTracker(Plugin):
    """Track actual tool-start events plus partial metrics emitted via callbacks."""

    name = "telemetry"

    def __init__(self):
        super().__init__()
        self.tool_calls = 0
        self._tool_use_ids: set[str] = set()
        self.partial_metrics = None

    @property
    def unique_tool_calls(self) -> int:
        return len(self._tool_use_ids)

    @hook
    def on_agent_initialized(self, event: AgentInitializedEvent) -> None:
        self.tool_calls = 0
        self._tool_use_ids = set()
        self.partial_metrics = None

    @hook
    def on_before_tool(self, event: BeforeToolCallEvent) -> None:
        self.tool_calls += 1
        tool_use_id = str(event.tool_use.get("toolUseId", "") or "")
        if tool_use_id:
            self._tool_use_ids.add(tool_use_id)

    def __call__(self, **kwargs):
        if "metrics" in kwargs:
            self.partial_metrics = kwargs["metrics"]


__all__ = [
    "event_loop_tracker",
    "ToolLimitSteeringHandler",
    "SubmitAnswerPlugin",
    "LoggingPlugin",
    "TelemetryTracker",
]
