"""
Strands-native agent runner for the Data Lake benchmark.

Replaces the hand-rolled agent_runner.py with a clean Strands Agent skeleton
that wires together:
  - built-in conversation management  (summarizing or sliding-window)
  - ToolLimitPlugin                   (stops after max_tool_calls)
  - LoggingCallbackHandler            (structured per-turn logging)

Holds ``DataLakeAgent`` itself; the mode axes it composes tools and prompts
from live in ``runner.modes``, and the parallel task runner that drives many
``DataLakeAgent`` instances lives in ``runner.batch``.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from strands import Agent
from strands.tools.executors import SequentialToolExecutor, ConcurrentToolExecutor
from strands.vended_plugins.skills import AgentSkills

from sana_evaluation.config import AgentConfig, RunConfig
from sana_evaluation.instrumentation import (
    TracePlugin,
    ReadTracePlugin,
    SearchCallBudgetHandler,
    event_loop_tracker,
    ToolLimitSteeringHandler,
    SubmitAnswerPlugin,
    LoggingPlugin,
    TelemetryTracker,
)
from sana_evaluation.instrumentation.loop_plugin import CategoryStagnationHandler
from sana_evaluation.helper.prompting import (
    compose_baseline_prompt,
    inject_debug_prompt,
    skill_paths_for_modes,
)
from sana_evaluation.helper.agent_runtime import invoke_with_watchdog
from sana_evaluation.helper.conversation import build_conversation_manager
from sana_evaluation.helper.result import AgentResult
from sana_evaluation.helper.sandbox import (
    _cleanup_isolated_sandbox,
    _create_isolated_sandbox,
)
from sana_evaluation.helper.text_utils import _clean_answer
from sana_evaluation.llm.llm_factory import build_model
from sana_evaluation.tools.lake import (
    cleanup_sandbox,
    clear_submitted_answer,
    configure_benchmark as configure_data_lake_benchmark,
    get_submitted_answer,
    search_prefix,
    set_sandbox_dir,
)
from sana_evaluation.tools.external.search_eval_tools import (
    build_search_tools,
    search_tool_names_in as search_tool_names_in_legacy,
)
from sana_evaluation.runner.modes import (
    _inject_search_budget_prompt,
    _resolve_condition,
    _tool_limit_exclusions_for_run,
    build_data_tools,
    build_mode_bundle,
)

logger = logging.getLogger(__name__)

# Naive sparse search tools -- the legacy (no mode axes) fallback path below
# needs the tool objects themselves, not just the availability flag, so this
# mirrors runner.modes's own guarded import rather than depending on names
# that modes.py binds only when the optional import there succeeds.
_NAIVE_SEARCH_TOOLS_AVAILABLE = False
try:
    from sana_evaluation.tools.external.search_naive_tools import (
        search_value as search_value_naive,
        search_schema as search_schema_naive,
    )
    _NAIVE_SEARCH_TOOLS_AVAILABLE = True
except ImportError:
    pass


class DataLakeAgent:
    """Strands-based agent for the Data Lake benchmark."""

    def __init__(
        self,
        agent_config: AgentConfig,
        run_config: Optional[RunConfig] = None,
    ) -> None:
        self.agent_config = agent_config
        self.run_config = run_config or RunConfig()
        self._model = build_model(agent_config)

    # ------------------------------------------------------------------
    # Subclass extension hooks (default no-ops; SANA-agnostic).
    # ------------------------------------------------------------------

    def _pre_build_setup(
        self,
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> None:
        """Hook for runtime toggles that must run before the Agent is constructed."""
        return None

    def _extra_prompt_text(
        self,
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> str:
        """Return additional prompt text appended after the search-budget block but before the task trailer."""
        return ""

    def _system_prompt_override(
        self,
        *,
        system_prompt: str,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return a full replacement system prompt, or None to keep the composed prompt."""
        return None

    def _extra_plugins(
        self,
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> List[Any]:
        """Return additional plugins to append before the Agent is constructed."""
        return []

    def _conversation_manager(
        self,
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> Optional[Any]:
        """Return a custom ConversationManager, or None to use the default."""
        return None

    def _decorate_tools(
        self,
        tools: List[Any],
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return a (possibly modified) tools list. Default: identity."""
        return tools

    def _decorate_plugins(
        self,
        plugins: List[Any],
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> List[Any]:
        """Return a (possibly modified) plugin list. Default: identity."""
        return plugins

    def _tool_limit_excluded_tools(
        self,
        *,
        search_tool_mode: Optional[str],
        plan_mode: Optional[str],
    ) -> Sequence[str]:
        """Return tool names excluded from the global tool-limit counter."""
        return ("skills", "plan", "plan_ideal")

    def _build_agent(
        self,
        telemetry: "TelemetryTracker",
        trace_attributes: Optional[Dict[str, Any]] = None,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        configure_data_lake_benchmark(getattr(self.run_config, "benchmark", None))
        cond = self.run_config.condition_config
        condition = _resolve_condition(cond)
        mode_overrides_enabled = any(
            [
                self.run_config.search_tool_mode,
                self.run_config.search_results_mode,
                self.run_config.plan_mode,
                self.run_config.computation_tool_mode,
            ]
        )

        if not mode_overrides_enabled:
            if not _NAIVE_SEARCH_TOOLS_AVAILABLE:
                raise RuntimeError("Naive sparse search tools are unavailable (import failed).")

        # Core data-manipulation tools shared across all conditions
        _data_tools = build_data_tools(
            no_s3=bool(getattr(self.run_config, "no_s3", False)),
            search_tool_mode=getattr(self.run_config, "search_tool_mode", None),
        )

        task_trailer = ""
        if mode_overrides_enabled:
            mode_bundle = build_mode_bundle(
                self.run_config,
                data_tools=_data_tools,
                task_context=task_context,
            )
            tools = mode_bundle.tools
            system_prompt = mode_bundle.system_prompt
            task_trailer = mode_bundle.task_trailer
            search_tool_names = mode_bundle.search_tool_names
            enable_skills = mode_bundle.enable_skills
            enable_stagnation = mode_bundle.enable_stagnation
            skill_paths = skill_paths_for_modes(
                mode_bundle.modes["search_tool"],
                mode_bundle.modes["plan"],
            )
            logger.info(
                "Mode axes active: search_tool=%s search_results=%s plan=%s plan_skills=%s",
                mode_bundle.modes["search_tool"],
                mode_bundle.modes["search_results"],
                mode_bundle.modes["plan"],
                mode_bundle.modes["plan_skills"],
            )
        else:
            raw_search_tools = [search_value_naive, search_schema_naive, search_prefix]
            system_prompt = compose_baseline_prompt("naive")
            search_tools = build_search_tools(
                raw_search_tools,
                fixed_k=self.run_config.search_k,
                search_descriptions=self.run_config.search_descriptions,
            )
            tools = search_tools + _data_tools
            enable_skills = False
            enable_stagnation = False
            skill_paths = skill_paths_for_modes("naive", "naive")

            search_tool_names = search_tool_names_in_legacy(search_tools)

        # Resolve the active modes (None on the legacy path) for hook calls.
        _hook_search_tool_mode: Optional[str]
        _hook_plan_mode: Optional[str]
        if mode_overrides_enabled:
            _hook_search_tool_mode = mode_bundle.modes.get("search_tool")
            _hook_plan_mode = mode_bundle.modes.get("plan")
        else:
            _hook_search_tool_mode = None
            _hook_plan_mode = None

        self._pre_build_setup(
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
        )

        system_prompt = _inject_search_budget_prompt(
            system_prompt,
            self.run_config.search_calls_limit,
            search_tool_names,
        )
        if not mode_overrides_enabled:
            system_prompt = inject_debug_prompt(system_prompt, self.run_config.debug_mode)

        prompt_override = self._system_prompt_override(
            system_prompt=system_prompt,
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
            task_context=task_context,
        )
        if prompt_override is not None:
            system_prompt = prompt_override

        extra_prompt = self._extra_prompt_text(
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
        )
        if extra_prompt:
            system_prompt = system_prompt.rstrip() + extra_prompt

        if task_trailer:
            system_prompt = system_prompt.rstrip() + task_trailer

        conv_manager = self._conversation_manager(
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
        )
        if conv_manager is None:
            conv_manager = build_conversation_manager(self.run_config)

        if self.run_config.tool_executor == "sequential":
            tool_executor = SequentialToolExecutor()
        else:
            tool_executor = ConcurrentToolExecutor()

        _tool_limit_handler = ToolLimitSteeringHandler(
            self.run_config.max_tool_calls,
            self.run_config.timeout_seconds,
            submit_only_max_tokens=self.run_config.submit_only_max_tokens,
            excluded_tools=_tool_limit_exclusions_for_run(
                base_excluded=self._tool_limit_excluded_tools(
                    search_tool_mode=_hook_search_tool_mode,
                    plan_mode=_hook_plan_mode,
                ),
                search_free=bool(self.run_config.search_free),
                search_tool_names=search_tool_names,
            ),
        )

        def _callback(**kwargs):
            telemetry(**kwargs)
            event_loop_tracker(**kwargs)
            if kwargs.get("force_stop", False):
                reason = kwargs.get("force_stop_reason", "")
                if "ValidationException" in reason or "too long" in reason:
                    _tool_limit_handler.signal_context_overflow()

        cond = self.run_config.condition_config
        plugins = [_tool_limit_handler, telemetry]
        if self.run_config.search_calls_limit is not None:
            plugins.append(
                SearchCallBudgetHandler(
                    max_search_calls=self.run_config.search_calls_limit,
                    search_tools=search_tool_names,
                )
            )
        plugins.extend([
            SubmitAnswerPlugin(),
            LoggingPlugin(),
        ])
        if enable_skills:
            plugins.append(AgentSkills(skills=skill_paths))
        if enable_stagnation and self.run_config.max_consecutive_category > 0:
            plugins.append(
                CategoryStagnationHandler(self.run_config.max_consecutive_category)
            )
        read_tracer = ReadTracePlugin()
        plugins.append(read_tracer)
        plugins.append(TracePlugin(cond.trace_output_dir))

        plugins.extend(
            self._extra_plugins(
                search_tool_mode=_hook_search_tool_mode,
                plan_mode=_hook_plan_mode,
            )
        )
        plugins = self._decorate_plugins(
            plugins,
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
        )

        tools = self._decorate_tools(
            list(tools),
            search_tool_mode=_hook_search_tool_mode,
            plan_mode=_hook_plan_mode,
            task_context=task_context,
        )

        return Agent(
            model=self._model,
            tools=tools,
            tool_executor=tool_executor,
            system_prompt=system_prompt,
            conversation_manager=conv_manager,
            plugins=plugins,
            callback_handler=_callback,
            trace_attributes=trace_attributes,
        ), read_tracer

    def run(
        self,
        question: str,
        session_id: Optional[str] = None,
        task_context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        start = time.time()

        logger.info("=" * 60)
        logger.info(f"NEW TASK: {self.agent_config.model_id}")
        logger.debug(f"QUESTION: {question}")
        logger.info("=" * 60)

        sandbox = _create_isolated_sandbox(str(os.getpid()))
        set_sandbox_dir(sandbox)
        clear_submitted_answer()

        telemetry = TelemetryTracker()
        trace_attributes = (
            {
                "gen_ai.conversation.id": session_id,
                "session.id": session_id,
            }
            if session_id
            else None
        )

        try:
            agent, read_tracer = self._build_agent(
                telemetry,
                trace_attributes=trace_attributes,
                task_context=task_context,
            )
            hard_deadline = (
                start
                + self.run_config.timeout_seconds
                + self.run_config.submit_grace_seconds
            )
            outcome = invoke_with_watchdog(
                agent,
                question,
                hard_deadline=hard_deadline,
                timeout_seconds=self.run_config.timeout_seconds,
                submit_grace_seconds=self.run_config.submit_grace_seconds,
            )
            response = outcome.response

            submitted = get_submitted_answer()
            if outcome.timed_out and not submitted:
                elapsed = time.time() - start
                return AgentResult(
                    answer="",
                    model="",
                    model_name=self.agent_config.model_name,
                    metrics=response.metrics if response is not None else telemetry.partial_metrics,
                    elapsed_time=elapsed,
                    success=False,
                    error=outcome.timeout_reason or "Timeout reached",
                )
            retries = 0
            while not submitted and retries < 2:
                logger.warning(
                    f"Agent finished without submit_answer. Nudging (attempt {retries + 1}/2)..."
                )
                outcome = invoke_with_watchdog(
                    agent,
                    "You provided a text response but you MUST use the `submit_answer` tool "
                    "to submit your final answer. Please call the tool now.",
                    hard_deadline=hard_deadline,
                    timeout_seconds=self.run_config.timeout_seconds,
                    submit_grace_seconds=self.run_config.submit_grace_seconds,
                )
                response = outcome.response
                submitted = get_submitted_answer()
                if outcome.timed_out and not submitted:
                    elapsed = time.time() - start
                    return AgentResult(
                        answer="",
                        model="",
                        model_name=self.agent_config.model_name,
                        metrics=response.metrics if response is not None else telemetry.partial_metrics,
                        elapsed_time=elapsed,
                        success=False,
                        error=outcome.timeout_reason or "Timeout reached",
                    )
                retries += 1

            answer = submitted["answer"] if submitted else _clean_answer(str(response))
            elapsed = time.time() - start
            sources = list(read_tracer.gold_datasets_read)

            logger.info(f"ANSWER: {answer} ({elapsed:.1f}s)")
            return AgentResult(
                answer=answer,
                model=agent.model.config,
                model_name=self.agent_config.model_name,
                reasoning=submitted["reasoning"] if submitted else "",
                sources=sources,
                metrics=response.metrics,
                elapsed_time=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = time.time() - start
            unique_tool_suffix = (
                f" across {telemetry.unique_tool_calls} tool uses"
                if telemetry.unique_tool_calls and telemetry.unique_tool_calls != telemetry.tool_calls
                else ""
            )
            logger.error(
                f"Agent crashed after {telemetry.tool_calls} tool starts{unique_tool_suffix} "
                f"({elapsed:.1f}s): {type(e).__name__}: {e}",
                exc_info=True,
            )

            # Return the Error as a Result, including the partial metrics!
            return AgentResult(
                answer="",
                model="",
                model_name=self.agent_config.model_name,
                metrics=telemetry.partial_metrics, # Salvaged metrics!
                elapsed_time=elapsed,
                success=False,
                error=f"{type(e).__name__}: {e}",
            )

        finally:
            try:
                cleanup_sandbox()
            except Exception as e:
                logger.warning(f"Sandbox cleanup failed: {e}")
            try:
                _cleanup_isolated_sandbox(sandbox)
            except Exception as e:
                logger.warning(f"Isolated sandbox cleanup failed: {e}")
