"""
Strands-native agent runner for the Data Lake benchmark.

Replaces the hand-rolled agent_runner.py with a clean Strands Agent skeleton
that wires together:
  - built-in conversation management  (summarizing or sliding-window)
  - ToolLimitPlugin                   (stops after max_tool_calls)
  - LoggingCallbackHandler            (structured per-turn logging)
"""

import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strands import Agent
from strands.tools.executors import SequentialToolExecutor, ConcurrentToolExecutor
from strands.tools.decorator import DecoratedFunctionTool
from strands.vended_plugins.skills import AgentSkills

import logging

from sana_evaluation.config import AgentConfig, ConditionConfig, RunConfig
from sana_evaluation.instrumentation import (
    TracePlugin,
    ReadTracePlugin,
    SearchCallBudgetHandler,
    set_trace_context,
    event_loop_tracker,
    ToolLimitSteeringHandler,
    SubmitAnswerPlugin,
    LoggingPlugin,
    TelemetryTracker,
)
from sana_evaluation.instrumentation.loop_plugin import CategoryStagnationHandler
from sana_evaluation.helper.logger import configure_worker_logging
from sana_evaluation.helper.prompting import (
    compose_baseline_prompt,
    compose_kramabench_prompt,
    compose_managed_prompt,
    compose_preloaded_block,
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
from sana_evaluation.tools.agent_tools import (
    clear_submitted_answer,
    get_submitted_answer,
    submit_answer,
    search,
    search_keyword,
    search_prefix,
)
from sana_evaluation.tools.agent_tools_v2 import (
    cleanup_sandbox,
    configure_benchmark as configure_data_lake_benchmark,
    execute_code,
    grep_file,
    list_files,
    parse_xml_records,
    peek_file,
    peek_multiple,
    query_file,
    read_file,
    set_sandbox_dir,
)
from sana_evaluation.tools.agent_tools import download
from sana_evaluation.tools.external.plan_tools import plan
from sana_evaluation.tools.external.ideal.plan_ideal import (
    inject_reasoning_chain_prompt,
    plan_ideal,
)
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    load_runtime_profile_for_context as load_ideal_profile_for_context,
    set_task_context as set_ideal_profile_task_context,
)
from sana_evaluation.tools.external.ideal.search_wrapper import (
    build_search_tools as build_search_tools_by_mode,
    search_tool_names_in as search_tool_names_in_mode,
)
from sana_evaluation.tools.external.search_eval_tools import (
    build_search_tools,
    search_tool_names_in as search_tool_names_in_legacy,
)

logger = logging.getLogger(__name__)

# Standard hybrid search tools
_STANDARD_SEARCH_TOOLS_AVAILABLE = False
try:
    from sana_evaluation.tools.external.search_standard_tools import (
        search_value as search_value_standard,
        search_schema,
        search_reranked,
    )
    _STANDARD_SEARCH_TOOLS_AVAILABLE = True
except ImportError:
    pass

# Naive sparse search tools
_NAIVE_SEARCH_TOOLS_AVAILABLE = False
try:
    from sana_evaluation.tools.external.search_naive_tools import (
        search_value as search_value_naive,
        search_schema as search_schema_naive,
    )
    _NAIVE_SEARCH_TOOLS_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Mode composition (inlined in this module)
# ---------------------------------------------------------------------------

_MODES = {"naive", "standard", "ideal", "preloaded"}
_RESULT_MODES = {"naive", "ideal"}
_COMPUTATION_MODES = {"standard", "ideal"}


@dataclass
class ModeBundle:
    tools: List[Any]
    system_prompt: str
    search_tool_names: Tuple[str, ...]
    enable_skills: bool
    enable_stagnation: bool
    modes: Dict[str, str]
    task_trailer: str = ""


def _normalize_mode(value: Optional[str], default: str, label: str) -> str:
    mode = (value or default).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"Unsupported {label} mode '{value}'. Expected one of: {', '.join(sorted(_MODES))}")
    return mode


def _normalize_result_mode(value: Optional[str], default: str, label: str) -> str:
    mode = (value or default).strip().lower()
    if mode not in _RESULT_MODES:
        raise ValueError(
            f"Unsupported {label} mode '{value}'. Expected one of: {', '.join(sorted(_RESULT_MODES))}"
        )
    return mode


def _normalize_computation_mode(value: Optional[str], default: str = "standard") -> str:
    mode = (value or default).strip().lower()
    if mode not in _COMPUTATION_MODES:
        raise ValueError(
            f"Unsupported computation_tool mode '{value}'. Expected one of: {', '.join(sorted(_COMPUTATION_MODES))}"
        )
    return mode


def build_search(
    mode: str,
    *,
    task_context: Optional[Dict[str, Any]] = None,
    search_lessguide: bool = False,
) -> List[DecoratedFunctionTool]:
    """Return the base search tool surface for a mode."""
    search_mode = _normalize_mode(mode, "standard", "search_tool")

    if search_mode == "naive":
        if not _NAIVE_SEARCH_TOOLS_AVAILABLE:
            raise RuntimeError("Naive sparse search tools are unavailable (import failed).")
        from sana_evaluation.tools.external.search_naive_tools import (
            search_schema as search_schema_sparse,
            search_value as search_value_sparse,
        )

        return [search_value_sparse, search_schema_sparse, search_prefix]

    if search_mode == "standard":
        if not _STANDARD_SEARCH_TOOLS_AVAILABLE:
            raise RuntimeError("Standard hybrid search tools are unavailable (import failed).")
        from sana_evaluation.tools.external.search_standard_tools import (
            search_schema as search_schema_hybrid,
            search_value as search_value_hybrid,
        )

        return [search_value_hybrid, search_schema_hybrid, search_prefix]

    if search_mode == "preloaded":
        return []

    import sana_evaluation.tools.external.ideal.search_ideal as search_ideal

    search_ideal.set_lessguide(search_lessguide)
    search_ideal.set_task_context(task_context or {})
    return [search_ideal.search_ideal]


def build_management(
    mode: str,
    *,
    search_tool_mode: str,
    task_context: Optional[Dict[str, Any]],
    profile_skills_enabled: bool = False,
    benchmark: str = "lakeqa",
) -> tuple[str, List[Any], bool, bool, str]:
    """Return stable system prompt, management tools, behavior toggles, and a task-specific trailer.

    The trailer (gold reasoning chain, preloaded dataset URIs) is task-specific and must be
    appended AFTER all variant-stable injections so the cacheable prefix stays intact across tasks.
    """
    management_mode = _normalize_mode(mode, "standard", "profile")

    trailer_sections: List[str] = []
    if management_mode == "ideal":
        set_ideal_profile_task_context(task_context or {})
        ideal_profile = load_ideal_profile_for_context(task_context)
        reasoning_trailer = inject_reasoning_chain_prompt("", ideal_profile.reasoning_chain_text).lstrip()
        if reasoning_trailer:
            trailer_sections.append(reasoning_trailer)
    if search_tool_mode == "preloaded":
        ideal_profile = load_ideal_profile_for_context(task_context)
        trailer_sections.append(compose_preloaded_block(ideal_profile.source_sequence))
    task_trailer = ("\n\n" + "\n\n".join(trailer_sections)) if trailer_sections else ""

    benchmark_name = (benchmark or "lakeqa").strip().lower()
    if management_mode == "naive":
        if benchmark_name == "kramabench":
            return compose_kramabench_prompt(search_tool_mode, include_skills=False), [], False, False, task_trailer
        return compose_baseline_prompt(search_tool_mode), [], False, False, task_trailer

    if benchmark_name == "kramabench":
        prompt = compose_kramabench_prompt(
            search_tool_mode,
            include_skills=bool(profile_skills_enabled),
        )
    else:
        prompt = compose_managed_prompt(
            search_tool_mode,
            include_skills=bool(profile_skills_enabled),
        )
    if management_mode == "standard":
        return prompt, [plan], bool(profile_skills_enabled), True, task_trailer

    return prompt, [plan_ideal], bool(profile_skills_enabled), True, task_trailer


def build_results(
    mode: str,
    *,
    base_search_tools: Sequence[DecoratedFunctionTool],
    fixed_k: Optional[int],
) -> List[DecoratedFunctionTool]:
    """Apply fixed-k and payload-shaping wrappers to search tools."""
    if not base_search_tools:
        return []
    results_mode = _normalize_result_mode(mode, "naive", "search_results")
    return build_search_tools_by_mode(
        base_search_tools,
        fixed_k=fixed_k,
        results_mode=results_mode,
    )


def build_mode_bundle(
    run_config: RunConfig,
    *,
    data_tools: Sequence[Any],
    task_context: Optional[Dict[str, Any]] = None,
) -> ModeBundle:
    """Build final tools/prompt/plugin toggles from multi-axis modes."""
    search_tool_mode = _normalize_mode(run_config.search_tool_mode, "standard", "search_tool")
    search_results_mode = _normalize_result_mode(run_config.search_results_mode, "naive", "search_results")
    profile_mode = _normalize_mode(
        run_config.profile_mode or run_config.profile_mode,
        "standard",
        "profile",
    )
    computation_tool_mode = _normalize_computation_mode(run_config.computation_tool_mode)
    benchmark = (getattr(run_config, "benchmark", None) or "lakeqa").strip().lower()

    if search_tool_mode == "ideal" or profile_mode == "ideal" or computation_tool_mode == "ideal":
        set_ideal_profile_task_context(task_context or {})

    raw_search_tools = build_search(
        search_tool_mode,
        task_context=task_context,
        search_lessguide=bool(run_config.search_lessguide),
    )
    search_tools = build_results(
        search_results_mode,
        base_search_tools=raw_search_tools,
        fixed_k=run_config.search_k,
    )
    system_prompt, management_tools, enable_skills, enable_stagnation, task_trailer = build_management(
        profile_mode,
        search_tool_mode=search_tool_mode,
        task_context=task_context,
        profile_skills_enabled=bool(run_config.profile_skills_enabled),
        benchmark=benchmark,
    )
    system_prompt = inject_debug_prompt(system_prompt, run_config.debug_mode)
    system_prompt = _inject_computation_file_family_prompt(
        system_prompt,
        computation_tool_mode=computation_tool_mode,
        benchmark=benchmark,
    )

    data_tool_list = _apply_computation_tool_mode(
        data_tools,
        computation_tool_mode=computation_tool_mode,
        task_context=task_context,
        benchmark=benchmark,
    )
    if computation_tool_mode == "ideal":
        system_prompt = _inject_ideal_computation_prompt(system_prompt, benchmark=benchmark)

    tools = list(search_tools) + list(management_tools) + list(data_tool_list)
    return ModeBundle(
        tools=tools,
        system_prompt=system_prompt,
        search_tool_names=search_tool_names_in_mode(search_tools),
        enable_skills=enable_skills,
        enable_stagnation=enable_stagnation,
        modes={
            "search_tool": search_tool_mode,
            "search_results": search_results_mode,
            "profile": profile_mode,
            "computation_tool": computation_tool_mode,
            "profile_skills": "on" if run_config.profile_skills_enabled else "off",
        },
        task_trailer=task_trailer,
    )


def _apply_computation_tool_mode(
    data_tools: Sequence[Any],
    *,
    computation_tool_mode: str,
    task_context: Optional[Dict[str, Any]],
    benchmark: str = "lakeqa",
) -> List[Any]:
    query_disabled = benchmark == "kramabench"
    if computation_tool_mode != "ideal":
        if not query_disabled:
            return list(data_tools)
        return [
            tool_obj
            for tool_obj in data_tools
            if _tool_name(tool_obj) != "query_file"
        ]

    from sana_evaluation.tools.external.ideal import computation_ideal

    computation_ideal.set_task_context(task_context or {})
    out: List[Any] = []
    for tool_obj in data_tools:
        tool_name = _tool_name(tool_obj)
        if tool_name == "query_file":
            if query_disabled:
                continue
            if not any(_tool_name(t) == "query_ideal" for t in out):
                out.append(computation_ideal.query_ideal)
            continue
        if tool_name == "execute_code":
            if not any(_tool_name(t) == "execute_ideal" for t in out):
                out.append(computation_ideal.execute_ideal)
            continue
        out.append(tool_obj)
    return out


def _tool_name(tool_obj: Any) -> Optional[str]:
    tool_name = getattr(tool_obj, "tool_name", None)
    if tool_name is None and hasattr(tool_obj, "tool_spec"):
        tool_name = tool_obj.tool_spec.get("name")
    return tool_name


def _inject_ideal_computation_prompt(system_prompt: str, *, benchmark: str = "lakeqa") -> str:
    if benchmark == "kramabench":
        section = (
            "\n\n## IDEAL COMPUTATION TOOLS\n"
            "- Use only `execute_ideal` for computation in this Kramabench run.\n"
            "- Use `execute_ideal(code, intent, dataset_id=..., file_path=... or s3_uri=...)` "
            "for computation.\n"
            "- Always write a concise intent describing the computation you are trying to perform.\n"
        )
    else:
        section = (
            "\n\n## IDEAL COMPUTATION TOOLS\n"
            "- `query_file` and `execute_code` are replaced in this run.\n"
            "- Use `query_ideal(..., intent=...)` for SQL-style computation and "
            "`execute_ideal(code, intent, dataset_id=..., file_path=... or s3_uri=...)` "
            "for Python computation.\n"
            "- Always write a concise intent describing the computation you are trying to perform.\n"
        )
    return system_prompt.rstrip() + section


def _inject_computation_file_family_prompt(
    system_prompt: str,
    *,
    computation_tool_mode: str,
    benchmark: str = "lakeqa",
) -> str:
    if computation_tool_mode == "ideal" and benchmark == "kramabench":
        blocked_tools = "`execute_ideal`"
    elif computation_tool_mode == "ideal":
        blocked_tools = "`query_file`, `execute_code`, `query_ideal`, or `execute_ideal`"
    elif benchmark == "kramabench":
        blocked_tools = "`execute_code`"
    else:
        blocked_tools = "`query_file` or `execute_code`"
    section = (
        "\n\n## COMPUTATION FILE FAMILY RULE\n"
        f"- Do not use {blocked_tools} when the target source is non-tabular or non-JSON.\n"
        "- Eligible computation sources are tabular or JSON-like files only: CSV/TSV/delimited tables, JSON, JSONL/NDJSON, or GeoJSON feature collections.\n"
        "- For XML/KML, use `parse_xml_records` for structured records/counts, or `peek_file`, `grep_file`, and `read_file` for inspection/search.\n"
        "- For Wikipedia/content.txt, prose/plain text, HTML, PDFs, binary files, or other non-tabular sources, use `read_file`, `grep_file`, or `peek_file` and extract the fact directly.\n"
        "- Do not download a non-tabular/non-JSON source just to parse it with Python.\n"
    )
    if computation_tool_mode != "ideal" and benchmark == "kramabench":
        section += (
            "- For Kramabench tabular computation, use `peek_file` to inspect structure, "
            "then download the file and use `execute_code`.\n"
        )
    return system_prompt.rstrip() + section


# Shared callback tracking and plugin classes are defined in
# sana_evaluation.instrumentation.agent_plugins.


# ---------------------------------------------------------------------------
# DataLakeAgent
# ---------------------------------------------------------------------------

def _base_condition(condition_label: str) -> str:
    """Strip optional experimental suffix from condition label."""
    if not condition_label:
        return "baseline"
    return str(condition_label).split("__", 1)[0]


def _resolve_condition(cond_cfg: ConditionConfig) -> str:
    """Resolve the base experiment condition label."""
    if getattr(cond_cfg, "base_condition", None):
        return str(cond_cfg.base_condition)
    return _base_condition(cond_cfg.condition)


def _inject_search_budget_prompt(
    system_prompt: str,
    search_calls_limit: Optional[int],
    search_tool_names: tuple[str, ...],
) -> str:
    """Append search-call budget instructions when configured."""
    if search_calls_limit is None or not search_tool_names:
        return system_prompt
    names = ", ".join(search_tool_names) if search_tool_names else "search tools"
    budget_note = (
        "\n\n## SEARCH CALL BUDGET\n"
        f"- You have at most {search_calls_limit} total calls to: {names}.\n"
        "- When this budget is exhausted, do not call search tools again.\n"
        "- Continue with non-search tools (list/read/query/grep/download/execute_code), "
        "then call submit_answer."
    )
    return system_prompt.rstrip() + budget_note


def _tool_limit_exclusions_for_run(
    *,
    base_excluded: Sequence[str],
    search_free: bool,
    search_tool_names: Sequence[str],
) -> Tuple[str, ...]:
    """Return tool-limit exclusions after applying run-level accounting flags."""
    names = list(base_excluded)
    if search_free:
        names.extend(search_tool_names)
    return tuple(dict.fromkeys(str(name) for name in names))


def _merge_sources_used(
    planner_sources: Optional[Sequence[Any]],
    delegation_subagent_stats: Optional[Dict[str, Any]],
) -> List[str]:
    """Merge parent-agent reads with gold reads observed inside delegation workers."""

    merged: List[str] = []
    seen: set[str] = set()
    delegation_sources = []
    if delegation_subagent_stats:
        delegation_sources = delegation_subagent_stats.get("delegation_gold_datasets_read") or []
    for value in [*(planner_sources or []), *delegation_sources]:
        dataset_id = str(value).strip()
        if dataset_id and dataset_id not in seen:
            merged.append(dataset_id)
            seen.add(dataset_id)
    return merged


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
        profile_mode: Optional[str],
    ) -> None:
        """Hook for runtime toggles that must run before the Agent is constructed."""
        return None

    def _extra_prompt_text(
        self,
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
    ) -> str:
        """Return additional prompt text appended after the search-budget block but before the task trailer."""
        return ""

    def _system_prompt_override(
        self,
        *,
        system_prompt: str,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return a full replacement system prompt, or None to keep the composed prompt."""
        return None

    def _extra_plugins(
        self,
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
    ) -> List[Any]:
        """Return additional plugins to append before the Agent is constructed."""
        return []

    def _conversation_manager(
        self,
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
    ) -> Optional[Any]:
        """Return a custom ConversationManager, or None to use the default."""
        return None

    def _decorate_tools(
        self,
        tools: List[Any],
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Return a (possibly modified) tools list. Default: identity."""
        return tools

    def _decorate_plugins(
        self,
        plugins: List[Any],
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
    ) -> List[Any]:
        """Return a (possibly modified) plugin list. Default: identity."""
        return plugins

    def _tool_limit_excluded_tools(
        self,
        *,
        search_tool_mode: Optional[str],
        profile_mode: Optional[str],
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
                self.run_config.profile_mode,
                self.run_config.computation_tool_mode,
            ]
        )

        if not mode_overrides_enabled:
            if not _NAIVE_SEARCH_TOOLS_AVAILABLE:
                raise RuntimeError("Naive sparse search tools are unavailable (import failed).")

        # Core data-manipulation tools shared across all conditions
        _data_tools = [
            list_files, peek_file, peek_multiple, read_file, grep_file,
            parse_xml_records, query_file, download, execute_code,
            submit_answer,
        ]

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
                mode_bundle.modes["profile"],
            )
            logger.info(
                "Mode axes active: search_tool=%s search_results=%s profile=%s profile_skills=%s",
                mode_bundle.modes["search_tool"],
                mode_bundle.modes["search_results"],
                mode_bundle.modes["profile"],
                mode_bundle.modes["profile_skills"],
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
        _hook_profile_mode: Optional[str]
        if mode_overrides_enabled:
            _hook_search_tool_mode = mode_bundle.modes.get("search_tool")
            _hook_profile_mode = mode_bundle.modes.get("profile")
        else:
            _hook_search_tool_mode = None
            _hook_profile_mode = None

        self._pre_build_setup(
            search_tool_mode=_hook_search_tool_mode,
            profile_mode=_hook_profile_mode,
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
            profile_mode=_hook_profile_mode,
            task_context=task_context,
        )
        if prompt_override is not None:
            system_prompt = prompt_override

        extra_prompt = self._extra_prompt_text(
            search_tool_mode=_hook_search_tool_mode,
            profile_mode=_hook_profile_mode,
        )
        if extra_prompt:
            system_prompt = system_prompt.rstrip() + extra_prompt

        if task_trailer:
            system_prompt = system_prompt.rstrip() + task_trailer

        conv_manager = self._conversation_manager(
            search_tool_mode=_hook_search_tool_mode,
            profile_mode=_hook_profile_mode,
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
                    profile_mode=_hook_profile_mode,
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
                profile_mode=_hook_profile_mode,
            )
        )
        plugins = self._decorate_plugins(
            plugins,
            search_tool_mode=_hook_search_tool_mode,
            profile_mode=_hook_profile_mode,
        )

        tools = self._decorate_tools(
            list(tools),
            search_tool_mode=_hook_search_tool_mode,
            profile_mode=_hook_profile_mode,
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


# ---------------------------------------------------------------------------
# Worker function (must be module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _run_task_worker(
    task: Dict[str, Any],
    task_index: int,
    agent_config: AgentConfig,
    run_config: RunConfig,
    run_id: str,
    batch_name: Optional[str],
    agent_class: Optional[type] = None,
) -> Dict[str, Any]:
    """Run a single task in a worker process.

    `agent_class` lets callers swap in a DataLakeAgent subclass (e.g. for SANA).
    Defaults to `DataLakeAgent`.
    """
    from sana_evaluation.helper.metrics import compute_exact_match, compute_f1_score, normalize_text

    log_model_name = agent_config.model_name or agent_config.model_id
    effort = (agent_config.extra_model_kwargs or {}).get("reasoning_effort")
    if effort:
        log_model_name = f"{log_model_name}-{effort}"

    configure_worker_logging(
        run_config,
        model=log_model_name,
        condition=run_config.condition_config.condition,
        task_id=task.get("id"),
    )

    # Eagerly load search backend state once per worker process.
    # For mode runs, explicit axes drive setup. Otherwise the baseline path drives setup.
    mode_search_tool = (run_config.search_tool_mode or "").strip().lower() or None
    if mode_search_tool is None:
        mode_search_tool = "naive"
    mode_computation_tool = (run_config.computation_tool_mode or "").strip().lower() or None
    if mode_computation_tool is None:
        mode_computation_tool = "standard"

    if mode_search_tool == "standard" and _STANDARD_SEARCH_TOOLS_AVAILABLE:
        try:
            import sana_evaluation.tools.external.search_standard_tools as _sa
            if run_config.search_db_path:
                _sa.set_db_path(run_config.search_db_path)
            _sa.setup()
        except Exception as e:
            logger.warning(f"Hybrid search setup failed: {e}")

    if mode_search_tool == "naive" and _NAIVE_SEARCH_TOOLS_AVAILABLE:
        # NOTE: do not call search_naive_tools.setup() for naive mode.
        # setup() triggers external-tools/hybrid_search/api.setup(), which eagerly
        # loads embedding/reranker models that are unnecessary for sparse-only
        # search paths and can heavily impact worker startup and memory.
        if run_config.search_db_path:
            try:
                import sana_evaluation.tools.external.search_naive_tools as _sb

                _sb.set_db_path(run_config.search_db_path)
            except Exception as e:
                logger.warning(f"Sparse search path override failed: {e}")

    if mode_search_tool == "ideal":
        if run_config.search_db_path:
            import sana_evaluation.tools.external.ideal.search_ideal as _si

            _si.set_db_path(run_config.search_db_path)

    try:
        logger.info(f"Starting task {task_index + 1}: {task.get('question', '')[:80]}...")

        agent_cls = agent_class or DataLakeAgent
        da = agent_cls(agent_config, run_config)

        cond = run_config.condition_config
        gold_ids = task.get("datasets_used", [])
        task_id = task.get("id", str(task_index))
        set_trace_context(task_id, gold_ids, cond.trace_output_dir)
        from sana_evaluation.instrumentation import ideal_subagent_costs as _ideal_costs

        _ideal_costs.reset_stats()

        try:
            from sana_evaluation.instrumentation import delegation_subagent_costs as _deleg_costs
        except ImportError:
            _deleg_costs = None
        if _deleg_costs is not None:
            _deleg_costs.reset_stats()

        task_context = {
            "task_id": task_id,
            "datasets_used": gold_ids,
            "reasoning_chain": task.get("reasoning_chain", []),
        }
        if mode_search_tool == "ideal":
            import sana_evaluation.tools.external.ideal.search_ideal as _si

            _si.set_task_context(task_context)

        result = da.run(task["question"], task_context=task_context)
        delegation_subagent_stats: Dict[str, Any] = {}
        if _deleg_costs is not None:
            delegation_subagent_stats = _deleg_costs.get_stats()

        result_dict: Dict[str, Any] = {
            "task_id": task.get("id", task_index),
            "model": agent_config.model_id,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": result.answer,
            "reasoning": result.reasoning,
            "sources_used": _merge_sources_used(result.sources, delegation_subagent_stats),
            "time": result.elapsed_time,
            "success": result.success,
            "error": result.error,
        }

        # Tokens
        result_dict["input_tokens"]     = result.input_tokens
        result_dict["cached_input_tokens"] = result.cached_input_tokens
        result_dict["uncached_input_tokens"] = result.uncached_input_tokens
        result_dict["output_tokens"]    = result.output_tokens
        result_dict["total_tokens"]     = result.total_tokens
        result_dict["cost_usd"]         = result.cost_usd

        # Tool counts
        _, total_calls = result.get_cumulative_tool_counts()
        result_dict["tool_calls_total"] = total_calls
        result_dict["api_tool_calls"]   = result.get_api_tool_calls()

        # Cycle count
        result_dict["cycle_count"]      = result.cycle_count

        # Per-tool breakdown (for tools CSV) — List[Dict], pickle-safe
        result_dict["tool_counts"]      = result.get_tool_counts()

        if mode_computation_tool == "ideal":
            from sana_evaluation.tools.external.ideal import computation_ideal as _ci

            result_dict.update(_ci.get_stats())

        ideal_subagent_stats = _ideal_costs.get_stats()
        result_dict.update(ideal_subagent_stats)
        ideal_cost = float(ideal_subagent_stats.get("ideal_subagent_cost_usd", 0.0) or 0.0)
        result_dict["total_cost_with_ideal_subagents_usd"] = result.cost_usd + ideal_cost

        if _deleg_costs is not None:
            result_dict.update(delegation_subagent_stats)
        delegation_cost = float(
            delegation_subagent_stats.get("delegation_subagent_cost_usd", 0.0) or 0.0
        )
        result_dict["total_cost_with_all_subagents_usd"] = (
            result.cost_usd + ideal_cost + delegation_cost
        )

        try:
            from sana_evaluation.tools.delegation_common import clear_delegation_runtime
        except ImportError:
            pass
        else:
            clear_delegation_runtime()

        # Compute accuracy metrics if ground truth available
        if task.get("answer"):
            gt = str(task["answer"])
            pred = result.answer
            result_dict["exact_match"] = compute_exact_match(pred, gt)
            result_dict["f1_score"] = compute_f1_score(pred, gt)
            result_dict["normalized_prediction"] = normalize_text(pred)
            result_dict["normalized_ground_truth"] = normalize_text(gt)


        result_dict["task_metadata"] = {
            "num_nodes": len(task.get("nodes", {})),
            "has_reasoning_chain": "reasoning_chain" in task,
        }

        if result.success:
            logger.info(
                f"Completed task {task_index + 1}: "
                f"input={result.input_tokens} cached_input={result.cached_input_tokens} "
                f"uncached_input={result.uncached_input_tokens} output={result.output_tokens} "
                f"tokens={result.total_tokens} cost=${result.cost_usd:.4f} "
                f"tools={total_calls} cycles={result.cycle_count}"
            )
        else:
            logger.warning(
                f"Task {task_index + 1} finished with failure: {result.error}"
            )
        return result_dict

    except Exception as e:
        logger.error(
            f"Worker error for task {task_index + 1} ({type(e).__name__}): {e}",
            exc_info=True,
        )
        return {
            "task_id": task.get("id", task_index),
            "model": agent_config.model_id,
            "question": task.get("question", ""),
            "ground_truth": task.get("answer", ""),
            "predicted_answer": "",
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


# ---------------------------------------------------------------------------
# BatchRunner
# ---------------------------------------------------------------------------

class BatchRunner:
    """Run DataLakeAgent on multiple tasks with parallel ProcessPoolExecutor."""

    # Subclasses can override this to swap in a DataLakeAgent subclass.
    _AGENT_CLASS: Optional[type] = None

    def __init__(
        self,
        agent_config: AgentConfig,
        run_config: Optional[RunConfig] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.agent_config = agent_config
        self.run_config = run_config or RunConfig()
        self.max_workers = max_workers or min(6, os.cpu_count() or 1)

    def run_tasks(
        self,
        tasks: List[Dict[str, Any]],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run agent on a list of tasks in parallel."""
        num_workers = max_workers or self.max_workers
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_label = batch_name or "batch"

        if verbose:
            print(f"\nRunning {len(tasks)} tasks with {num_workers} workers...")

        results: List[tuple] = []

        import multiprocessing as _mp
        mp_ctx = _mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as executor:
            future_to_index = {
                executor.submit(
                    _run_task_worker,
                    task,
                    i,
                    self.agent_config,
                    self.run_config,
                    run_id,
                    batch_label,
                    self._AGENT_CLASS,
                ): i
                for i, task in enumerate(tasks)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                    if verbose:
                        em = result.get("exact_match")
                        status = f" EM={em:.2f}" if em is not None else ""
                        print(f"  Task {idx + 1}/{len(tasks)} done{status}")
                except Exception as e:
                    logger.error(
                        f"Future for task {idx + 1} raised {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                    results.append((idx, {
                        "task_id": tasks[idx].get("id", idx),
                        "model": self.agent_config.model_id,
                        "question": tasks[idx].get("question", ""),
                        "ground_truth": tasks[idx].get("answer", ""),
                        "predicted_answer": "",
                        "success": False,
                        "error": f"{type(e).__name__}: {e}",
                    }))
                    if verbose:
                        print(f"  Task {idx + 1}/{len(tasks)} FAILED: {type(e).__name__}: {e}")

        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def run_from_files(
        self,
        task_files: List[str],
        verbose: bool = False,
        max_workers: Optional[int] = None,
        batch_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load tasks from JSON files and run them."""
        import json

        tasks = []
        for path in task_files:
            with open(path) as f:
                task = json.load(f)
                task["id"] = path
                tasks.append(task)

        if not batch_name and task_files:
            try:
                batch_name = Path(os.path.commonpath(task_files)).name
            except Exception:
                pass

        return self.run_tasks(
            tasks, verbose=verbose, max_workers=max_workers, batch_name=batch_name
        )
