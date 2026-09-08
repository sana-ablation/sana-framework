"""
Mode axis resolution for the Data Lake benchmark runner.

Builds the tool surface, system prompt, and behavior toggles for each of the
search_tool / search_results / plan / computation_tool axes and merges
them into a ModeBundle (``build_mode_bundle``). Also holds the small
per-run bookkeeping helpers -- condition-label resolution, search-budget
prompt injection, tool-limit exclusions, and gold-source merging -- that
``runner.agent.DataLakeAgent`` and ``runner.batch`` both consume.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from strands.tools.decorator import DecoratedFunctionTool

from sana_evaluation.config import ConditionConfig, RunConfig
from sana_evaluation.helper.prompting import (
    _normalize_mode,
    compose_baseline_prompt,
    compose_kramabench_prompt,
    compose_managed_prompt,
    compose_preloaded_block,
    inject_debug_prompt,
)
from sana_evaluation.tools.lake import (
    download,
    execute_code,
    grep_file,
    list_files,
    parse_xml_records,
    peek_file,
    peek_multiple,
    query_file,
    read_file,
    search_prefix,
    submit_answer,
)
from sana_evaluation.tools.fetch import download_web
from sana_evaluation.tools.plan import plan
from sana_evaluation.tools.oracle.plan import (
    inject_reasoning_chain_prompt,
    plan_ideal,
)
from sana_evaluation.profiles import (
    load_runtime_profile_for_context as load_ideal_profile_for_context,
    set_task_context as set_ideal_profile_task_context,
)
from sana_evaluation.tools.search.wrapper import (
    build_search_tools as build_search_tools_by_mode,
    search_tool_names_in as search_tool_names_in_mode,
)

# Standard hybrid search tools
_STANDARD_SEARCH_TOOLS_AVAILABLE = False
try:
    from sana_evaluation.tools.search.standard import (
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
    from sana_evaluation.tools.search.naive import (
        search_value as search_value_naive,
        search_schema as search_schema_naive,
    )
    _NAIVE_SEARCH_TOOLS_AVAILABLE = True
except ImportError:
    pass

# search_results controls how much metadata rides along with each search hit --
# minimal is dataset_id + s3_uri, rich adds the LLM description, the schema and a
# data snippet. It is not one of the paper's axes, so it is named for what it
# varies rather than borrowed from the naive/ideal tier vocabulary those axes use.
# The old names remain accepted so existing scripts and result trees still
# resolve.
_RESULT_MODES = {"minimal", "rich"}
_RESULT_MODE_ALIASES = {"naive": "minimal", "ideal": "rich"}
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


def _normalize_result_mode(value: Optional[str], default: str = "rich", label: str = "search_results") -> str:
    mode = (value or default).strip().lower()
    mode = _RESULT_MODE_ALIASES.get(mode, mode)
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


def _validate_search_mode_combination(
    *,
    search_tool_mode: str,
    search_results_mode: str,
    plan_mode: str,
    computation_tool_mode: str,
    no_s3: bool = False,
) -> None:
    """Reject axis combinations that would make a web-search run unmeasurable.

    Every ideal axis is backed by the task's runtime profile, which is authored
    against data-lake sources. Combining any of them with web search produces a
    run whose result is independent of what web search actually returned:

    - computation_tool=ideal: execute_ideal/query_ideal return the authored
      ``record.answer`` for a semantically matching record, and _records_for_target
      falls back to *all* records when the submitted source matches none. The
      agent is handed gold node answers no matter what it retrieved.
    - plan=ideal: the gold reasoning chain is injected into the prompt.
    - search_results=rich: reshape_search_payload expects lake-shaped result
      fields that web results do not carry.
    """
    if no_s3 and search_tool_mode != "web":
        raise ValueError(
            "--no-s3 removes every data-lake tool, so it is only meaningful with "
            f"--search_tool web (got '{search_tool_mode}'). Without lake tools and "
            "without web search the agent has no retrieval path at all."
        )

    if search_tool_mode != "web":
        return

    conflicts = [
        ("--computation_tool ideal", computation_tool_mode == "ideal"),
        ("--plan ideal", plan_mode == "ideal"),
        # Normalised here so the deprecated spelling (--search_results ideal) is
        # caught too; callers may pass either.
        ("--search_results rich",
         _normalize_result_mode(search_results_mode, "rich", "search_results") == "rich"),
    ]
    active = [label for label, hit in conflicts if hit]
    if active:
        raise ValueError(
            "search_tool=web cannot be combined with "
            + ", ".join(active)
            + ". Ideal axes are backed by data-lake runtime profiles, so the run's "
            "outcome would not depend on what web search retrieved. Use "
            "--search_results naive --plan naive|standard --computation_tool standard."
        )


def build_search(
    mode: str,
    *,
    task_context: Optional[Dict[str, Any]] = None,
    fixed_k: Optional[int] = None,
) -> List[DecoratedFunctionTool]:
    """Return the base search tool surface for a mode."""
    search_mode = _normalize_mode(mode, "standard", "search_tool")

    if search_mode == "web":
        # Web search is not reshaped by the results axis (see search_wrapper._WEB_TOOLS),
        # so --k has to be applied here rather than by build_search_results.
        from sana_evaluation.tools.search.web import (
            search_web,
            set_max_results,
        )

        set_max_results(fixed_k)
        return [search_web]

    if search_mode == "naive":
        if not _NAIVE_SEARCH_TOOLS_AVAILABLE:
            raise RuntimeError("Naive sparse search tools are unavailable (import failed).")
        from sana_evaluation.tools.search.naive import (
            search_schema as search_schema_sparse,
            search_value as search_value_sparse,
        )

        return [search_value_sparse, search_schema_sparse, search_prefix]

    if search_mode == "standard":
        if not _STANDARD_SEARCH_TOOLS_AVAILABLE:
            raise RuntimeError("Standard hybrid search tools are unavailable (import failed).")
        from sana_evaluation.tools.search.standard import (
            search_schema as search_schema_hybrid,
            search_value as search_value_hybrid,
        )

        return [search_value_hybrid, search_schema_hybrid, search_prefix]

    if search_mode == "preloaded":
        return []

    import sana_evaluation.tools.oracle.search as search_ideal

    search_ideal.set_task_context(task_context or {})
    return [search_ideal.search_ideal]


def build_plan(
    mode: str,
    *,
    search_tool_mode: str,
    task_context: Optional[Dict[str, Any]],
    plan_skills_enabled: bool = False,
    benchmark: str = "lakeqa",
    no_s3: bool = False,
) -> tuple[str, List[Any], bool, bool, str]:
    """Return stable system prompt, planning tools, behavior toggles, and a task-specific trailer.

    The trailer (gold reasoning chain, preloaded dataset URIs) is task-specific and must be
    appended AFTER all variant-stable injections so the cacheable prefix stays intact across tasks.
    """
    plan_mode = _normalize_mode(mode, "standard", "plan")

    trailer_sections: List[str] = []
    if plan_mode == "ideal":
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
    if plan_mode == "naive":
        if benchmark_name == "kramabench":
            return compose_kramabench_prompt(search_tool_mode, include_skills=False), [], False, False, task_trailer
        return compose_baseline_prompt(search_tool_mode, no_s3=no_s3), [], False, False, task_trailer

    if benchmark_name == "kramabench":
        prompt = compose_kramabench_prompt(
            search_tool_mode,
            include_skills=bool(plan_skills_enabled),
        )
    else:
        prompt = compose_managed_prompt(
            search_tool_mode,
            include_skills=bool(plan_skills_enabled),
            no_s3=no_s3,
        )
    if plan_mode == "standard":
        return prompt, [plan], bool(plan_skills_enabled), True, task_trailer

    return prompt, [plan_ideal], bool(plan_skills_enabled), True, task_trailer


def build_search_results(
    mode: str,
    *,
    base_search_tools: Sequence[DecoratedFunctionTool],
    fixed_k: Optional[int],
) -> List[DecoratedFunctionTool]:
    """Configure the search_results axis (minimal | rich).

    Named for the axis, not for run results: this shapes search *payloads*.
    AgentResult and the CSV writers are unrelated and live in runner/record.py
    and runner/reporting.py.
    """
    if not base_search_tools:
        return []
    results_mode = _normalize_result_mode(mode, "naive", "search_results")
    return build_search_tools_by_mode(
        base_search_tools,
        fixed_k=fixed_k,
        results_mode=results_mode,
    )


_S3_DATA_TOOLS = (
    "list_files", "peek_file", "peek_multiple", "read_file", "grep_file",
    "parse_xml_records", "query_file",
)


def build_data_tools(*, no_s3: bool = False, search_tool_mode: Optional[str] = None) -> List[Any]:
    """Return the core data-manipulation tool surface.

    Without the lake, every S3-backed tool is dropped rather than left in place
    to fail at call time, and ``download`` is swapped for the web fetcher. What
    remains is fetch-then-compute.

    Web search implies this. The lake tools reject an ``http(s)`` URL, and web
    mode has no lake search to find lake sources with, so leaving them in place
    only offers the agent tools that cannot work -- and would contradict the web
    overlay, which tells it there is no data lake.
    """
    if no_s3 or _normalize_mode(search_tool_mode, "naive", "search_tool") == "web":
        return [download_web, execute_code, submit_answer]
    return [
        list_files, peek_file, peek_multiple, read_file, grep_file,
        parse_xml_records, query_file, download, execute_code,
        submit_answer,
    ]


def build_mode_bundle(
    run_config: RunConfig,
    *,
    data_tools: Sequence[Any],
    task_context: Optional[Dict[str, Any]] = None,
) -> ModeBundle:
    """Build final tools/prompt/plugin toggles from multi-axis modes."""
    search_tool_mode = _normalize_mode(run_config.search_tool_mode, "standard", "search_tool")
    search_results_mode = _normalize_result_mode(run_config.search_results_mode, "rich", "search_results")
    plan_mode = _normalize_mode(
        run_config.plan_mode or run_config.plan_mode,
        "standard",
        "plan",
    )
    computation_tool_mode = _normalize_computation_mode(run_config.computation_tool_mode)
    benchmark = (getattr(run_config, "benchmark", None) or "lakeqa").strip().lower()

    if search_tool_mode == "ideal" or plan_mode == "ideal" or computation_tool_mode == "ideal":
        set_ideal_profile_task_context(task_context or {})

    _validate_search_mode_combination(
        search_tool_mode=search_tool_mode,
        search_results_mode=search_results_mode,
        plan_mode=plan_mode,
        computation_tool_mode=computation_tool_mode,
        no_s3=bool(getattr(run_config, "no_s3", False)),
    )

    raw_search_tools = build_search(
        search_tool_mode,
        task_context=task_context,
        fixed_k=run_config.search_k,
    )
    search_tools = build_search_results(
        search_results_mode,
        base_search_tools=raw_search_tools,
        fixed_k=run_config.search_k,
    )
    system_prompt, plan_tools, enable_skills, enable_stagnation, task_trailer = build_plan(
        plan_mode,
        search_tool_mode=search_tool_mode,
        task_context=task_context,
        plan_skills_enabled=bool(run_config.plan_skills_enabled),
        benchmark=benchmark,
        no_s3=bool(getattr(run_config, "no_s3", False)),
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

    tools = list(search_tools) + list(plan_tools) + list(data_tool_list)
    return ModeBundle(
        tools=tools,
        system_prompt=system_prompt,
        search_tool_names=search_tool_names_in_mode(search_tools),
        enable_skills=enable_skills,
        enable_stagnation=enable_stagnation,
        modes={
            "search_tool": search_tool_mode,
            "search_results": search_results_mode,
            "plan": plan_mode,
            "computation_tool": computation_tool_mode,
            "plan_skills": "on" if run_config.plan_skills_enabled else "off",
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

    from sana_evaluation.tools.oracle import computation as computation_ideal

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


