"""Ideal search tool for query-driven retrieval over the authored source pool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from strands import Agent, tool

from sana_evaluation.config import AgentConfig
from sana_evaluation.instrumentation.agent_plugins import LoggingPlugin
from sana_evaluation.instrumentation.ideal_subagent_costs import record_subagent_call
from sana_evaluation.llm.llm_factory import build_model
from sana_evaluation.tools.external.ideal.subagent_models import search_model_name
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    load_runtime_profile_for_context,
    set_runtime_profiles_root as _set_runtime_profiles_root_shared,
    set_task_context as _set_task_context_shared,
)
from sana_evaluation.tools.external.ideal.benchmark_paths import canonical_source_uri
from sana_evaluation.tools.external.ideal import search_wrapper

logger = logging.getLogger(__name__)

_DATASET_NOT_FOUND = "Dataset not found"
_JUDGE_SYSTEM_PROMPT = (
    "You are a dataset selector. Given a query and candidate datasets, call `pick` exactly once "
    "with the most relevant s3_uris, or an empty list if none are close. For aggregate queries "
    '(year ranges, "all of", multiple regions), pick the matching group. For filtered subset '
    "questions, pick the dataset with records and fields needed to filter that subset; it does "
    "not need to mention the exact filter value. Pick multiple datasets only when the query asks "
    "to compare, join, subtract, or aggregate across them. Never pick an s3_uri not in the list."
)

# The eval runner uses one task at a time per process; parallel runs fork subprocesses.
_CANDIDATES: list[tuple[str, str]] = []
_USED_S3_URIS: set[str] = set()
_LESSGUIDE = False


def set_db_path(path: str) -> None:
    """Retained for compatibility; ideal retrieval ignores db_path."""
    _ = path


def set_runtime_profiles_root(path: str | Path) -> None:
    """Override runtime profiles root (mainly for tests)."""
    _set_runtime_profiles_root_shared(path)
    _CANDIDATES.clear()
    _USED_S3_URIS.clear()


def set_lessguide(enabled: bool) -> None:
    """Configure whether search_ideal omits plan-exhaustion guidance fields."""
    global _LESSGUIDE
    _LESSGUIDE = bool(enabled)


def set_task_context(task_context: Dict[str, Any]) -> None:
    """Load the active task runtime profile and materialize the candidate pool."""
    task_context = dict(task_context or {})
    _set_task_context_shared(task_context)
    profile = load_runtime_profile_for_context(task_context)
    _CANDIDATES.clear()
    _CANDIDATES.extend(
        (_canonical_uri(source_path), _dataset_id_from_source(source_path))
        for source_path in profile.source_sequence
    )
    _USED_S3_URIS.clear()


def reset_state() -> None:
    """Reset in-process candidate/judge state."""
    _CANDIDATES.clear()
    _USED_S3_URIS.clear()
    set_lessguide(False)
    _set_task_context_shared({})


def _dataset_id_from_source(source: str) -> str:
    value = str(source or "").strip()
    if "/" not in value:
        return value
    parts = value.split("/")
    if len(parts) >= 2 and parts[1].strip():
        return parts[1].strip()
    return value


def _canonical_uri(source_path: str) -> str:
    return canonical_source_uri(source_path)


def _description_for_uri(uri: str) -> str:
    search_wrapper._load_desc_cache()
    return search_wrapper._DESC_BY_URI.get(uri, "")


def _format_judge_prompt(query: str, remaining: list[tuple[str, str]]) -> str:
    lines = [f"Query: {query}", "", "Candidates:"]
    for index, (s3_uri, dataset_id) in enumerate(remaining, start=1):
        description = _description_for_uri(s3_uri)
        line = f"{index}. s3_uri={s3_uri}  dataset_id={dataset_id}"
        if description:
            line += f"  description={description}"
        lines.append(line)
    return "\n".join(lines)


def _judge_model():
    return build_model(AgentConfig(model_name=search_model_name()))


def _build_judge(remaining: list[tuple[str, str]]) -> tuple[Agent, Dict[str, Any]]:
    valid_uris = {uri for uri, _ in remaining}
    state: Dict[str, Any] = {"picked": None, "reason": None}

    @tool
    def pick(s3_uris: list[str], reason: str) -> str:
        """Record the s3_uris most relevant to the query. Call EXACTLY once."""
        if not isinstance(s3_uris, list):
            raise ValueError("s3_uris must be a list of strings")
        bad = [uri for uri in s3_uris if uri not in valid_uris]
        if bad:
            raise ValueError(
                f"Picked s3_uri(s) not in candidate list: {bad}. "
                "You must pick from the provided list only."
            )
        state["picked"] = list(dict.fromkeys(s3_uris))
        state["reason"] = reason
        return f"Recorded {len(state['picked'])} pick(s)."

    judge = Agent(
        model=_judge_model(),
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        tools=[pick],
        plugins=[LoggingPlugin()],
        callback_handler=None,
    )
    return judge, state


def _apply_lessguide(payload: dict) -> dict:
    if not _LESSGUIDE:
        return payload
    out = dict(payload)
    out.pop("plan_exhausted", None)
    return out


def _dataset_not_found_response(query: str, *, plan_exhausted: bool) -> dict:
    return _apply_lessguide(
        {
            "results": [],
            "count": 0,
            "query": query,
            "message": _DATASET_NOT_FOUND,
            "plan_exhausted": plan_exhausted,
        }
    )


@tool
def search_ideal(query: str, top_k: int = 100) -> dict:
    """Return the planned sources most relevant to ``query``."""
    _ = top_k

    if not _CANDIDATES:
        return {"error": "search_ideal called before set_task_context; no runtime profile loaded."}

    remaining = [(uri, dsid) for uri, dsid in _CANDIDATES if uri not in _USED_S3_URIS]
    if not remaining:
        return _dataset_not_found_response(query, plan_exhausted=True)

    judge, state = _build_judge(remaining)
    try:
        judge_result = judge(_format_judge_prompt(query, remaining))
    except Exception as exc:
        record_subagent_call(
            tool="search_ideal",
            subagent_kind="judge",
            model_name=search_model_name(),
            agent_result=None,
            success=False,
            candidate_count=len(remaining),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    record_subagent_call(
        tool="search_ideal",
        subagent_kind="judge",
        model_name=search_model_name(),
        agent_result=judge_result,
        success=True,
        candidate_count=len(remaining),
        selected_count=len(state["picked"] or []),
        decision_recorded=state["picked"] is not None,
    )

    if state["picked"] is None:
        logger.warning("search_ideal judge recorded no pick; returning dataset not found")
        return _dataset_not_found_response(query, plan_exhausted=False)

    picked = state["picked"]
    if not picked:
        logger.info("search_ideal judge found no similar dataset reason=%r", state["reason"])
        return _dataset_not_found_response(query, plan_exhausted=False)

    _USED_S3_URIS.update(picked)
    logger.info(
        "search_ideal judge picked %d uri(s) reason=%r",
        len(picked),
        state["reason"],
    )

    dsid_by_uri = dict(remaining)
    results = [{"s3_uri": uri, "dataset_id": dsid_by_uri[uri]} for uri in picked]
    return _apply_lessguide(
        {
            "results": results,
            "count": len(results),
            "query": query,
            "plan_exhausted": len(_USED_S3_URIS) >= len(_CANDIDATES),
        }
    )
