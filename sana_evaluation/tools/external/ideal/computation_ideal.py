"""Ideal computation tools backed by authored runtime profile records."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from strands import Agent, tool

from sana_evaluation.config import AgentConfig
from sana_evaluation.instrumentation.agent_plugins import LoggingPlugin
from sana_evaluation.instrumentation.ideal_subagent_costs import record_subagent_call
from sana_evaluation.llm.llm_factory import build_model
from sana_evaluation.tools.external.ideal.subagent_models import (
    repair_model_name,
    semantic_model_name,
)
from sana_evaluation.tools.agent_tools import _get_sandbox_dir, execute_code as _execute_code_tool
from sana_evaluation.tools.agent_tools_v2 import peek_file as _peek_file_tool
from sana_evaluation.tools.agent_tools_v2 import query_file as _query_file_tool
from sana_evaluation.tools.external.ideal.runtime_profile_store import (
    IdealComputationRecord,
    IdealRuntimeProfile,
    load_runtime_profile_for_context,
    set_task_context as _set_task_context_shared,
)
from sana_evaluation.tools.external.ideal.benchmark_paths import (
    canonical_source_uri,
    source_key,
)

logger = logging.getLogger(__name__)

_MAX_REPAIR_ATTEMPTS = 2
_MAX_ENHANCED_CONTEXT_CHARS = 10_000
_MAX_TARGET_CONTEXT_CHARS = 12_000

_SEMANTIC_JUDGE_SYSTEM_PROMPT = (
    "You judge semantic equivalence between a submitted computation "
    "tool call and authored computation records."
)
_QUERY_REPAIR_SYSTEM_PROMPT = "You repair DuckDB SQL for a benchmark tool call."
_CODE_REPAIR_SYSTEM_PROMPT = "You repair Python data-analysis code for a benchmark tool call."

_SEMANTIC_JUDGE_PROMPT_TEMPLATE = """Tool: {tool_name}
Submitted intent:
{intent}

Submitted {payload_label}:
{payload}

Candidate authored records:
{candidates_json}

Decide sequentially:
1. Call select_match first. Select the candidate's `selection_index` only if the submitted {payload_label} and/or intent are semantically equivalent to that authored record's computation for the same source and expected output shape. Select 0 if no authored record matches. Never select a record merely because its answer would be useful.
2. If and only if you selected 0, call record_intent_match. Set matches=true only if the submitted {payload_label} reasonably attempts the submitted intent. Set matches=false if the payload is unrelated, contradictory, answer-only, or only inspects schema/sample rows while the intent asks for a substantive computation. If the intent itself asks to inspect schema or sample rows, an inspection payload can match."""

_QUERY_REPAIR_PROMPT_TEMPLATE = """Rewrite the SQL so it accomplishes the user's intent against table alias t.
Call submit_repaired_sql exactly once. Return only executable DuckDB SQL via the tool.

Intent:
{intent}

Original SQL:
{sql}

Previous error:
{previous_error}

Runtime profile records:
{profile_records}

Target dataset context:
{target_dataset_context}"""

_CODE_REPAIR_PROMPT_TEMPLATE = """Rewrite the Python code so it accomplishes the user's intent in the sandbox.
The code must print the result. Call submit_repaired_code exactly once.

Intent:
{intent}

Original code:
{code}

Previous error:
{previous_error}

Runtime profile records:
{profile_records}

Target dataset context:
{target_dataset_context}

Sandbox context:
{sandbox_context}"""

_TASK_CONTEXT: Dict[str, Any] = {}
_ACTIVE_PROFILE: Optional[IdealRuntimeProfile] = None
_STATS: Dict[str, int] = {
    "execute_ideal_agent_repair_calls": 0,
    "query_ideal_agent_repair_calls": 0,
}

_base_query_file = getattr(_query_file_tool, "_tool_func", _query_file_tool)
_base_execute_code = getattr(_execute_code_tool, "_tool_func", _execute_code_tool)
_base_peek_file = getattr(_peek_file_tool, "_tool_func", _peek_file_tool)


@dataclass
class _SemanticDecision:
    record: Optional[IdealComputationRecord] = None
    intent_matches_payload: bool = False
    reason: str = ""


def set_task_context(task_context: Dict[str, Any]) -> None:
    """Load the active task's ideal computation records."""
    global _TASK_CONTEXT, _ACTIVE_PROFILE
    _TASK_CONTEXT = dict(task_context or {})
    _set_task_context_shared(_TASK_CONTEXT)
    _ACTIVE_PROFILE = load_runtime_profile_for_context(_TASK_CONTEXT)
    reset_stats()


def reset_state() -> None:
    """Reset in-process ideal computation state."""
    global _TASK_CONTEXT, _ACTIVE_PROFILE
    _TASK_CONTEXT = {}
    _ACTIVE_PROFILE = None
    _set_task_context_shared({})
    reset_stats()


def reset_stats() -> None:
    """Reset per-task ideal computation instrumentation counters."""
    for key in _STATS:
        _STATS[key] = 0


def get_stats() -> Dict[str, int]:
    """Return per-task ideal computation instrumentation counters."""
    return dict(_STATS)


def _record_repair_event(tool_name: str, event: str, **fields: Any) -> None:
    try:
        from sana_evaluation.instrumentation.trace_plugin import write_trace_record
    except Exception:
        return
    write_trace_record({"tool": tool_name, "event": event, **fields})


def _active_profile() -> IdealRuntimeProfile:
    if _ACTIVE_PROFILE is not None:
        return _ACTIVE_PROFILE
    return load_runtime_profile_for_context(_TASK_CONTEXT)


def _dataset_id_from_source(value: Any) -> str:
    text = source_key(str(value or ""))
    if not text:
        return ""
    text = text.lstrip("/")
    parts = text.split("/")
    if len(parts) >= 2 and parts[0] in {"datagov", "wikipedia"}:
        return parts[1]
    return parts[0]


def _canonical_uri(source: str) -> str:
    return canonical_source_uri(source)


def _s3_uri_for_record(record: IdealComputationRecord) -> str:
    return _canonical_uri(record.source)


def _file_path_for_record(record: IdealComputationRecord) -> str:
    marker = f"/{record.dataset_id}/"
    if marker in record.source:
        return record.source.split(marker, 1)[1]
    parts = record.source.split("/")
    if len(parts) >= 3 and parts[1] == record.dataset_id:
        return "/".join(parts[2:])
    return record.source


def _query_answer_payload(record: IdealComputationRecord) -> Dict[str, Any]:
    return {
        "success": True,
        "dataset_id": record.dataset_id,
        "file_path": _file_path_for_record(record),
        "s3_uri": _s3_uri_for_record(record),
        "columns": ["answer"],
        "rows": [[record.answer]],
        "row_count": 1,
        "truncated": False,
    }


def _blocked_query_payload(record: IdealComputationRecord) -> Dict[str, Any]:
    return {
        "success": False,
        "error": str(record.answer),
        "dataset_id": record.dataset_id,
        "file_path": _file_path_for_record(record),
        "s3_uri": _s3_uri_for_record(record),
        "recommendation": "Use execute_ideal or download-style code for this source; query_ideal only supports files that can be queried directly.",
    }


def _execute_answer_payload(record: IdealComputationRecord) -> Dict[str, Any]:
    return {
        "output": str(record.answer),
        "success": True,
        "dataset_id": record.dataset_id,
        "file_path": _file_path_for_record(record),
        "s3_uri": _s3_uri_for_record(record),
    }


def _is_success(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("success") is False:
        return False
    return "error" not in result


def _profile_records_for_prompt(profile: IdealRuntimeProfile, records: Iterable[IdealComputationRecord]) -> str:
    rows = []
    for index, record in enumerate(records, start=1):
        rows.append(
            {
                "selection_index": index,
                "node_id": record.node_id,
                "dataset_id": record.dataset_id,
                "source": record.source,
                "intent": record.intent,
                "payload": record.payload,
                "expected_output": str(record.answer),
                "blocked": bool(getattr(record, "blocked", False)),
            }
        )
    return json.dumps(
        {
            "task_id": profile.task_id,
            "profile_path": str(profile.profile_path),
            "records": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


def _enhanced_peek_context(source: str) -> str:
    if not source:
        return ""
    s3_uri = _canonical_uri(source)
    payload: Dict[str, Any] = {"s3_uri": s3_uri}
    try:
        peek = _base_peek_file(s3_uri=s3_uri, max_rows=5)
    except Exception as exc:
        peek = {"error": f"enhanced peek failed: {type(exc).__name__}: {exc}"}
    payload["enhanced_peek_file"] = peek

    try:
        from sana_evaluation.helper.peek_profile import load_dataset_profile, select_dataset_profile_fields
        profile = load_dataset_profile(s3_uri)
    except Exception:
        profile = None
    profile = select_dataset_profile_fields(profile)
    if profile:
        payload["dataset_profile"] = profile
        description = profile.get("llm_description") or profile.get("description")
        if description:
            payload["dataset_description"] = str(description)

    return json.dumps(payload, ensure_ascii=False)[:_MAX_ENHANCED_CONTEXT_CHARS]


def _target_sources_for_records(records: Iterable[IdealComputationRecord]) -> List[str]:
    out: List[str] = []
    for record in records:
        source = record.source
        if source and source not in out:
            out.append(source)
        if len(out) >= 3:
            break
    return out


def _target_dataset_context(records: Iterable[IdealComputationRecord]) -> str:
    contexts = []
    for source in _target_sources_for_records(records):
        context = _enhanced_peek_context(source)
        if context:
            contexts.append({"source": source, "context": json.loads(context)})
    if not contexts:
        return ""
    return json.dumps(contexts, ensure_ascii=False)[:_MAX_TARGET_CONTEXT_CHARS]


def _sandbox_context() -> str:
    try:
        sandbox = _get_sandbox_dir()
    except Exception:
        return ""
    files: List[str] = []
    if Path(sandbox).exists():
        for path in Path(sandbox).rglob("*"):
            if path.is_file():
                try:
                    files.append(str(path.relative_to(sandbox)))
                except ValueError:
                    files.append(str(path))
    return json.dumps({"sandbox_dir": str(sandbox), "files": sorted(files)[:200]}, indent=2)


def _repair_model():
    return build_model(
        AgentConfig(
            model_name=repair_model_name(),
            max_tokens=4096,
        )
    )


def _semantic_model():
    return build_model(AgentConfig(model_name=semantic_model_name()))


def _normalized_file_path(value: Any) -> str:
    text = source_key(str(value or ""))
    text = text.lstrip("/")
    parts = text.split("/")
    if len(parts) >= 3 and parts[0] in {"datagov", "wikipedia"}:
        return "/".join(parts[2:])
    return text


def _records_for_target(
    records: Iterable[IdealComputationRecord],
    *,
    dataset_id: str = "",
    file_path: str = "",
    s3_uri: str = "",
    fallback_all: bool = False,
) -> List[IdealComputationRecord]:
    rows = list(records)
    if not rows:
        return []

    target_dataset = _dataset_id_from_source(dataset_id) or _dataset_id_from_source(s3_uri)
    target_file = _normalized_file_path(file_path or s3_uri)

    if s3_uri:
        wanted_uri = _canonical_uri(s3_uri)
        exact = [record for record in rows if _canonical_uri(record.source) == wanted_uri]
        if exact:
            return exact

    if target_dataset and target_file:
        exact_file = [
            record
            for record in rows
            if record.dataset_id == target_dataset
            and _normalized_file_path(record.source) == target_file
        ]
        if exact_file:
            return exact_file

    if target_file:
        file_matches = [
            record
            for record in rows
            if (not target_dataset or record.dataset_id == target_dataset)
            and (
                _normalized_file_path(record.source) == target_file
                or _normalized_file_path(record.source).endswith(f"/{target_file}")
                or target_file.endswith(f"/{_normalized_file_path(record.source)}")
            )
        ]
        if file_matches:
            return file_matches

    if target_dataset:
        dataset_matches = [record for record in rows if record.dataset_id == target_dataset]
        if dataset_matches:
            return dataset_matches

    grep_terms = [
        term.lower()
        for term in {
            str(dataset_id or "").strip(),
            str(file_path or "").strip(),
            Path(target_file).name if target_file else "",
        }
        if term
    ]
    if grep_terms:
        grep_matches = [
            record
            for record in rows
            if (not target_dataset or record.dataset_id == target_dataset)
            and any(term in f"{record.dataset_id}/{record.source}".lower() for term in grep_terms)
        ]
        if grep_matches:
            return grep_matches

    return rows if fallback_all else []


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "y", "1", "match", "matches"}:
        return True
    if text in {"false", "no", "n", "0", "mismatch", "does not match"}:
        return False
    return bool(value)


def _semantic_decision(
    *,
    profile: IdealRuntimeProfile,
    tool_name: str,
    payload_label: str,
    payload: str,
    intent: str,
    candidates: Iterable[IdealComputationRecord],
) -> _SemanticDecision:
    records = list(candidates)

    state: Dict[str, Any] = {
        "index": None,
        "match_reason": "",
        "intent_matches": None,
        "intent_reason": "",
    }

    @tool
    def select_match(index: int, reason: str) -> str:
        """Select the equivalent candidate by 1-based index, or 0 for no match."""
        if not isinstance(index, int):
            raise ValueError("index must be an integer")
        if index < 0 or index > len(records):
            raise ValueError(f"index must be between 0 and {len(records)}")
        state["index"] = index
        state["match_reason"] = reason
        return "semantic match recorded"

    @tool
    def record_intent_match(matches: bool, reason: str) -> str:
        """Record whether the submitted payload matches its stated intent."""
        state["intent_matches"] = _coerce_bool(matches)
        state["intent_reason"] = str(reason or "")
        return "intent match recorded"

    candidates_json = _profile_records_for_prompt(profile, records)
    prompt = _SEMANTIC_JUDGE_PROMPT_TEMPLATE.format(
        tool_name=tool_name,
        intent=intent,
        payload_label=payload_label,
        payload=payload,
        candidates_json=candidates_json,
    )
    judge = Agent(
        model=_semantic_model(),
        system_prompt=_SEMANTIC_JUDGE_SYSTEM_PROMPT,
        tools=[select_match, record_intent_match],
        plugins=[LoggingPlugin()],
        callback_handler=None,
    )
    try:
        judge_result = judge(prompt)
    except Exception as exc:
        record_subagent_call(
            tool=tool_name,
            subagent_kind="semantic_judge",
            model_name=semantic_model_name(),
            agent_result=None,
            success=False,
            candidate_count=len(records),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    record_subagent_call(
        tool=tool_name,
        subagent_kind="semantic_judge",
        model_name=semantic_model_name(),
        agent_result=judge_result,
        success=True,
        candidate_count=len(records),
        selected_index=state["index"],
        decision_recorded=state["index"] is not None,
        intent_matches_payload=state["intent_matches"],
        intent_decision_recorded=state["intent_matches"] is not None,
    )
    index = state["index"]
    if index is None:
        logger.warning("%s semantic judge recorded no decision", tool_name)
        return _SemanticDecision(
            record=None,
            intent_matches_payload=False,
            reason="semantic judge recorded no decision",
        )
    if index == 0:
        if state["intent_matches"] is None:
            logger.warning("%s semantic judge recorded no intent/payload decision", tool_name)
            return _SemanticDecision(
                record=None,
                intent_matches_payload=False,
                reason="semantic judge recorded no intent/payload decision",
            )
        logger.info(
            "%s semantic judge found no equivalent record reason=%r; intent_matches=%s intent_reason=%r",
            tool_name,
            state["match_reason"],
            state["intent_matches"],
            state["intent_reason"],
        )
        return _SemanticDecision(
            record=None,
            intent_matches_payload=bool(state["intent_matches"]),
            reason=str(state["intent_reason"] or state["match_reason"] or ""),
        )

    logger.info("%s semantic judge matched record index=%s reason=%r", tool_name, index, state["match_reason"])
    return _SemanticDecision(record=records[index - 1], intent_matches_payload=True, reason=str(state["match_reason"] or ""))


def _semantic_record_match(
    *,
    profile: IdealRuntimeProfile,
    tool_name: str,
    payload_label: str,
    payload: str,
    intent: str,
    candidates: Iterable[IdealComputationRecord],
) -> Optional[IdealComputationRecord]:
    return _semantic_decision(
        profile=profile,
        tool_name=tool_name,
        payload_label=payload_label,
        payload=payload,
        intent=intent,
        candidates=candidates,
    ).record


def _repair_query(
    *,
    profile: IdealRuntimeProfile,
    sql: str,
    intent: str,
    records: Optional[Iterable[IdealComputationRecord]] = None,
    previous_error: str = "",
    attempt: int | None = None,
) -> Dict[str, str]:
    state: Dict[str, str] = {}

    @tool
    def submit_repaired_sql(sql: str, reason: str) -> str:
        """Submit the repaired SQL. Call exactly once."""
        state["sql"] = sql
        state["reason"] = reason
        return "repaired SQL recorded"

    prompt_records = list(records) if records is not None else list(profile.ideal_query)
    if not prompt_records:
        prompt_records = list(profile.ideal_query)
    prompt = _QUERY_REPAIR_PROMPT_TEMPLATE.format(
        intent=intent,
        sql=sql,
        previous_error=previous_error,
        profile_records=_profile_records_for_prompt(profile, prompt_records),
        target_dataset_context=_target_dataset_context(prompt_records),
    )
    repair_agent = Agent(
        model=_repair_model(),
        system_prompt=_QUERY_REPAIR_SYSTEM_PROMPT,
        tools=[submit_repaired_sql],
        plugins=[LoggingPlugin()],
        callback_handler=None,
    )
    try:
        repair_result = repair_agent(prompt)
    except Exception as exc:
        record_subagent_call(
            tool="query_ideal",
            subagent_kind="repair_agent",
            model_name=repair_model_name(),
            agent_result=None,
            attempt=attempt,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    record_subagent_call(
        tool="query_ideal",
        subagent_kind="repair_agent",
        model_name=repair_model_name(),
        agent_result=repair_result,
        attempt=attempt,
        success=True,
        repair_tool_called=bool(state.get("sql")),
    )
    return state


def _repair_code(
    *,
    profile: IdealRuntimeProfile,
    code: str,
    intent: str,
    records: Optional[Iterable[IdealComputationRecord]] = None,
    previous_error: str = "",
    attempt: int | None = None,
) -> Dict[str, str]:
    state: Dict[str, str] = {}

    @tool
    def submit_repaired_code(code: str, reason: str) -> str:
        """Submit the repaired Python code. Call exactly once."""
        state["code"] = code
        state["reason"] = reason
        return "repaired code recorded"

    prompt_records = list(records) if records is not None else list(profile.ideal_code)
    if not prompt_records:
        prompt_records = list(profile.ideal_code)
    prompt = _CODE_REPAIR_PROMPT_TEMPLATE.format(
        intent=intent,
        code=code,
        previous_error=previous_error,
        profile_records=_profile_records_for_prompt(profile, prompt_records),
        target_dataset_context=_target_dataset_context(prompt_records),
        sandbox_context=_sandbox_context(),
    )
    repair_agent = Agent(
        model=_repair_model(),
        system_prompt=_CODE_REPAIR_SYSTEM_PROMPT,
        tools=[submit_repaired_code],
        plugins=[LoggingPlugin()],
        callback_handler=None,
    )
    try:
        repair_result = repair_agent(prompt)
    except Exception as exc:
        record_subagent_call(
            tool="execute_ideal",
            subagent_kind="repair_agent",
            model_name=repair_model_name(),
            agent_result=None,
            attempt=attempt,
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    record_subagent_call(
        tool="execute_ideal",
        subagent_kind="repair_agent",
        model_name=repair_model_name(),
        agent_result=repair_result,
        attempt=attempt,
        success=True,
        repair_tool_called=bool(state.get("code")),
    )
    return state


@tool
def query_ideal(
    dataset_id: str | None = None,
    file_path: str | None = None,
    sql: str = "",
    s3_uri: str | None = None,
    intent: str = "",
) -> Dict[str, Any]:
    """
    Ideal SQL tool for tabular or JSON-like sources only.
    Do not use query_ideal for non-tabular/non-JSON sources such as
    Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, or binary files.
    For XML/KML structured records, use parse_xml_records instead.
    """
    profile = _active_profile()
    dataset_id = dataset_id or ""
    file_path = file_path or ""
    s3_uri = s3_uri or ""
    candidate_records = _records_for_target(
        profile.ideal_query,
        dataset_id=dataset_id,
        file_path=file_path,
        s3_uri=s3_uri,
        fallback_all=True,
    )
    runnable_records = [
        record for record in candidate_records if not getattr(record, "blocked", False)
    ]
    blocked_record = next(
        (record for record in candidate_records if getattr(record, "blocked", False)),
        None,
    )
    if not runnable_records and blocked_record is not None:
        return _blocked_query_payload(blocked_record)

    decision = _semantic_decision(
        profile=profile,
        tool_name="query_ideal",
        payload_label="SQL",
        payload=sql,
        intent=intent,
        candidates=runnable_records,
    )
    record = decision.record
    if record is not None:
        return _query_answer_payload(record)

    if blocked_record is not None:
        return _blocked_query_payload(blocked_record)

    if decision.intent_matches_payload:
        try:
            last_result = _base_query_file(
                dataset_id=dataset_id,
                file_path=file_path,
                sql=sql,
                s3_uri=s3_uri,
            )
        except Exception as exc:
            last_result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        if _is_success(last_result):
            return dict(last_result)
        previous_error = str(last_result.get("error") or last_result)
    else:
        previous_error = f"Intent/payload mismatch: {decision.reason}".strip()
        last_result = {"success": False, "error": previous_error}
    for attempt in range(1, _MAX_REPAIR_ATTEMPTS + 1):
        logger.info("query_ideal repair agent invoked attempt=%s", attempt)
        _STATS["query_ideal_agent_repair_calls"] += 1
        _record_repair_event(
            "query_ideal",
            "repair_agent_invoked",
            attempt=attempt,
            previous_error=previous_error,
        )
        try:
            repaired = _repair_query(
                profile=profile,
                sql=sql,
                intent=intent,
                records=candidate_records or profile.ideal_query,
                previous_error=previous_error,
                attempt=attempt,
            )
        except Exception as exc:
            error = f"SQL repair failed: {type(exc).__name__}: {exc}"
            _record_repair_event(
                "query_ideal",
                "repair_agent_failed",
                attempt=attempt,
                error=error,
            )
            last_result = {"success": False, "error": error}
            previous_error = error
            continue
        repaired_sql = str(repaired.get("sql") or "").strip()
        _record_repair_event(
            "query_ideal",
            "repair_agent_completed",
            attempt=attempt,
            repair_reason=str(repaired.get("reason") or ""),
            submitted_sql=repaired_sql,
        )
        if not repaired_sql:
            last_result = {"success": False, "error": "SQL repair returned empty SQL."}
            previous_error = last_result["error"]
            continue
        last_result = _base_query_file(
            dataset_id=dataset_id,
            file_path=file_path,
            sql=repaired_sql,
            s3_uri=s3_uri,
        )
        if _is_success(last_result):
            return dict(last_result)
        previous_error = str(last_result.get("error") or last_result)

    return dict(last_result)


@tool
def execute_ideal(
    code: str,
    intent: str,
    dataset_id: str | None = None,
    file_path: str | None = None,
    s3_uri: str | None = None,
) -> Dict[str, Any]:
    """
    Ideal Python tool for tabular or JSON-like sources only.
    Do not use execute_ideal for non-tabular/non-JSON sources such as
    Wikipedia/content.txt, prose/plain text, XML/KML, HTML, PDFs, or binary files.
    For XML/KML structured records, use parse_xml_records instead.
    """
    profile = _active_profile()
    dataset_id = dataset_id or ""
    file_path = file_path or ""
    s3_uri = s3_uri or ""
    candidate_records = _records_for_target(
        profile.ideal_code,
        dataset_id=dataset_id,
        file_path=file_path,
        s3_uri=s3_uri,
        fallback_all=True,
    )
    decision = _semantic_decision(
        profile=profile,
        tool_name="execute_ideal",
        payload_label="Python code",
        payload=code,
        intent=intent,
        candidates=candidate_records,
    )
    record = decision.record
    if record is not None:
        return _execute_answer_payload(record)

    if decision.intent_matches_payload:
        try:
            last_result = _base_execute_code(code)
        except Exception as exc:
            last_result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        if _is_success(last_result):
            return dict(last_result)
        previous_error = str(last_result.get("error") or last_result)
    else:
        previous_error = f"Intent/payload mismatch: {decision.reason}".strip()
        last_result = {"success": False, "error": previous_error}
    for attempt in range(1, _MAX_REPAIR_ATTEMPTS + 1):
        logger.info("execute_ideal repair agent invoked attempt=%s", attempt)
        _STATS["execute_ideal_agent_repair_calls"] += 1
        _record_repair_event(
            "execute_ideal",
            "repair_agent_invoked",
            attempt=attempt,
            previous_error=previous_error,
        )
        try:
            repaired = _repair_code(
                profile=profile,
                code=code,
                intent=intent,
                records=candidate_records or profile.ideal_code,
                previous_error=previous_error,
                attempt=attempt,
            )
        except Exception as exc:
            error = f"Code repair failed: {type(exc).__name__}: {exc}"
            _record_repair_event(
                "execute_ideal",
                "repair_agent_failed",
                attempt=attempt,
                error=error,
            )
            last_result = {"success": False, "error": error}
            previous_error = error
            continue
        repaired_code = str(repaired.get("code") or "").strip()
        _record_repair_event(
            "execute_ideal",
            "repair_agent_completed",
            attempt=attempt,
            repair_reason=str(repaired.get("reason") or ""),
            submitted_code=repaired_code,
        )
        if not repaired_code:
            last_result = {"success": False, "error": "Code repair returned empty code."}
            previous_error = last_result["error"]
            continue
        last_result = _base_execute_code(repaired_code)
        if _is_success(last_result):
            return dict(last_result)
        previous_error = str(last_result.get("error") or last_result)

    return dict(last_result)


__all__ = [
    "execute_ideal",
    "query_ideal",
    "reset_state",
    "get_stats",
    "set_task_context",
]
