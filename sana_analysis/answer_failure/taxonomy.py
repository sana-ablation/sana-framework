"""Shared taxonomy for answer-failure audits, validation, and grouping."""
from __future__ import annotations


ANSWER_FAILURE_TYPE_DEFINITIONS = {
    "question_or_constraint_misread": (
        "The agent misunderstood the task, answer type, constraint, time window, "
        "geography, entity, aggregation instruction, or comparison."
    ),
    "planning_decomposition_mismatch": (
        "A reference plan or ideal chain specifies required reasoning hops, but "
        "the agent's executed trajectory materially followed a different decomposition."
    ),
    "wrong_source_or_dataset": (
        "The agent used the wrong dataset, file, table, database, source family, "
        "source version, or external source."
    ),
    "wrong_scope_or_filter": (
        "The agent used the right general source but selected the wrong entity, "
        "branch, time period, geography, jurisdiction, population, source slice, "
        "row subset, predicate, filter, or inclusion/exclusion criteria."
    ),
    "computation_or_aggregation_error": (
        "The correct evidence population was plausibly selected, but the agent made "
        "an error in math, aggregation, ranking, grouping, deduplication, join "
        "cardinality, denominator, sign, unit conversion, or rounding."
    ),
    "low_yield_search_loop": (
        "Repeated searches or discovery calls revealed no useful new evidence or "
        "rediscovered already-known sources."
    ),
    "schema_or_shape_inspection_loop": (
        "Repeated schema, header, preview, metadata, catalog, or file-shape "
        "inspection replaced evidence extraction."
    ),
    "query_execution_error_loop": (
        "Repeated SQL, code, query, repair, malformed-call, or near-duplicate "
        "execution failures blocked progress."
    ),
    "same_hop_repetition": (
        "The agent kept redoing one evidence hop after enough useful information "
        "existed to move on."
    ),
    "incomplete_evidence_early_answer": (
        "The agent submitted before required evidence was gathered, without a clear "
        "hard budget/tool/data blocker."
    ),
    "incomplete_evidence_budget_exhausted": (
        "The agent was still doing necessary work when turn, time, token, or tool "
        "budget ran out."
    ),
    "tool_or_data_blocker": (
        "Needed evidence was blocked by unavailable data, oversized/unsupported "
        "formats, tool crashes, context limits, or access failures."
    ),
    "extraction_or_parsing_error": (
        "The right source and relevant evidence region were found, but the agent "
        "read the wrong field, cell, nested value, text span, or structured item."
    ),
    "evidence_available_answer_error": (
        "The correct evidence or computed answer was present in the run, but the "
        "final response selected, synthesized, formatted, or submitted the wrong answer."
    ),
    "semantic_or_gold_label_issue": (
        "The expected answer, semantic label, or benchmark task appears wrong or "
        "materially ambiguous."
    ),
    "other_or_unclear": "A grounded failure is visible but does not fit another type cleanly.",
    "ungroundable": (
        "Logs, task data, or evidence are missing/malformed enough that no reliable "
        "diagnosis can be made."
    ),
}

ANSWER_FAILURE_TYPES = set(ANSWER_FAILURE_TYPE_DEFINITIONS)

FAILURE_STAGES = {
    "task_understanding",
    "planning",
    "source_selection",
    "data_access",
    "query_or_code_construction",
    "evidence_gathering",
    "computation",
    "extraction",
    "finalization",
    "submission",
    "validation",
}

FAILURE_STAGE_ORDER = [
    "task_understanding",
    "planning",
    "source_selection",
    "data_access",
    "query_or_code_construction",
    "evidence_gathering",
    "computation",
    "extraction",
    "finalization",
    "submission",
    "validation",
]

BLOCKER_SUBTYPES = {
    "tool_budget_exhausted",
    "turn_or_time_budget_exhausted",
    "token_or_context_exhausted",
    "runner_or_event_loop_exception",
    "tool_status_or_transport_error",
    "malformed_tool_call",
    "data_source_missing_or_unavailable",
    "unsupported_or_oversized_data_access",
    "ideal_repair_failure",
    "missing_or_malformed_log",
}

BOUNDARY_RULES = [
    "If logs, task data, or evidence are too missing or malformed to diagnose, use ungroundable.",
    "If the expected answer, semantic label, rubric, or benchmark task appears wrong or materially ambiguous, use semantic_or_gold_label_issue.",
    "If a tool or source made needed evidence inaccessible, use tool_or_data_blocker.",
    "If the agent misunderstood the user question or explicit constraints before selecting evidence, use question_or_constraint_misread. If the misunderstanding only manifests as a wrong source, scope, filter, or computation later, record it as a subtype or secondary note.",
    "If a reference plan or ideal chain exists and the executed trajectory materially skipped, substituted, or reordered required reasoning hops, use planning_decomposition_mismatch.",
    "If the agent used the wrong dataset, file, table, database, source family, source version, or external source, use wrong_source_or_dataset.",
    "If the agent used the right general source but selected the wrong entity, branch, time period, geography, jurisdiction, population, source slice, row subset, predicate, filter, or inclusion/exclusion criteria, use wrong_scope_or_filter.",
    "If the right source and evidence region were found but the agent read the wrong field, cell, text span, nested value, or structured item, use extraction_or_parsing_error.",
    "If the correct evidence population was selected but math, aggregation, ranking, grouping, deduplication, join cardinality, denominator, units, sign, or rounding were wrong, use computation_or_aggregation_error.",
    "If the correct evidence or computed answer appeared in the run but the final response reported something else, use evidence_available_answer_error.",
    "If the agent submitted before gathering required evidence despite no clear hard budget/data/tool blocker, use incomplete_evidence_early_answer.",
    "If the agent was still doing necessary work when turn, time, token, context, or tool budget ran out, use incomplete_evidence_budget_exhausted.",
    "If repeated searches, schema inspections, query repairs, or repeated hops consumed progress without advancing evidence gathering, use the appropriate loop label.",
    "Use other_or_unclear only for a grounded failure that does not fit the taxonomy.",
]

ANSWER_FAILURE_FIGURE_GROUPS = {
    "question_or_constraint_misread": "Task/planning failures",
    "planning_decomposition_mismatch": "Task/planning failures",
    "wrong_source_or_dataset": "Wrong source target failures",
    "wrong_scope_or_filter": "Execution/computation failures",
    "computation_or_aggregation_error": "Execution/computation failures",
    "extraction_or_parsing_error": "Execution/computation failures",
    "incomplete_evidence_budget_exhausted": "Incomplete evidence failures",
    "incomplete_evidence_early_answer": "Incomplete evidence failures",
    "query_execution_error_loop": "Turn-waste failures",
    "low_yield_search_loop": "Turn-waste failures",
    "schema_or_shape_inspection_loop": "Turn-waste failures",
    "same_hop_repetition": "Turn-waste failures",
    "evidence_available_answer_error": "Finalization failures",
    "tool_or_data_blocker": "Tool blocker failures",
}

OMITTED_ANSWER_FAILURE_TYPES = {
    "semantic_or_gold_label_issue",
    "other_or_unclear",
    "ungroundable",
}


def format_answer_failure_type_definitions() -> str:
    return "\n".join(
        f"- {failure_type}: {definition}"
        for failure_type, definition in ANSWER_FAILURE_TYPE_DEFINITIONS.items()
    )


def format_failure_stages() -> str:
    return ", ".join(FAILURE_STAGE_ORDER)


def format_blocker_subtypes() -> str:
    return "\n".join(f"- {subtype}" for subtype in sorted(BLOCKER_SUBTYPES))


def format_boundary_rules() -> str:
    return "\n".join(f"- {rule}" for rule in BOUNDARY_RULES)
