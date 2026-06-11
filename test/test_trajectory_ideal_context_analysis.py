import json
from pathlib import Path

from sana_analysis.running_analysis.trajectory_ideal_context_analysis import (
    TARGET_MODES,
    build_judge_prompt,
    build_trajectory_rows,
    default_journal_path,
    default_tmp_root,
    parse_log_run_status,
    summarize_rows,
    write_outputs,
)


def _write_log(
    path: Path,
    *,
    runner: str = "gpt-test",
    plan_tool: str | None = None,
    limit_marker: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "2026 | INFO | x | NEW TASK: " + runner,
        "2026 | INFO | x | --- Turn 1 ---",
    ]
    if plan_tool:
        lines.append(f'2026 | INFO | x | Executing: {plan_tool}({{"plan_text": "do work"}})')
        lines.append("2026 | DEBUG | x | Tool result: Plan recorded.")
    lines.extend(
        [
            '2026 | INFO | x | Executing: search_ideal({"query": "dataset"})',
            '2026 | DEBUG | x | Tool result: {"results": [{"dataset_id": "ds"}]}',
            '2026 | INFO | x | Executing: execute_ideal({"node_id": "1"})',
            '2026 | DEBUG | x | Tool result: {"answer": "A"}',
            "2026 | INFO | x | --- Turn 2 ---",
        ]
    )
    if limit_marker:
        lines.append(f"2026 | DEBUG | x | Tool result: Tool call cancelled. {limit_marker}")
    lines.extend(
        [
            '2026 | INFO | x | Executing: submit_answer({"answer": "A"})',
            "2026 | INFO | x | ANSWER: A (1.2s)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_task(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "question": "Which value should be returned?",
                "answer": "A",
                "nodes": {
                    "1": {
                        "source": "datagov/example/files/rows.txt",
                        "subquestion": "Find the relevant value.",
                        "fact": "answer = 'A'",
                        "answer": "A",
                    }
                },
                "reasoning_chain": [
                    "Node 1: Find the relevant value -> A.",
                    "Final answer: A",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_plan(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_sequence": ["example"],
                "source_sequence": ["datagov/example/files/rows.txt"],
                "reasoning_chain_text": ["1. Look up the relevant value and return it."],
                "original_reasoning_chain": ["Node 1: Find the relevant value -> A."],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_trajectory_rows_for_nii_dii_and_iii_with_task_and_plan_context(tmp_path: Path) -> None:
    task_rel = Path("tasks_mini/k-1-d-1/task_1.log")
    task_json = Path("tasks_mini/k-1-d-1/task_1.json")
    model = "openai_gpt-test"
    for mode_label, mode in TARGET_MODES.items():
        _write_log(
            tmp_path / "logs" / "modes" / model / mode / task_rel,
            plan_tool="plan_ideal" if mode_label == "iii" else "plan" if mode_label == "dii" else None,
        )
    _write_task(tmp_path / task_json)
    _write_plan(tmp_path / "plans_mini/k-1-d-1/task_1.json")

    rows = build_trajectory_rows(
        benchmark="lakeqa",
        log_root=tmp_path / "logs",
        repo_root=tmp_path,
    )

    assert {(row["mode_label"], row["audit_status"]) for row in rows} == {
        ("nii", "pending"),
        ("dii", "pending"),
        ("iii", "pending"),
    }
    assert all(row["task_id"] == task_json.as_posix() for row in rows)
    assert all(row["trajectory_event_count"] != "0" for row in rows)
    assert all("Node 1: Find the relevant value" in row["ideal_reasoning_chain"] for row in rows)
    assert all("Node 1 | source=datagov/example/files/rows.txt" in row["node_sequence"] for row in rows)
    assert all("Look up the relevant value" in row["plan_reasoning_text"] for row in rows)
    assert all(row["run_status"] == "answered" for row in rows)
    assert all(row["limit_reached"] == "false" for row in rows)
    assert all(row["log_max_turn"] == "2" for row in rows)


def test_parse_log_run_status_detects_tool_limit_timeout_and_turn_count(tmp_path: Path) -> None:
    tool_limit_log = tmp_path / "tool_limit.log"
    _write_log(tool_limit_log, limit_marker="Tool limit reached (30/30 calls used).")

    timeout_log = tmp_path / "timeout.log"
    _write_log(timeout_log, limit_marker="Timeout reached (603.6s elapsed).")

    normal_log = tmp_path / "normal.log"
    _write_log(normal_log)

    assert parse_log_run_status(tool_limit_log) == {
        "run_status": "tool_limit_reached",
        "limit_reached": "true",
        "log_max_turn": "2",
    }
    assert parse_log_run_status(timeout_log)["run_status"] == "timeout_reached"
    assert parse_log_run_status(timeout_log)["limit_reached"] == "true"
    assert parse_log_run_status(normal_log) == {
        "run_status": "answered",
        "limit_reached": "false",
        "log_max_turn": "2",
    }


def test_build_trajectory_rows_marks_missing_task_context_not_comparable(tmp_path: Path) -> None:
    task_rel = Path("tasks_mini/k-1-d-1/task_1.log")
    model = "openai_gpt-test"
    _write_log(tmp_path / "logs" / "modes" / model / TARGET_MODES["nii"] / task_rel)

    rows = build_trajectory_rows(
        benchmark="lakeqa",
        log_root=tmp_path / "logs",
        repo_root=tmp_path,
        mode_labels=["nii"],
    )

    assert len(rows) == 1
    assert rows[0]["trajectory_alignment"] == "not_comparable"
    assert rows[0]["divergence_type"] == "task_context_missing"
    assert rows[0]["audit_status"] == "complete"


def test_build_judge_prompt_uses_ideal_context_instead_of_pair_comparison(tmp_path: Path) -> None:
    log_path = tmp_path / "task.log"
    _write_log(log_path)
    row = {
        "benchmark": "lakeqa",
        "mode_label": "nii",
        "model_variant": "openai_gpt-test",
        "runner_model": "gpt-test",
        "task_id": "tasks_mini/k/task.json",
        "mode": TARGET_MODES["nii"],
        "log_path": str(log_path),
        "start_line": "0",
        "trajectory_event_count": "4",
        "run_status": "tool_limit_reached",
        "limit_reached": "true",
        "log_max_turn": "2",
        "task_path": "tasks_mini/k/task.json",
        "plan_path": "plans_mini/k/task.json",
        "question": "Which value should be returned?",
        "ideal_answer": "A",
        "ideal_reasoning_chain": "Node 1: Find the relevant value -> A.",
        "node_sequence": "Node 1 | source=datagov/example/files/rows.txt | subquestion=Find it | answer=A",
        "plan_dataset_sequence": "example",
        "plan_source_sequence": "datagov/example/files/rows.txt",
        "plan_reasoning_text": "1. Look up the relevant value and return it.",
        "trajectory_alignment": "",
        "divergence_type": "",
        "trajectory_summary": "",
        "followed_steps": "",
        "missed_or_divergent_steps": "",
        "alignment_reason": "",
        "evidence": "",
        "auditor_model": "",
        "audit_status": "pending",
    }

    prompt = build_judge_prompt(row)

    assert "one actual agent trajectory against the task's ideal reasoning context" in prompt
    assert "IDEAL REASONING CHAIN" in prompt
    assert "NODE SEQUENCE" in prompt
    assert "PLAN CONTEXT" in prompt
    assert "ACTUAL TRAJECTORY" in prompt
    assert "RUN STATUS" in prompt
    assert "tool_limit_reached" in prompt
    assert "COMPARISON TRAJECTORY" not in prompt


def test_summarize_rows_counts_followed_broadly_and_not_comparable() -> None:
    rows = [
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "mode_label": "nii",
            "mode": TARGET_MODES["nii"],
            "trajectory_alignment": "followed",
            "audit_status": "complete",
        },
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "mode_label": "nii",
            "mode": TARGET_MODES["nii"],
            "trajectory_alignment": "mostly_followed",
            "audit_status": "complete",
        },
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "mode_label": "nii",
            "mode": TARGET_MODES["nii"],
            "trajectory_alignment": "not_comparable",
            "audit_status": "complete",
        },
    ]

    summary, breakdown = summarize_rows(rows)

    assert summary[0]["n_total"] == 3
    assert summary[0]["n_followed_strict"] == 1
    assert summary[0]["n_followed_or_mostly"] == 2
    assert summary[0]["followed_or_mostly_pct_comparable"] == 100.0
    assert {row["trajectory_alignment"] for row in breakdown} == {
        "followed",
        "mostly_followed",
        "not_comparable",
    }


def test_run_id_suffixes_outputs_journal_and_tmp_root(tmp_path: Path) -> None:
    rows = [
        {
            "benchmark": "lakeqa",
            "mode_label": "nii",
            "model_variant": "openai_gpt-test",
            "runner_model": "gpt-test",
            "task_id": "tasks_mini/k/task.json",
            "mode": TARGET_MODES["nii"],
            "log_path": "task.log",
            "start_line": "0",
            "trajectory_event_count": "1",
            "run_status": "answered",
            "limit_reached": "false",
            "log_max_turn": "1",
            "task_path": "tasks_mini/k/task.json",
            "plan_path": "plans_mini/k/task.json",
            "question": "Which value should be returned?",
            "ideal_answer": "A",
            "ideal_reasoning_chain": "Node 1: Find the relevant value -> A.",
            "node_sequence": "Node 1 | source=datagov/example/files/rows.txt",
            "plan_dataset_sequence": "example",
            "plan_source_sequence": "datagov/example/files/rows.txt",
            "plan_reasoning_text": "Look up the relevant value.",
            "trajectory_alignment": "followed",
            "divergence_type": "none",
            "trajectory_summary": "",
            "followed_steps": "",
            "missed_or_divergent_steps": "",
            "alignment_reason": "",
            "evidence": "",
            "auditor_model": "gpt-5.4-mini",
            "audit_status": "complete",
        }
    ]

    outputs = write_outputs(tmp_path, rows, run_id="second pass")

    assert {path.name for path in outputs} == {
        "trajectory_ideal_context_second_pass_audit.csv",
        "trajectory_ideal_context_second_pass_summary.csv",
        "trajectory_ideal_context_second_pass_label_breakdown.csv",
        "trajectory_ideal_context_second_pass_summary.json",
    }
    assert default_journal_path(tmp_path, "second pass").name == "trajectory_ideal_context_second_pass_journal.jsonl"
    assert default_tmp_root(tmp_path, "second pass") == tmp_path / "tmp" / "second_pass"
