from pathlib import Path

from sana_analysis.running_analysis.trajectory_pair_analysis import (
    IDEAL_MODE,
    PAIR_MODES,
    build_pair_rows,
    extract_trajectory_excerpt,
    judge_pending_rows,
    load_tmp_audits,
    merge_existing_audits,
    resolve_judge_limit,
    summarize_rows,
    validate_judge_payload,
    write_outputs,
)


def _write_log(path: Path, *, runner: str = "gpt-test", plan_tool: str | None = None) -> None:
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
            '2026 | INFO | x | Executing: pick({"s3_uris": ["ignored-internal"]})',
            '2026 | DEBUG | x | Tool result: {"results": [{"dataset_id": "ds"}]}',
            '2026 | INFO | x | Executing: peek_file({"dataset_id": "ds", "file_path": "files/rows.txt"})',
            '2026 | DEBUG | x | Tool result: {"dataset_id": "ds", "preview_text": "x"}',
            '2026 | INFO | x | Executing: submit_answer({"answer": "A"})',
            "2026 | INFO | x | ANSWER: A (1.2s)",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_extract_trajectory_excludes_plan_and_internal_tools(tmp_path: Path) -> None:
    log_path = tmp_path / "task_1.log"
    _write_log(log_path, plan_tool="plan_ideal")

    events = extract_trajectory_excerpt(log_path, start_after_line=3)

    joined = "\n".join(events)
    assert "plan_ideal" not in joined
    assert "pick(" not in joined
    assert "search_ideal" in joined
    assert "peek_file" in joined
    assert "submit_answer" in joined


def test_build_pair_rows_creates_nii_and_dii_pairs(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    task_rel = Path("tasks_mini/k-1-d-1/task_1.log")
    model = "openai_gpt-test"
    _write_log(root / "modes" / model / IDEAL_MODE / task_rel, plan_tool="plan_ideal")
    _write_log(root / "modes" / model / PAIR_MODES["nii_vs_iii"] / task_rel)
    _write_log(root / "modes" / model / PAIR_MODES["dii_vs_iii"] / task_rel, plan_tool="plan")

    rows = build_pair_rows(benchmark="lakeqa", log_root=root)

    assert {(row["pair_label"], row["audit_status"]) for row in rows} == {
        ("nii_vs_iii", "pending"),
        ("dii_vs_iii", "pending"),
    }
    assert all(row["task_id"] == "tasks_mini/k-1-d-1/task_1.json" for row in rows)
    assert all(row["comparison_event_count"] != "0" for row in rows)
    assert all(row["ideal_event_count"] != "0" for row in rows)


def test_build_pair_rows_marks_missing_ideal_as_not_comparable(tmp_path: Path) -> None:
    root = tmp_path / "logs"
    task_rel = Path("tasks_mini/k-1-d-1/task_1.log")
    model = "openai_gpt-test"
    _write_log(root / "modes" / model / PAIR_MODES["nii_vs_iii"] / task_rel)

    rows = build_pair_rows(benchmark="lakeqa", log_root=root, pair_labels=["nii_vs_iii"])

    assert len(rows) == 1
    assert rows[0]["trajectory_similarity"] == "not_comparable"
    assert rows[0]["divergence_type"] == "ideal_log_missing"
    assert rows[0]["audit_status"] == "complete"


def test_summarize_rows_counts_close_partial_pending_and_not_comparable() -> None:
    rows = [
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "pair_label": "nii_vs_iii",
            "trajectory_similarity": "resembles_completely",
            "audit_status": "complete",
        },
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "pair_label": "nii_vs_iii",
            "trajectory_similarity": "partially_resembles",
            "audit_status": "complete",
        },
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "pair_label": "nii_vs_iii",
            "trajectory_similarity": "not_comparable",
            "audit_status": "complete",
        },
        {
            "benchmark": "lakeqa",
            "model_variant": "openai_gpt-test",
            "pair_label": "nii_vs_iii",
            "trajectory_similarity": "",
            "audit_status": "pending",
        },
    ]

    summary, breakdown = summarize_rows(rows)

    assert summary[0]["n_total"] == 4
    assert summary[0]["n_resembles_or_close"] == 1
    assert summary[0]["resembles_or_close_fraction_all"] == "1/4"
    assert summary[0]["resembles_or_close_pct_all"] == 25.0
    assert summary[0]["n_partial_or_better"] == 2
    assert summary[0]["partial_or_better_fraction_all"] == "2/4"
    assert summary[0]["partial_or_better_pct_all"] == 50.0
    assert summary[0]["n_not_comparable"] == 1
    assert summary[0]["n_pending_label"] == 1
    assert summary[0]["pending_fraction_all"] == "1/4"
    assert {row["trajectory_similarity"] for row in breakdown} == {
        "resembles_completely",
        "partially_resembles",
        "not_comparable",
        "pending",
    }


def test_write_outputs_creates_audit_summary_and_json(tmp_path: Path) -> None:
    rows = [
        {
            "benchmark": "lakeqa",
            "pair_label": "nii_vs_iii",
            "model_variant": "openai_gpt-test",
            "runner_model": "gpt-test",
            "task_id": "tasks_mini/k/task.json",
            "comparison_mode": PAIR_MODES["nii_vs_iii"],
            "ideal_mode": IDEAL_MODE,
            "comparison_log": "a.log",
            "ideal_log": "b.log",
            "comparison_start_line": "0",
            "ideal_start_line": "1",
            "comparison_event_count": "1",
            "ideal_event_count": "1",
            "trajectory_similarity": "closely_resembles",
            "divergence_type": "minor_detour",
            "comparison_trajectory_summary": "",
            "ideal_trajectory_summary": "",
            "aligned_steps": "",
            "divergent_steps": "",
            "similarity_reason": "",
            "evidence": "",
            "auditor_model": "gpt-5.4-mini",
            "audit_status": "complete",
        }
    ]

    outputs = write_outputs(tmp_path, rows)

    assert {path.name for path in outputs} == {
        "trajectory_pair_audit.csv",
        "trajectory_pair_summary.csv",
        "trajectory_pair_label_breakdown.csv",
        "trajectory_pair_summary.json",
    }
    assert all(path.exists() for path in outputs)


def test_judge_pending_rows_writes_prompt_response_json_and_journal(tmp_path: Path, monkeypatch) -> None:
    comparison_log = tmp_path / "comparison.log"
    ideal_log = tmp_path / "ideal.log"
    _write_log(comparison_log)
    _write_log(ideal_log, plan_tool="plan_ideal")
    row = {
        "benchmark": "lakeqa",
        "pair_label": "nii_vs_iii",
        "model_variant": "openai_gpt-test",
        "runner_model": "gpt-test",
        "task_id": "tasks_mini/k/task.json",
        "comparison_mode": PAIR_MODES["nii_vs_iii"],
        "ideal_mode": IDEAL_MODE,
        "comparison_log": str(comparison_log),
        "ideal_log": str(ideal_log),
        "comparison_start_line": "0",
        "ideal_start_line": "3",
        "comparison_event_count": "1",
        "ideal_event_count": "1",
        "trajectory_similarity": "",
        "divergence_type": "",
        "comparison_trajectory_summary": "",
        "ideal_trajectory_summary": "",
        "aligned_steps": "",
        "divergent_steps": "",
        "similarity_reason": "",
        "evidence": "",
        "auditor_model": "",
        "audit_status": "pending",
    }

    def fake_call_judge_model(*_args, **kwargs):
        text = """{
          "trajectory_similarity": "closely_resembles",
          "divergence_type": "minor_detour",
          "comparison_trajectory_summary": "comparison",
          "ideal_trajectory_summary": "ideal",
          "aligned_steps": "search; peek; submit",
          "divergent_steps": "",
          "similarity_reason": "same path",
          "evidence": "lines",
          "audit_status": "complete"
        }"""
        kwargs["last_message_path"].write_text(text, encoding="utf-8")
        return text

    monkeypatch.setattr("sana_analysis.running_analysis.trajectory_pair_analysis.call_judge_model", fake_call_judge_model)

    judged = judge_pending_rows(
        [row],
        backend="codex",
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        model="gpt-5.4-mini",
        reasoning_effort="low",
        limit=0,
        timeout=10,
        tmp_root=tmp_path / "tmp",
        journal_path=tmp_path / "journal.jsonl",
        max_retries=2,
    )

    assert judged == 1
    assert row["trajectory_similarity"] == "closely_resembles"
    audit_jsons = list((tmp_path / "tmp").rglob("*.json"))
    prompts = list((tmp_path / "tmp").rglob("*.prompt.txt"))
    last_messages = list((tmp_path / "tmp").rglob("*.last_message.txt"))
    assert len(audit_jsons) == 1
    assert len(prompts) == 2
    assert len(last_messages) == 2
    assert "closely_resembles" in audit_jsons[0].read_text()
    assert "COMPARISON TRAJECTORY" in prompts[0].read_text()
    assert "closely_resembles" in last_messages[0].read_text()
    assert "status" in (tmp_path / "journal.jsonl").read_text()


def test_tmp_json_audits_are_loaded_by_full_pair_key(tmp_path: Path) -> None:
    comparison_log = tmp_path / "comparison.log"
    ideal_log = tmp_path / "ideal.log"
    _write_log(comparison_log)
    _write_log(ideal_log, plan_tool="plan_ideal")
    rows = [
        {
            "benchmark": "lakeqa",
            "pair_label": "nii_vs_iii",
            "model_variant": "openai_gpt-test",
            "runner_model": "gpt-test",
            "task_id": "tasks_mini/k/task.json",
            "comparison_mode": PAIR_MODES["nii_vs_iii"],
            "ideal_mode": IDEAL_MODE,
            "comparison_log": str(comparison_log),
            "ideal_log": str(ideal_log),
            "comparison_start_line": "0",
            "ideal_start_line": "3",
            "comparison_event_count": "1",
            "ideal_event_count": "1",
            "trajectory_similarity": "",
            "divergence_type": "",
            "comparison_trajectory_summary": "",
            "ideal_trajectory_summary": "",
            "aligned_steps": "",
            "divergent_steps": "",
            "similarity_reason": "",
            "evidence": "",
            "auditor_model": "",
            "audit_status": "pending",
        },
        {
            "benchmark": "lakeqa",
            "pair_label": "dii_vs_iii",
            "model_variant": "openai_gpt-test",
            "runner_model": "gpt-test",
            "task_id": "tasks_mini/k/task.json",
            "comparison_mode": PAIR_MODES["dii_vs_iii"],
            "ideal_mode": IDEAL_MODE,
            "comparison_log": str(comparison_log),
            "ideal_log": str(ideal_log),
            "comparison_start_line": "0",
            "ideal_start_line": "3",
            "comparison_event_count": "1",
            "ideal_event_count": "1",
            "trajectory_similarity": "",
            "divergence_type": "",
            "comparison_trajectory_summary": "",
            "ideal_trajectory_summary": "",
            "aligned_steps": "",
            "divergent_steps": "",
            "similarity_reason": "",
            "evidence": "",
            "auditor_model": "",
            "audit_status": "pending",
        },
    ]
    audit_path = (
        tmp_path
        / "tmp"
        / "lakeqa"
        / "openai_gpt-test"
        / "nii_vs_iii"
        / "row.json"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        """{
          "row_key": {
            "benchmark": "lakeqa",
            "pair_label": "nii_vs_iii",
            "model_variant": "openai_gpt-test",
            "task_id": "tasks_mini/k/task.json"
          },
          "row_update": {
            "trajectory_similarity": "closely_resembles",
            "divergence_type": "minor_detour",
            "comparison_trajectory_summary": "comparison",
            "ideal_trajectory_summary": "ideal",
            "aligned_steps": "same",
            "divergent_steps": "",
            "similarity_reason": "reason",
            "evidence": "evidence",
            "auditor_model": "gpt-5.4-mini",
            "audit_status": "complete"
          }
        }""",
        encoding="utf-8",
    )

    merge_existing_audits(rows, load_tmp_audits(tmp_path / "tmp"))

    assert rows[0]["trajectory_similarity"] == "closely_resembles"
    assert rows[1]["trajectory_similarity"] == ""


def test_validate_judge_payload_requires_known_label_and_required_fields() -> None:
    assert validate_judge_payload(
        {
            "trajectory_similarity": "closely_resembles",
            "divergence_type": "minor_detour",
            "comparison_trajectory_summary": "comparison",
            "ideal_trajectory_summary": "ideal",
            "aligned_steps": "",
            "divergent_steps": "",
            "similarity_reason": "reason",
            "evidence": "evidence",
            "audit_status": "complete",
        }
    ) == []
    errors = validate_judge_payload({"trajectory_similarity": "close", "audit_status": "done"})
    assert any("trajectory_similarity" in error for error in errors)
    assert any("audit_status" in error for error in errors)


def test_resolve_judge_limit_prompts_for_minimal_interactive_judge_run() -> None:
    messages: list[str] = []

    limit = resolve_judge_limit(
        requested_limit=None,
        pending_count=17,
        is_interactive=True,
        read_input=lambda _prompt: "5",
        print_fn=messages.append,
    )

    assert limit == 5
    assert any("Pending rows matching filters: 17" in message for message in messages)
    assert any("type a number" in message for message in messages)


def test_resolve_judge_limit_noninteractive_aborts_without_limit() -> None:
    messages: list[str] = []

    limit = resolve_judge_limit(
        requested_limit=None,
        pending_count=17,
        is_interactive=False,
        print_fn=messages.append,
    )

    assert limit is None
    assert any("no judge calls were made" in message for message in messages)
    assert any("--limit 0" in message for message in messages)


def test_resolve_judge_limit_explicit_zero_runs_all() -> None:
    messages: list[str] = []

    limit = resolve_judge_limit(
        requested_limit=0,
        pending_count=17,
        is_interactive=False,
        print_fn=messages.append,
    )

    assert limit == 0
    assert any("Judging all 17 pending row" in message for message in messages)


def test_judge_pending_rows_retries_invalid_json_until_valid(tmp_path: Path, monkeypatch) -> None:
    comparison_log = tmp_path / "comparison.log"
    ideal_log = tmp_path / "ideal.log"
    _write_log(comparison_log)
    _write_log(ideal_log, plan_tool="plan_ideal")
    row = {
        "benchmark": "lakeqa",
        "pair_label": "nii_vs_iii",
        "model_variant": "openai_gpt-test",
        "runner_model": "gpt-test",
        "task_id": "tasks_mini/k/task.json",
        "comparison_mode": PAIR_MODES["nii_vs_iii"],
        "ideal_mode": IDEAL_MODE,
        "comparison_log": str(comparison_log),
        "ideal_log": str(ideal_log),
        "comparison_start_line": "0",
        "ideal_start_line": "3",
        "comparison_event_count": "1",
        "ideal_event_count": "1",
        "trajectory_similarity": "",
        "divergence_type": "",
        "comparison_trajectory_summary": "",
        "ideal_trajectory_summary": "",
        "aligned_steps": "",
        "divergent_steps": "",
        "similarity_reason": "",
        "evidence": "",
        "auditor_model": "",
        "audit_status": "pending",
    }
    responses = iter(
        [
            '{"trajectory_similarity": "close", "audit_status": "done"}',
            """{
              "trajectory_similarity": "partially_resembles",
              "divergence_type": "operation_mismatch",
              "comparison_trajectory_summary": "comparison",
              "ideal_trajectory_summary": "ideal",
              "aligned_steps": "search",
              "divergent_steps": "operation",
              "similarity_reason": "valid after repair",
              "evidence": "lines",
              "audit_status": "complete"
            }""",
        ]
    )

    monkeypatch.setattr("sana_analysis.running_analysis.trajectory_pair_analysis.call_judge_model", lambda *_args, **_kwargs: next(responses))

    judged = judge_pending_rows(
        [row],
        backend="codex",
        output_dir=tmp_path / "out",
        repo_root=tmp_path,
        model="gpt-5.4-mini",
        reasoning_effort="low",
        limit=0,
        timeout=10,
        tmp_root=tmp_path / "tmp",
        journal_path=tmp_path / "journal.jsonl",
        max_retries=2,
    )

    assert judged == 1
    assert row["trajectory_similarity"] == "partially_resembles"
    assert len(list((tmp_path / "tmp").rglob("*.attempt1.last_message.txt"))) == 1
    assert len(list((tmp_path / "tmp").rglob("*.attempt2.last_message.txt"))) == 1
    assert '"status": "retry"' in (tmp_path / "journal.jsonl").read_text()
