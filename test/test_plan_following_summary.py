import csv
from pathlib import Path

from sana_analysis.running_analysis.plan_following_summary import (
    load_follow_plan_rows,
    summarize_follow_plan,
    write_outputs,
)


def _write_follow_plan(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "model_variant",
        "runner_model",
        "mode",
        "plan_family",
        "semantic_match",
        "follow_plan_label",
        "audit_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_plan_following_summary_counts_strict_broad_and_not_comparable(tmp_path: Path) -> None:
    mode = "search_i_results_i_plani_computei_k5_skills_off"
    csv_path = (
        tmp_path
        / "results_semantic"
        / "modes"
        / "openai_gpt-5-mini"
        / mode
        / "follow_plan.csv"
    )
    _write_follow_plan(
        csv_path,
        [
            {
                "task_id": "task_1",
                "model_variant": "openai_gpt-5-mini",
                "runner_model": "gpt-5-mini",
                "mode": mode,
                "plan_family": "iii",
                "semantic_match": "1",
                "follow_plan_label": "followed",
                "audit_status": "complete",
            },
            {
                "task_id": "task_2",
                "model_variant": "openai_gpt-5-mini",
                "runner_model": "gpt-5-mini",
                "mode": mode,
                "plan_family": "iii",
                "semantic_match": "0",
                "follow_plan_label": "mostly_followed",
                "audit_status": "complete",
            },
            {
                "task_id": "task_3",
                "model_variant": "openai_gpt-5-mini",
                "runner_model": "gpt-5-mini",
                "mode": mode,
                "plan_family": "iii",
                "semantic_match": "0",
                "follow_plan_label": "not_comparable",
                "audit_status": "complete",
            },
        ],
    )

    rows = load_follow_plan_rows(tmp_path)
    summary, breakdown = summarize_follow_plan(rows)

    assert len(summary) == 1
    assert summary[0]["benchmark"] == "lakeqa"
    assert summary[0]["n_total"] == 3
    assert summary[0]["n_comparable"] == 2
    assert summary[0]["n_followed_strict"] == 1
    assert summary[0]["followed_strict_pct_all"] == 33.3
    assert summary[0]["followed_strict_pct_comparable"] == 50.0
    assert summary[0]["n_followed_or_mostly"] == 2
    assert summary[0]["followed_or_mostly_pct_all"] == 66.7
    assert summary[0]["followed_or_mostly_pct_comparable"] == 100.0
    assert summary[0]["mean_semantic_match"] == 0.3333
    assert {row["follow_plan_label"] for row in breakdown} == {
        "followed",
        "mostly_followed",
        "not_comparable",
    }


def test_plan_following_summary_writes_csv_and_json(tmp_path: Path) -> None:
    outputs = write_outputs(
        [
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "mode": "mode",
                "n_total": 1,
            }
        ],
        [
            {
                "benchmark": "lakeqa",
                "model_variant": "openai_gpt-5-mini",
                "mode": "mode",
                "follow_plan_label": "followed",
                "n": 1,
            }
        ],
        tmp_path / "summary",
    )

    assert {path.name for path in outputs} == {
        "plan_following_summary.csv",
        "plan_following_label_breakdown.csv",
        "plan_following_summary.json",
    }
    assert all(path.exists() for path in outputs)
