#!/usr/bin/env python3
"""Plan or run targeted answer-failure repair and validator reruns."""
from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sana_analysis.running_analysis.answer_failure_validation import is_non_correct_row
from sana_analysis.running_analysis.answer_failure_validation import validate_answer_failure_root


DETERMINISTIC_USABLE_STATUSES = {"valid", "valid_with_warnings"}
STALE_MODEL_VALIDATION_STATUSES = {"", "invalid_untrusted"}


@dataclass(frozen=True)
class EvalRerunCandidate:
    source_eval_path: Path
    answer_failure_eval_path: Path
    hard_invalid_rows: int
    missing_log_rows: int
    stale_model_validator_rows: int
    rows_with_events: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def source_eval_path_for_answer_failure_eval(
    answer_failure_eval_path: Path,
    *,
    answer_failure_root: Path,
    source_root: Path,
) -> Path:
    return source_root / answer_failure_eval_path.relative_to(answer_failure_root)


def candidate_for_answer_failure_eval(
    answer_failure_eval_path: Path,
    *,
    answer_failure_root: Path,
    source_root: Path,
) -> EvalRerunCandidate | None:
    source_eval_path = source_eval_path_for_answer_failure_eval(
        answer_failure_eval_path,
        answer_failure_root=answer_failure_root,
        source_root=source_root,
    )
    if not source_eval_path.exists():
        return None

    hard_invalid_rows = 0
    missing_log_rows = 0
    stale_model_validator_rows = 0
    rows_with_events = 0

    for row in _read_csv(answer_failure_eval_path):
        if not is_non_correct_row(row):
            continue
        if str(row.get("answer_failure_event_count", "")).strip():
            rows_with_events += 1
        deterministic_status = str(row.get("answer_failure_validation_status", ""))
        model_status = str(row.get("answer_failure_model_validation_status", ""))
        if deterministic_status == "invalid":
            hard_invalid_rows += 1
        elif deterministic_status == "missing_log":
            missing_log_rows += 1
        elif (
            deterministic_status in DETERMINISTIC_USABLE_STATUSES
            and model_status in STALE_MODEL_VALIDATION_STATUSES
        ):
            stale_model_validator_rows += 1

    return EvalRerunCandidate(
        source_eval_path=source_eval_path,
        answer_failure_eval_path=answer_failure_eval_path,
        hard_invalid_rows=hard_invalid_rows,
        missing_log_rows=missing_log_rows,
        stale_model_validator_rows=stale_model_validator_rows,
        rows_with_events=rows_with_events,
    )


def discover_candidates(
    *,
    answer_failure_root: Path,
    source_root: Path,
) -> list[EvalRerunCandidate]:
    candidates: list[EvalRerunCandidate] = []
    for eval_path in sorted(answer_failure_root.rglob("eval_results.csv")):
        candidate = candidate_for_answer_failure_eval(
            eval_path,
            answer_failure_root=answer_failure_root,
            source_root=source_root,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def command_for_candidate(
    candidate: EvalRerunCandidate,
    *,
    phase: str,
    repair_invalid_rounds: int,
    concurrency: int,
) -> list[str]:
    if phase == "invalid-repair":
        rounds = repair_invalid_rounds
    elif phase == "stale-validator":
        rounds = 0
    else:
        raise ValueError(f"unsupported phase: {phase}")

    command = [
        "uv",
        "run",
        "python",
        "-m",
        "sana_analysis.answer_failure_audit_runner",
        "--eval-path",
        candidate.source_eval_path.as_posix(),
        "--repair-invalid-rounds",
        str(rounds),
        "--concurrency",
        str(concurrency),
    ]
    if phase == "invalid-repair":
        command.append("--model-validator-repaired-only")
    else:
        command.append("--model-validator-stale-only")
    return command


def candidates_for_phase(candidates: list[EvalRerunCandidate], phase: str) -> list[EvalRerunCandidate]:
    if phase == "invalid-repair":
        return [candidate for candidate in candidates if candidate.hard_invalid_rows > 0]
    if phase == "stale-validator":
        return [candidate for candidate in candidates if candidate.stale_model_validator_rows > 0]
    raise ValueError(f"unsupported phase: {phase}")


def _shell_join(command: list[str]) -> str:
    return shlex.join(command)


def print_phase(
    candidates: list[EvalRerunCandidate],
    *,
    phase: str,
    repair_invalid_rounds: int,
    concurrency: int,
) -> None:
    selected = candidates_for_phase(candidates, phase)
    print(f"# Phase: {phase}")
    print(f"# Files: {len(selected)}")
    print(
        "# Rows: "
        f"hard_invalid={sum(candidate.hard_invalid_rows for candidate in selected)} "
        f"missing_log={sum(candidate.missing_log_rows for candidate in selected)} "
        f"stale_model_validator={sum(candidate.stale_model_validator_rows for candidate in selected)}"
    )
    for candidate in selected:
        print(
            "# "
            f"hard_invalid={candidate.hard_invalid_rows} "
            f"missing_log={candidate.missing_log_rows} "
            f"stale_model_validator={candidate.stale_model_validator_rows} "
            f"rows_with_events={candidate.rows_with_events} "
            f"{candidate.answer_failure_eval_path.as_posix()}"
        )
        print(
            _shell_join(
                command_for_candidate(
                    candidate,
                    phase=phase,
                    repair_invalid_rounds=repair_invalid_rounds,
                    concurrency=concurrency,
                )
            )
        )


def run_phase(
    candidates: list[EvalRerunCandidate],
    *,
    phase: str,
    repair_invalid_rounds: int,
    concurrency: int,
) -> None:
    for candidate in candidates_for_phase(candidates, phase):
        command = command_for_candidate(
            candidate,
            phase=phase,
            repair_invalid_rounds=repair_invalid_rounds,
            concurrency=concurrency,
        )
        print(_shell_join(command), flush=True)
        subprocess.run(command, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default="results_semantic", type=Path)
    parser.add_argument("--answer-failure-root", default="results_semantic_answer_failures", type=Path)
    parser.add_argument("--logs-dir", default="logs/modes", type=Path)
    parser.add_argument(
        "--phase",
        choices=["invalid-repair", "stale-validator", "all"],
        default="all",
        help="Which rerun queue to print or execute.",
    )
    parser.add_argument("--repair-invalid-rounds", type=int, default=2)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Passed to answer_failure_audit_runner for row-audit and model-validator concurrency within each file.",
    )
    parser.add_argument(
        "--skip-refresh-validation",
        action="store_true",
        help="Do not refresh deterministic validation columns before building each queue phase.",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing a dry-run plan.")
    return parser


def refresh_validation(answer_failure_root: Path, logs_dir: Path) -> dict:
    return validate_answer_failure_root(
        source_root=answer_failure_root,
        logs_dir=logs_dir,
        rewrite=True,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    phases = ["invalid-repair", "stale-validator"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if not args.skip_refresh_validation:
            outputs = refresh_validation(args.answer_failure_root, args.logs_dir)
            print(
                "# Refreshed deterministic validation: "
                f"audited_rows={outputs['audited_rows']} invalid_or_missing_log={len(outputs['invalid_rows'])}"
            )
        candidates = discover_candidates(
            answer_failure_root=args.answer_failure_root,
            source_root=args.source_root,
        )
        if args.execute:
            run_phase(
                candidates,
                phase=phase,
                repair_invalid_rounds=args.repair_invalid_rounds,
                concurrency=args.concurrency,
            )
        else:
            print_phase(
                candidates,
                phase=phase,
                repair_invalid_rounds=args.repair_invalid_rounds,
                concurrency=args.concurrency,
            )


if __name__ == "__main__":
    main()
