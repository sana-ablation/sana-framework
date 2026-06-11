#!/usr/bin/env python3
"""Build a semantic-equivalence relabeling of an existing results tree."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path


EXTRA_FIELDS = [
    "lexical_exact_match",
    "lexical_f1_score",
    "semantic_equivalent",
    "semantic_match_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results-ec2")
    parser.add_argument("--output", default="results-ec2-semantic")
    return parser.parse_args()


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def strip_outer_brackets(text: str) -> str:
    value = str(text or "").strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    return value


def normalize_fragment(text: str) -> str:
    value = strip_outer_brackets(text)
    value = unicodedata.normalize("NFKD", value)
    value = value.lower()
    value = re.sub(r"\bu\.?\s*s\.?\b", " united states ", value)
    value = re.sub(r"[^\w\s,/]", " ", value)
    value = " ".join(value.split())
    return value


def split_answer_items(text: str) -> list[str]:
    value = normalize_fragment(text)
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def canonical_item(text: str) -> str:
    tokens = normalize_fragment(text).split()
    while len(tokens) >= 2 and tokens[:2] == ["united", "states"]:
        tokens = tokens[2:]
    if tokens and tokens[-1] == "county":
        tokens = tokens[:-1]
    return " ".join(tokens)


def person_name_match(expected: str, predicted: str) -> bool:
    expected_tokens = canonical_item(expected).split()
    predicted_tokens = canonical_item(predicted).split()
    if not expected_tokens or not predicted_tokens:
        return False
    if expected_tokens[-1] != predicted_tokens[-1]:
        return False
    if len(expected_tokens) != len(predicted_tokens):
        return False

    for left, right in zip(expected_tokens[:-1], predicted_tokens[:-1]):
        if left == right:
            continue
        if len(left) == 1 and right.startswith(left):
            continue
        if len(right) == 1 and left.startswith(right):
            continue
        return False

    return True


def semantic_equivalent(expected: str, predicted: str) -> tuple[bool, str]:
    expected_items = split_answer_items(expected)
    predicted_items = split_answer_items(predicted)

    if not expected_items or not predicted_items:
        return False, ""

    if len(expected_items) == 1 and len(predicted_items) == 1:
        expected_canonical = canonical_item(expected_items[0])
        predicted_canonical = canonical_item(predicted_items[0])
        if expected_canonical == predicted_canonical:
            return True, "canonical_single_match"
        if person_name_match(expected_items[0], predicted_items[0]):
            return True, "person_name_match"
        return False, ""

    expected_canonical = sorted(canonical_item(item) for item in expected_items)
    predicted_canonical = sorted(canonical_item(item) for item in predicted_items)
    if expected_canonical == predicted_canonical:
        return True, "canonical_list_match"
    return False, ""


def semantic_record(expected: str, predicted: str, lexical_exact_match: float) -> tuple[float, str]:
    if lexical_exact_match >= 1.0:
        return 1.0, "lexical_exact_match"

    matched, reason = semantic_equivalent(expected, predicted)
    return (1.0 if matched else 0.0), reason


def update_agent_results(path: Path) -> list[dict]:
    updated_rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            expected = row.get("ground_truth", row.get("expected_answer", ""))
            predicted = row.get("predicted_answer", "")
            lexical_exact_match = as_float(row.get("exact_match"))
            lexical_f1_score = as_float(row.get("f1_score"))
            semantic_match, reason = semantic_record(expected, predicted, lexical_exact_match)

            row["lexical_exact_match"] = lexical_exact_match
            row["lexical_f1_score"] = lexical_f1_score
            row["semantic_equivalent"] = semantic_match
            row["semantic_match_reason"] = reason
            row["exact_match"] = semantic_match
            row["f1_score"] = 1.0 if semantic_match else lexical_f1_score
            updated_rows.append(row)

    with path.open("w") as handle:
        for row in updated_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

    return updated_rows


def csv_field_order(existing_fieldnames: list[str] | None) -> list[str]:
    existing_fieldnames = existing_fieldnames or []
    preferred_prefix = [
        "task_id",
        "model",
        "expected_answer",
        "predicted_answer",
        "exact_match",
        "f1_score",
        "lexical_exact_match",
        "lexical_f1_score",
        "semantic_equivalent",
        "semantic_match_reason",
    ]
    remaining = [field for field in existing_fieldnames if field not in preferred_prefix]
    return preferred_prefix + remaining


def update_eval_results(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        original_fieldnames = reader.fieldnames
        rows = list(reader)

    updated_rows: list[dict] = []
    for row in rows:
        expected = row.get("expected_answer", "")
        predicted = row.get("predicted_answer", "")
        lexical_exact_match = as_float(row.get("exact_match"))
        lexical_f1_score = as_float(row.get("f1_score"))
        semantic_match, reason = semantic_record(expected, predicted, lexical_exact_match)

        row["lexical_exact_match"] = lexical_exact_match
        row["lexical_f1_score"] = lexical_f1_score
        row["semantic_equivalent"] = semantic_match
        row["semantic_match_reason"] = reason
        row["exact_match"] = semantic_match
        row["f1_score"] = 1.0 if semantic_match else lexical_f1_score
        updated_rows.append(row)

    fieldnames = csv_field_order(original_fieldnames)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in updated_rows:
            writer.writerow(row)

    return updated_rows


def write_summary(output_root: Path, summary_rows: list[dict], change_rows: list[dict]) -> None:
    summary_path = output_root / "semantic_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_mode",
                "task_count",
                "lexical_exact_matches",
                "semantic_exact_matches",
                "semantic_upgrades",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    changes_path = output_root / "semantic_changes.csv"
    with changes_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "model_mode",
                "task_id",
                "expected_answer",
                "predicted_answer",
                "lexical_exact_match",
                "semantic_equivalent",
                "semantic_match_reason",
            ],
        )
        writer.writeheader()
        for row in change_rows:
            writer.writerow(row)


def build_semantic_results(source_root: Path, output_root: Path) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")

    shutil.copytree(source_root, output_root)

    summary_rows: list[dict] = []
    change_rows: list[dict] = []

    for agent_path in sorted(output_root.glob("modes/*/*/agent_results.jsonl")):
        model = agent_path.parents[1].name
        mode = agent_path.parent.name
        model_mode = f"{model}/{mode}"

        eval_path = agent_path.parent / "eval_results.csv"
        agent_rows = update_agent_results(agent_path)
        eval_rows = update_eval_results(eval_path)

        if len(agent_rows) != len(eval_rows):
            raise ValueError(f"Row mismatch between {agent_path} and {eval_path}")

        lexical_exact_matches = sum(as_float(row.get("lexical_exact_match")) for row in eval_rows)
        semantic_exact_matches = sum(as_float(row.get("semantic_equivalent")) for row in eval_rows)
        semantic_upgrades = sum(
            1
            for row in eval_rows
            if as_float(row.get("semantic_equivalent")) > as_float(row.get("lexical_exact_match"))
        )
        summary_rows.append(
            {
                "model": model,
                "model_mode": model_mode,
                "task_count": len(eval_rows),
                "lexical_exact_matches": int(lexical_exact_matches),
                "semantic_exact_matches": int(semantic_exact_matches),
                "semantic_upgrades": semantic_upgrades,
            }
        )

        for row in eval_rows:
            if as_float(row.get("semantic_equivalent")) > as_float(row.get("lexical_exact_match")):
                change_rows.append(
                    {
                        "model": model,
                        "model_mode": model_mode,
                        "task_id": row.get("task_id", ""),
                        "expected_answer": row.get("expected_answer", ""),
                        "predicted_answer": row.get("predicted_answer", ""),
                        "lexical_exact_match": row.get("lexical_exact_match", 0.0),
                        "semantic_equivalent": row.get("semantic_equivalent", 0.0),
                        "semantic_match_reason": row.get("semantic_match_reason", ""),
                    }
                )

    write_summary(output_root, summary_rows, change_rows)


def main() -> None:
    args = parse_args()
    build_semantic_results(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
