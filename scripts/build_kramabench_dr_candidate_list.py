#!/usr/bin/env python3
"""Build a direct Kramabench dr-input candidate list for mini promotion."""

from __future__ import annotations

import argparse
import fnmatch
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
KRAMA_ROOT = REPO / "other-benchmarks" / "Kramabench"
WORKLOAD_ROOT = KRAMA_ROOT / "workload"
DR_INPUT_ROOT = KRAMA_ROOT / "dr-input"
SOLUTIONS_ROOT = KRAMA_ROOT / "solutions"
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks/kramabench/tasks-mini/tasks"
DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
GLOB_CHARS = set("*?[")


def normalize_file_refs(raw: Any) -> list[str]:
    if raw is None:
        items: list[Any] = []
    elif isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]

    refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        ref = str(item).strip().replace("\\", "/").lstrip("./")
        if not ref or ref in {".", "./"}:
            continue
        while "//" in ref:
            ref = ref.replace("//", "/")
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def parse_difficulty(source_id: str) -> str:
    parts = source_id.split("-")
    if len(parts) >= 3 and parts[1] in {"easy", "medium", "hard"}:
        return parts[1]
    return "unknown"


def task_required_refs(task: dict[str, Any]) -> list[str]:
    top_refs = normalize_file_refs(task.get("data_sources"))
    refs = list(top_refs)
    for subtask in task.get("subtasks") or []:
        sub_refs = normalize_file_refs(subtask.get("data_sources")) or top_refs
        refs.extend(sub_refs)
    return ordered_unique(refs)


def task_normalized_data_sources(task: dict[str, Any]) -> list[str]:
    top_refs = normalize_file_refs(task.get("data_sources"))
    return top_refs or task_required_refs(task)


def _is_glob_ref(ref: str) -> bool:
    return any(char in ref for char in GLOB_CHARS)


def bundle_files(bundle: Path) -> list[Path]:
    return sorted(path for path in bundle.rglob("*") if path.is_file())


def match_ref_in_bundle(bundle: Path, ref: str) -> list[str]:
    files = bundle_files(bundle)
    rels = [path.relative_to(bundle).as_posix() for path in files]
    ref_name = Path(ref).name

    if _is_glob_ref(ref):
        return [
            rel
            for rel in rels
            if fnmatch.fnmatch(rel, ref) or fnmatch.fnmatch(Path(rel).name, ref_name)
        ]

    return [rel for rel in rels if rel == ref or Path(rel).name == ref_name]


def _json_path(path: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _load_domain_tasks(domain: str) -> list[dict[str, Any]]:
    return json.loads((WORKLOAD_ROOT / f"{domain}.json").read_text(encoding="utf-8"))


def _candidate_payload(
    *,
    domain: str,
    task: dict[str, Any],
    output_root: Path,
) -> dict[str, Any] | tuple[str, dict[str, Any]]:
    source_id = str(task.get("id", ""))
    difficulty = parse_difficulty(source_id)
    subtasks = task.get("subtasks") or []
    normalized_sources = task_normalized_data_sources(task)
    required_refs = task_required_refs(task)
    bundle = DR_INPUT_ROOT / domain / source_id

    if difficulty != "hard":
        return "not_hard", {}
    if len(subtasks) > _candidate_payload.max_subtasks:
        return "too_many_subtasks", {}
    if len(normalized_sources) < _candidate_payload.min_datasets:
        return "too_few_datasets", {}
    if not bundle.is_dir():
        return "missing_dr_input_bundle", {}

    ref_matches = {ref: match_ref_in_bundle(bundle, ref) for ref in required_refs}
    missing_refs = [ref for ref, matches in ref_matches.items() if not matches]
    if missing_refs:
        return "missing_dr_input_file", {"missing_refs": missing_refs}

    solution_path = SOLUTIONS_ROOT / domain / f"{source_id}.py"
    return {
        "candidate_version": "kramabench_dr_direct_v1",
        "source_id": source_id,
        "domain": domain,
        "difficulty": difficulty,
        "question": task.get("query", ""),
        "answer": task.get("answer"),
        "answer_type": task.get("answer_type"),
        "runtime": task.get("runtime"),
        "subtask_count": len(subtasks),
        "normalized_data_sources": normalized_sources,
        "required_file_refs": required_refs,
        "dr_input_bundle": _json_path(bundle),
        "dr_input_files": [path.relative_to(bundle).as_posix() for path in bundle_files(bundle)],
        "dr_input_ref_matches": ref_matches,
        "workload_path": _json_path(WORKLOAD_ROOT / f"{domain}.json"),
        "solution_path": _json_path(solution_path),
        "solution_exists": solution_path.exists(),
        "s3_mirror_prefix": f"datagov/kramabench-{source_id}/files/",
        "task": task,
    }


_candidate_payload.min_datasets = 3  # type: ignore[attr-defined]
_candidate_payload.max_subtasks = 6  # type: ignore[attr-defined]


def select_candidates(
    *,
    target_count: int = 135,
    min_datasets: int = 3,
    max_subtasks: int = 6,
    seed: int = 20260514,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[list[dict[str, Any]], Counter]:
    _candidate_payload.min_datasets = min_datasets  # type: ignore[attr-defined]
    _candidate_payload.max_subtasks = max_subtasks  # type: ignore[attr-defined]

    eligible: list[dict[str, Any]] = []
    rejected: Counter = Counter()
    for domain in DOMAINS:
        for task in sorted(_load_domain_tasks(domain), key=lambda item: str(item.get("id", ""))):
            result = _candidate_payload(domain=domain, task=task, output_root=output_root)
            if isinstance(result, tuple):
                reason, _details = result
                rejected[reason] += 1
            else:
                eligible.append(result)

    if len(eligible) > target_count:
        rng = random.Random(seed)
        eligible = sorted(rng.sample(eligible, target_count), key=lambda item: item["source_id"])

    return eligible, rejected


def build_candidate_list(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    target_count: int = 135,
    min_datasets: int = 3,
    max_subtasks: int = 6,
    seed: int = 20260514,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = REPO / output_root

    candidates, rejected = select_candidates(
        target_count=target_count,
        min_datasets=min_datasets,
        max_subtasks=max_subtasks,
        seed=seed,
        output_root=output_root,
    )

    candidates_dir = output_root / "candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for payload in candidates:
        candidate_path = candidates_dir / f"{payload['source_id']}.json"
        candidate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "source_id": payload["source_id"],
                "domain": payload["domain"],
                "difficulty": payload["difficulty"],
                "subtask_count": payload["subtask_count"],
                "dataset_count": len(payload["normalized_data_sources"]),
                "normalized_data_sources": payload["normalized_data_sources"],
                "dr_input_bundle": payload["dr_input_bundle"],
                "candidate_path": _json_path(candidate_path),
                "s3_mirror_prefix": payload["s3_mirror_prefix"],
            }
        )

    selection_mode = (
        "sampled_without_replacement"
        if len(candidates) == target_count and sum(rejected.values()) + len(candidates) > target_count
        else "take_all_eligible_because_pool_smaller_than_target"
    )
    candidate_list = {
        "candidate_set": "kramabench_dr_direct_hard_min3datasets_max6subtasks",
        "candidate_version": "kramabench_dr_direct_v1",
        "source": "other-benchmarks/Kramabench/workload",
        "target_count": target_count,
        "selected_count": len(candidates),
        "selection_mode": selection_mode,
        "selection_seed": seed,
        "filters": {
            "difficulty": "hard",
            "min_normalized_datasets_used": min_datasets,
            "max_subtasks": max_subtasks,
            "requires_dr_input_bundle": True,
            "requires_required_refs_in_dr_input": True,
            "ref_matching": "relative_path_or_basename",
        },
        "rejected_counts": dict(sorted(rejected.items())),
        "candidates": summaries,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_list.json").write_text(
        json.dumps(candidate_list, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-count", type=int, default=135)
    parser.add_argument("--min-datasets", type=int, default=3)
    parser.add_argument("--max-subtasks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args(argv)

    candidate_list = build_candidate_list(
        output_root=args.output_root,
        target_count=args.target_count,
        min_datasets=args.min_datasets,
        max_subtasks=args.max_subtasks,
        seed=args.seed,
    )
    print(f"wrote {candidate_list['selected_count']} candidates to {_json_path(Path(args.output_root) / 'candidate_list.json')}")
    print(f"target_count: {candidate_list['target_count']}")
    print(f"selection_mode: {candidate_list['selection_mode']}")
    print("rejected_counts:")
    for reason, count in candidate_list["rejected_counts"].items():
        print(f"  {reason}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
