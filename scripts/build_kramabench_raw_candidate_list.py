#!/usr/bin/env python3
"""Build a raw all-task Kramabench candidate list for mini promotion."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
KRAMA_ROOT = REPO / "other-benchmarks" / "Kramabench"
WORKLOAD_ROOT = KRAMA_ROOT / "workload"
SOLUTIONS_ROOT = KRAMA_ROOT / "solutions"
DATA_ROOT = KRAMA_ROOT / "data"
DR_INPUT_ROOT = KRAMA_ROOT / "dr-input"
DEFAULT_OUTPUT_ROOT = REPO / "benchmarks/kramabench/tasks-mini/tasks"
DOMAINS = ("archeology", "astronomy", "biomedical", "environment", "legal", "wildfire")
GLOB_CHARS = set("*?[")
FILE_REF_RE = re.compile(r"[^'\"\s{}()]+\.(?:csv|xlsx|xls|txt|html|json|dat|sp3|cdf|npz|gpkg)")
IGNORED_LITERAL_REFS = {"datetime.dat"}


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
        refs.extend(normalize_file_refs(subtask.get("data_sources")) or top_refs)
    return ordered_unique(refs)


def task_normalized_data_sources(task: dict[str, Any]) -> list[str]:
    top_refs = normalize_file_refs(task.get("data_sources"))
    return top_refs or task_required_refs(task)


def _is_glob_ref(ref: str) -> bool:
    return any(char in ref for char in GLOB_CHARS)


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def match_ref(root: Path, ref: str) -> list[str]:
    rels = [path.relative_to(root).as_posix() for path in _files(root)]
    ref_name = Path(ref).name
    if _is_glob_ref(ref):
        return [
            rel
            for rel in rels
            if fnmatch.fnmatch(rel, ref) or fnmatch.fnmatch(Path(rel).name, ref_name)
        ]
    return [rel for rel in rels if rel == ref or Path(rel).name == ref_name]


def solution_file_refs(solution_path: Path) -> list[str]:
    if not solution_path.exists():
        return []

    refs: list[str] = []
    seen: set[str] = set()
    source = solution_path.read_text(encoding="utf-8")
    for match in FILE_REF_RE.finditer(source):
        ref = match.group(0).strip("/").replace("data_path}/", "")
        if "/input/" in ref:
            ref = ref.split("/input/", 1)[1]
        if ref.startswith("data/") or ref in IGNORED_LITERAL_REFS:
            continue
        if ref.startswith("http") or "`" in ref or ref.startswith(".*") or "[" in ref or "mask" in ref:
            continue
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def _json_path(path: Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _load_domain_tasks(domain: str) -> list[dict[str, Any]]:
    return json.loads((WORKLOAD_ROOT / f"{domain}.json").read_text(encoding="utf-8"))


def build_candidate_payload(domain: str, task: dict[str, Any]) -> dict[str, Any]:
    source_id = str(task.get("id", ""))
    solution_path = SOLUTIONS_ROOT / domain / f"{source_id}.py"
    raw_data_root = DATA_ROOT / domain / "input"
    dr_input_bundle = DR_INPUT_ROOT / domain / source_id

    required_refs = task_required_refs(task)
    solution_refs = solution_file_refs(solution_path)
    raw_ref_matches = {ref: match_ref(raw_data_root, ref) for ref in required_refs}
    solution_ref_matches = {ref: match_ref(raw_data_root, ref) for ref in solution_refs}
    dr_ref_matches = (
        {ref: match_ref(dr_input_bundle, ref) for ref in required_refs}
        if dr_input_bundle.is_dir()
        else {}
    )

    missing_required_refs = [ref for ref, matches in raw_ref_matches.items() if not matches]
    missing_solution_refs = [ref for ref, matches in solution_ref_matches.items() if not matches]
    has_complete_dr_input = bool(dr_ref_matches) and all(dr_ref_matches.values())

    return {
        "candidate_version": "kramabench_raw_all_v1",
        "source_id": source_id,
        "domain": domain,
        "difficulty": parse_difficulty(source_id),
        "source_mode": "full_domain_data",
        "question": task.get("query", ""),
        "answer": task.get("answer"),
        "answer_type": task.get("answer_type"),
        "runtime": task.get("runtime"),
        "subtask_count": len(task.get("subtasks") or []),
        "normalized_data_sources": task_normalized_data_sources(task),
        "required_file_refs": required_refs,
        "raw_data_root": _json_path(raw_data_root),
        "raw_data_files": [path.relative_to(raw_data_root).as_posix() for path in _files(raw_data_root)],
        "raw_data_ref_matches": raw_ref_matches,
        "missing_required_refs": missing_required_refs,
        "solution_path": _json_path(solution_path),
        "solution_exists": solution_path.exists(),
        "solution_file_refs": solution_refs,
        "solution_ref_matches": solution_ref_matches,
        "missing_solution_refs": missing_solution_refs,
        "dr_input_bundle": _json_path(dr_input_bundle) if dr_input_bundle.is_dir() else None,
        "dr_input_files": [
            path.relative_to(dr_input_bundle).as_posix()
            for path in _files(dr_input_bundle)
        ],
        "dr_input_ref_matches": dr_ref_matches,
        "has_complete_dr_input": has_complete_dr_input,
        "source_status": (
            "ready"
            if solution_path.exists() and not missing_solution_refs
            else "needs_source_resolution"
        ),
        "workload_path": _json_path(WORKLOAD_ROOT / f"{domain}.json"),
        "s3_mirror_prefix": f"datagov/kramabench-{source_id}/files/",
        "task": task,
    }


def build_candidate_list(*, output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = REPO / output_root

    candidates_dir = output_root / "candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for task in sorted(_load_domain_tasks(domain), key=lambda item: str(item.get("id", ""))):
            payloads.append(build_candidate_payload(domain, task))

    summaries: list[dict[str, Any]] = []
    for payload in payloads:
        candidate_path = candidates_dir / f"{payload['source_id']}.json"
        candidate_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(
            {
                "source_id": payload["source_id"],
                "domain": payload["domain"],
                "difficulty": payload["difficulty"],
                "source_mode": payload["source_mode"],
                "source_status": payload["source_status"],
                "subtask_count": payload["subtask_count"],
                "dataset_count": len(payload["normalized_data_sources"]),
                "normalized_data_sources": payload["normalized_data_sources"],
                "candidate_path": _json_path(candidate_path),
                "raw_data_root": payload["raw_data_root"],
                "dr_input_bundle": payload["dr_input_bundle"],
                "has_complete_dr_input": payload["has_complete_dr_input"],
                "missing_solution_refs": payload["missing_solution_refs"],
                "s3_mirror_prefix": payload["s3_mirror_prefix"],
            }
        )

    candidate_list = {
        "candidate_set": "kramabench_raw_all_main_workload",
        "candidate_version": "kramabench_raw_all_v1",
        "source": "other-benchmarks/Kramabench/workload",
        "selected_count": len(payloads),
        "selection_mode": "all_main_workload_no_filters",
        "filters": {},
        "domains": list(DOMAINS),
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
    args = parser.parse_args(argv)

    candidate_list = build_candidate_list(output_root=args.output_root)
    print(f"wrote {candidate_list['selected_count']} candidates to {_json_path(Path(args.output_root) / 'candidate_list.json')}")
    print(f"selection_mode: {candidate_list['selection_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
