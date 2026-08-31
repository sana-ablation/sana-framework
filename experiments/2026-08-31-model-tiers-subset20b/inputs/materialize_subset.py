#!/usr/bin/env python3
"""Materialize the subset20b task tree from its manifest.

The layout is not arbitrary. `run_eval.py` sets `task["id"] = path`, and
`runtime_profile_store._profile_location_from_task` locates a task's runtime
profile by scanning that id for the literal path segment

    benchmarks/<benchmark>/tasks-mini/tasks

A tree that drops the segment does not error — it resolves to a *wrong* profile
path and the run fails later with a confusing missing-file message. So the
subset is written under a prefix that preserves it:

    <out>/benchmarks/lakeqa/tasks-mini/tasks/<dir>/<task>.json

Verified resolutions:
    benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/task_2.json   -> runtime-profiles/k-1-d-1/task_2.json  OK
    tmp/x/benchmarks/lakeqa/tasks-mini/tasks/k-1-d-1/...     -> runtime-profiles/k-1-d-1/task_2.json  OK
    tmp/x/tasks/k-1-d-1/task_2.json                          -> runtime-profiles/tmp/x/tasks/...      WRONG
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "benchmarks/lakeqa/tasks-mini/tasks"
PROFILES = REPO / "benchmarks/lakeqa/tasks-mini/runtime-profiles"
MANIFEST = Path(__file__).resolve().parent / "subset20b-manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "tmp/subset20b"), help="output root")
    ap.add_argument("--manifest", default=str(MANIFEST))
    args = ap.parse_args()

    tasks = json.loads(Path(args.manifest).read_text())
    dst_root = Path(args.out) / "benchmarks/lakeqa/tasks-mini/tasks"
    if Path(args.out).exists():
        shutil.rmtree(args.out)

    missing = []
    for task in tasks:
        src, prof = SRC / f"{task}.json", PROFILES / f"{task}.json"
        if not src.exists():
            missing.append(f"task {src}")
            continue
        if not prof.exists():
            missing.append(f"profile {prof}")
            continue
        dst = dst_root / f"{task}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if missing:
        print("MISSING:", *missing, sep="\n  ", file=sys.stderr)
        return 1

    written = sorted(dst_root.glob("*/task_*.json"))
    print(f"materialized {len(written)} tasks under {dst_root}")
    try:
        task_set = dst_root.relative_to(Path.cwd())
    except ValueError:
        task_set = dst_root
    print(f"pass to the runner as:  --task-set {task_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
