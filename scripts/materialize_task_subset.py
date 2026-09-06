#!/usr/bin/env python3
"""Build a named task set from a manifest of task ids.

A manifest is a JSON list of ids relative to the source task tree, without the
.json suffix:

    ["k-3-d-2/task_11", "k-4-d-3/task_5", ...]

Tasks are copied, not linked, so a set is self-contained and a source edit does
not silently change a set someone has already reported numbers from.

Runtime profiles are NOT copied. A task keeps its ``<bucket>/<task>.json`` path
relative to the set's ``tasks/`` directory, and runtime_profile_store resolves
``benchmarks/<benchmark>/<set>/tasks/...`` against the benchmark's shared
runtime-profiles directory, so every set reuses one copy.

    python scripts/materialize_task_subset.py \\
        --manifest benchmarks/lakeqa/nano20/manifest.json \\
        --source benchmarks/lakeqa/tasks-mini/tasks \\
        --out benchmarks/lakeqa/nano20/tasks
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def materialize(manifest: Path, source: Path, out: Path, *, clean: bool = False) -> tuple[int, list[str]]:
    ids = json.loads(manifest.read_text())
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{manifest} must contain a JSON list of task id strings")

    if clean and out.exists():
        shutil.rmtree(out)

    missing: list[str] = []
    written = 0
    for task_id in ids:
        src = source / f"{task_id}.json"
        if not src.is_file():
            missing.append(task_id)
            continue
        dst = out / f"{task_id}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written += 1
    return written, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path,
                    help="task tree to copy from, e.g. benchmarks/lakeqa/tasks-mini/tasks")
    ap.add_argument("--out", required=True, type=Path,
                    help="destination tasks/ directory, e.g. benchmarks/lakeqa/nano20/tasks")
    ap.add_argument("--clean", action="store_true",
                    help="remove the destination first, so a shrunken manifest does not "
                         "leave orphaned tasks behind")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"source task tree not found: {args.source}", file=sys.stderr)
        return 1

    written, missing = materialize(args.manifest, args.source, args.out, clean=args.clean)
    print(f"wrote {written} tasks to {args.out}")
    if missing:
        # Loud, and a non-zero exit: a set that is quietly short is worse than
        # one that fails to build, because the shortfall shows up as an
        # unexplained change in denominators much later.
        print(f"MISSING {len(missing)} task(s) not found in {args.source}:", file=sys.stderr)
        for task_id in missing:
            print(f"  {task_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
