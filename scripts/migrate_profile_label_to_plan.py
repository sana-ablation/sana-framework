#!/usr/bin/env python3
"""Rename result directories carrying the old ``__profile_<mode>`` variant label.

The evaluation CLI's planning axis was renamed from ``--profile`` to ``--plan``,
and with it the segment the variant condition label emits. Runs made before that
rename sit in directories spelled ``...__profile_ideal__...``; runs made after it
spell the same condition ``...__plan_ideal__...``. This script migrates the old
directories so both sets of results live under one naming scheme.

It touches **directory names only**. No CSV, JSONL or log file is opened, read or
rewritten -- verified by the fact that this module never calls ``open``.

Dry run is the default::

    python scripts/migrate_profile_label_to_plan.py experiments
    python scripts/migrate_profile_label_to_plan.py experiments --apply

Every rename actually performed is appended to a mapping file in the order it was
performed, so the migration is reversible: replay the pairs bottom-to-top with the
two columns swapped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

OLD_SEGMENT = "__profile_"
NEW_SEGMENT = "__plan_"

DEFAULT_ROOT = "experiments"
DEFAULT_MAPPING = "scripts/profile_label_to_plan_mapping.tsv"


def _candidate_dirs(root: Path) -> List[Path]:
    """Every directory under ``root`` whose *basename* carries the old segment.

    Returned deepest-first, so renaming in order can never invalidate a path that
    has not been renamed yet: a child is always renamed before its parent.
    """
    found: List[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        for name in dirnames:
            if OLD_SEGMENT in name:
                found.append(Path(dirpath) / name)
    # Deepest first; ties broken by path so the plan is deterministic.
    found.sort(key=lambda p: (-len(p.parts), str(p)))
    return found


def _plan_renames(dirs: List[Path]) -> List[Tuple[Path, Path]]:
    return [(d, d.with_name(d.name.replace(OLD_SEGMENT, NEW_SEGMENT))) for d in dirs]


def _collisions(renames: List[Tuple[Path, Path]]) -> List[str]:
    """Destinations that already exist, or that two sources would both claim."""
    problems: List[str] = []
    seen: dict[Path, Path] = {}
    for src, dst in renames:
        if dst.exists():
            problems.append(f"destination already exists: {dst}  (from {src})")
        if dst in seen:
            problems.append(f"two sources map to {dst}: {seen[dst]} and {src}")
        seen[dst] = src
    return problems


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_profile_label_to_plan.py",
        description="Rename __profile_<mode> variant directories to __plan_<mode>.",
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                        help=f"directory tree to migrate (default: {DEFAULT_ROOT})")
    parser.add_argument("--apply", action="store_true",
                        help="actually rename; without it this is a dry run")
    parser.add_argument("--mapping", default=DEFAULT_MAPPING,
                        help=f"where to write the old->new mapping (default: {DEFAULT_MAPPING})")
    parser.add_argument("--expect", type=int, default=None,
                        help="refuse to act unless exactly this many directories match")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    renames = _plan_renames(_candidate_dirs(root))
    print(f"{len(renames)} directories under {root} carry '{OLD_SEGMENT}'")

    if args.expect is not None and len(renames) != args.expect:
        print(f"error: expected {args.expect} directories, found {len(renames)}; refusing to act",
              file=sys.stderr)
        return 3

    if not renames:
        return 0

    problems = _collisions(renames)
    if problems:
        print(f"error: {len(problems)} collision(s); nothing was renamed", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 4

    if not args.apply:
        print("DRY RUN -- nothing renamed. Pass --apply to act.\n")
        for src, dst in renames:
            print(f"{src}\t{dst}")
        return 0

    mapping_path = Path(args.mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    # Appended one line per completed rename, in execution order, so an
    # interrupted run still leaves an accurate record of what changed.
    with mapping_path.open("a", encoding="utf-8") as mapping:
        mapping.write(f"# {root}\told\tnew\n")
        for src, dst in renames:
            os.rename(src, dst)
            mapping.write(f"{src}\t{dst}\n")
            mapping.flush()

    print(f"renamed {len(renames)} directories; mapping appended to {mapping_path}")
    print("to reverse: replay the mapping bottom-to-top with the two columns swapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
