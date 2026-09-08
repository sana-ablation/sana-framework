#!/usr/bin/env python3
"""Rewrite result directories to the canonical variant-label layout.

Two changes landed in ``_variant_condition_label`` together:

* the three ablation axes lead -- ``search``, ``plan``, ``compute`` -- and the
  search-result richness modifier trails them, rather than sitting second;
* that modifier is recorded canonically. ``--results`` accepts ``ideal`` and
  ``naive`` as aliases for ``rich`` and ``minimal``, and the label used to keep
  whichever spelling the caller typed, so one condition could land in two
  different directories and ``--only-new`` would resume against neither.

This script rewrites directories written before that change. It also accepts the
older ``profile_`` spelling of the planning axis, so a tree pulled from a host
that never ran ``migrate_profile_label_to_plan.py`` is fixed in one pass rather
than two.

It touches **directory names only**. No CSV, JSONL or log file is opened, read
or rewritten -- this module never calls ``open`` except on the mapping file.

Dry run is the default::

    python scripts/migrate_variant_label_layout.py experiments
    python scripts/migrate_variant_label_layout.py experiments --apply

Every rename performed is appended to a mapping file in execution order, so the
migration is reversible: replay the pairs bottom-to-top with the columns
swapped. Running it twice is a no-op -- ``canonical_label`` is idempotent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SEPARATOR = "__"

# The planning axis answers to both spellings; it is always written as `plan`.
PLAN_PREFIXES = ("plan_", "profile_")
SEARCH_PREFIX = "search_"
COMPUTE_PREFIX = "compute_"
RESULTS_PREFIX = "results_"

# `--results` aliases, resolved to the canonical vocabulary. A value that is
# neither an alias nor canonical is passed through untouched rather than guessed.
RESULTS_CANONICAL = {"ideal": "rich", "naive": "minimal"}

DEFAULT_ROOT = "experiments"
DEFAULT_MAPPING = "scripts/variant_label_layout_mapping.tsv"


def canonical_label(name: str) -> str:
    """Return ``name`` in canonical layout, or unchanged if it is not a variant.

    A name is only rewritten when all four axes are present. Anything else --
    a plain directory, or a variant missing an axis -- is returned as given, so
    the caller skips it instead of inventing a default for the missing piece.
    """
    parts = name.split(SEPARATOR)
    if len(parts) < 4:
        return name

    search = plan = compute = results = None
    tail: List[str] = []
    for part in parts:
        if part.startswith(SEARCH_PREFIX) and search is None:
            search = part[len(SEARCH_PREFIX):]
        elif part.startswith(PLAN_PREFIXES) and plan is None:
            plan = part.split("_", 1)[1]
        elif part.startswith(COMPUTE_PREFIX) and compute is None:
            compute = part[len(COMPUTE_PREFIX):]
        elif part.startswith(RESULTS_PREFIX) and results is None:
            results = part[len(RESULTS_PREFIX):]
        else:
            tail.append(part)

    if None in (search, plan, compute, results):
        return name

    results = RESULTS_CANONICAL.get(results, results)
    ordered = [
        f"{SEARCH_PREFIX}{search}",
        f"plan_{plan}",
        f"{COMPUTE_PREFIX}{compute}",
        f"{RESULTS_PREFIX}{results}",
    ]
    return SEPARATOR.join(ordered + tail)


def candidate_dirs(root: Path) -> List[Path]:
    """Every directory under ``root`` whose basename looks like a variant label.

    Returned deepest-first, so renaming in order can never invalidate a path
    that has not been renamed yet: a child is always renamed before its parent.
    """
    found: List[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        for name in dirnames:
            if name.startswith(SEARCH_PREFIX) and SEPARATOR in name:
                found.append(Path(dirpath) / name)
    found.sort(key=lambda p: (-len(p.parts), str(p)))
    return found


def plan_renames(dirs: List[Path]) -> List[Tuple[Path, Path]]:
    """Only directories whose canonical form differs from their current name."""
    renames: List[Tuple[Path, Path]] = []
    for d in dirs:
        canonical = canonical_label(d.name)
        if canonical != d.name:
            renames.append((d, d.with_name(canonical)))
    return renames


def collisions(renames: List[Tuple[Path, Path]]) -> List[str]:
    """Destinations that already exist, or that two sources would both claim."""
    problems: List[str] = []
    seen: Dict[Path, Path] = {}
    for src, dst in renames:
        if dst.exists():
            problems.append(f"destination already exists: {dst}  (from {src})")
        if dst in seen:
            problems.append(f"two sources map to {dst}: {seen[dst]} and {src}")
        seen[dst] = src
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="migrate_variant_label_layout.py",
        description="Rewrite variant directories to search/plan/compute/results order "
                    "with a canonical results spelling.",
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                        help=f"directory tree to migrate (default: {DEFAULT_ROOT})")
    parser.add_argument("--apply", action="store_true",
                        help="actually rename; without it this is a dry run")
    parser.add_argument("--mapping", default=DEFAULT_MAPPING,
                        help=f"where to write the old->new mapping (default: {DEFAULT_MAPPING})")
    parser.add_argument("--expect", type=int, default=None,
                        help="refuse to act unless exactly this many directories need renaming")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2

    found = candidate_dirs(root)
    renames = plan_renames(found)
    print(f"{len(found)} variant directories under {root}; {len(renames)} need renaming")

    if args.expect is not None and len(renames) != args.expect:
        print(f"error: expected {args.expect} renames, found {len(renames)}; refusing to act",
              file=sys.stderr)
        return 3

    if not renames:
        return 0

    problems = collisions(renames)
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
