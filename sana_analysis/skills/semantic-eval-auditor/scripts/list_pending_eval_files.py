#!/usr/bin/env python3
"""List eval_results.csv files that have not yet been mirrored into the _semantic tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="results-ec2")
    parser.add_argument("--output", default="")
    parser.add_argument("--include-complete", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def default_output_root(source_root: Path) -> Path:
    return source_root.with_name(f"{source_root.name}_semantic")


def eval_search_root(source_root: Path) -> Path:
    modes_root = source_root / "modes"
    if modes_root.is_dir():
        return modes_root
    return source_root


def iter_eval_paths(source_root: Path) -> list[Path]:
    return sorted(eval_search_root(source_root).rglob("eval_results.csv"))


def pending_eval_paths(
    *,
    source_root: Path,
    output_root: Path,
    include_complete: bool = False,
) -> list[dict[str, str | bool]]:
    pending: list[dict[str, str | bool]] = []
    for source_eval in iter_eval_paths(source_root):
        relative = source_eval.relative_to(source_root)
        mirrored = output_root / relative
        done = mirrored.exists()
        if done and not include_complete:
            continue
        pending.append(
            {
                "source_eval": str(source_eval),
                "relative_eval": relative.as_posix(),
                "mirrored_eval": str(mirrored),
                "done": done,
            }
        )
    return pending


def render_text(entries: list[dict[str, str | bool]], *, include_complete: bool) -> str:
    if not entries:
        return "No pending eval_results.csv files found."

    lines = []
    if include_complete:
        lines.append("Eval files:")
    else:
        lines.append("Pending eval files:")
    for idx, entry in enumerate(entries, start=1):
        suffix = " [done]" if entry["done"] else ""
        lines.append(f"{idx}. {entry['relative_eval']}{suffix}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve() if args.output else default_output_root(source_root)
    entries = pending_eval_paths(
        source_root=source_root,
        output_root=output_root,
        include_complete=args.include_complete,
    )
    if args.limit > 0:
        entries = entries[: args.limit]
    if args.json:
        print(json.dumps(entries, indent=2))
        return
    print(render_text(entries, include_complete=args.include_complete))


if __name__ == "__main__":
    main()
