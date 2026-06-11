#!/usr/bin/env python3
"""Prepare SANA result trees for mode-based analysis scripts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUMMARY_FILENAMES = ("agent_results.jsonl", "eval_results.csv", "tools_breakdown.csv")


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree_contents(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    for child in source.rglob("*"):
        if child.is_file():
            _copy_file(child, target / child.relative_to(source))


def _raw_result_dirs(sana_root: Path) -> list[Path]:
    if not sana_root.is_dir():
        return []
    return sorted(path.parent for path in sana_root.glob("*/*/*/eval_results.csv"))


def _log_source_for_result(result_dir: Path, model: str) -> Path | None:
    candidates = [
        result_dir.parent / model.replace("_", "-"),
        result_dir.parent / result_dir.name.replace("_", "-"),
        result_dir,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.log")):
            return candidate
    return None


def prepare_sana_modes(source_root: str | Path = "sana-results") -> int:
    """Copy raw SANA runs into the modes/logs/traces compatibility layout."""

    root = Path(source_root)
    prepared = 0
    for result_dir in _raw_result_dirs(root / "sana"):
        relative = result_dir.relative_to(root / "sana")
        model, variant = relative.parts[0], relative.parts[1]

        mode_dir = root / "modes" / model / variant
        for filename in SUMMARY_FILENAMES:
            source = result_dir / filename
            if source.is_file():
                _copy_file(source, mode_dir / filename)

        log_source = _log_source_for_result(result_dir, model)
        if log_source is not None:
            _copy_tree_contents(log_source, root / "logs" / "modes" / model / variant)

        trace_source = root / "traces" / "sana" / model / variant
        _copy_tree_contents(trace_source, root / "traces" / "modes" / model / variant)
        prepared += 1

    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sana-results")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_sana_modes(args.source)
    print(f"Prepared {prepared} SANA run(s) under {Path(args.source) / 'modes'}")
    if not args.prepare_only:
        print("Mode analysis is not implemented here; use --prepare-only.")


if __name__ == "__main__":
    main()
