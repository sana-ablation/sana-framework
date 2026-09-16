#!/usr/bin/env python3
"""List paired standard-plan and ideal-plan logs, extracting explicit plan text.

A CLI over `sana_analysis.metrics.plan_ablation_analysis`. The logic lives in the
package because skill directories are hyphenated and therefore unimportable, and
both the judging runner and the tests need to call it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sana_analysis.metrics.plan_ablation_analysis import (  # noqa: E402
    PAIR_COLUMNS,
    build_plan_pair_rows,
    render_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_root", help="Log root such as logs or log-kramabench")
    parser.add_argument("--model", default="")
    parser.add_argument("--plan-d-mode", default="")
    parser.add_argument("--plan-i-mode", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--all-modes",
        action="store_true",
        help="Pair every standard-plan mode with every ideal-plan mode, instead of "
             "the one canonical arm of each.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = build_plan_pair_rows(
        Path(args.log_root).resolve(),
        model=args.model or None,
        plan_d_mode=args.plan_d_mode or None,
        plan_i_mode=args.plan_i_mode or None,
        task=args.task or None,
        include_all_modes=args.all_modes,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.json:
        print(json.dumps([{key: row[key] for key in PAIR_COLUMNS} for row in rows], indent=2))
        return
    print(render_text(rows))


if __name__ == "__main__":
    main()
