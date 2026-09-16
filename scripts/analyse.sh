#!/usr/bin/env bash
# Analyse every round of an experiment, from the experiment directory alone.
#
#   ./scripts/analyse.sh --experiment 2026-08-31-model-tiers-subset20b
#   ./scripts/analyse.sh --experiment experiments/my-sweep --round rep2
#
# Derives --results-dir, --base-results-dir, --traces-dir and --tasks-dir for
# each round, picks the judged path when a complete <round>_semantic mirror
# exists and the exact_match-only path otherwise, and writes a combined
# across-round summary with mean and spread per condition.
#
# Costs nothing and calls no model. To audit rounds that have no mirror first,
# use analyse_with_autoaudit.sh, which spends money on a judge.
#
# Output: <experiment>/analysis/<round>/ per round, and
#         <experiment>/analysis/combined/.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

# `python -m sana_analysis.*` needs the repo root on sys.path, and the package is
# not installed -- so run from there, not from wherever the caller stood. Without
# this, an absolute-path invocation from anywhere else dies on ModuleNotFoundError.
cd "$REPO" || exit 1

exec "$PY" -m sana_analysis.analyse_experiment "$@"
