#!/usr/bin/env bash
# analyse.sh, but audit unjudged rounds with a model first.
#
#   ./scripts/analyse_with_autoaudit.sh --experiment 2026-09-08-luna-postrefactor
#
# THIS SPENDS MONEY. It runs the semantic auditor over every round that has no
# complete <round>_semantic mirror, then hands over to analyse.sh. Rounds whose
# mirror already exists and validates are skipped by the auditor itself, so
# re-running costs nothing for work already done.
#
# Use analyse.sh instead when you just want numbers: it calls no model.
#
# Every argument is passed through to analyse.sh unchanged; --experiment and
# --round are also read here to decide what to audit.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/.venv/bin/python}"
[ -x "$PY" ] || PY=python3
AUDITOR="$REPO/sana_analysis/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py"

say () { echo "[autoaudit] $*"; }
die () { echo "error: $*" >&2; exit 1; }

[ -f "$AUDITOR" ] || die "auditor not found at $AUDITOR"

# Read --experiment / --round without consuming them: analyse.sh needs them too.
# Both `--experiment foo` and `--experiment=foo` are recognised, matching
# analyse.sh's argparse pass-through.
EXPERIMENT=""
ROUND=""
prev=""
for arg in "$@"; do
  case "$arg" in
    --experiment=*) EXPERIMENT="${arg#--experiment=}" ;;
    --round=*) ROUND="${arg#--round=}" ;;
  esac
  case "$prev" in
    --experiment) EXPERIMENT="$arg" ;;
    --round) ROUND="$arg" ;;
  esac
  prev="$arg"
done
[ -n "$EXPERIMENT" ] || die "--experiment is required"

PLAN_ARGS=(--experiment "$EXPERIMENT" --print-audit-plan)
[ -n "$ROUND" ] && PLAN_ARGS+=(--round "$ROUND")

PLAN="$("$PY" -m sana_analysis.analyse_experiment "${PLAN_ARGS[@]}")" \
  || die "could not read the audit plan for $EXPERIMENT"

if [ -z "$PLAN" ]; then
  say "every round already has a complete semantic mirror; nothing to audit"
else
  say "rounds needing an audit:"
  echo "$PLAN" | while IFS=$'\t' read -r name _src _mirror _logs; do
    say "  $name"
  done
  # The auditor skips cells whose mirror already exists and validates, so a
  # partially audited round costs only its remaining cells.
  while IFS=$'\t' read -r name src mirror logs; do
    [ -n "$name" ] || continue
    say "auditing $name -> $mirror"
    ARGS=(--source "$src" --output "$mirror")
    if [ -n "$logs" ]; then
      ARGS+=(--logs "$logs")
    else
      say "  note: $name has no logs directory; log_error_* evidence will be limited"
    fi
    "$PY" "$AUDITOR" "${ARGS[@]}" || die "audit failed for $name"
  done <<< "$PLAN"
fi

say "handing over to analyse.sh"
exec "$REPO/scripts/analyse.sh" "$@"
