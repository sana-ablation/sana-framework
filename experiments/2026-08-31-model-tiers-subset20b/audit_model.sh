#!/bin/bash
# Semantic-audit one model's cells without re-judging its neighbours.
#
# Two properties of the auditor force the shim below. It walks every
# eval_results.csv under --source with no skip-existing pass, so pointing it at
# a round tree would re-judge the three models already audited -- expensive, and
# the judge is not deterministic, so their published numbers would move. And
# infer_mode_context insists on a modes/<model>/<variant>/ layout relative to
# --source, so --source cannot simply be the model directory.
#
# The shim is a scratch tree containing only this model, in that layout. It is a
# copy rather than a symlink because Path.rglob does not recurse into symlinked
# directories on every Python version this repo runs on.
#
#   ./audit_model.sh 1 2 3
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
MODEL=${MODEL:-openai_gpt-5.6-luna}
PY=${PY:-.venv/bin/python}
AUDITOR=${AUDITOR:-.agents/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py}
SHIM=${SHIM:-tmp/audit-shim}
# The script's default of 320 is too small: a reasoning judge spends the budget
# thinking and returns truncated JSON, which surfaces as an APIConnectionError
# rather than as the length error it is.
ROW_TOKENS=${ROW_TOKENS:-1500}

[ -f "$AUDITOR" ] || { echo "auditor not found: $AUDITOR" >&2; exit 1; }

for ROUND in "$@"; do
  if [ "$ROUND" = 1 ]; then
    SRC=$EXP/results; LOG=$EXP/logs; OUT=$EXP/results_semantic
  else
    SRC=$EXP/results-rep$ROUND; LOG=$EXP/logs-rep$ROUND; OUT=$EXP/results-rep${ROUND}_semantic
  fi
  if [ ! -d "$SRC/modes/$MODEL" ]; then
    echo "round $ROUND: no $MODEL under $SRC -- skipping"; continue
  fi

  S=$SHIM/r$ROUND
  rm -rf "$S"; mkdir -p "$S/src/modes" "$S/logs/modes"
  cp -R "$SRC/modes/$MODEL" "$S/src/modes/$MODEL"
  [ -d "$LOG/modes/$MODEL" ] && cp -R "$LOG/modes/$MODEL" "$S/logs/modes/$MODEL"

  echo "=== round $ROUND: auditing $MODEL ($(find "$S/src" -name eval_results.csv | wc -l | tr -d ' ') files) ==="
  $PY "$AUDITOR" --source "$S/src" --logs "$S/logs" --output "$S/out" \
      --row-max-output-tokens "$ROW_TOKENS"

  mkdir -p "$OUT/modes"
  rm -rf "${OUT:?}/modes/${MODEL:?}"
  cp -R "$S/out/modes/$MODEL" "$OUT/modes/$MODEL"
  echo "=== round $ROUND -> $OUT/modes/$MODEL ==="
done
