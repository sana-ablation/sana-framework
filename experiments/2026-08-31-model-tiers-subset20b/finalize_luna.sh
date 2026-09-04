#!/bin/bash
# Bring gpt-5.6-luna's three rounds home and rebuild everything that reads them.
#
# Ordering matters and is not obvious, so it is fixed here rather than left to
# be retyped:
#
#   1. pull    -- results and logs; the audit needs the logs for blank answers
#   2. backfill-- cells that ran before luna had a price recorded 0.0, and the
#                 subagent totals were written as 0+0, so this must precede any
#                 table that reports cost
#   3. audit   -- per-model, so the three already-audited models are not
#                 re-judged (the judge is not deterministic)
#   4. backfill again -- the audited tree is a separate copy of the rows
#   5. regenerate figures, then tex, then the PDF
#
#   ./finalize_luna.sh
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
REMOTE=${REMOTE:-sana}
MODEL=${MODEL:-openai_gpt-5.6-luna}
PY=${PY:-.venv/bin/python}

say () { echo; echo "=== $* ==="; }

say "1/6 pull results and logs from $REMOTE"
for R in "results:logs" "results-rep2:logs-rep2" "results-rep3:logs-rep3"; do
  RES=${R%%:*}; LOG=${R##*:}
  for KIND in "$RES" "$LOG"; do
    # No leading ~: bash tilde-expands after a colon in an assignment, so
    # "$REMOTE:~/..." silently became this machine's home path and every pull
    # failed into the "(no ... yet)" branch. A relative remote path is already
    # resolved against the remote home.
    SRC=$REMOTE:sana-framework/$EXP/$KIND/modes/$MODEL/
    DST=$EXP/$KIND/modes/$MODEL/
    mkdir -p "$DST"
    if rsync -az "$SRC" "$DST"; then
      echo "  pulled $KIND ($(find "$DST" -name eval_results.csv 2>/dev/null | wc -l | tr -d " ") cells)"
    elif ssh -o BatchMode=yes "$REMOTE" "test -d sana-framework/$EXP/$KIND/modes/$MODEL"; then
      echo "  FAILED to pull $KIND, but it exists on $REMOTE" >&2
      exit 1
    else
      echo "  (no $KIND on $REMOTE yet)"
    fi
  done
done

say "2/6 backfill cost on the raw trees"
$PY "$EXP/backfill_cost.py" --model "$MODEL" --apply

say "3/6 semantic audit (this model only)"
"./$EXP/audit_model.sh" 1 2 3

say "4/6 backfill cost on the audited trees"
$PY "$EXP/backfill_cost.py" --model "$MODEL" --apply

say "5/6 regenerate figures and tex"
$PY "$EXP/make_axis_delta_figure.py"
$PY experiments/2026-08-31-web-arm-subset20b/make_search_axis_figure.py
$PY "$EXP/gen_tex.py"
$PY experiments/2026-08-31-web-arm-subset20b/gen_tex.py

say "6/6 compile"
( cd experiments && tectonic build_pdf.tex --outdir . 2>&1 | grep -Ei "^error|Writing" )

say "done"
