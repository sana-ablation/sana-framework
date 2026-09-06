#!/bin/bash
# End-to-end mode ablation for gpt-5.6-luna: every axis set to the same level.
#
# The grid so far is leave-one-out -- all axes at the oracle, one degraded. This
# asks the complementary question: what does a wholly naive, a wholly standard,
# and a wholly oracle agent score? The degradations are not independent, so the
# end-to-end drop is not the sum of the single-axis drops, which is the point.
#
#   Naive     search=naive     results=ideal  profile=naive     compute=standard
#   Standard  search=standard  results=ideal  profile=standard  compute=standard
#   Oracle    search=ideal     results=ideal  profile=ideal      compute=ideal
#
# search_results is pinned to ideal in all three, matching the rest of this
# grid: every leave-one-out cell already runs --search_results ideal, so varying
# it here would make these cells incomparable with the ones they sit beside, and
# would confound the mode contrast with a fifth moving part.
#
# computation_tool has no naive level (only {standard, ideal}), so the Naive
# mode takes standard -- forced, not chosen.
#
# Oracle is byte-identical in configuration to the existing reference cell --
# same four axes, same --k 5 -- so it is NOT re-run here. The three rounds
# already collected are reused, which also keeps the comparison free of any
# run-to-run drift between the two sweeps.
#
#   ROUNDS="1 2 3" ./run_e2e_luna.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-openai/gpt-5.6-luna}
TASKSET=${TASKSET:-experiments/2026-08-31-web-arm-subset20b/inputs/benchmarks/lakeqa/tasks-mini/tasks}
PARALLEL=${PARALLEL:-4}
ROUNDS=${ROUNDS:-"1 2 3"}
TIMEOUT=${TIMEOUT:-900}
GRACE=${GRACE:-60}
export HYBRID_TORCH_DTYPE=${HYBRID_TORCH_DTYPE:-float32}

# name | search | results | profile | compute
CELLS=(
  "e2e-naive|naive|ideal|naive|standard"
  "e2e-standard|standard|ideal|standard|standard"
)

log () { echo "[e2e $(date -u +%H:%M:%SZ)] $*"; }

run_cell () {
  local ROUND="$1" NAME="$2" SEARCH="$3" RESULTS="$4" PROFILE="$5" COMPUTE="$6"
  local RES LOGS
  if [ "$ROUND" = 1 ]; then RES=$EXP/results; LOGS=$EXP/logs
  else RES=$EXP/results-rep$ROUND; LOGS=$EXP/logs-rep$ROUND; fi
  mkdir -p "$RES" "$LOGS"
  echo "=================================================================="
  echo ">>> round$ROUND $NAME (search=$SEARCH results=$RESULTS profile=$PROFILE compute=$COMPUTE)  $(date -u +%H:%M:%SZ)"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --pool-tasks --only-new --task-set "$TASKSET" \
    --model-name "$MODEL" \
    --search_tool "$SEARCH" \
    --search_results "$RESULTS" \
    --profile "$PROFILE" \
    --computation_tool "$COMPUTE" \
    --k 5 \
    --parallel "$PARALLEL" \
    --db-path lance_data \
    --results-output-dir "$RES" \
    --logs-output-dir "$LOGS" \
    --timeout "$TIMEOUT" --submit-grace-seconds "$GRACE" \
    --openai-prompt-cache-retention 24h \
    --verbose >> "$EXP/logs-e2e-round$ROUND-$NAME.log" 2>&1
  echo "<<< round$ROUND $NAME exit=$? $(date -u +%H:%M:%SZ)"
}

for ROUND in $ROUNDS; do
  log "round $ROUND"
  for CELL in "${CELLS[@]}"; do
    IFS='|' read -r NAME SEARCH RESULTS PROFILE COMPUTE <<< "$CELL"
    run_cell "$ROUND" "$NAME" "$SEARCH" "$RESULTS" "$PROFILE" "$COMPUTE"
  done
done

log "E2E LUNA COMPLETE"
