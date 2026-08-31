#!/bin/bash
# Model tier x per-axis ablation on the LakeQA subset20b split.
#
# Leave-one-out: every axis at the oracle, then one axis degraded per cell.
# The accuracy DROP from each degradation localises the bottleneck.
#
#   REFERENCE          search=ideal     profile=ideal    compute=ideal
#   search=naive       search=naive     profile=ideal    compute=ideal   (figure: BM25)
#   search=standard    search=standard  profile=ideal    compute=ideal   (figure: PNEUMA)
#   search=preloaded   search=preloaded profile=ideal    compute=ideal   (figure: Preloaded)
#   profile=naive      search=ideal    profile=naive    compute=ideal
#   profile=standard   search=ideal    profile=standard compute=ideal
#   compute=standard   search=ideal    profile=ideal    compute=standard
#
# computation_tool has only {standard, ideal} - no naive - so the compute
# ablation is a single cell.
#
# --k 5 and --search_results ideal match the original 135-task experiments, so
# variant directory names line up with the historical runs and sana_analysis
# keeps working unchanged.
#
# METRIC: these runs emit raw exact_match. The paper metric is semantic_match,
# produced afterwards by the semantic-eval-auditor agent skill into a parallel
# results_semantic/ tree. See README "Required before running" item 1.
set -u

PY=${PY:-.venv/bin/python}
TASKSET=${TASKSET:-tmp/subset20b/benchmarks/lakeqa/tasks-mini/tasks}
PARALLEL=${PARALLEL:-8}
RESULTS=${RESULTS:-tmp/results-tiers}
LOGS=${LOGS:-tmp/logs-tiers}
MODELS=${MODELS:-"openai/gpt-5.4-nano openai/gpt-5-mini openai/gpt-5.2"}

# float16 is emulated on x86 CPU at ~2x the cost; only the search=standard/naive
# cells embed a query locally. No GPU required either way.
export HYBRID_TORCH_DTYPE=${HYBRID_TORCH_DTYPE:-float32}

# The web-arm run observed invoke_with_watchdog's threading.Timer firing ~40 min
# late (3000s against a 630s deadline). 900s is headroom, not a fix - see
# README "Required before running" for the client-side HTTP timeout.
TIMEOUT=${TIMEOUT:-900}
GRACE=${GRACE:-60}

if [ ! -d "$TASKSET" ]; then
  echo "task set missing: $TASKSET" >&2
  echo "run: $PY experiments/2026-08-31-model-tiers-subset20b/inputs/materialize_subset.py" >&2
  exit 1
fi

# cell name | search | profile | compute
CELLS=(
  "reference|ideal|ideal|ideal"
  "search-naive|naive|ideal|ideal"
  "search-standard|standard|ideal|ideal"
  "search-preloaded|preloaded|ideal|ideal"
  "profile-naive|ideal|naive|ideal"
  "profile-standard|ideal|standard|ideal"
  "compute-standard|ideal|ideal|standard"
)

run_cell () {
  local MODEL="$1" NAME="$2" SEARCH="$3" PROFILE="$4" COMPUTE="$5"
  local SLUG; SLUG=$(echo "${MODEL}-${NAME}" | tr '/.' '__')
  echo "=================================================================="
  echo ">>> $MODEL  $NAME (search=$SEARCH profile=$PROFILE compute=$COMPUTE)  $(date +%H:%M:%S)"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --task-set "$TASKSET" \
    --model-name "$MODEL" \
    --search_tool "$SEARCH" \
    --profile "$PROFILE" \
    --computation_tool "$COMPUTE" \
    --search_results ideal \
    --k 5 \
    --parallel "$PARALLEL" \
    --db-path lance_data \
    --results-output-dir "$RESULTS" \
    --logs-output-dir "$LOGS" \
    --timeout "$TIMEOUT" --submit-grace-seconds "$GRACE" \
    --openai-prompt-cache-retention 24h \
    --verbose > "${LOGS}-stdout-${SLUG}.log" 2>&1
  echo "<<< $MODEL  $NAME  exit=$?  $(date +%H:%M:%S)"
}

mkdir -p "$RESULTS" "$LOGS"

# Cheapest model first: a config error surfaces for ~$1, not ~$75.
for MODEL in $MODELS; do
  for CELL in "${CELLS[@]}"; do
    IFS='|' read -r NAME SEARCH PROFILE COMPUTE <<< "$CELL"
    run_cell "$MODEL" "$NAME" "$SEARCH" "$PROFILE" "$COMPUTE"
  done
done

echo "ABLATION GRID COMPLETE $(date +%H:%M:%S)"
