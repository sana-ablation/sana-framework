#!/bin/bash
# Model tier x search-tool grid on the LakeQA subset20b split.
#
# Held fixed: results=naive, profile=standard, compute=standard, skills=off.
# Same fixed axes as experiments/2026-08-30-web-arm-subset20, so the two
# experiments compose into one grid.
#
# compute=standard, not ideal: execute_ideal/query_ideal return the authored
# record.answer regardless of what was retrieved, so an ideal-compute cell
# measures the oracle rather than the model.
#
# No --k: every arm runs at its own default, matching the web-arm run.
set -u

PY=${PY:-.venv/bin/python}
TASKSET=${TASKSET:-tmp/subset20b/benchmarks/lakeqa/tasks-mini/tasks}
PARALLEL=${PARALLEL:-8}
RESULTS=${RESULTS:-tmp/results-tiers}
LOGS=${LOGS:-tmp/logs-tiers}

# float16 is emulated on x86 CPU and runs ~2x slower than float32; only
# matters for the standard/naive arms, which embed the query locally.
export HYBRID_TORCH_DTYPE=${HYBRID_TORCH_DTYPE:-float32}

# The web-arm run found invoke_with_watchdog's threading.Timer firing ~40 min
# late (3000s against a 630s deadline). 900s is headroom, not a fix — see
# README "Required before running".
TIMEOUT=${TIMEOUT:-900}
GRACE=${GRACE:-60}

if [ ! -d "$TASKSET" ]; then
  echo "task set missing: $TASKSET" >&2
  echo "run: $PY experiments/2026-08-31-model-tiers-subset20b/inputs/materialize_subset.py" >&2
  exit 1
fi

run_cell () {
  local MODEL="$1" ARM="$2"
  local SLUG; SLUG=$(echo "${MODEL}-${ARM}" | tr '/.' '__')
  echo "=================================================================="
  echo ">>> $MODEL / search=$ARM   started $(date +%H:%M:%S)"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --task-set "$TASKSET" \
    --model-name "$MODEL" \
    --search_tool "$ARM" \
    --search_results naive --profile standard --computation_tool standard \
    --parallel "$PARALLEL" \
    --db-path lance_data \
    --results-output-dir "$RESULTS" \
    --logs-output-dir "$LOGS" \
    --timeout "$TIMEOUT" --submit-grace-seconds "$GRACE" \
    --openai-prompt-cache-retention 24h \
    --verbose > "${LOGS%/}-stdout-${SLUG}.log" 2>&1
  echo "<<< $MODEL / search=$ARM   exit=$?   finished $(date +%H:%M:%S)"
}

mkdir -p "$RESULTS" "$LOGS"

# Cheapest model first: a config error surfaces for ~$1 instead of ~$10.
for MODEL in openai/gpt-5.4-nano openai/gpt-5-mini openai/gpt-5.2; do
  for ARM in ideal standard naive; do
    run_cell "$MODEL" "$ARM"
  done
done

echo "MODEL TIER GRID COMPLETE $(date +%H:%M:%S)"
