#!/bin/bash
# Rounds 2 and 3 of the web-arm retrieval comparison.
#
# Four arms: web (--no-s3), ideal, standard, naive. Held fixed at
# search_results=naive, profile=standard, computation_tool=standard, since the
# ideal computation tools resolve against authored lake records and would make a
# web run's outcome independent of what it retrieved.
#
# No --k: web search pools results across queries into one ranked list, so a
# small k starves whole sub-questions rather than limiting each.
#
# Each round needs its own results tree -- --only-new keys off the existing CSV,
# so a replicate sharing a tree would skip every task and measure nothing.
#
#   WAIT_FOR=<tmux session> ./run_webarm_replicates.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-web-arm-subset20b
TASKSET=${TASKSET:-$EXP/inputs/benchmarks/lakeqa/tasks-mini/tasks}
PY=${PY:-.venv/bin/python}
MODEL=${MODEL:-openai/gpt-5-mini}
PARALLEL=${PARALLEL:-4}
WAIT_FOR=${WAIT_FOR:-}

log () { echo "[webarm $(date -u +%H:%M:%SZ)] $*"; }

if [ -n "$WAIT_FOR" ]; then
  log "waiting for session $WAIT_FOR"
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done
  log "session $WAIT_FOR finished"
fi

run_arm () {
  local ROUND="$1" NAME="$2"; shift 2
  echo "=================================================================="
  echo ">>> round$ROUND $NAME  $(date -u +%H:%M:%SZ)"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --pool-tasks --only-new --task-set "$TASKSET" \
    --benchmark lakeqa \
    --model-name "$MODEL" \
    --search_results naive --profile standard --computation_tool standard \
    --parallel "$PARALLEL" \
    --db-path lance_data \
    --max-tool-calls 30 \
    --timeout 600 --submit-grace-seconds 30 \
    --results-output-dir "$EXP/results-rep$ROUND" \
    --logs-output-dir "$EXP/logs-rep$ROUND" \
    --openai-prompt-cache-retention 24h \
    --verbose "$@" >> "$EXP/logs-rep$ROUND-stdout-$NAME.log" 2>&1
  echo "<<< round$ROUND $NAME exit=$? $(date -u +%H:%M:%SZ)"
}

for ROUND in 2 3; do
  mkdir -p "$EXP/logs-rep$ROUND"
  log "starting round $ROUND"
  run_arm "$ROUND" web      --search_tool web --no-s3
  run_arm "$ROUND" ideal    --search_tool ideal
  run_arm "$ROUND" standard --search_tool standard
  run_arm "$ROUND" naive    --search_tool naive
  log "round $ROUND complete"
done

log "WEB-ARM REPLICATES COMPLETE"
