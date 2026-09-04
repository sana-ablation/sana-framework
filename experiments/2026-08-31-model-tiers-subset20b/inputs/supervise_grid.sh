#!/bin/bash
# Keep the model-tier grid running across crashes.
#
# Three runs were killed by the kernel OOM killer, each leaving the tmux session
# dead until a human noticed -- once for six hours. --only-new means a restart
# resumes rather than repeats, so the only cost of a crash is the partial cell in
# flight. This loop supplies the restart.
#
#   MODELS="openai/gpt-5.4-nano openai/gpt-5-mini" ./supervise_grid.sh
#
# Stops on: grid completion, MAX_RESTARTS, or no forward progress between two
# consecutive attempts (which would mean a task fails deterministically, and
# restarting forever would just burn money).
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
MAX_RESTARTS=${MAX_RESTARTS:-25}
export MODELS=${MODELS:-"openai/gpt-5.4-nano openai/gpt-5-mini"}
export TASKSET=${TASKSET:-experiments/2026-08-31-web-arm-subset20b/inputs/benchmarks/lakeqa/tasks-mini/tasks}
export RESULTS=${RESULTS:-$EXP/results}
export LOGS=${LOGS:-$EXP/logs}
export PARALLEL=${PARALLEL:-8}
# Deliberately NOT setting SANA_WORKER_MEMORY_CAP_GB. RLIMIT_DATA counts
# virtual reservations, so any cap fires during module import and fails the
# task -- see _apply_worker_memory_cap. Forcing it here also silently
# overrode that helper's disabled default.

rows_done () {
  find "$RESULTS" -name eval_results.csv 2>/dev/null \
    | xargs -r wc -l 2>/dev/null | awk '/total|eval_results/{s+=$1} END{print s+0}'
}

mkdir -p "$LOGS"
for attempt in $(seq 1 "$MAX_RESTARTS"); do
  before=$(rows_done)
  echo "=== supervisor attempt $attempt/$MAX_RESTARTS  rows_done=$before  $(date -u +%H:%M:%SZ) ==="

  # tee: the completion marker below is grepped out of driver.log, and
  # run_model_tiers.sh writes it to stdout.
  ./$EXP/inputs/run_model_tiers.sh 2>&1 | tee -a "$LOGS/driver.log"
  rc=${PIPESTATUS[0]}

  after=$(rows_done)
  echo "=== attempt $attempt ended rc=$rc  rows_done=$before -> $after  $(date -u +%H:%M:%SZ) ==="

  if grep -q "ABLATION GRID COMPLETE" "$LOGS/driver.log" 2>/dev/null; then
    echo "SUPERVISOR: grid complete"; exit 0
  fi
  if [ "$after" -le "$before" ]; then
    echo "SUPERVISOR: no forward progress on attempt $attempt -- stopping rather than looping" >&2
    exit 1
  fi
  echo "SUPERVISOR: progress made, restarting in 20s"
  sleep 20
done

echo "SUPERVISOR: hit MAX_RESTARTS=$MAX_RESTARTS" >&2
exit 1
