#!/bin/bash
# Run the remaining experiments back to back, unattended.
#
#   1. wait for the in-flight nano+mini grid to finish
#   2. gpt-5.2, all 7 cells, default reasoning effort   (~$87-140)
#   3. replicate of nano+mini, all 7 cells              (~$22)
#
# The replicate writes to a SEPARATE results tree. --only-new keys off the
# existing CSV, so a replicate sharing `results/` would skip every task as
# already done and measure nothing.
#
# Runs strictly in sequence: two grids at PARALLEL=8 would double memory
# pressure on a box that has already been OOM-killed several times.
#
#   WAIT_FOR=<tmux session> ./run_queue.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
SUP=./$EXP/inputs/supervise_grid.sh
WAIT_FOR=${WAIT_FOR:-}
# 8 workers OOMed the box in the search cells, where each worker loads its own
# embedding model (~3.7 GB x 8 = 30 GB of 31 GB). 4 measured clean at ~14 GB.
PARALLEL=${PARALLEL:-4}
NANO_MINI="openai/gpt-5.4-nano openai/gpt-5-mini"

log () { echo "[queue $(date -u +%H:%M:%SZ)] $*"; }

if [ -n "$WAIT_FOR" ]; then
  log "waiting for session $WAIT_FOR"
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done
  log "session $WAIT_FOR finished"
fi

# --- 2. gpt-5.2 -------------------------------------------------------------
log "starting gpt-5.2 (7 cells, default reasoning effort)"
MODELS="openai/gpt-5.2" \
RESULTS=$EXP/results \
LOGS=$EXP/logs \
PARALLEL=$PARALLEL \
  $SUP
log "gpt-5.2 finished rc=$?"

# --- 3. replicate of nano + mini -------------------------------------------
# Separate tree so --only-new cannot skip the tasks. Same config otherwise, so
# the spread between run 1 and run 2 is the error bar on every cell.
log "starting replicate of nano+mini into results-rep2"
MODELS="$NANO_MINI" \
RESULTS=$EXP/results-rep2 \
LOGS=$EXP/logs-rep2 \
PARALLEL=$PARALLEL \
  $SUP
log "replicate finished rc=$?"

log "QUEUE COMPLETE"
