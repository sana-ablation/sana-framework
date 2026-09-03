#!/bin/bash
# Round 2 for gpt-5.2, then round 3 for all three tiers.
#
#   1. gpt-5.2 into results-rep2   (7 cells)   -- gives 5.2 its first error bar
#   2. all three into results-rep3 (21 cells)  -- third replicate for everyone
#
# Each round needs its own results tree: --only-new keys off the existing CSV, so
# a replicate sharing a tree would skip every task and measure nothing.
#
# Sequential, PARALLEL=4. Eight workers exhausted the box in the search cells,
# where each worker loads its own embedding model (~3.7 GB).
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
SUP=./$EXP/inputs/supervise_grid.sh
PARALLEL=${PARALLEL:-4}
ALL="openai/gpt-5.4-nano openai/gpt-5-mini openai/gpt-5.2"

log () { echo "[queue2 $(date -u +%H:%M:%SZ)] $*"; }

log "round 2: gpt-5.2 -> results-rep2"
MODELS="openai/gpt-5.2" RESULTS=$EXP/results-rep2 LOGS=$EXP/logs-rep2 PARALLEL=$PARALLEL $SUP
log "round 2 finished rc=$?"

log "round 3: all three tiers -> results-rep3"
MODELS="$ALL" RESULTS=$EXP/results-rep3 LOGS=$EXP/logs-rep3 PARALLEL=$PARALLEL $SUP
log "round 3 finished rc=$?"

log "QUEUE2 COMPLETE"
