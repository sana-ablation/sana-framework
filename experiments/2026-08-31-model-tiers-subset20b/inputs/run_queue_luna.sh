#!/bin/bash
# gpt-5.6-luna across the same 7-cell leave-one-out ablation, three rounds.
#
# The model id is pinned in full: bare "gpt-5.6" resolves to gpt-5.6-sol, a
# different model, exactly as "gpt-5.2" and "gpt-5.2-codex" are different.
#
# Rounds land in the three EXISTING results trees rather than new ones. Each tree
# is keyed by modes/<model>/<variant>, so luna adds a sibling directory and every
# analysis script picks it up with no change. --only-new keys off each cell's own
# CSV, so the other models' completed rows are untouched.
set -u

cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b
SUP=./$EXP/inputs/supervise_grid.sh
PARALLEL=${PARALLEL:-4}
MODEL=${MODEL:-openai/gpt-5.6-luna}
# Which replicate rounds to run. Round 1 was completed separately, so a
# resumed sweep is launched as ROUNDS="2 3".
ROUNDS=${ROUNDS:-"1 2 3"}

log () { echo "[luna $(date -u +%H:%M:%SZ)] $*"; }

for ROUND in $ROUNDS; do
  if [ "$ROUND" = 1 ]; then R=$EXP/results; L=$EXP/logs
  else R=$EXP/results-rep$ROUND; L=$EXP/logs-rep$ROUND; fi
  mkdir -p "$L"
  log "round $ROUND -> $R"
  MODELS="$MODEL" RESULTS="$R" LOGS="$L" PARALLEL="$PARALLEL" $SUP
  log "round $ROUND finished rc=$?"
done

log "LUNA QUEUE COMPLETE"
