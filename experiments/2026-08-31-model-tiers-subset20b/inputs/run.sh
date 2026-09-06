#!/usr/bin/env bash
# Driver for the model-tier ablation, invoked by experiments/run_experiment.sh.
#
# Defaults to the full grid over every tier. Narrow it with the same env vars
# the underlying scripts already take:
#
#   MODELS="openai/gpt-5-mini" ROUNDS="1" ./run.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
EXP=experiments/2026-08-31-model-tiers-subset20b

MODELS=${MODELS:-"openai/gpt-5.4-nano openai/gpt-5-mini openai/gpt-5.2 openai/gpt-5.6-luna"}
ROUNDS=${ROUNDS:-"1 2 3"}
PARALLEL=${PARALLEL:-4}

for ROUND in $ROUNDS; do
  if [ "$ROUND" = 1 ]; then R=$EXP/results; L=$EXP/logs
  else R=$EXP/results-rep$ROUND; L=$EXP/logs-rep$ROUND; fi
  echo "=== round $ROUND -> $R ==="
  MODELS="$MODELS" RESULTS="$R" LOGS="$L" PARALLEL="$PARALLEL" \
    ./$EXP/inputs/supervise_grid.sh
done
echo "MODEL TIERS COMPLETE"
