#!/usr/bin/env bash
# Driver for the retrieval-source comparison, invoked by
# experiments/run_experiment.sh. Rounds are selectable; the underlying script
# already knows the four arms.
#
#   ROUNDS="1 2 3" ./run.sh
set -uo pipefail
cd "$(dirname "$0")/../../.." || exit 1
exec ./experiments/2026-08-31-web-arm-subset20b/inputs/run_webarm_replicates.sh
