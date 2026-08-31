#!/usr/bin/env bash
# Launch the four-arm sweep on the remote box, detached in tmux.
#
#   REMOTE_HOST=user@host REMOTE_IDENTITY=key.pem ./run_remote.sh
#
# Calls run_mode_eval directly rather than setup_run.py: run_mode_eval already
# takes --task-set, setup_run.py hardcodes _DEFAULT_TASK_SET.
#
# Arms run sequentially in one tmux session. The box is always-on, so the macOS
# sleep that cost the local pilot 5 tasks cannot recur (threading.Timer waits on
# a monotonic clock that stops during sleep; wall-clock runtime kept counting).
set -euo pipefail

HOST="${REMOTE_HOST:?set REMOTE_HOST}"
KEY="${REMOTE_IDENTITY:-}"
DIR="${REMOTE_DIR:-~/sana-framework}"
SESSION="${SESSION:-webarm-$(date +%Y%m%d-%H%M%S)}"
MODEL="${MODEL:-openai/gpt-5-mini}"

SSH=(ssh); [[ -n "$KEY" ]] && SSH=(ssh -i "$KEY")

read -r -d '' SWEEP <<'INNER' || true
set -u
PY=.venv/bin/python
TASKSET=experiments/2026-08-31-web-arm-subset20b/inputs/benchmarks/lakeqa/tasks-mini/tasks
OUT=experiments/2026-08-31-web-arm-subset20b

run_arm () {
  local NAME="$1"; shift
  echo "=================================================================="
  echo ">>> $NAME  started $(date -u +%H:%M:%S)Z"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --only-new --task-set "$TASKSET" \
    --model-name "__MODEL__" \
    --search_results naive --profile standard --computation_tool standard \
    --parallel 4 \
    --db-path lance_data \
    --results-output-dir "$OUT/results" \
    --logs-output-dir "$OUT/logs" \
    --openai-prompt-cache-retention 24h \
    --verbose "$@" >> "$OUT/logs/arm-$NAME.log" 2>&1
  echo "<<< $NAME  exit=$?  finished $(date -u +%H:%M:%S)Z"
}

mkdir -p "$OUT/logs"
run_arm web      --search_tool web --no-s3
run_arm ideal    --search_tool ideal
run_arm standard --search_tool standard
run_arm naive    --search_tool naive
echo "ALL ARMS COMPLETE $(date -u +%H:%M:%S)Z"
INNER

SWEEP="${SWEEP//__MODEL__/$MODEL}"

"${SSH[@]}" "$HOST" "set -e
  cd $DIR
  mkdir -p experiments/2026-08-31-web-arm-subset20b/logs
  cat > .sweep.sh <<'EOSWEEP'
$SWEEP
EOSWEEP
  chmod +x .sweep.sh
  tmux new-session -d -s $SESSION \"bash -lc 'cd $DIR && ./.sweep.sh 2>&1 | tee -a experiments/2026-08-31-web-arm-subset20b/logs/driver.log'\"
  echo 'started tmux session: $SESSION'
"

cat <<EOF

session: $SESSION
attach : ssh ${KEY:+-i $KEY }$HOST -t 'tmux attach -t $SESSION'
tail   : ssh ${KEY:+-i $KEY }$HOST 'tail -f $DIR/experiments/2026-08-31-web-arm-subset20b/logs/driver.log'
pull   : ./pull_remote.sh
EOF
