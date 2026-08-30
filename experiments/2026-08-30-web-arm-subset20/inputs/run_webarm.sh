#!/bin/bash
# gpt-5-mini x search-tool axis on the 20-task LakeQA subset.
# Held fixed: results=naive, profile=standard, compute=standard (compute=ideal
# is impossible for web, so every arm is pinned to standard for comparability).
# No --k: web search pools results across queries into one ranked list, so a
# small k silently starves whole sub-questions.
set -u
PY=.venv/bin/python
TASKSET=tmp/subset20/benchmarks/lakeqa/tasks-mini/tasks
MODEL=openai/gpt-5-mini

run_arm () {
  local NAME="$1"; shift
  echo "=================================================================="
  echo ">>> $NAME  started $(date +%H:%M:%S)"
  echo "=================================================================="
  $PY -m sana_evaluation.run_mode_eval \
    --all-tasks --task-set "$TASKSET" \
    --model-name "$MODEL" \
    --search_results naive --profile standard --computation_tool standard \
    --parallel 4 \
    --db-path lance_data \
    --results-output-dir tmp/results-webarm \
    --logs-output-dir tmp/logs-webarm \
    --openai-prompt-cache-retention 24h \
    --verbose "$@" > "tmp/webarm-${NAME}.log" 2>&1
  echo "<<< $NAME  exit=$?  finished $(date +%H:%M:%S)"
}

# Web first: it is the novel path, so it fails fast if something is wrong.
run_arm web      --search_tool web --no-s3
run_arm ideal    --search_tool ideal
run_arm standard --search_tool standard
run_arm naive    --search_tool naive

echo "ALL ARMS COMPLETE $(date +%H:%M:%S)"
