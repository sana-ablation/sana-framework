#!/usr/bin/env bash
# Score the pulled results with the semantic judge. Runs LOCALLY — the auditor is
# a standalone OpenAI-API script (stdlib + `from openai import OpenAI`), imports
# nothing from this repo, and needs only the pulled results/ and logs/ plus
# OPENAI_API_KEY. The eval box never needs it.
#
#   ./audit_semantic.sh              # judge with the script's default model
#   MODEL=gpt-5.2-codex ./audit_semantic.sh
#
# Writes a sibling results_semantic/ tree, preserving lexical exact_match and
# adding semantic_match / semantic_reason / semantic_bucket / log_error_bucket /
# log_error_evidence. The judge model is part of the measurement — record it.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
AUDITOR="$ROOT/.agents/skills/semantic-eval-auditor/scripts/rewrite_semantic_eval_results.py"
MODEL="${MODEL:-gpt-5.2-codex}"
EFFORT="${EFFORT:-medium}"

[[ -f "$AUDITOR" ]] || {
  echo "auditor missing: $AUDITOR" >&2
  echo "it lives in the gitignored .agents/ — copy it from the exploratory-qa-eval checkout" >&2
  exit 1
}
[[ -d "$HERE/results" ]] || { echo "no results/ yet — run ./pull_remote.sh first" >&2; exit 1; }

set -a; [[ -f "$ROOT/.env" ]] && . "$ROOT/.env"; set +a

"$ROOT/.venv/bin/python" "$AUDITOR" \
  --source "$HERE/results" \
  --logs "$HERE/logs" \
  --model "$MODEL" \
  --reasoning-effort "$EFFORT" \
  "$@"

echo
echo "judged with: $MODEL (effort=$EFFORT)"
"$ROOT/.venv/bin/python" "$HERE/summarize.py"
