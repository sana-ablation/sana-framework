#!/usr/bin/env bash
# Provision a bare remote box to run the sana-framework eval.
#
# Assumes NOTHING on the remote except ssh, tar, and python3. No git needed:
# `git archive` runs here and ships a tarball. Only lance_data cannot be
# regenerated remotely (rebuilding the FTS index needs datalake_with_schema.parquet,
# which is not on disk), so it is copied wholesale.
#
#   REMOTE_HOST=user@host REMOTE_IDENTITY=key.pem ./bootstrap_remote.sh
#
set -euo pipefail

HOST="${REMOTE_HOST:?set REMOTE_HOST, e.g. ec2-user@1.2.3.4}"
KEY="${REMOTE_IDENTITY:-}"
DIR="${REMOTE_DIR:-~/sana-framework}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

SSH=(ssh); [[ -n "$KEY" ]] && SSH=(ssh -i "$KEY")
SCP=(scp); [[ -n "$KEY" ]] && SCP=(scp -i "$KEY")

say() { printf '\n=== %s ===\n' "$*"; }

say "1/6 remote prerequisites"
"${SSH[@]}" "$HOST" 'set -e
  echo "host:   $(uname -srm)"
  echo "python: $(command -v python3 || echo MISSING)  $(python3 -V 2>&1 || true)"
  echo "tar:    $(command -v tar || echo MISSING)"
  echo "tmux:   $(command -v tmux || echo MISSING)"
  echo "disk:   $(df -h ~ | tail -1)"
'

say "2/6 shipping code (git archive; no git needed on remote)"
"${SSH[@]}" "$HOST" "mkdir -p $DIR"
git -C "$ROOT" archive --format=tar HEAD | gzip \
  | "${SSH[@]}" "$HOST" "tar xzf - -C $DIR"

say "3/6 shipping .env and .agents (gitignored, so not in the archive)"
"${SCP[@]}" "$ROOT/.env" "$HOST:$DIR/.env"
# .agents holds the semantic-eval-auditor that produces semantic_match.
tar czf - -C "$ROOT" --exclude='__pycache__' .agents | "${SSH[@]}" "$HOST" "tar xzf - -C $DIR"

say "4/6 shipping lance_data (~759 MB, cannot be rebuilt remotely)"
tar czf - -C "$ROOT" lance_data | "${SSH[@]}" "$HOST" "tar xzf - -C $DIR"

say "5/6 building venv"
"${SSH[@]}" "$HOST" "set -e
  cd $DIR
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  echo 'installed:' \$(.venv/bin/pip list 2>/dev/null | wc -l) 'packages'
"

say "6/6 preflight for all four arms"
"${SSH[@]}" "$HOST" "cd $DIR && .venv/bin/python - <<'PY'
import glob, io, os, contextlib
from sana_evaluation.env import load_repo_dotenv
load_repo_dotenv()
from sana_evaluation.config import RunConfig, ConditionConfig
from sana_evaluation.preflight import run_preflight, PreflightError

TASKS = 'experiments/2026-08-31-web-arm-subset20b/inputs/benchmarks/lakeqa/tasks-mini/tasks'
files = sorted(glob.glob(os.path.join(TASKS, '*', '*.json')))
print('task files:', len(files))
for st, no_s3 in [('web', True), ('ideal', False), ('standard', False), ('naive', False)]:
    rc = RunConfig(search_db_path='lance_data', search_tool_mode=st,
                   search_results_mode='naive', profile_mode='standard',
                   computation_tool_mode='standard', benchmark='lakeqa', no_s3=no_s3,
                   condition_config=ConditionConfig(condition='probe', base_condition='baseline'))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_preflight(rc, files)
        print(f'  OK   search={st} no_s3={no_s3}')
    except PreflightError as e:
        print(f'  FAIL search={st}: {str(e).strip().splitlines()[-1][:120]}')
PY"

say "verifying the semantic judge landed"
"${SSH[@]}" "$HOST" "ls $DIR/.agents/skills/semantic-eval-auditor/scripts/ 2>/dev/null \
  || echo 'ABSENT — semantic_match cannot be produced without it'"

say "done. next: ./run_remote.sh"
