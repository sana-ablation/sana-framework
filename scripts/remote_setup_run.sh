#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_setup_run.sh [options] smoke|full [setup_run args...]

Options:
  --host HOST        SSH host. Default: $REMOTE_HOST or ec2-user@ec2-18-191-139-16.us-east-2.compute.amazonaws.com
  -i, --identity PEM SSH identity file. Default: $REMOTE_IDENTITY or asw2215.pem
  --repo PATH        Remote repo path. Default: $REMOTE_REPO or ~/eval_eqa/exploratory-qa-eval
  --venv PATH        Remote venv activate path, relative to repo unless absolute.
                     Default: $REMOTE_VENV or .venv/bin/activate
  --runner NAME      setup_run family: sana. Default: sana
  --python BIN       Python executable on remote. Default: python3
  --session NAME     tmux session name. Default: eval-YYYYmmdd-HHMMSS
  --log PATH         Log path, relative to repo unless absolute.
                     Default: run_logs/SESSION.log
  --attach           Attach to tmux after starting the run
  -h, --help         Show this help

Examples:
  scripts/remote_setup_run.sh full \
    --benchmark kramabench --search ideal --plans standard --compute ideal \
    --k 5 --parallel 4 --model openai/gpt-5-mini --db lance_kramabench_infused \
    --timeout 600 --submit-grace-seconds 30

  scripts/remote_setup_run.sh --session sana-sprint-k5 full \
    --mode ideal --sana-feature sprint --sprint-mode commitment --k 5 --parallel 4 \
    --model gpt5.4-nano --db lance_data
EOF
}

remote_host="${REMOTE_HOST:-ec2-user@ec2-18-191-139-16.us-east-2.compute.amazonaws.com}"
identity_file="${REMOTE_IDENTITY:-asw2215.pem}"
remote_repo="${REMOTE_REPO:-~/eval_eqa/exploratory-qa-eval}"
remote_venv="${REMOTE_VENV:-.venv/bin/activate}"
runner_name="sana"
python_bin="python3"
session_name="eval-$(date +%Y%m%d-%H%M%S)"
log_path=""
attach=0

while (($#)); do
  case "$1" in
    --host)
      remote_host="${2:?--host requires a value}"
      shift 2
      ;;
    -i|--identity)
      identity_file="${2:?--identity requires a value}"
      shift 2
      ;;
    --repo)
      remote_repo="${2:?--repo requires a value}"
      shift 2
      ;;
    --venv)
      remote_venv="${2:?--venv requires a value}"
      shift 2
      ;;
    --runner)
      runner_name="${2:?--runner requires a value}"
      shift 2
      ;;
    --python)
      python_bin="${2:?--python requires a value}"
      shift 2
      ;;
    --session)
      session_name="${2:?--session requires a value}"
      shift 2
      ;;
    --log)
      log_path="${2:?--log requires a value}"
      shift 2
      ;;
    --attach)
      attach=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    smoke|full)
      break
      ;;
    *)
      echo "Unknown wrapper option or missing subcommand: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (($# == 0)); then
  echo "Missing setup_run subcommand: smoke or full" >&2
  usage >&2
  exit 2
fi

if [[ -z "$log_path" ]]; then
  log_path="run_logs/${session_name}.log"
fi

case "$runner_name" in
  sana)
    runner_path="sana_evaluation/setup_run.py"
    ;;
  *)
    echo "--runner must be sana, got: $runner_name" >&2
    exit 2
    ;;
esac

quote_words() {
  local quoted=()
  local word
  for word in "$@"; do
    quoted+=("$(printf '%q' "$word")")
  done
  printf '%s ' "${quoted[@]}"
}

setup_args="$(quote_words "$@")"
session_q="$(printf '%q' "$session_name")"
repo_q="$(printf '%q' "$remote_repo")"
venv_q="$(printf '%q' "$remote_venv")"
log_q="$(printf '%q' "$log_path")"
python_q="$(printf '%q' "$python_bin")"
runner_path_q="$(printf '%q' "$runner_path")"
ssh_args=()
if [[ -n "$identity_file" ]]; then
  ssh_args=(-i "$identity_file")
fi

read -r -d '' remote_script <<EOF || true
set -euo pipefail
expand_path() {
  case "\$1" in
    "~") printf '%s\n' "\$HOME" ;;
    "~/"*) printf '%s/%s\n' "\$HOME" "\${1#~/}" ;;
    *) printf '%s\n' "\$1" ;;
  esac
}
remote_repo=\$(expand_path ${repo_q})
remote_venv=\$(expand_path ${venv_q})
remote_log=\$(expand_path ${log_q})
cd "\$remote_repo"
if [[ ! -f "\$remote_venv" ]]; then
  echo "Venv activate script not found: ${remote_venv}" >&2
  exit 1
fi
if tmux has-session -t ${session_q} 2>/dev/null; then
  echo "tmux session already exists: ${session_name}" >&2
  echo "Attach with: ssh ${remote_host} -t 'tmux attach -t ${session_name}'" >&2
  exit 1
fi
mkdir -p "\$(dirname "\$remote_log")"
tmux new-session -d -s ${session_q} "bash -lc 'set -euo pipefail; cd \"\$remote_repo\"; source \"\$remote_venv\"; ${python_q} ${runner_path_q} ${setup_args}2>&1 | tee -a \"\$remote_log\"'"
echo "Started tmux session: ${session_name}"
echo "Log: ${remote_repo}/${log_path}"
echo "Attach: ssh ${identity_file:+-i ${identity_file} }${remote_host} -t 'tmux attach -t ${session_name}'"
echo "Tail: ssh ${identity_file:+-i ${identity_file} }${remote_host} 'tail -f ${remote_repo}/${log_path}'"
EOF

ssh "${ssh_args[@]}" "$remote_host" "$remote_script"

if ((attach)); then
  ssh "${ssh_args[@]}" -t "$remote_host" "tmux attach -t $(printf '%q' "$session_name")"
fi
