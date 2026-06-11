#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_pull_outputs.sh [options] REMOTE_ITEM [LOCAL_PATH]
  scripts/remote_pull_outputs.sh [options] --preset strands
  scripts/remote_pull_outputs.sh [options] --preset sana

Options:
  --host HOST        SSH host. Default: $REMOTE_HOST or ec2-user@ec2-18-191-139-16.us-east-2.compute.amazonaws.com
  -i, --identity PEM SSH identity file. Default: $REMOTE_IDENTITY or asw2215.pem
  --repo PATH        Remote repo path. Default: $REMOTE_REPO or ~/eval_eqa/exploratory-qa-eval
  --preset NAME      Pull common outputs: strands or sana
  --backup           Automatically move existing local targets aside with a timestamp
  --overwrite        Replace existing local targets without asking
  -h, --help         Show this help

Examples:
  scripts/remote_pull_outputs.sh --host ec2-user@ec2-18-191-139-16.us-east-2.compute.amazonaws.com sana-results ./sana-results
  scripts/remote_pull_outputs.sh --host ec2-user@ec2-18-221-146-210.us-east-2.compute.amazonaws.com results ./results-ec2
  scripts/remote_pull_outputs.sh --preset strands --backup
EOF
}

remote_host="${REMOTE_HOST:-ec2-user@ec2-18-191-139-16.us-east-2.compute.amazonaws.com}"
identity_file="${REMOTE_IDENTITY:-asw2215.pem}"
remote_repo="${REMOTE_REPO:-~/eval_eqa/exploratory-qa-eval}"
preset=""
backup=0
overwrite=0

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
    --preset)
      preset="${2:?--preset requires a value}"
      shift 2
      ;;
    --backup)
      backup=1
      shift
      ;;
    --overwrite)
      overwrite=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

if ((backup && overwrite)); then
  echo "Choose only one of --backup or --overwrite." >&2
  exit 2
fi

declare -a transfers=()
case "$preset" in
  "")
    if (($# < 1 || $# > 2)); then
      echo "Provide REMOTE_ITEM and optional LOCAL_PATH, or use --preset." >&2
      usage >&2
      exit 2
    fi
    remote_item="$1"
    local_path="${2:-./$(basename "$remote_item")}"
    transfers+=("${remote_item}" "${local_path}")
    ;;
  strands)
    if (($#)); then
      echo "--preset strands does not accept extra positional arguments." >&2
      exit 2
    fi
    transfers+=("results" "./results-ec2" "logs" "./logs-ec2")
    ;;
  sana)
    if (($#)); then
      echo "--preset sana does not accept extra positional arguments." >&2
      exit 2
    fi
    transfers+=("sana-results" "./sana-results")
    ;;
  *)
    echo "--preset must be strands or sana, got: $preset" >&2
    exit 2
    ;;
esac

ssh_args=()
scp_args=()
if [[ -n "$identity_file" ]]; then
  ssh_args=(-i "$identity_file")
  scp_args=(-i "$identity_file")
fi

backup_existing() {
  local target="$1"
  local timestamp
  timestamp="$(date +%Y%m%d-%H%M%S)"
  local backup_target="${target%/}.bak-${timestamp}"
  mv "$target" "$backup_target"
  echo "Moved existing ${target} to ${backup_target}"
}

prepare_local_target() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    return
  fi

  if ((overwrite)); then
    rm -rf "$target"
    return
  fi

  if ((backup)); then
    backup_existing "$target"
    return
  fi

  echo
  echo "Local target already exists: ${target}"
  echo "Save the old results before copying new ones?"
  echo "  b = move existing target aside with a timestamp, then copy"
  echo "  o = overwrite existing target"
  echo "  s = skip this transfer"
  echo "  q = quit"
  read -r -p "Choose [b/o/s/q]: " choice
  case "$choice" in
    b|B)
      backup_existing "$target"
      ;;
    o|O)
      rm -rf "$target"
      ;;
    s|S)
      return 1
      ;;
    q|Q)
      exit 0
      ;;
    *)
      echo "Unrecognized choice; skipping ${target}." >&2
      return 1
      ;;
  esac
}

for ((i = 0; i < ${#transfers[@]}; i += 2)); do
  remote_item="${transfers[i]}"
  local_path="${transfers[i + 1]}"
  if ! prepare_local_target "$local_path"; then
    continue
  fi
  mkdir -p "$(dirname "$local_path")"
  echo "Copying ${remote_host}:${remote_repo}/${remote_item} -> ${local_path}"
  scp "${scp_args[@]}" -r "${remote_host}:${remote_repo}/${remote_item}" "$local_path"
done
