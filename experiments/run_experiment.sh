#!/usr/bin/env bash
# One entry point for every experiment: provision, run, pull, inspect.
#
#   ./run_experiment.sh <experiment-dir> <command>
#
# Local by default. Set REMOTE_HOST and the same command runs on that box
# instead -- provisioning it first if needed, and pulling results back when the
# sweep finishes. Nothing about the experiment changes between the two; only
# where the driver executes.
#
#   ./run_experiment.sh 2026-08-31-model-tiers-subset20b run
#   REMOTE_HOST=sana ./run_experiment.sh 2026-08-31-model-tiers-subset20b run
#   REMOTE_HOST=sana ./run_experiment.sh 2026-08-31-model-tiers-subset20b pull
#
# Contract: each experiment directory supplies inputs/run.sh as its driver.
# That is the only thing this script needs to know about it.
#
# Written because the two experiments here had each grown their own copy of the
# remote plumbing -- web-arm-subset20b has bootstrap/run/pull/watch scripts with
# its own directory name baked into pull_remote.sh, and model-tiers-subset20b
# reimplemented the driver side without any of it. Sharing the plumbing is what
# makes a third experiment cheap.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXP_ROOT="$REPO/experiments"

EXP="${1:-}"
CMD="${2:-help}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-sana-framework}"     # relative: resolved against the remote home
REMOTE_IDENTITY="${REMOTE_IDENTITY:-}"
SESSION="${SESSION:-exp-${EXP##*/}}"

die () { echo "error: $*" >&2; exit 1; }
say () { echo "[exp $(date -u +%H:%M:%SZ)] $*"; }

usage () {
  # Print the contiguous comment header, stopping at the first line of code, so
  # the usage text cannot drift from the file's length.
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"
  echo
  echo "commands: bootstrap | run | status | logs | pull | stop"
  echo
  echo "experiments:"
  for d in "$EXP_ROOT"/*/; do
    [ -f "$d/inputs/run.sh" ] && echo "  $(basename "$d")" || echo "  $(basename "$d")  (no inputs/run.sh)"
  done
  exit "${1:-0}"
}

[ -z "$EXP" ] || [ "$CMD" = help ] && usage 0
[ -d "$EXP_ROOT/$EXP" ] || die "no such experiment: $EXP"
DRIVER="$EXP_ROOT/$EXP/inputs/run.sh"
if [ ! -f "$DRIVER" ] && [ "$CMD" != pull ] && [ "$CMD" != status ]; then
  die "$EXP has no inputs/run.sh -- that file is the driver this script invokes"
fi

ssh_cmd () {
  local -a s=(ssh -o BatchMode=yes -o ConnectTimeout=20)
  [ -n "$REMOTE_IDENTITY" ] && s+=(-i "$REMOTE_IDENTITY")
  "${s[@]}" "$REMOTE_HOST" "$@"
}
rsync_to () {
  local -a r=(rsync -az --exclude '__pycache__' --exclude '*.pyc')
  [ -n "$REMOTE_IDENTITY" ] && r+=(-e "ssh -i $REMOTE_IDENTITY")
  "${r[@]}" "$@"
}

# No leading ~ in remote paths: bash tilde-expands after a colon in an
# assignment, so "$HOST:~/dir" silently becomes THIS machine's home and every
# transfer fails into whatever the caller treats as "nothing there yet". A
# relative path is already resolved against the remote home.
remote_path () { printf '%s:%s/%s' "$REMOTE_HOST" "$REMOTE_DIR" "$1"; }

case "$CMD" in

bootstrap)
  [ -n "$REMOTE_HOST" ] || die "bootstrap is remote-only; set REMOTE_HOST"
  say "shipping source to $REMOTE_HOST:$REMOTE_DIR"
  ssh_cmd "mkdir -p $REMOTE_DIR" || die "cannot reach $REMOTE_HOST"
  for path in sana_evaluation sana_analysis benchmarks requirements.txt "experiments/$EXP"; do
    [ -e "$REPO/$path" ] || continue
    rsync_to "$REPO/$path" "$(remote_path "$(dirname "$path")")/" \
      || die "failed to ship $path"
    say "  shipped $path"
  done
  say "note: lance_data is not shipped -- it cannot be regenerated remotely; copy it once by hand"
  ;;

run)
  say "driver: experiments/$EXP/inputs/run.sh"
  if [ -n "$REMOTE_HOST" ]; then
    ssh_cmd "test -f $REMOTE_DIR/experiments/$EXP/inputs/run.sh" \
      || die "driver missing on $REMOTE_HOST -- run 'bootstrap' first"
    ssh_cmd "cd $REMOTE_DIR && tmux kill-session -t $SESSION 2>/dev/null; \
             tmux new-session -d -s $SESSION \
             'cd $REMOTE_DIR && ./experiments/$EXP/inputs/run.sh > experiments/$EXP/run.log 2>&1'"
    say "started in tmux session '$SESSION' on $REMOTE_HOST"
    say "follow with: REMOTE_HOST=$REMOTE_HOST $0 $EXP logs"
  else
    ( cd "$REPO" && "./experiments/$EXP/inputs/run.sh" 2>&1 | tee "$EXP_ROOT/$EXP/run.log" )
  fi
  ;;

status)
  if [ -n "$REMOTE_HOST" ]; then
    ssh_cmd "tmux ls 2>/dev/null | grep '^$SESSION' || echo '(no session $SESSION)'; \
             echo -n 'eval procs: '; pgrep -cf 'run_mode[_]eval' 2>/dev/null || echo 0; \
             tail -3 $REMOTE_DIR/experiments/$EXP/run.log 2>/dev/null"
  else
    tail -3 "$EXP_ROOT/$EXP/run.log" 2>/dev/null || echo "(no local run.log)"
  fi
  ;;

logs)
  if [ -n "$REMOTE_HOST" ]; then
    ssh_cmd "tail -n 40 -f $REMOTE_DIR/experiments/$EXP/run.log"
  else
    tail -n 40 -f "$EXP_ROOT/$EXP/run.log"
  fi
  ;;

pull)
  [ -n "$REMOTE_HOST" ] || die "pull is remote-only; set REMOTE_HOST"
  pulled=0
  for tree in results results-rep2 results-rep3 logs logs-rep2 logs-rep3; do
    src="$(remote_path "experiments/$EXP/$tree")/"
    dst="$EXP_ROOT/$EXP/$tree/"
    if ssh_cmd "test -d $REMOTE_DIR/experiments/$EXP/$tree" 2>/dev/null; then
      mkdir -p "$dst"
      # A failed transfer of a tree that demonstrably exists is fatal. The soft
      # "(nothing yet)" branch is what let a whole replicate round go missing
      # unnoticed once.
      rsync_to "$src" "$dst" || die "failed to pull $tree (it exists on $REMOTE_HOST)"
      say "  pulled $tree ($(find "$dst" -name eval_results.csv 2>/dev/null | wc -l | tr -d ' ') cells)"
      pulled=$((pulled + 1))
    fi
  done
  [ "$pulled" -gt 0 ] || say "nothing to pull yet"
  ;;

stop)
  if [ -n "$REMOTE_HOST" ]; then
    ssh_cmd "tmux kill-session -t $SESSION 2>/dev/null && echo killed || echo '(no session)'"
  else
    die "stop is remote-only; interrupt the local run instead"
  fi
  ;;

*) usage 1 ;;
esac
