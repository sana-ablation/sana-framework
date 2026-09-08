#!/usr/bin/env bash
# One entry point for every experiment: provision, run, pull, inspect.
#
#   ./scripts/run_experiment.sh <experiment-dir> <command>
#
# Local by default. Set REMOTE_HOST and the same command runs on that box
# instead -- provisioning it first if needed, and pulling results back when the
# sweep finishes. Nothing about the experiment changes between the two; only
# where the driver executes.
#
#   ./scripts/run_experiment.sh my-sweep run
#   REMOTE_HOST=box ./scripts/run_experiment.sh my-sweep bootstrap
#   REMOTE_HOST=box ./scripts/run_experiment.sh my-sweep run
#   REMOTE_HOST=box ./scripts/run_experiment.sh my-sweep pull
#   REMOTE_HOST=box ./scripts/run_experiment.sh "sweep-a sweep-b" queue
#
# queue runs several experiments back to back in one session, sequentially --
# overlapping sweeps is what OOM-killed this box, since the search cells load a
# ~3.7 GB embedding model per worker. stop ends the session and the eval workers
# it started; killing the session alone leaves them running and writing rows.
#
# Contract: an experiment is any directory under experiments/ that supplies
# inputs/run.sh. That file is the only thing this script needs to know about
# it -- put whatever driver you like behind it. experiments/ is gitignored, so
# your sweeps stay yours; this runner is the reusable part.
#
# Two remote-transfer hazards are designed out rather than left to the caller.
# Remote paths never start with ~, because bash tilde-expands after a colon in
# an assignment and quietly turns a remote path into a local one. And a pull
# that fails for a tree which demonstrably exists on the remote is fatal, not a
# soft "nothing there yet" note -- that soft branch hides a missing result tree
# behind what looks like an ordinary empty run.
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
  echo "commands: bootstrap | run | queue | status | logs | pull | stop"
  echo
  echo "experiments:"
  for d in "$EXP_ROOT"/*/; do
    [ -f "$d/inputs/run.sh" ] && echo "  $(basename "$d")" || echo "  $(basename "$d")  (no inputs/run.sh)"
  done
  exit "${1:-0}"
}

[ -z "$EXP" ] || [ "$CMD" = help ] && usage 0
# queue takes a space-separated list; every other command takes one experiment
# and validates it here.
if [ "$CMD" != queue ]; then
  [ -d "$EXP_ROOT/$EXP" ] || die "no such experiment: $EXP"
fi
DRIVER="$EXP_ROOT/$EXP/inputs/run.sh"
if [ ! -f "$DRIVER" ] && [ "$CMD" != pull ] && [ "$CMD" != status ] \
   && [ "$CMD" != queue ] && [ "$CMD" != stop ]; then
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
    # Refuse rather than clobber. SESSION defaults to exp-<dirname>, so a second
    # `run` against the same experiment would otherwise silently kill a sweep
    # that is hours in.
    if ssh_cmd "tmux has-session -t $SESSION 2>/dev/null"; then
      die "session '$SESSION' is already running on $REMOTE_HOST. \
Use '$0 $EXP status' to check it, or '$0 $EXP stop' to end it first."
    fi
    ssh_cmd "cd $REMOTE_DIR && \
             tmux new-session -d -s $SESSION \
             'cd $REMOTE_DIR && ./experiments/$EXP/inputs/run.sh > experiments/$EXP/run.log 2>&1'"
    say "started in tmux session '$SESSION' on $REMOTE_HOST"
    say "follow with: REMOTE_HOST=$REMOTE_HOST $0 $EXP logs"
  else
    ( cd "$REPO" && "./experiments/$EXP/inputs/run.sh" 2>&1 | tee "$EXP_ROOT/$EXP/run.log" )
  fi
  ;;

queue)
  # Run several experiments back to back in ONE session, unattended.
  #
  #   ./scripts/run_experiment.sh "a b c" queue
  #
  # Sequential on purpose. Two sweeps at PARALLEL=4 double memory pressure, and
  # the search cells load a ~3.7 GB embedding model per worker -- that is what
  # OOM-killed this box repeatedly when grids overlapped. It also means a queued
  # sweep inherits a machine in a known state rather than one still unwinding.
  #
  # Each experiment keeps its own driver and its own results tree; the queue only
  # decides the order.
  for e in $EXP; do
    [ -f "$EXP_ROOT/$e/inputs/run.sh" ] || die "queued experiment '$e' has no inputs/run.sh"
  done
  QUEUE_SESSION="${QUEUE_SESSION:-exp-queue}"
  script="set -u\n"
  for e in $EXP; do
    script="${script}echo \"=== queue: $e \$(date -u +%H:%M:%SZ) ===\"\n"
    script="${script}./experiments/$e/inputs/run.sh > experiments/$e/run.log 2>&1\n"
    script="${script}echo \"=== queue: $e done rc=\$? \$(date -u +%H:%M:%SZ) ===\"\n"
  done
  script="${script}echo QUEUE COMPLETE\n"

  if [ -n "$REMOTE_HOST" ]; then
    if ssh_cmd "tmux has-session -t $QUEUE_SESSION 2>/dev/null"; then
      die "queue session '$QUEUE_SESSION' already running; '$0 \"$EXP\" stop' to end it"
    fi
    ssh_cmd "cd $REMOTE_DIR && printf '%b' '$script' > .queue.sh && chmod +x .queue.sh && \
             tmux new-session -d -s $QUEUE_SESSION 'cd $REMOTE_DIR && ./.queue.sh > queue.log 2>&1'"
    say "queued [$EXP] in session '$QUEUE_SESSION' on $REMOTE_HOST"
    say "follow with: REMOTE_HOST=$REMOTE_HOST QUEUE_SESSION=$QUEUE_SESSION $0 '$EXP' logs"
  else
    ( cd "$REPO" && printf '%b' "$script" > .queue.sh && chmod +x .queue.sh && ./.queue.sh )
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
  # Results made before the --profile -> --plan rename reached $REMOTE_HOST still
  # carry __profile_ in their variant directory names, and arrive beside the
  # migrated __plan_ ones. Fix with (idempotent, dry-run by default):
  #   python scripts/migrate_profile_label_to_plan.py experiments --apply
  # See "After pull: migrate the variant label" in scripts/README.md.
  ;;

stop)
  # Kill the session AND the eval processes it started. Killing only the session
  # leaves the python workers orphaned: they keep running, keep holding S3
  # connections and several GB of embedding model each, and keep writing rows
  # into the results tree -- so a "stopped" sweep silently carries on and the
  # next run contends with it for memory.
  [ -n "$REMOTE_HOST" ] || die "stop is remote-only; interrupt the local run instead"
  QUEUE_SESSION="${QUEUE_SESSION:-exp-queue}"
  ssh_cmd "SESSIONS='$SESSION $QUEUE_SESSION' REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -u
for s in $SESSIONS; do
  if tmux has-session -t "$s" 2>/dev/null; then
    tmux kill-session -t "$s"; echo "killed session $s"
  fi
done
# Match by PID, never by pattern: a pkill -f on the eval module name also
# matches this very command line and would kill the shell running it.
pids=$(pgrep -f 'run_mode[_]eval' | tr '\n' ' ')
if [ -n "$pids" ]; then
  kill $pids 2>/dev/null
  sleep 3
  still=$(pgrep -f 'run_mode[_]eval' | tr '\n' ' ')
  if [ -n "$still" ]; then kill -9 $still 2>/dev/null; echo "force-killed stragglers: $still"; fi
  echo "stopped eval workers: $pids"
else
  echo "no eval workers running"
fi
rm -f "$REMOTE_DIR/.queue.sh"
REMOTE
  ;;

*) usage 1 ;;
esac
