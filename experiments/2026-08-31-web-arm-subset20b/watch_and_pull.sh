#!/usr/bin/env bash
# Poll the remote sweep and pull results down each time an arm finishes, so
# progress is visible locally without waiting for all four.
#
#   REMOTE_HOST=user@host REMOTE_IDENTITY=key.pem ./watch_and_pull.sh
#
# Emits one line per event (arm started / arm finished+pulled / complete), so it
# can be driven by a monitor. Exits when the sweep prints ALL ARMS COMPLETE.
set -uo pipefail

HOST="${REMOTE_HOST:?set REMOTE_HOST}"
KEY="${REMOTE_IDENTITY:-}"
DIR="${REMOTE_DIR:-~/sana-framework}"
INTERVAL="${INTERVAL:-60}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REMOTE_EXP="$DIR/experiments/2026-08-31-web-arm-subset20b"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
[[ -n "$KEY" ]] && SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15)

seen=""

pull() {
  # Tar over ssh: needs only tar on both ends, no rsync.
  "${SSH[@]}" "$HOST" "cd $REMOTE_EXP 2>/dev/null && tar czf - results logs 2>/dev/null" \
    | tar xzf - -C "$HERE" 2>/dev/null || return 1
}

summarise() {
  [[ -x "$HERE/../../.venv/bin/python" ]] || return 0
  "$HERE/../../.venv/bin/python" "$HERE/summarize.py" 2>/dev/null | sed 's/^/    /' || true
}

while :; do
  # One round trip: fetch the driver log's event lines.
  log="$("${SSH[@]}" "$HOST" "cat $REMOTE_EXP/logs/driver.log 2>/dev/null" || true)"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    case "$seen" in *"$line"*) continue ;; esac
    seen+="$line"$'\n'

    case "$line" in
      ">>> "*)
        echo "START  ${line#>>> }"
        ;;
      "<<< "*)
        if pull; then
          echo "PULLED ${line#<<< }"
          summarise
        else
          echo "PULL FAILED after: ${line#<<< }"
        fi
        ;;
    esac
  done <<< "$(printf '%s\n' "$log" | grep -E '^(>>>|<<<) ' || true)"

  if printf '%s' "$log" | grep -q "ALL ARMS COMPLETE"; then
    pull && echo "DONE  all arms complete, final pull ok" || echo "DONE  all arms complete, final pull FAILED"
    summarise
    exit 0
  fi

  sleep "$INTERVAL"
done
