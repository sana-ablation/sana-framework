#!/usr/bin/env bash
# Pull the sweep's results and logs back into this experiment directory.
#   REMOTE_HOST=user@host REMOTE_IDENTITY=key.pem ./pull_remote.sh
set -euo pipefail
HOST="${REMOTE_HOST:?set REMOTE_HOST}"
KEY="${REMOTE_IDENTITY:-}"
DIR="${REMOTE_DIR:-~/sana-framework}"
DEST="$(cd "$(dirname "$0")" && pwd)"
SSH=(ssh); [[ -n "$KEY" ]] && SSH=(ssh -i "$KEY")

"${SSH[@]}" "$HOST" "cd $DIR/experiments/2026-08-31-web-arm-subset20b && tar czf - results logs" \
  | tar xzf - -C "$DEST"

echo "pulled into $DEST"
du -sh "$DEST/results" "$DEST/logs" 2>/dev/null || true
