#!/bin/bash
# Copy the sweep's scratch output into the durable experiment dir.
# Run after the sweep completes; tmp/ is gitignored and gets wiped.
set -eu
DEST="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DEST/../.." && pwd)"

rsync -a --delete "$ROOT/tmp/results-webarm/" "$DEST/results/"
rsync -a --delete "$ROOT/tmp/logs-webarm/"    "$DEST/logs/"
mkdir -p "$DEST/logs/arm-stdout"
cp "$ROOT"/tmp/webarm-*.log "$DEST/logs/arm-stdout/" 2>/dev/null || true

echo "archived to $DEST"
du -sh "$DEST"
