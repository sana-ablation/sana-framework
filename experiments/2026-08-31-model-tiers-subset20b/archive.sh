#!/bin/bash
# Copy the grid's scratch output into the durable experiment dir.
# Run after the grid completes; tmp/ is gitignored and gets wiped.
set -eu
DEST="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DEST/../.." && pwd)"

rsync -a --delete "$ROOT/tmp/results-tiers/" "$DEST/results/"
rsync -a --delete "$ROOT/tmp/logs-tiers/"    "$DEST/logs/"
mkdir -p "$DEST/logs/cell-stdout"
cp "$ROOT"/tmp/logs-tiers-stdout-*.log "$DEST/logs/cell-stdout/" 2>/dev/null || true

echo "archived to $DEST"
du -sh "$DEST"
