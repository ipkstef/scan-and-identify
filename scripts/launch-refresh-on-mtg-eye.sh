#!/usr/bin/env bash
# Launch the catalog refresh on mtg-eye in a detached tmux session.
#
# Run from your LOCAL machine. ssh's to mtg-eye, pulls latest code,
# rebuilds the locally-tagged scan-and-identify:latest image (the one
# refresh-catalog.sh runs the build CLI inside — distinct from the
# ghcr.io/ipkstef/scan-and-identify:latest image prod is running on),
# then launches ./scripts/refresh-catalog.sh inside a detached `tmux`
# with output teed to a host log file.
#
# Detached tmux means: closing your local terminal does NOT kill the build.
# Attach/detach freely without affecting the run.
#
# Usage:
#     ./scripts/launch-refresh-on-mtg-eye.sh
#
# When it returns, the build is RUNNING in the background on mtg-eye.
# The script prints the attach/tail commands you'll use to watch progress.

set -euo pipefail

REMOTE=mtg-eye

ssh "$REMOTE" 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail

SESSION=catalog-refresh
CACHE_DIR="$HOME/cv-build/imgs"
REPO="$HOME/scan-and-identify"
IMG=scan-and-identify:latest

# --- pre-flight guards ---------------------------------------------------
if docker ps --format '{{.Command}}' | grep -q 'build-catalog'; then
    echo "ABORT: a build-catalog container is already running. Inspect with: docker ps" >&2
    docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}' >&2
    exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ABORT: tmux session '$SESSION' already exists. Inspect with: tmux ls; tmux attach -t $SESSION" >&2
    tmux ls >&2
    exit 1
fi
if [ ! -d "$CACHE_DIR" ]; then
    echo "ABORT: expected image cache at $CACHE_DIR (containing flat <pid>.jpg files) — not found." >&2
    exit 1
fi
if [ ! -x "$REPO/scripts/refresh-catalog.sh" ]; then
    echo "ABORT: $REPO/scripts/refresh-catalog.sh not found or not executable." >&2
    exit 1
fi

# --- sync code + rebuild local refresh-runtime image ---------------------
# Rebuilds the locally-tagged scan-and-identify:latest. Prod runs on
# ghcr.io/ipkstef/scan-and-identify:latest (a different image reference)
# so this does NOT touch the production container.
cd "$REPO"
echo "==> git pull"
git pull --ff-only origin main
echo
if docker image inspect "$IMG" >/dev/null 2>&1; then
    BACKUP="scan-and-identify:pre-refresh-$(date +%Y%m%d-%H%M)"
    docker tag "$IMG" "$BACKUP"
    echo "==> Tagged previous $IMG as $BACKUP (rollback: docker tag $BACKUP $IMG)"
fi
echo "==> Rebuilding $IMG from $(git rev-parse --short HEAD)"
docker build -t "$IMG" .
echo

# --- launch detached tmux ------------------------------------------------
LOG="$REPO/refresh-$(date +%Y%m%d-%H%M).log"
echo "Cache dir : $CACHE_DIR (contains $(ls "$CACHE_DIR" | wc -l) files)"
echo "Log file  : $LOG"
echo "Session   : $SESSION"
echo

tmux new -d -s "$SESSION" \
    "cd '$REPO' && IMAGE_CACHE='$CACHE_DIR' ./scripts/refresh-catalog.sh 2>&1 | tee '$LOG'"

sleep 2
echo "--- tmux sessions ---"
tmux ls
echo
echo "Watch live:   ssh -t mtg-eye tmux attach -t $SESSION"
echo "              (detach without killing: Ctrl-b then d)"
echo "Tail log:     ssh mtg-eye \"tail -f $LOG\""
echo "Check status: ssh mtg-eye \"docker ps --filter ancestor=$IMG; tmux ls\""
REMOTE_SCRIPT
