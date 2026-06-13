#!/usr/bin/env bash
# Launch the catalog refresh on mtg-eye in a detached tmux session.
#
# Run from your LOCAL machine. ssh's to mtg-eye, refuses to start if a
# refresh is already in progress, then kicks off ./scripts/refresh-catalog.sh
# inside `tmux` with output teed to a host log file.
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
SESSION=catalog-refresh

ssh "$REMOTE" 'bash -s' <<'REMOTE_SCRIPT'
set -euo pipefail

SESSION=catalog-refresh
CACHE_DIR="$HOME/cv-build/imgs"
REPO="$HOME/scan-and-identify"

# --- guards ---------------------------------------------------------------
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

# --- launch ---------------------------------------------------------------
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
echo "Watch live:   ssh -t $(hostname -s 2>/dev/null || echo mtg-eye) tmux attach -t $SESSION"
echo "              (detach without killing: Ctrl-b then d)"
echo "Tail log:     ssh mtg-eye \"tail -f $LOG\""
echo "Check still running:"
echo "              ssh mtg-eye \"docker ps --filter ancestor=scan-and-identify:latest; tmux ls\""
REMOTE_SCRIPT
