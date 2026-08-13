#!/usr/bin/env bash
# Lightweight watcher for the two backgrounded SOTA runs.
# Replaces capture_sota_runtime.sh which used `sample` (unreliable on macOS).
# Polls every 60s; logs cpu/rss/results.json state. Exits when both PIDs done.

set -u

CAP_DIR="${CAP_DIR:-data/experiments/runtime_capture}"
PIDS_DEFAULT="85694 85695"
PIDS="${PIDS:-${PIDS_DEFAULT}}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "$CAP_DIR"
CO="$CAP_DIR/coordinator.log"
echo "$(date -u +%FT%TZ) watch_sota_results started pids=$PIDS interval=${INTERVAL}s" >> "$CO"

while :; do
  ALL_DONE=1
  for pid in $PIDS; do
    case "$pid" in
      85694) out="data/experiments/sota-v2-s0" ;;
      85695) out="data/experiments/sota-v2-s1" ;;
      *)     out="data/experiments/unknown-$pid" ;;
    esac
    log="$CAP_DIR/${pid}.log"
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "$(date -u +%FT%TZ) process gone pid=$pid" >> "$log"
      [ -f "$out/results.json" ] || ALL_DONE=0
      continue
    fi
    cpu="$(ps -o cputime= -p "$pid" 2>/dev/null | tr -d ' ')"
    rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')KB"
    etime="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ -f "$out/results.json" ]; then
      echo "$(date -u +%FT%TZ) results.json present pid=$pid etime=$etime cpu=$cpu rss=$rss" >> "$log"
    else
      echo "$(date -u +%FT%TZ) running pid=$pid etime=$etime cpu=$cpu rss=$rss done=NO" >> "$log"
      ALL_DONE=0
    fi
  done
  [ "$ALL_DONE" -eq 1 ] && { echo "$(date -u +%FT%TZ) all pids done" >> "$CO"; exit 0; }
  sleep "$INTERVAL"
done