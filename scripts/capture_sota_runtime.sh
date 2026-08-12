#!/usr/bin/env bash
# Periodic runtime capture for backgrounded SOTA training runs.
#
# Why this exists: the two `sota_unordered_nophysics` runs (PID 85694, 85695)
# were launched without stdout redirection. Their per-epoch print() lines are
# permanently lost. On macOS without sudo we cannot attach py-spy or read
# Python locals from a native-code GIL-released frame, so the only observable
# signal is:
#   1. C-level call stack via `sample` (snapshot, non-blocking)
#   2. CPU time growth (= forward progress through PyTorch ops)
#   3. Filesystem changes (`results.json` / `best_model.pt` appearing)
#
# This script polls every 60 seconds and writes a structured log to
# data/experiments/runtime_capture/<pid>.log. When `results.json` lands for a
# PID, that PID's polling stops; the parent script exits when both PIDs have
# finished.

set -u

CAP_DIR="${CAP_DIR:-data/experiments/runtime_capture}"
PIDS_DEFAULT="85694 85695"
PIDS="${PIDS:-${PIDS_DEFAULT}}"
INTERVAL="${INTERVAL:-60}"

mkdir -p "$CAP_DIR"

now_iso() { date -u +%FT%TZ; }
cpu_time() { ps -o cputime= -p "$1" 2>/dev/null | tr -d ' '; }
rss_kb()   { ps -o rss=    -p "$1" 2>/dev/null | tr -d ' '; }
elapsed()  { ps -o etime=  -p "$1" 2>/dev/null | tr -d ' '; }
top_frame() {
  # Best-effort "what is it doing right now" — second non-header line of `sample` summary.
  sample "$1" 1 -mayDie 2>/dev/null | awk '/Call graph:/,0' | grep -m1 -E '\(in libtorch_cpu|\(in libpython3|\(in libomp|PyEval_EvalCode|THPEngine_run_backward' | sed 's/^[[:space:]]*+[[:space:]]*//; s/[[:space:]]\+/ /g'
}
has_results() { ls "$2/results.json" >/dev/null 2>&1; }

run_log() {
  local pid="$1" out_dir="$2" log="$CAP_DIR/${pid}.log"
  local first_ts last_cpu samples_done=0
  first_ts="$(now_iso)"
  last_cpu="$(cpu_time "$pid")"
  echo "$first_ts start pid=$pid out=$out_dir cputime=${last_cpu}s" >> "$log"

  while :; do
    sleep "$INTERVAL"
    local ts cur_cpu rss top done
    ts="$(now_iso)"
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "$ts exit pid=$pid (process gone)" >> "$log"
      return 0
    fi
    cur_cpu="$(cpu_time "$pid")"
    rss="$(rss_kb "$pid")"
    top="$(top_frame "$pid" | head -c 200)"
    has_results "$pid" "$out_dir" && done="YES" || done="NO"
    samples_done=$((samples_done + 1))
    echo "$ts sample=$samples_done cputime=${cur_cpu}s rss=${rss}KB done=$done top=${top}" >> "$log"
    if [ "$done" = "YES" ]; then
      echo "$ts complete results.json present at $out_dir" >> "$log"
      return 0
    fi
  done
}

# Fan out one poller per PID; the wrapper waits for all.
pids_running=()
for pid in $PIDS; do
  case "$pid" in
    85694) out_dir="data/experiments/sota-v2-s0" ;;
    85695) out_dir="data/experiments/sota-v2-s1" ;;
    *)     out_dir="data/experiments/unknown-$pid" ;;
  esac
  run_log "$pid" "$out_dir" &
  pids_running+=("$!")
done

echo "$(now_iso) capturers started: ${pids_running[*]}" >> "$CAP_DIR/coordinator.log"
for job in "${pids_running[@]}"; do
  wait "$job"
done
echo "$(now_iso) all capturers exited" >> "$CAP_DIR/coordinator.log"