#!/usr/bin/env bash
# Bundle of post-processing steps for the SOTA s0/s1 runs.
# Triggered by cron job ad2c02f6 (or run manually).
#
# Idempotent: skips steps whose output already exists.

set -u
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
SOTA_OUT_S0="data/experiments/sota-v2-s0"
SOTA_OUT_S1="data/experiments/sota-v2-s1"
CONFIG="configs/experiments/sota_unordered_nophysics.yaml"
CAP="data/experiments/runtime_capture"
RUNS_SOTA="data/experiments/runs_sota"

mkdir -p "$CAP"

step() { printf '\n=== %s ===\n' "$*"; }

# --- Step 0: verify both results.json present ---
if [ ! -f "$SOTA_OUT_S0/results.json" ] || [ ! -f "$SOTA_OUT_S1/results.json" ]; then
  echo "ABORT: results.json missing in s0 or s1 — try next cron tick."
  ps -o pid,etime,cputime -p 85694,85695 2>/dev/null || true
  exit 1
fi
step "Step 0: results present"
ls -la "$SOTA_OUT_S0" "$SOTA_OUT_S1"

# --- Step 1: recalibrate each run (skip if already done) ---
for seed in 0 1; do
  out="data/experiments/sota-v2-s$seed"
  if [ -f "$out/recalibrated_results.json" ]; then
    echo "recalibrated_results.json exists for seed=$seed, skip"
    continue
  fi
  step "Step 1: recalibrate seed=$seed"
  "$VENV" scripts/recalibrate_run.py --run "$out" --config "$CONFIG"
done

# --- Step 2: independent verification ---
step "Step 2: verify_results.py on SOTA"
"$VENV" scripts/verify_results.py \
  --runs-dir data/experiments \
  --include sota-v2-s0,sota-v2-s1 \
  --write "$CAP/verify_sota.json" || echo "verify_results.py failed (non-fatal)"

# --- Step 3: aggregate tables ---
step "Step 3: aggregate_results.py"
mkdir -p "$RUNS_SOTA/sota_unordered_nophysics"
ln -sfn ../../../sota-v2-s0 "$RUNS_SOTA/sota_unordered_nophysics/seed0"
ln -sfn ../../../sota-v2-s1 "$RUNS_SOTA/sota_unordered_nophysics/seed1"

"$VENV" scripts/aggregate_results.py \
  --root data/experiments/runs --format markdown \
  > "$CAP/aggregate_main.md" || echo "aggregate main failed (non-fatal)"
"$VENV" scripts/aggregate_results.py \
  --root "$RUNS_SOTA" --format markdown \
  > "$CAP/aggregate_sota.md" || echo "aggregate sota failed (non-fatal)"

# --- Step 4-5: update docs (manual edit step — print instructions) ---
step "Step 4-5: update paper_results_draft.md + progress_log.md"
echo "Auto-update not yet wired — edit these files manually:"
echo "  docs/research/paper_results_draft.md"
echo "  docs/research/progress_log.md"
echo "Insert SOTA s0/s1 numbers + comparison vs ablation_unordered."

# --- Step 6: commit + push ---
step "Step 6: commit + push (REVIEW before running)"
cat <<'EOF'
NEXT (when ready to push):
  git add -A
  git commit -m "feat: SOTA s0/s1 complete; recalibrate; update results draft"
  git push -u origin agent/phase2b-probabilistic-core
EOF

step "Done"
echo "Wall: $(date)"
echo "Process 85694: $(ps -o etime= -p 85694 2>/dev/null || echo gone)"
echo "Process 85695: $(ps -o etime= -p 85695 2>/dev/null || echo gone)"