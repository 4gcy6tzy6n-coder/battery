#!/usr/bin/env bash
# Post-processing for the 4 SOTA second-batch runs (physics s0/s1 + TCN s0/s1).
# Triggered by cron, idempotent.

set -u
cd "$(dirname "$0")/.."

VENV=".venv/bin/python"
DIRS=(data/experiments/sota-v2-physics-s0
      data/experiments/sota-v2-physics-s1
      data/experiments/sota-tcn-s0
      data/experiments/sota-tcn-s1)
CFGS=(configs/experiments/sota_unordered_physics.yaml
      configs/experiments/sota_unordered_physics.yaml
      configs/experiments/sota_tcn_unordered.yaml
      configs/experiments/sota_tcn_unordered.yaml)

CAP=data/experiments/runtime_capture
mkdir -p "$CAP/figures"

# Step 0: check all 4 results.json present
ALL_DONE=1
for d in "${DIRS[@]}"; do
  if [ ! -f "$d/results.json" ]; then
    ALL_DONE=0
    break
  fi
done
if [ "$ALL_DONE" -ne 1 ]; then
  echo "ABORT: not all 4 results.json present — try next cron tick"
  for d in "${DIRS[@]}"; do
    name=$(basename "$d")
    pid=$(pgrep -f "run_experiment.*$name" || true)
    [ -n "$pid" ] && ps -o pid,etime,cputime -p "$pid"
  done
  exit 1
fi

echo "Step 0: all 4 results.json present"
ls -la "${DIRS[@]}"

# Step 1: recalibrate each (skip if already done)
for i in 0 1 2 3; do
  d="${DIRS[$i]}"
  cfg="${CFGS[$i]}"
  if [ -f "$d/recalibrated_results.json" ]; then
    echo "recalibrated already exists for $d — skip"
    continue
  fi
  echo "Step 1: recalibrate $d"
  "$VENV" scripts/recalibrate_run.py --run "$d" --config "$cfg"
done

# Step 2: build symlink root for aggregate
echo "Step 2: aggregate"
mkdir -p data/experiments/runs_sota/sota_unordered_physics
mkdir -p data/experiments/runs_sota/sota_tcn_unordered
ln -sfn /Users/yyl/Desktop/workshop/battery/data/experiments/sota-v2-physics-s0 data/experiments/runs_sota/sota_unordered_physics/seed0
ln -sfn /Users/yyl/Desktop/workshop/battery/data/experiments/sota-v2-physics-s1 data/experiments/runs_sota/sota_unordered_physics/seed1
ln -sfn /Users/yyl/Desktop/workshop/battery/data/experiments/sota-tcn-s0 data/experiments/runs_sota/sota_tcn_unordered/seed0
ln -sfn /Users/yyl/Desktop/workshop/battery/data/experiments/sota-tcn-s1 data/experiments/runs_sota/sota_tcn_unordered/seed1

"$VENV" scripts/aggregate_results.py --root data/experiments/runs_sota --format markdown > "$CAP/aggregate_all_sota.md"
cat "$CAP/aggregate_all_sota.md"

# Step 3: figures
echo "Step 3: figures"
"$VENV" scripts/make_figures.py \
  --runs "sota-noPhys=data/experiments/sota-v2-s0,sota+phys=data/experiments/sota-v2-physics-s0,sota+TCN=data/experiments/sota-tcn-s0" \
  --output "$CAP/figures" || echo "figures step failed (non-fatal)"

# Step 4: copy figures into tracked docs/research/figures/
echo "Step 4: copy figures to docs/"
mkdir -p docs/research/figures
cp "$CAP/figures/calibration.png" docs/research/figures/calibration_sota_all.png 2>/dev/null
cp "$CAP/figures/physics_consistency.png" docs/research/figures/physics_consistency_sota_all.png 2>/dev/null

# Step 5: commit
echo "Step 5: commit (REVIEW before push)"
echo "git add docs/research/progress_log.md docs/research/paper_manuscript.md docs/research/figures/"

echo "DONE"
echo "Wall: $(date)"