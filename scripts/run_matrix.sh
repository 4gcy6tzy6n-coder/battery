#!/usr/bin/env bash
# Run the full experiment matrix (configs x seeds) sequentially, logging progress.
# Usage: bash scripts/run_matrix.sh [--seeds "0 1 2"] [--configs "main,nophysics,..."]
set -u

ROOT="data/experiments/runs"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-main,ablation_nophysics,ablation_unordered,ablation_fixedweight,baseline_point,baseline_mcdropout}"

mkdir -p "$ROOT"
LOG="$ROOT/matrix.log"
echo "=== matrix start $(date +%FT%T) ===" >> "$LOG"

IFS=',' read -ra NAMES <<< "$CONFIGS"
for name in "${NAMES[@]}"; do
  for seed in $SEEDS; do
    out="$ROOT/$name/seed$seed"
    echo ">> [$name seed=$seed] start $(date +%FT%T)" >> "$LOG"
    .venv/bin/python scripts/run_experiment.py \
      --config "configs/experiments/$name.yaml" \
      --output "$out" \
      --seed "$seed" >> "$LOG" 2>&1
    status=$?
    echo ">> [$name seed=$seed] exit=$status done $(date +%FT%T)" >> "$LOG"
  done
done
echo "=== matrix complete $(date +%FT%T) ===" >> "$LOG"
