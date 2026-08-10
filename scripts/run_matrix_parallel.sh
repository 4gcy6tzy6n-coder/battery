#!/usr/bin/env bash
# Run the experiment matrix with bounded parallelism (xargs -P).
# Each worker pins PyTorch to a few threads so workers do not contend.
# Usage: SEEDS="0 1 2" WORKERS=3 THREADS=3 bash scripts/run_matrix_parallel.sh
set -u

export ROOT="${ROOT:-data/experiments/runs}"
export SEEDS="${SEEDS:-0 1 2}"
export CONFIGS="${CONFIGS:-main,ablation_nophysics,ablation_unordered,ablation_fixedweight,baseline_point,baseline_mcdropout}"
export WORKERS="${WORKERS:-3}"
export THREADS="${THREADS:-3}"
export LOG="$ROOT/matrix_parallel.log"

mkdir -p "$ROOT"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"

jobs=()
for name in ${CONFIGS//,/ }; do
  for seed in $SEEDS; do
    jobs+=("$name $seed")
  done
done

echo "=== parallel matrix start $(date +%FT%T) | ${#jobs[@]} runs | ${WORKERS}x${THREADS} threads ===" >> "$LOG"
printf '%s\n' "${jobs[@]}" | xargs -P "$WORKERS" -I '{}' bash -c '
  name=${1%% *}
  seed=${1##* }
  out="$ROOT/$name/seed$seed"
  echo ">> [$name seed=$seed] start $(date +%FT%T)" >> "$LOG"
  .venv/bin/python scripts/run_experiment.py \
    --config "configs/experiments/$name.yaml" --output "$out" --seed "$seed" >> "$LOG" 2>&1
  echo ">> [$name seed=$seed] exit=$? done $(date +%FT%T)" >> "$LOG"
' _ {}
echo "=== parallel matrix complete $(date +%FT%T) ===" >> "$LOG"
