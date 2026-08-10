#!/usr/bin/env bash
# Run the experiment matrix with bounded parallelism (xargs -P).
# Each worker pins PyTorch to a few threads so workers do not contend.
# Usage: SEEDS="0 1 2" WORKERS=3 THREADS=3 bash scripts/run_matrix_parallel.sh
set -u

ROOT="data/experiments/runs"
SEEDS="${SEEDS:-0 1 2}"
CONFIGS="${CONFIGS:-main,ablation_nophysics,ablation_unordered,ablation_fixedweight,baseline_point,baseline_mcdropout}"
WORKERS="${WORKERS:-3}"
THREADS="${THREADS:-3}"
LOG="$ROOT/matrix_parallel.log"

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
  name=$(echo "{}" | cut -d" " -f1)
  seed=$(echo "{}" | cut -d" " -f2)
  out="'$ROOT'/$name/seed$seed"
  echo ">> [$name seed=$seed] start $(date +%FT%T)" >> "'$LOG'"
  .venv/bin/python scripts/run_experiment.py \
    --config "configs/experiments/$name.yaml" --output "$out" --seed "$seed" >> "'$LOG'" 2>&1
  echo ">> [$name seed=$seed] exit=$? done $(date +%FT%T)" >> "'$LOG'"
'
echo "=== parallel matrix complete $(date +%FT%T) ===" >> "$LOG"
