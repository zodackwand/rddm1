#!/bin/bash
# Exp 01: puts every job of the experiment in the Slurm queue with dependencies.
# Usage (from anywhere inside the repo): bash experiments/exp01_subject_balance_cond_drop/submit.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP=experiments/exp01_subject_balance_cond_drop
mkdir -p "$EXP/logs"

# 1) Subject id of every window (CPU). Existing windows are only checked, never rewritten.
ids=$(sbatch --parsable --job-name=exp01-ids --partition=compute --nodes=1 --ntasks=1 \
  --cpus-per-task=32 --mem-per-cpu=3700 --time=04:00:00 --output="$EXP/logs/ids_%j.log" \
  --wrap "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; .venv/bin/python preprocess.py --ids-only")
echo "ids: $ids"

# 2) Three runs, 1 GPU each (the per-user limit is 3), batch 128 as v1; 3) evaluation after each run
for name in bal300 bal300_cd01 cd01; do
  case $name in
    bal300)      extra="--per-subject 300" ;;
    bal300_cd01) extra="--per-subject 300 --cond-drop 0.1" ;;
    cd01)        extra="--cond-drop 0.1" ;;
  esac
  mkdir -p "$EXP/results/$name"
  train=$(sbatch --parsable --dependency=afterok:$ids --job-name=exp01-$name --output="$EXP/logs/train_${name}_%j.log" \
    train.slurm --path runs/exp01_$name $extra --save-every 50)
  ev=$(sbatch --parsable --dependency=afterok:$train --job-name=exp01-eval-$name --output="$EXP/logs/eval_${name}_%j.log" \
    evaluate.slurm --path runs/exp01_$name --out "$EXP/results/$name/eval")
  curve=$(sbatch --parsable --dependency=afterok:$train --time=08:00:00 --job-name=exp01-curve-$name \
    --output="$EXP/logs/curve_${name}_%j.log" \
    diagnostics.slurm diag2_epoch_curve.py --path runs/exp01_$name --out "$EXP/results/$name/curve.json")
  echo "$name: train $train, eval $ev, curve $curve"
done
