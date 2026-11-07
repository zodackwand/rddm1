#!/bin/bash
#SBATCH --job-name=rddm-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --time=04:00:00
#SBATCH --output=diag_%j.log

# One diagnostic per job, for example:
#   sbatch diagnostics.slurm diag1_ppg_ablation.py --path runs/rddm_v2 --authors ../authors_weights --out diag/d1.json
.venv/bin/python "$@"
