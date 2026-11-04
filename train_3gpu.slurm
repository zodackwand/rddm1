#!/bin/bash
#SBATCH --job-name=rddm-train-3gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.log

# Batch 512 as in the paper: 171 per GPU, accumulated in micro-batches that fit in memory
.venv/bin/torchrun --standalone --nproc_per_node=3 train.py --batch-size 512 --micro-batch 86 "$@"
