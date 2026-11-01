#!/bin/bash
#SBATCH --job-name=rddm-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --time=48:00:00
#SBATCH --output=train_%j.log

.venv/bin/python train.py "$@"
