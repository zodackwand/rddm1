#!/bin/bash
#SBATCH --job-name=rddm-gen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:1
#SBATCH --time=01:00:00
#SBATCH --output=gen_subjects_%j.log

.venv/bin/python gen_subjects.py "$@"
