#!/bin/bash
#SBATCH --job-name=rddm-shift
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3700
#SBATCH --time=04:00:00
#SBATCH --output=shift_%j.log

# Shifted train windows, one offset per job: sbatch shift.slurm --offset 1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

.venv/bin/python preprocess.py "$@"
