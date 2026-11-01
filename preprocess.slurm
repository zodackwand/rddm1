#!/bin/bash
#SBATCH --job-name=rddm-preprocess
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3700
#SBATCH --time=04:00:00
#SBATCH --output=preprocess_%j.log

# One process per subject, so numeric libraries must stay single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

.venv/bin/python preprocess.py "$@"

.venv/bin/python check_data.py
