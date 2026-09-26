#!/usr/bin/env bash
#SBATCH --job-name=relax_sweeps
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/relax_sweeps_%j.out
#SBATCH --error=logs/relax_sweeps_%j.err
# Relaxation sweeps of the current handwired constructions (CPU only); EXTRA="--v1" for the earlier ones.
# Run from relaxation/:  sbatch run_sweeps.sbatch
set -euo pipefail
# activate a Python environment with requirements.txt installed
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -u run_sweeps.py --n_inst 20 --T 500 --procs 16 ${EXTRA:-}
