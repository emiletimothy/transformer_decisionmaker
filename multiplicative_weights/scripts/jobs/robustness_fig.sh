#!/usr/bin/env bash
#SBATCH --job-name=mw_robustness
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Robustness figure of the continuous transformer (figures_per_model.py --only robustness).
# Usage (from multiplicative_weights/):  sbatch scripts/jobs/robustness_fig.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/"; exit 1; }
# activate a Python environment with requirements.txt installed
python3 -u scripts/figures_per_model.py --only robustness
