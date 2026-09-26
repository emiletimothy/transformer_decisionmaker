#!/usr/bin/env bash
#SBATCH --job-name=mwu_eval_matched
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Matched-memory comparison (continuous residual vs discrete vs the raw-history model vs
# MW / Bayes / majority vote), both seeds -> figures/comparison/matched_memory/.
# Usage (from multiplicative_weights/):  sbatch scripts/jobs/eval_matched_memory.sh [extra args]
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/:  cd multiplicative_weights && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
# activate a Python environment with requirements.txt installed
python3 -u scripts/eval_matched_memory.py "$@"
