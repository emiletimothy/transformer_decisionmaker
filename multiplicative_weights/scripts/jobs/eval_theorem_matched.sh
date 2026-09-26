#!/usr/bin/env bash
#SBATCH --job-name=mwu_eval_theorem
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# Theorem-matched models vs weighted majority (distance, regret, probe, adversarial tests)
# -> figures/continuous_residual/theorem_matched/.
# Usage (from multiplicative_weights/):  sbatch scripts/jobs/eval_theorem_matched.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/:  cd multiplicative_weights && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
# activate a Python environment with requirements.txt installed
python3 -u scripts/eval_theorem_matched.py "$@"
