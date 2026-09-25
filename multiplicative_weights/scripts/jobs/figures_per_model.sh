#!/usr/bin/env bash
#SBATCH --job-name=mwu_figures
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#
# The submitted paper's MWU figure layouts, redrawn for the final recurrent models
# -> figures/continuous_residual/{evaluation,attention,overview}/.
# Usage (from multiplicative_weights/):  sbatch scripts/jobs/figures_per_model.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/:  cd multiplicative_weights && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
export PATH=/system/linux/miniforge-3.13/bin:$PATH
python3 -u scripts/figures_per_model.py "$@"
