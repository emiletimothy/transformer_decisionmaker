#!/usr/bin/env bash
#SBATCH --job-name=ql_figures
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#
# The 4_evaluate.py figure set of the three final models -> figures/<model>/evaluation/.
# Closed-loop panels show the transformer greedy (solid) and eps-greedy (dashed, eps = --epsilon).
# Usage (from tabular_q_learning/):  sbatch scripts/jobs/figures.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from tabular_q_learning/:  cd tabular_q_learning && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
# activate a Python environment with requirements.txt installed
export WANDB_MODE=disabled
for m in continuous_residual continuous_overwrite discrete; do
  python3 -u scripts/4_evaluate.py --model "$m" --n_steps 50 --nonstationary_switch_step 50 \
    --nonstationary_post_steps 100 --alpha 0.2 --gamma 0.9 --epsilon 0.2 --eval_seed 9999 --n_eval_mdps 10
done
