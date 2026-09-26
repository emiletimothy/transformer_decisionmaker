#!/usr/bin/env bash
#SBATCH --job-name=ql_main_fig_data
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Data behind the main-text Q-learning figure: the combined-row data of the continuous and
# discrete models (4_evaluate.py, legacy parts) and the held-out Q-probe predictions.
# Usage (from tabular_q_learning/):  sbatch scripts/jobs/main_figure_data.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from tabular_q_learning/"; exit 1; }
# activate a Python environment with requirements.txt installed
export WANDB_MODE=disabled
python3 -u scripts/eval_q_probe.py
for m in continuous_residual discrete; do
  python3 -u scripts/4_evaluate.py --model "$m" --parts legacy --n_steps 50 --nonstationary_switch_step 50 \
    --nonstationary_post_steps 100 --alpha 0.2 --gamma 0.9 --epsilon 0.2 --eval_seed 9999 --n_eval_mdps 10
done
