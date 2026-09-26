#!/usr/bin/env bash
#SBATCH --job-name=ql_evaluate
#SBATCH --partition=<partition>        # set to your cluster's GPU partition
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#
# Every evaluation of the three final models (paths in paths.py):
#   figures/comparison/  clean steps, memory-length fit, Q-table probe, (alpha, gamma) fit
#   figures/<model>/     closed_loop/ (1000-step suite, model's own a*) and evaluation/ (4_evaluate figures)
# Usage (from tabular_q_learning/):  sbatch scripts/jobs/evaluate.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from tabular_q_learning/:  cd tabular_q_learning && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
# activate a Python environment with requirements.txt installed
export WANDB_MODE=disabled
python3 -u scripts/eval_clean_steps.py --max_seqs 2000
python3 -u scripts/eval_memory_fit.py
python3 -u scripts/eval_q_probe.py
python3 -u scripts/eval_alpha_gamma.py --max_seqs 1500
for m in continuous_residual continuous_overwrite discrete; do
  python3 -u scripts/eval_closed_loop.py --model "$m" --n_mdps 50 --alpha 0.2
  python3 -u scripts/4_evaluate.py --model "$m" --n_steps 50 --nonstationary_switch_step 50 \
    --nonstationary_post_steps 100 --alpha 0.2 --gamma 0.9 --epsilon 0.2 --eval_seed 9999 --n_eval_mdps 10
done
python3 -u scripts/eval_drift.py
