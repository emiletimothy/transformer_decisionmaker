#!/usr/bin/env bash
#SBATCH --job-name=mwu_train
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:30:00
#SBATCH --output=logs/%x_%j.out
#
# Train one recurrent MWU model (the recipe of every final model): 16-stage curriculum
# T = 5 ... 200, lr 1e-3, batch 64, <= 8 epochs per stage (patience 3), a fresh
# 20k-sequence training set every epoch. Saves checkpoints/<model>_seed<seed>/final.pt.
#   model = continuous_residual   latent M_t = M_{t-1} + W h[UPD]   (headline)
#           continuous_overwrite  latent M_t = h[UPD]
#           discrete              latent snapped to a vocabulary token
#           theorem_matched       continuous_residual on the noisy-true-expert data of Theorem 3.1
# Usage (from multiplicative_weights/):  sbatch scripts/jobs/train.sh continuous_residual 42
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/:  cd multiplicative_weights && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
export PATH=/system/linux/miniforge-3.13/bin:$PATH
MODEL=${1:?model: continuous_residual | continuous_overwrite | discrete | theorem_matched}
SEED=${2:-42}
case "$MODEL" in
  continuous_residual)  FLAGS="--context_mode continuous --residual_latent" ;;
  continuous_overwrite) FLAGS="--context_mode continuous" ;;
  discrete)             FLAGS="--context_mode discrete" ;;
  theorem_matched)      FLAGS="--context_mode continuous --residual_latent --data_model noisy_expert" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
python3 -u scripts/train.py $FLAGS --seed "$SEED" --fresh_data \
  --n_train 20000 --batch_size 64 --lr 1e-3 --max_T 200 \
  --stage_lengths 5 10 15 20 25 30 35 40 45 50 65 80 95 130 165 200 \
  --max_epochs_per_stage 8 --patience 3 --time_budget_min 240
