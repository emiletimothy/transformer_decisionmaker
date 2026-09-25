#!/usr/bin/env bash
#SBATCH --job-name=ql_exploration
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Closed loop with an eps-greedy behaviour policy on the model's own SELECT argmax
# (eps = 0, 0.05, 0.1, 0.2, 0.3; the tabular eps-greedy baseline uses 0.2), 1000-step
# long-horizon families, 50 MDPs -> figures/<model>/closed_loop_exploration/.
# Usage (from tabular_q_learning/):  sbatch scripts/jobs/eval_exploration.sh
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from tabular_q_learning/:  cd tabular_q_learning && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
export PATH=/system/linux/miniforge-3.13/bin:$PATH
export WANDB_MODE=disabled
for m in continuous_residual continuous_overwrite discrete; do
  python3 -u scripts/eval_closed_loop.py --model "$m" --out_subdir closed_loop_exploration \
    --parts long_horizon --n_mdps 50 --alpha 0.2 \
    --astar_modes self,self_eps0.05,self_eps0.1,self_eps0.2,self_eps0.3
done
