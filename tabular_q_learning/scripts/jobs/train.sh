#!/usr/bin/env bash
#SBATCH --job-name=ql_train
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#
# Data and training of the final Q-learning models.
#  * data: alpha 0.2, gamma 0.9, Gaussian reward noise sigma 0.3, episodes of 100-200 steps,
#    task mix orig / shifted / goal-state -> data/qlv3_dataset.pt, then the shortcut analysis
#    (how often simpler rules give the teacher's label).
#  * training: 4 layers, 8 heads, d_model 256, BPTT window 50, max_steps 200, 20 epochs
#    (discrete: 15) -> checkpoints/<model>/coconut_transformer_qlv3-<variant>.pt
#      continuous_residual   slot write c <- c + W h[UPDATE]   (headline)
#      continuous_overwrite  slot write c <- h[UPDATE]
#      discrete              slot snapped to a vocabulary token (straight-through Gumbel, tau 2 -> 0.5)
# Usage (from tabular_q_learning/):
#   sbatch scripts/jobs/train.sh data
#   sbatch --dependency=afterok:<data job> scripts/jobs/train.sh continuous_residual
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from tabular_q_learning/:  cd tabular_q_learning && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
export PATH=/system/linux/miniforge-3.13/bin:$PATH
DATA=data/qlv3_dataset.pt
MODEL=${1:?data | continuous_residual | continuous_overwrite | discrete}
if [[ "$MODEL" == data ]]; then
  python3 -u scripts/1_generate_data.py --n_sequences 22000 --alpha 0.2 --gamma 0.9 \
    --reward_noise 0.3 --min_steps 100 --max_steps 200 \
    --task_mix orig shifted goal --reward_shift 0.5 --goal_step_cost 0.1 \
    --seed 44 --output "$DATA"
  python3 -u scripts/data_shortcuts.py --data "$DATA" --max_seqs 4000
  exit 0
fi
case "$MODEL" in
  continuous_residual)  VARIANT=residual;  FLAGS="--context_mode continuous --residual_context"; EPOCHS=${EPOCHS:-20} ;;
  continuous_overwrite) VARIANT=overwrite; FLAGS="--context_mode continuous";                    EPOCHS=${EPOCHS:-20} ;;
  discrete)             VARIANT=discrete;  FLAGS="--context_mode discrete --gumbel_tau_start 2.0 --gumbel_tau_end 0.5"; EPOCHS=${EPOCHS:-15} ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
python3 -u scripts/3_train.py --data_path "$DATA" --checkpoint_dir "checkpoints/$MODEL" \
  --n_layers 4 --n_heads 8 --d_model 256 --d_ff 1024 --dropout 0.1 \
  --epochs "$EPOCHS" --batch_size 64 --lr 3e-4 --weight_decay 1e-2 --eval_every 500 \
  --max_steps 200 --truncate_bptt 50 \
  --run_name "qlv3-$VARIANT" $FLAGS \
  --use_wandb --wandb_tags qlv3 "context-write-$VARIANT"
