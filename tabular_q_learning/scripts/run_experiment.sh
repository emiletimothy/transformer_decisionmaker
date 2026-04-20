#!/usr/bin/env bash
#SBATCH --job-name=coconut_ql
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=coconut_experiment%j.out
#SBATCH --error=coconut_experiment%j.err
#SBATCH --mail-user=abdullah_ateyeh@berkeley.edu
#SBATCH --mail-type=END,FAIL
# 
# Run from: tabular_q_learning/
#   sbatch scripts/run_experiment.sh

set -euo pipefail

export WANDB_API_KEY=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

# Parse optional overrides
N_SEQUENCES=50000
EPOCHS=28
BATCH_SIZE=64
RUN_NAME="coconut-v3-$(date +%Y%m%d-%H%M)"
NO_COCONUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_sequences) N_SEQUENCES="$2"; shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
    --run_name)    RUN_NAME="$2";    shift 2 ;;
    --no_coconut)  NO_COCONUT="--no_coconut"; shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p data checkpoints figures

echo "============================================================"
echo " COCONUT Q-Learning Transformer — Full Experiment Pipeline"
echo "============================================================"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo " SLURM Job:  $SLURM_JOB_ID"
  echo " Node:       ${SLURMD_NODENAME:-local}"
fi
python3 -c "
import torch
if torch.cuda.is_available():
    print(f' GPU:         {torch.cuda.get_device_name(0)}')
    print(f' VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print(' GPU:         None (CPU only)')
print(f' PyTorch:     {torch.__version__}')
"
echo " n_sequences : $N_SEQUENCES"
echo " epochs      : $EPOCHS"
echo " batch_size  : $BATCH_SIZE"
echo " no_coconut  : ${NO_COCONUT:-false}"
echo " run_name    : $RUN_NAME"
echo "============================================================"
echo ""

# # ---- Step 1: Data Generation ----
# echo "=== Step 1/4: Generating training data ==="
# echo "Start: $(date)"
# python3 scripts/1_generate_data.py \
#   --n_sequences "$N_SEQUENCES" \
#   --n_states 4 \
#   --n_actions 2 \
#   --min_steps 10 \
#   --max_steps 50 \
#   --alpha 0.1 \
#   --gamma 0.9 \
#   --seed 42 \
#   --output data/coconut_dataset.pt
# echo "Done: $(date)"
# echo ""

# # ---- Step 2: Model Summary ----
# echo "=== Step 2/4: Model architecture summary ==="
# python3 scripts/2_model.py
# echo ""

# # ---- Step 3: Training ----
# echo "=== Step 3/4: Training ==="
# echo "Start: $(date)"
# python3 scripts/3_train.py \
#   --data_path data/coconut_dataset.pt \
#   --checkpoint_dir checkpoints \
#   --n_layers 4 \
#   --n_heads 8 \
#   --d_model 256 \
#   --d_ff 1024 \
#   --dropout 0.1 \
#   --max_seq_len 1024 \
#   --epochs "$EPOCHS" \
#   --batch_size "$BATCH_SIZE" \
#   --lr 1e-4 \
#   --weight_decay 1e-2 \
#   --eval_every 500 \
#   --run_name "$RUN_NAME" \
#   --use_wandb \
#   ${NO_COCONUT}
# echo "Done: $(date)"
# echo ""

# ---- Step 4: Evaluation ----
echo "=== Step 4/4: Evaluation ==="
echo "Start: $(date)"
python3 scripts/4_evaluate.py \
  --checkpoint checkpoints/coconut_transformer_hao.pt \
  --figures_dir figures \
  --n_steps 50 \
  --alpha 0.1 \
  --gamma 0.9 \
  --epsilon 0.2 \
  --eval_seed 9999 \
  --n_eval_mdps 10 \
  --n_probe_train 1000 \
  --n_probe_eval 100 \
  --probe_epochs 10 \
  ${NO_COCONUT}
echo "Done: $(date)"
echo ""

# echo "============================================================"
# echo " Pipeline complete."
# echo " Checkpoint : checkpoints/coconut_transformer.pt"
# echo " Figures    : figures/action_agreement.png"
# echo "              figures/probe_scatter.png"
# echo "              figures/probe_frobenius.png"
# echo "              figures/training_curves.png"
# echo "              figures/ablation_accuracy.png"
# echo "============================================================"
