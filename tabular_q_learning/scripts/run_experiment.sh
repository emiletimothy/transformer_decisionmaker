#!/usr/bin/env bash
#SBATCH --job-name=coconut_ql
#SBATCH --partition=yss
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=coconut_experiment.out
#SBATCH --error=coconut_experiment.err
#
# Run from: tabular_q_learning/
#   sbatch scripts/run_experiment.sh

set -euo pipefail

export WANDB_API_KEY=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

# Parse optional overrides
N_SEQUENCES=50000
EPOCHS=20
BATCH_SIZE=64

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n_sequences) N_SEQUENCES="$2"; shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
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
echo "============================================================"
echo ""

# ---- Step 1: Data Generation ----
echo "=== Step 1/4: Generating training data ==="
echo "Start: $(date)"
python3 scripts/1_generate_data.py \
  --n_sequences "$N_SEQUENCES" \
  --n_states 5 \
  --n_actions 2 \
  --min_steps 10 \
  --max_steps 50 \
  --alpha 0.1 \
  --gamma 0.9 \
  --epsilon 0.2 \
  --trap_prob 0.2 \
  --random_walk_frac 0.3 \
  --seed 42 \
  --output data/coconut_dataset.pt
echo "Done: $(date)"
echo ""

# ---- Step 2: Model Summary ----
echo "=== Step 2/4: Model architecture summary ==="
python3 scripts/2_model.py
echo ""

# ---- Step 3: Training ----
echo "=== Step 3/4: Training ==="
echo "Start: $(date)"
python3 scripts/3_train.py \
  --data_path data/coconut_dataset.pt \
  --checkpoint_dir checkpoints \
  --n_layers 4 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 512 \
  --dropout 0.1 \
  --max_seq_len 1024 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --warmup_steps 500 \
  --eval_every 500 \
  --use_wandb
echo "Done: $(date)"
echo ""

# ---- Step 4: Evaluation ----
echo "=== Step 4/4: Evaluation ==="
echo "Start: $(date)"
python3 scripts/4_evaluate.py \
  --checkpoint checkpoints/coconut_transformer.pt \
  --figures_dir figures \
  --n_steps 50 \
  --alpha 0.1 \
  --gamma 0.9 \
  --epsilon 0.2 \
  --eval_seed 9999
echo "Done: $(date)"
echo ""

echo "============================================================"
echo " Pipeline complete."
echo " Checkpoint : checkpoints/coconut_transformer.pt"
echo " Figures    : figures/frobenius_norm.png"
echo "              figures/qtable_comparison.png"
echo "============================================================"
