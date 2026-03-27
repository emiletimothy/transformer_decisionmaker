#!/usr/bin/env bash
# =============================================================================
# AWS GPU Training Script for MW Transformer
#
# Usage:
#   1. Launch a g4dn.2xlarge (T4) or g5.xlarge (A10G) with Deep Learning AMI
#   2. SSH in and clone the repo:
#        git clone https://github.com/emiletimothy/transformer_decisionmaker.git
#        cd transformer_decisionmaker
#   3. Run this script:
#        bash multiplicative_weights/scripts/run_aws_train.sh
#
# The Deep Learning AMI comes with PyTorch + CUDA pre-installed.
# If using a vanilla Ubuntu AMI, uncomment the pip install section below.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/multiplicative_weights/scripts"
FIGURES_DIR="$REPO_ROOT/multiplicative_weights/figures"

mkdir -p "$FIGURES_DIR"

# --- Install dependencies ---
pip install -r "$REPO_ROOT/requirements.txt"

# --- Verify CUDA is available ---
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# --- wandb (authenticate via WANDB_API_KEY env var or `wandb login` beforehand) ---
WANDB_ARGS="--wandb_entity emiletimothyanand --wandb_project transformers_online_decision_makers --wandb_run_name cuda-$(date +%Y%m%d-%H%M%S)"

echo "=================================================="
echo "Starting MW Transformer training on CUDA"
echo "=================================================="

cd "$SCRIPT_DIR"

nohup python3 -u gradient_train_transformer_for_mwu.py \
  --cot_mode continuous --n_thought_steps 4 --max_stages 10 \
  --max_steps 50 --n_train 3000 --n_val 500 --save_dir ../figures \
  --device cuda \
  $WANDB_ARGS \
  > "$FIGURES_DIR/training_output.log" 2>&1 &

PID=$!
echo "Training started (PID: $PID)"
echo "Log: $FIGURES_DIR/training_output.log"
echo "Monitor with: tail -f $FIGURES_DIR/training_output.log"
echo ""
echo "To check if still running: ps -p $PID"
echo "To stop: kill $PID"
