#!/bin/bash
# Run training and evaluation for the Learned Encoder Transformer
#
# Usage:
#   bash run_train_and_eval.bash                    # defaults (continuous CoT)
#   bash run_train_and_eval.bash --cot_mode discrete
#   bash run_train_and_eval.bash --device cuda --wandb_project my_project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  Learned Encoder Transformer — Train & Evaluate"
echo "=================================================="

# Default arguments (can be overridden via CLI)
python train_transformer_for_encoder.py \
    --input_dim 16 \
    --latent_dim 4 \
    --hidden_dim 32 \
    --n_clusters 5 \
    --cluster_std 0.3 \
    --max_steps 10 \
    --n_train 3000 \
    --n_val 500 \
    --max_stages 10 \
    --ae_epochs 50 \
    --cot_mode continuous \
    --n_thought_steps 4 \
    --save_dir ../figures \
    --seed 42 \
    "$@"

echo ""
echo "=================================================="
echo "  Running standalone evaluation"
echo "=================================================="

python eval.py \
    --model_path ../figures/learned_encoder_transformer.pt \
    --n_test 200 \
    --max_steps 10 \
    --save_dir ../figures/eval \
    --seed 123

echo ""
echo "Done! Results in neural_network/figures/"
