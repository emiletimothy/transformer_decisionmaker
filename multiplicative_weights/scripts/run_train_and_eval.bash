#!/usr/bin/env bash
cd "$(dirname "$0")"

nohup python3 -u gradient_train_transformer_for_mwu.py \
  --cot_mode continuous --n_thought_steps 4 --max_stages 10 \
  --max_steps 50 --n_train 3000 --n_val 500 --save_dir ../figures \
  --device mps \
  > ../figures/training_output.log 2>&1 &

echo "Training started (PID: $!)"
echo "Log: $(cd .. && pwd)/figures/training_output.log"
echo "Monitor with: tail -f ../figures/training_output.log"