#!/bin/bash
# Wait for nothink to finish, then run think1024
# Usage: bash run_qwen3_sequence.sh

cd /home/jupyter-xcao/mwu

echo "=== Waiting for nothink to finish ==="
# Wait for the nothink process to complete
while ps aux | grep "interactive_api_run.*qwen3_14b_nothink" | grep -v grep > /dev/null 2>&1; do
    sleep 30
done
echo "=== nothink finished ==="

echo ""
echo "=== Starting think1024 ==="
python interactive_api_run.py cases \
    --idx 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --model qwen3_14b_think1024 \
    --prompt interactive_online interactive_weather interactive_mw_name interactive_explicit_update \
    --dataset mw_dataset_stratified.json \
    --steps 100 \
    2>&1

echo ""
echo "=== think1024 finished ==="
echo "=== All done ==="
