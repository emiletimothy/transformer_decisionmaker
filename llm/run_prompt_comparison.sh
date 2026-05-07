#!/bin/bash
cd /home/jupyter-xcao/mwu

DATASET="mw_dataset.json"
IDX_10="0 1 2 3 4 5 6 7 8 9"
IDX_20="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

echo "[$(date)] === Phase 1: new prompts, first 10 cases ==="

echo "[$(date)] Running weather_no_morereliable..."
python interactive_api_run.py cases --idx $IDX_10 \
    --model ds --prompt interactive_weather_no_morereliable \
    --dataset $DATASET --steps 100 2>&1

echo "[$(date)] Running online_acc..."
python interactive_api_run.py cases --idx $IDX_10 \
    --model ds --prompt interactive_online_acc \
    --dataset $DATASET --steps 100 2>&1

echo "[$(date)] === Phase 1 done ==="

# Check elapsed time - if under 4 hours, run idx 10-19 for all 4 prompts
echo "[$(date)] === Phase 2: all 4 prompts, cases 10-19 ==="

python interactive_api_run.py cases --idx $IDX_20 \
    --model ds \
    --prompt interactive_weather interactive_weather_no_morereliable interactive_online interactive_online_acc \
    --dataset $DATASET --steps 100 2>&1

echo "[$(date)] === All done ==="
