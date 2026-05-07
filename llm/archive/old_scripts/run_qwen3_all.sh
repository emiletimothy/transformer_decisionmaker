#!/bin/bash
cd /home/jupyter-xcao/mwu

DATASET="mw_dataset_stratified.json"
IDX="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
PROMPTS="interactive_online interactive_weather interactive_mw_name interactive_explicit_update"

echo "[$(date)] === Starting nothink (remaining) ==="
python interactive_api_run.py cases --idx $IDX \
    --model qwen3_14b_nothink \
    --prompt $PROMPTS \
    --dataset $DATASET --steps 100 2>&1

NOTHINK_EXIT=$?
echo "[$(date)] === nothink finished (exit=$NOTHINK_EXIT) ==="

echo ""
echo "[$(date)] === Starting think1024 ==="
python interactive_api_run.py cases --idx $IDX \
    --model qwen3_14b_think1024 \
    --prompt $PROMPTS \
    --dataset $DATASET --steps 100 2>&1

THINK_EXIT=$?
echo "[$(date)] === think1024 finished (exit=$THINK_EXIT) ==="

echo ""
echo "[$(date)] === All done! nothink=$NOTHINK_EXIT think=$THINK_EXIT ==="
