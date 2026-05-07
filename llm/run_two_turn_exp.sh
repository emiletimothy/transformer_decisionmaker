#!/bin/bash
cd /home/jupyter-xcao/mwu

IDX="0 1 2 3 4 5 6 7 8 9"
PROMPTS="interactive_online interactive_weather"
DATASET="mw_dataset.json"

echo "[$(date)] === nothink: online + weather ==="
python interactive_api_run.py cases --idx $IDX \
    --model qwen3_14b_nothink \
    --prompt $PROMPTS \
    --dataset $DATASET --steps 100 2>&1
echo "[$(date)] === nothink done (exit=$?) ==="

echo ""
echo "[$(date)] === think1024: online + weather ==="
python interactive_api_run.py cases --idx $IDX \
    --model qwen3_14b_think1024 \
    --prompt $PROMPTS \
    --dataset $DATASET --steps 100 2>&1
echo "[$(date)] === think1024 done (exit=$?) ==="

echo ""
echo "[$(date)] === All done ==="
