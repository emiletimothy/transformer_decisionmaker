#!/bin/bash
cd /home/jupyter-xcao/mwu

PROMPTS="interactive_weather_v2 interactive_weather_nonote_v2 interactive_online_v2 interactive_online_nonote_v2"
MODEL=qwen3_14b_nothink
STEPS=100

run_dataset() {
  local DS_FILE=$1
  local DS_NAME=$2
  local DEST="archive/qwen_${DS_NAME}_30cases"

  echo "[$(date)] === Starting $DS_NAME ==="

  for p in $PROMPTS; do
    echo "[$(date)] --- $p ---"
    for idx in $(seq 0 29); do
      python3 interactive_api_run.py cases --idx $idx --model $MODEL --prompt $p --dataset "$DS_FILE" --steps $STEPS
    done
  done

  echo "[$(date)] === Moving $DS_NAME results ==="
  mkdir -p "$DEST"
  for p in weather_v2 weather_nonote_v2 online_v2 online_nonote_v2; do
    if [ -d "exp_results/$p/$MODEL" ]; then
      mkdir -p "$DEST/$p"
      mv "exp_results/$p/$MODEL" "$DEST/$p/$MODEL"
    fi
  done
  echo "[$(date)] === Done $DS_NAME ==="
}

run_dataset "mw_dataset.json" "stratified"
run_dataset "mw_dataset_flat.json" "flat"
run_dataset "mw_dataset_antisignal.json" "antisignal"

echo "[$(date)] === All done ==="
