#!/bin/bash
cd /home/jupyter-xcao/mwu

PROMPTS="weather_v2 weather_nonote_v2 online_v2 online_nonote_v2"
DATASET="mw_dataset_antisignal.json"
STEPS=100
MODEL=ds
N_CASES=20

echo "[$(date)] === Anti-signal dataset experiment ==="
echo "Prompts: $PROMPTS"
echo "Dataset: $DATASET, Steps: $STEPS, Cases: 0-$((N_CASES-1))"

for prompt in $PROMPTS; do
  echo "[$(date)] --- Running prompt: $prompt ---"
  for idx in $(seq 0 $((N_CASES-1))); do
    python3 interactive_api_run.py cases --idx $idx --model $MODEL --prompt "interactive_${prompt}" --dataset "$DATASET" --steps $STEPS
  done
  echo "[$(date)] --- Done: $prompt ---"
done

echo "[$(date)] === All done ==="
