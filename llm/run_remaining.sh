#!/bin/bash
cd /home/jupyter-xcao/mwu

PROMPTS="weather_v2 weather_nonote_v2 online_v2 online_nonote_v2"
STEPS=100
MODEL=ds

echo "[$(date)] === Running cases 20-29 for all three datasets ==="

# --- Stratified (cases 20-29) ---
DATASET="mw_dataset.json"
echo "[$(date)] === Dataset: stratified ==="
for prompt in $PROMPTS; do
  echo "[$(date)] --- $prompt ---"
  for idx in $(seq 20 29); do
    python3 interactive_api_run.py cases --idx $idx --model $MODEL --prompt "interactive_${prompt}" --dataset "$DATASET" --steps $STEPS
  done
done

# Move stratified results out before flat runs
echo "[$(date)] === Moving stratified results ==="
mkdir -p /tmp/antisignal_staging/stratified
for p in $PROMPTS; do
  if [ -d "exp_results/$p" ]; then
    mv "exp_results/$p" "/tmp/antisignal_staging/stratified/"
  fi
done

# --- Flat (cases 20-29) ---
DATASET="mw_dataset_flat.json"
echo "[$(date)] === Dataset: flat ==="
for prompt in $PROMPTS; do
  echo "[$(date)] --- $prompt ---"
  for idx in $(seq 20 29); do
    python3 interactive_api_run.py cases --idx $idx --model $MODEL --prompt "interactive_${prompt}" --dataset "$DATASET" --steps $STEPS
  done
done

# Move flat results out before antisignal runs
echo "[$(date)] === Moving flat results ==="
mkdir -p /tmp/antisignal_staging/flat
for p in $PROMPTS; do
  if [ -d "exp_results/$p" ]; then
    mv "exp_results/$p" "/tmp/antisignal_staging/flat/"
  fi
done

# --- Antisignal (cases 20-29) ---
DATASET="mw_dataset_antisignal.json"
echo "[$(date)] === Dataset: antisignal ==="
for prompt in $PROMPTS; do
  echo "[$(date)] --- $prompt ---"
  for idx in $(seq 20 29); do
    python3 interactive_api_run.py cases --idx $idx --model $MODEL --prompt "interactive_${prompt}" --dataset "$DATASET" --steps $STEPS
  done
done

# Move antisignal 20-29 results
echo "[$(date)] === Moving antisignal 20-29 results ==="
mkdir -p /tmp/antisignal_staging/antisignal_2029
for p in $PROMPTS; do
  if [ -d "exp_results/$p" ]; then
    mv "exp_results/$p" "/tmp/antisignal_staging/antisignal_2029/"
  fi
done

echo "[$(date)] === All 120 runs done ==="
