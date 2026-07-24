#!/usr/bin/env bash
#SBATCH --job-name=coconut_ql_discrete
#SBATCH --partition=yss,jsteinhardt
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=coconut_discrete_experiment%j.out
#SBATCH --error=coconut_discrete_experiment%j.err
#
# Discrete-token context experiment.
#
# Trains a NEW model identical to the continuous (latent) model in every way —
# same dataset/curriculum (data/coconut_dataset.pt), same architecture, same
# hyperparameters, same per-step token count — except the recurrent context is
# forced to be a genuine vocabulary token each step (context_mode=discrete).
# Then evaluates BOTH checkpoints and overlays the metrics.
#
# Run from: tabular_q_learning/
#   sbatch scripts/run_discrete_experiment.sh
#   # or override the continuous baseline:
#   sbatch scripts/run_discrete_experiment.sh --continuous_ckpt <path.pt>

set -euo pipefail

export WANDB_API_KEY=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

# ---- Defaults (match the original continuous run in run_experiment.sh) ----
EPOCHS=40
BATCH_SIZE=64
DATA_PATH="data/coconut_dataset.pt"
CONTINUOUS_CKPT="checkpoints/coconut_transformer_coconut-v4-mixed-20260502-1646.pt"
RUN_NAME="discrete-v1-$(date +%Y%m%d-%H%M)"
GUMBEL_TAU_START=2.0
GUMBEL_TAU_END=0.5
SKIP_TRAIN=0
SKIP_EVAL=0
DISCRETE_CKPT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)           EPOCHS="$2";            shift 2 ;;
    --batch_size)       BATCH_SIZE="$2";        shift 2 ;;
    --data_path)        DATA_PATH="$2";         shift 2 ;;
    --continuous_ckpt)  CONTINUOUS_CKPT="$2";   shift 2 ;;
    --run_name)         RUN_NAME="$2";          shift 2 ;;
    --gumbel_tau_start) GUMBEL_TAU_START="$2";  shift 2 ;;
    --gumbel_tau_end)   GUMBEL_TAU_END="$2";    shift 2 ;;
    --discrete_ckpt)    DISCRETE_CKPT_OVERRIDE="$2"; shift 2 ;;
    --skip_train)       SKIP_TRAIN=1;           shift 1 ;;
    --skip_eval)        SKIP_EVAL=1;            shift 1 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -n "$DISCRETE_CKPT_OVERRIDE" ]]; then
  DISCRETE_CKPT="$DISCRETE_CKPT_OVERRIDE"
else
  DISCRETE_CKPT="checkpoints/coconut_transformer_${RUN_NAME}.pt"
fi

mkdir -p checkpoints figures figures/continuous figures/discrete figures/comparison

echo "============================================================"
echo " Discrete vs Continuous Context — Q-Learning Transformer"
echo "============================================================"
python3 -c "
import torch
print(f' GPU:      {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else ' GPU:      None (CPU)')
print(f' PyTorch:  {torch.__version__}')
"
echo " data_path        : $DATA_PATH  (reused — identical data/curriculum)"
echo " continuous_ckpt  : $CONTINUOUS_CKPT"
echo " discrete run_name: $RUN_NAME"
echo " discrete_ckpt    : $DISCRETE_CKPT"
echo " epochs/batch     : $EPOCHS / $BATCH_SIZE"
echo " gumbel_tau       : $GUMBEL_TAU_START -> $GUMBEL_TAU_END"
echo "============================================================"
echo ""

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: dataset not found at $DATA_PATH."
  echo "  Generate it once with scripts/1_generate_data.py (seed 42) or run run_experiment.sh."
  exit 1
fi

# Preserve the continuous run's training log before discrete training overwrites
# checkpoints/training_log.npz (3_train.py writes a fixed filename).
CONT_LOG="checkpoints/training_log_continuous.npz"
DISC_LOG="checkpoints/training_log_discrete.npz"
if [[ -f "checkpoints/training_log.npz" && ! -f "$CONT_LOG" ]]; then
  cp "checkpoints/training_log.npz" "$CONT_LOG"
  echo "Backed up existing training log -> $CONT_LOG"
fi

# ---- Step 1: Train the discrete-context model ----
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "=== Step 1/3: Training discrete-context model ==="
  echo "Start: $(date)"
  python3 scripts/3_train.py \
    --data_path "$DATA_PATH" \
    --checkpoint_dir checkpoints \
    --n_layers 4 \
    --n_heads 8 \
    --d_model 256 \
    --d_ff 1024 \
    --dropout 0.1 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --eval_every 500 \
    --run_name "$RUN_NAME" \
    --context_mode discrete \
    --gumbel_tau_start "$GUMBEL_TAU_START" \
    --gumbel_tau_end "$GUMBEL_TAU_END" \
    --use_wandb
  echo "Done: $(date)"
  # Snapshot the discrete run's training log.
  if [[ -f "checkpoints/training_log.npz" ]]; then
    cp "checkpoints/training_log.npz" "$DISC_LOG"
  fi
  echo ""
else
  echo "=== Step 1/3: SKIPPED (--skip_train) ==="
fi

# ---- Step 2: Evaluate both checkpoints (same metrics, per-model dirs) ----
if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "=== Step 2/3: Per-model evaluation (all metrics, labeled) ==="
  for pair in "continuous:$CONTINUOUS_CKPT" "discrete:$DISCRETE_CKPT"; do
    label="${pair%%:*}"; ckpt="${pair#*:}"
    if [[ ! -f "$ckpt" ]]; then
      echo "  WARNING: $label checkpoint not found at $ckpt — skipping."
      continue
    fi
    echo "  -> evaluating $label ($ckpt)"
    python3 scripts/4_evaluate.py \
      --checkpoint "$ckpt" \
      --figures_dir "figures/$label" \
      --label "$label" \
      --n_steps 50 \
      --alpha 0.1 --gamma 0.9 --epsilon 0.2 \
      --eval_seed 9999 --n_eval_mdps 10 \
      --n_probe_train 1000 --n_probe_eval 100 --probe_epochs 10
  done
  echo ""

  # ---- Step 3: Overlaid comparison figures ----
  echo "=== Step 3/3: Overlaid continuous-vs-discrete comparison ==="
  python3 scripts/5_compare_context_modes.py \
    --continuous_ckpt "$CONTINUOUS_CKPT" \
    --discrete_ckpt "$DISCRETE_CKPT" \
    --figures_dir figures/comparison \
    --continuous_log "$CONT_LOG" \
    --discrete_log "$DISC_LOG" \
    --n_steps 50 --long_horizon_steps 200 \
    --alpha 0.1 --gamma 0.9 --epsilon 0.2 \
    --eval_seed 9999 --n_eval_mdps 10 \
    --n_probe_train 500 --n_probe_eval 100 --probe_epochs 10
else
  echo "=== Steps 2-3: SKIPPED (--skip_eval) ==="
fi

echo "============================================================"
echo " Complete."
echo "   continuous figures : figures/continuous/"
echo "   discrete   figures : figures/discrete/"
echo "   comparison figures : figures/comparison/"
echo "============================================================"
