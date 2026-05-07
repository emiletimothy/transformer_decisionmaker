#!/usr/bin/env bash
#SBATCH --job-name=compare_ql_transformer
#SBATCH --partition=jsteinhardt,yss
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=%x.out
#SBATCH --error=%x.err

# Define our directories based on where the job was submitted
SCRIPT_DIR="$SLURM_SUBMIT_DIR"
# Resolve the absolute path for the figures directory (../figures)
FIGURES_DIR="$(readlink -m "$SCRIPT_DIR/../figures")"

echo "=================================================="
echo "Job:       $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:      $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "=================================================="

# Print GPU info if available
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU detected — running on CPU')
"

echo ""
echo "Running compare_q_learning_transformer.py"
echo ""

# Move to the directory where the script was submitted from
cd "$SCRIPT_DIR"

python3 compare_q_learning_transformer.py \
  --n_states 4 \
  --n_actions 2 \
  --T 500 \
  --alpha 0.1 \
  --gamma 0.9 \
  --epsilon 0.1 \
  --seed 42 \
  --save_dir "$FIGURES_DIR"

echo ""
echo "=================================================="
echo "Done: $(date)"
echo "Figure saved to: $FIGURES_DIR"
echo "=================================================="