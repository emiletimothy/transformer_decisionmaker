#!/usr/bin/env bash
#SBATCH --job-name=mwu_eval_mechanism
#SBATCH --partition=yss,jsteinhardt
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#
# Mechanism tests (state probes, retention rho, behavioural fit, causal steering) of the final
# continuous-residual and discrete models -> figures/comparison/mechanism/mech_final.json.
# Other models:  sbatch scripts/jobs/eval_mechanism.sh --ckpts a.pt b.pt --labels a b --out x.json
set -euo pipefail
[[ -f paths.py ]] || { echo "submit from multiplicative_weights/:  cd multiplicative_weights && sbatch scripts/jobs/$(basename "$0")"; exit 1; }
export PATH=/system/linux/miniforge-3.13/bin:$PATH
python3 -u scripts/eval_mechanism.py "$@"
