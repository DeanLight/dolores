#!/usr/bin/env bash
#
# Slurm batch script for deep-reasoner-eval-experiment.
#
# Usage:
#   sbatch scripts/slurm_eval.sh <main_config.yaml>
#
# Logs are written to:
#   logs/slurm/<job-id>.log
#
# Check status:
#   squeue --name eval_experiment -u $USER
#   sacct  --name eval_experiment -u $USER --format=JobID,JobName,State,Start,End
#
#SBATCH --output=logs/slurm/%j.log
#SBATCH --error=logs/slurm/%j.log
#SBATCH --job-name=eval_experiment
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00

set -euo pipefail

# ── parse args ────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch $0 <main_config.yaml>" >&2
    exit 1
fi

CONFIG="$1"

# ── environment ───────────────────────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/13.0.0

# Pass HuggingFace token so vllm can download gated/rate-limited models
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ── run ───────────────────────────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

echo "[slurm] job=${SLURM_JOB_ID} started at $(date)"
echo "[slurm] config=${CONFIG}"
echo "──────────────────────────────────────────────────"

python -m dolores.experiment "$CONFIG"

echo "──────────────────────────────────────────────────"
echo "[slurm] job=${SLURM_JOB_ID} finished at $(date)"
