#!/usr/bin/env bash
#
# SLURM batch script for one unified run config. Delegates to scripts/run.sh,
# which starts vLLM, runs the config and stops vLLM.
#
# Do not invoke directly — use sbatch_runs.sh, which sets the resources and env
# vars from the config. Or submit manually:
#
#   sbatch --export=ALL,CONFIG=configs/deep_reasoner/qwen3_32b/oolong.yaml \
#     --gres=gpu:1 --time=16:00:00 --mem=200G --cpus-per-task=8 \
#     scripts/slurm_run.sh
#
# Required env vars:
#   CONFIG           — path to the run config
#
# Optional env vars:
#   TENSOR_PARALLEL  — vLLM tensor-parallel-size (default: from the config)
#   LIMIT            — run only the first N tasks
#   MAX_WORKERS      — override the config's max_workers
#   TOKEN_BUDGET     — override token_budget (smoke tests)
#   SAMPLES          — override samples (smoke tests)
#   VLLM_WAIT        — seconds to wait for vLLM /health (default: 600)
#
#SBATCH --output=logs/slurm/%j.log
#SBATCH --error=logs/slurm/%j.log
#SBATCH --job-name=dolores_run
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=16:00:00

set -euo pipefail

: "${CONFIG:?CONFIG env var is required}"

cd "$SLURM_SUBMIT_DIR"

ARGS=(--config "$CONFIG")
[[ -n "${TENSOR_PARALLEL:-}" ]] && ARGS+=(--tensor-parallel "$TENSOR_PARALLEL")
[[ -n "${LIMIT:-}"           ]] && ARGS+=(--limit "$LIMIT")
[[ -n "${MAX_WORKERS:-}"     ]] && ARGS+=(--max-workers "$MAX_WORKERS")
[[ -n "${TOKEN_BUDGET:-}"    ]] && ARGS+=(--token-budget "$TOKEN_BUDGET")
[[ -n "${SAMPLES:-}"         ]] && ARGS+=(--samples "$SAMPLES")
[[ -n "${VLLM_WAIT:-}"       ]] && ARGS+=(--vllm-wait "$VLLM_WAIT")

echo "[slurm] job=${SLURM_JOB_ID} started at $(date)"
echo "[slurm] config=${CONFIG}"
echo "──────────────────────────────────────────────────"

bash scripts/run.sh "${ARGS[@]}"

echo "──────────────────────────────────────────────────"
echo "[slurm] job=${SLURM_JOB_ID} finished at $(date)"
