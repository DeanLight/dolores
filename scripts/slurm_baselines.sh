#!/usr/bin/env bash
#
# Slurm batch script for smolagents baselines (react, codeact, rlm, deepresearch).
# Delegates to run_baseline.sh which handles vllm start/wait/stop.
#
# Do not invoke directly — use sbatch_baselines.sh which sets all env vars.
# Or submit manually:
#
#   sbatch \
#     --export=ALL,BASELINE=react,BENCHMARK=phantomwiki,MODEL=Qwen/Qwen3-32B \
#     --gres=gpu:2 --time=08:00:00 --mem=200G --cpus-per-task=16 \
#     scripts/slurm_baselines.sh
#
# Required env vars:
#   BASELINE    — react | codeact | rlm | deepresearch
#   BENCHMARK   — phantomwiki | synthworlds | deepresearchqa | oolong
#   MODEL       — HuggingFace model ID, e.g. Qwen/Qwen3-32B
#
# Optional env vars (with defaults):
#   MAX_WORKERS    — parallel subprocess workers  (default: 32 for rlm, 8 for others)
#   GPU_MEM        — gpu_memory_utilization        (default: 0.9)
#   TENSOR_PARALLEL — tensor-parallel-size         (default: auto from model name)
#   VLLM_WAIT      — seconds to wait for /health   (default: 600)
#   PW_SIZE        — PhantomWiki corpus size        (default: 500)
#   PW_SEED        — PhantomWiki seed               (default: 1)
#   LIMIT          — oolong/deepresearchqa item cap (default: unset = all)
#   SEED           — oolong/deepresearchqa seed     (default: 42)
#   NO_THINKING    — set to "1" to pass --no-thinking
#
#SBATCH --output=logs/slurm/%j.log
#SBATCH --error=logs/slurm/%j.log
#SBATCH --job-name=baseline_eval
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00

set -euo pipefail

: "${BASELINE:?BASELINE env var is required}"
: "${BENCHMARK:?BENCHMARK env var is required}"
: "${MODEL:?MODEL env var is required}"

# ── environment ────────────────────────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/13.0.0

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ── build run_baseline.sh args ────────────────────────────────────────────────
cd "$SLURM_SUBMIT_DIR"

ARGS=(
    --baseline "$BASELINE"
    --benchmark "$BENCHMARK"
    --model "$MODEL"
)

[[ -n "${MAX_WORKERS:-}"     ]] && ARGS+=(--max-workers     "$MAX_WORKERS")
[[ -n "${GPU_MEM:-}"         ]] && ARGS+=(--gpu-mem         "$GPU_MEM")
[[ -n "${TENSOR_PARALLEL:-}" ]] && ARGS+=(--tensor-parallel "$TENSOR_PARALLEL")
[[ -n "${VLLM_WAIT:-}"       ]] && ARGS+=(--vllm-wait       "$VLLM_WAIT")
[[ -n "${PW_SIZE:-}"         ]] && ARGS+=(--pw-size         "$PW_SIZE")
[[ -n "${PW_SEED:-}"         ]] && ARGS+=(--pw-seed         "$PW_SEED")
[[ -n "${SEED:-}"            ]] && ARGS+=(--seed            "$SEED")
[[ -n "${LIMIT:-}"           ]] && ARGS+=(--limit           "$LIMIT")
[[ "${NO_THINKING:-0}" == "1" ]] && ARGS+=(--no-thinking)

echo "[slurm] job=${SLURM_JOB_ID} started at $(date)"
echo "[slurm] baseline=${BASELINE} benchmark=${BENCHMARK} model=${MODEL}"
echo "──────────────────────────────────────────────────"

bash scripts/run_baseline.sh "${ARGS[@]}"

echo "──────────────────────────────────────────────────"
echo "[slurm] job=${SLURM_JOB_ID} finished at $(date)"
