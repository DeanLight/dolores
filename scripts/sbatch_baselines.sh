#!/usr/bin/env bash
#
# Submit one sbatch job per baseline × benchmark combo.
#
# Usage:
#   bash scripts/sbatch_baselines.sh --model Qwen/Qwen3-32B [OPTIONS]
#
# Required:
#   --model MODEL          HuggingFace model ID (e.g. Qwen/Qwen3-32B)
#
# Filters (default: all combos):
#   --baseline BASELINE    Only submit jobs for this baseline (react|codeact|rlm|deepresearch)
#   --benchmark BENCHMARK  Only submit jobs for this benchmark (phantomwiki|synthworlds|deepresearchqa|oolong)
#
# Resource overrides (applied to all submitted jobs):
#   --gpus N               GPU count      (default: 2)
#   --time HH:MM:SS        Wall time      (default: 08:00:00)
#   --mem MEM              Memory         (default: 200G)
#   --cpus N               CPUs per task  (default: 16)
#   --qos QOS              SLURM QOS      (default: normal)
#
# Baseline-specific overrides (passed as env vars to slurm_baselines.sh):
#   --max-workers N        Parallel workers       (default: 32)
#   --vllm-gpu-mem F       GPU memory fraction    (default: 0.9)
#   --vllm-wait N          vllm health timeout s  (default: 600)
#   --pw-size N            PhantomWiki corpus size (default: 500)
#   --pw-seed N            PhantomWiki seed        (default: 1)
#   --limit N              oolong/deepresearchqa cap (default: unset)
#   --seed N               oolong/deepresearchqa seed (default: 42)
#   --no-thinking          Pass --no-thinking to the baseline
#
# Flags:
#   --dry-run              Print what would be submitted without submitting
#
# Examples:
#   # All combos for one model:
#   bash scripts/sbatch_baselines.sh --model Qwen/Qwen3-32B
#
#   # Only react × phantomwiki:
#   bash scripts/sbatch_baselines.sh --model Qwen/Qwen3-32B --baseline react --benchmark phantomwiki
#
#   # All oolong combos, dry-run:
#   bash scripts/sbatch_baselines.sh --model Qwen/Qwen3-32B --benchmark oolong --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── all valid baseline × benchmark combos ─────────────────────────────────────
# Format: "baseline:benchmark"
ALL_COMBOS=(
    "react:phantomwiki"
    "react:synthworlds"
    "react:deepresearchqa"
    "codeact:oolong"
    "codeact:phantomwiki"
    "codeact:synthworlds"
    "codeact:deepresearchqa"
    "rlm:oolong"
    "rlm:phantomwiki"
    "rlm:synthworlds"
    "rlm:deepresearchqa"
    "deepresearch:phantomwiki"
    "deepresearch:synthworlds"
    "deepresearch:deepresearchqa"
)

# ── parse args ────────────────────────────────────────────────────────────────
MODEL=""
FILTER_BASELINE=""
FILTER_BENCHMARK=""
DRY_RUN=false

SLURM_GPUS=2
SLURM_TIME="08:00:00"
SLURM_MEM="200G"
SLURM_CPUS=16
SLURM_QOS="normal"

MAX_WORKERS=32
VLLM_GPU_MEM=0.9
VLLM_WAIT=600
PW_SIZE=500
PW_SEED=1
LIMIT=""
SEED=42
NO_THINKING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";             shift 2 ;;
        --baseline)      FILTER_BASELINE="$2";   shift 2 ;;
        --benchmark)     FILTER_BENCHMARK="$2";  shift 2 ;;
        --gpus)          SLURM_GPUS="$2";        shift 2 ;;
        --time)          SLURM_TIME="$2";        shift 2 ;;
        --mem)           SLURM_MEM="$2";         shift 2 ;;
        --cpus)          SLURM_CPUS="$2";        shift 2 ;;
        --qos)           SLURM_QOS="$2";         shift 2 ;;
        --max-workers)   MAX_WORKERS="$2";       shift 2 ;;
        --vllm-gpu-mem)  VLLM_GPU_MEM="$2";      shift 2 ;;
        --vllm-wait)     VLLM_WAIT="$2";         shift 2 ;;
        --pw-size)       PW_SIZE="$2";           shift 2 ;;
        --pw-seed)       PW_SEED="$2";           shift 2 ;;
        --limit)         LIMIT="$2";             shift 2 ;;
        --seed)          SEED="$2";              shift 2 ;;
        --no-thinking)   NO_THINKING=1;          shift ;;
        --dry-run)       DRY_RUN=true;           shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required" >&2
    exit 1
fi

# ── filter combos ─────────────────────────────────────────────────────────────
COMBOS=()
for combo in "${ALL_COMBOS[@]}"; do
    baseline="${combo%%:*}"
    benchmark="${combo##*:}"
    [[ -n "$FILTER_BASELINE"  && "$baseline"  != "$FILTER_BASELINE"  ]] && continue
    [[ -n "$FILTER_BENCHMARK" && "$benchmark" != "$FILTER_BENCHMARK" ]] && continue
    COMBOS+=("$combo")
done

if [[ ${#COMBOS[@]} -eq 0 ]]; then
    echo "No combos match the given filters." >&2
    exit 1
fi

echo "[sbatch_baselines] model=${MODEL}"
echo "[sbatch_baselines] ${#COMBOS[@]} combo(s) to submit:"
for combo in "${COMBOS[@]}"; do
    echo "  ${combo}"
done

[[ "$DRY_RUN" == true ]] && echo "[sbatch_baselines] DRY RUN — no jobs submitted." && exit 0

# ── submit one job per combo ──────────────────────────────────────────────────
mkdir -p logs/slurm

for combo in "${COMBOS[@]}"; do
    BASELINE="${combo%%:*}"
    BENCHMARK="${combo##*:}"

    JOB_NAME="${BASELINE}_${BENCHMARK}"

    EXPORT_VARS="ALL"
    EXPORT_VARS+=",BASELINE=${BASELINE}"
    EXPORT_VARS+=",BENCHMARK=${BENCHMARK}"
    EXPORT_VARS+=",MODEL=${MODEL}"
    EXPORT_VARS+=",MAX_WORKERS=${MAX_WORKERS}"
    EXPORT_VARS+=",VLLM_GPU_MEM=${VLLM_GPU_MEM}"
    EXPORT_VARS+=",VLLM_WAIT=${VLLM_WAIT}"
    EXPORT_VARS+=",PW_SIZE=${PW_SIZE}"
    EXPORT_VARS+=",PW_SEED=${PW_SEED}"
    EXPORT_VARS+=",SEED=${SEED}"
    EXPORT_VARS+=",NO_THINKING=${NO_THINKING}"
    [[ -n "$LIMIT" ]] && EXPORT_VARS+=",LIMIT=${LIMIT}"

    echo "[sbatch_baselines] Submitting: ${JOB_NAME} gpus=${SLURM_GPUS} time=${SLURM_TIME} mem=${SLURM_MEM}"

    JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --gres="gpu:${SLURM_GPUS}" \
        --time="${SLURM_TIME}" \
        --mem="${SLURM_MEM}" \
        --cpus-per-task="${SLURM_CPUS}" \
        --qos="${SLURM_QOS}" \
        --export="${EXPORT_VARS}" \
        "${SCRIPT_DIR}/slurm_baselines.sh")

    echo "[sbatch_baselines] Submitted job ${JOB_ID} (${JOB_NAME})"

    printf '{"job_id": "%s", "baseline": "%s", "benchmark": "%s", "model": "%s"}\n' \
        "$JOB_ID" "$BASELINE" "$BENCHMARK" "$MODEL" > "logs/slurm/${JOB_ID}.json"
done

echo "[sbatch_baselines] Done. Submitted ${#COMBOS[@]} job(s)."
