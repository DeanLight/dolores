#!/usr/bin/env bash
#
# Run a smolagents baseline with a managed local vllm server.
# Starts vllm with model-appropriate flags, waits for /health, runs the
# baseline against it, then stops vllm on exit (even if the baseline fails).
#
# Usage:
#   bash scripts/run_baseline.sh --baseline react --benchmark phantomwiki --model Qwen/Qwen3-32B
#
# Required:
#   --baseline BASELINE    react | codeact | rlm | deepresearch
#   --benchmark BENCHMARK  phantomwiki | synthworlds | deepresearchqa | oolong
#   --model MODEL          HuggingFace model ID, e.g. Qwen/Qwen3-32B
#
# Optional:
#   --max-workers N        Parallel workers (default: 32 for rlm, 8 for others)
#   --gpu-mem F            gpu_memory_utilization (default: 0.9)
#   --tensor-parallel N    tensor-parallel-size (auto-detected from model if omitted)
#   --vllm-wait N          Seconds to wait for vllm /health (default: 600)
#   --port N               vllm port (default: auto-assigned free port)
#   --api-key KEY          API key passed to both vllm and the baseline (default: your_secret)
#   --pw-size N            PhantomWiki corpus size (default: 500)
#   --pw-seed N            PhantomWiki seed (default: 1)
#   --limit N              oolong/deepresearchqa item cap (default: unset = all)
#   --seed N               oolong/deepresearchqa sampling seed (default: 42)
#   --no-thinking          Pass --no-thinking to the baseline
#   --extra-vllm-args ARGS Extra args appended to vllm serve (quoted string)

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
module load gcc/13.4.0
module load cuda/13.0.0

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ── parse args ────────────────────────────────────────────────────────────────
BASELINE=""
BENCHMARK=""
MODEL=""
MAX_WORKERS=""
GPU_MEM="0.9"
TENSOR_PARALLEL=""
VLLM_WAIT=600
PORT=""
API_KEY="your_secret"
PW_SIZE=500
PW_SEED=1
LIMIT=""
SEED=42
NO_THINKING=false
EXTRA_VLLM_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)         BASELINE="$2";          shift 2 ;;
        --benchmark)        BENCHMARK="$2";         shift 2 ;;
        --model)            MODEL="$2";             shift 2 ;;
        --max-workers)      MAX_WORKERS="$2";       shift 2 ;;
        --gpu-mem)          GPU_MEM="$2";           shift 2 ;;
        --tensor-parallel)  TENSOR_PARALLEL="$2";   shift 2 ;;
        --vllm-wait)        VLLM_WAIT="$2";         shift 2 ;;
        --port)             PORT="$2";              shift 2 ;;
        --api-key)          API_KEY="$2";           shift 2 ;;
        --pw-size)          PW_SIZE="$2";           shift 2 ;;
        --pw-seed)          PW_SEED="$2";           shift 2 ;;
        --limit)            LIMIT="$2";             shift 2 ;;
        --seed)             SEED="$2";              shift 2 ;;
        --no-thinking)      NO_THINKING=true;       shift ;;
        --extra-vllm-args)  EXTRA_VLLM_ARGS="$2";  shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${BASELINE:?--baseline is required}"
: "${BENCHMARK:?--benchmark is required}"
: "${MODEL:?--model is required}"

# ── defaults ──────────────────────────────────────────────────────────────────
if [[ -z "$MAX_WORKERS" ]]; then
    [[ "$BASELINE" == "rlm" ]] && MAX_WORKERS=32 || MAX_WORKERS=8
fi

# Auto-detect tensor-parallel-size from model name if not set.
if [[ -z "$TENSOR_PARALLEL" ]]; then
    case "$MODEL" in
        *70B*|*72B*) TENSOR_PARALLEL=4 ;;
        *)           TENSOR_PARALLEL=2 ;;
    esac
fi

# ── model-specific vllm flags ─────────────────────────────────────────────────
case "$MODEL" in
    *Qwen3*|*qwen3*)
        VLLM_MODEL_FLAGS="--reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice"
        ;;
    *Llama-3.3*|*llama-3.3*)
        VLLM_MODEL_FLAGS="--tool-call-parser llama3_json --enable-auto-tool-choice"
        ;;
    *)
        VLLM_MODEL_FLAGS=""
        ;;
esac

# ── pick a free port ──────────────────────────────────────────────────────────
if [[ -z "$PORT" ]]; then
    PORT=$(python3 -c "
import socket
s = socket.socket()
s.bind(('', 0))
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
print(s.getsockname()[1])
s.close()
")
fi

API_BASE="http://localhost:${PORT}/v1"

# ── start vllm ────────────────────────────────────────────────────────────────
VLLM_CMD=(
    vllm serve "$MODEL"
    --port "$PORT"
    --api-key "$API_KEY"
    --enable-prefix-caching
    --seed 42
    --gpu-memory-utilization "$GPU_MEM"
    --tensor-parallel-size "$TENSOR_PARALLEL"
)
# shellcheck disable=SC2206
[[ -n "$VLLM_MODEL_FLAGS" ]] && VLLM_CMD+=($VLLM_MODEL_FLAGS)
# shellcheck disable=SC2206
[[ -n "$EXTRA_VLLM_ARGS" ]] && VLLM_CMD+=($EXTRA_VLLM_ARGS)

echo "[run_baseline] Starting vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" &
VLLM_PID=$!

# Ensure vllm is stopped on exit regardless of how the script exits.
trap 'echo "[run_baseline] Stopping vllm (pid=${VLLM_PID})"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# ── wait for /health ──────────────────────────────────────────────────────────
echo "[run_baseline] Waiting for vllm at port ${PORT} (timeout ${VLLM_WAIT}s)..."
DEADLINE=$(( $(date +%s) + VLLM_WAIT ))
LAST_PRINT=0
ATTEMPT=0
while true; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[run_baseline] vllm ready (attempt ${ATTEMPT})"
        break
    fi
    NOW=$(date +%s)
    if (( NOW >= DEADLINE )); then
        echo "[run_baseline] vllm did not become healthy within ${VLLM_WAIT}s" >&2
        exit 1
    fi
    if (( NOW - LAST_PRINT >= 30 )); then
        echo "[run_baseline] still waiting... elapsed=$(( NOW - (DEADLINE - VLLM_WAIT) ))s attempt=${ATTEMPT}"
        LAST_PRINT=$NOW
    fi
    sleep 5
done

# ── build baseline command ────────────────────────────────────────────────────
# deepresearch uses LiteLLM format (hosted_vllm/<model>); others use plain model ID.
if [[ "$BASELINE" == "deepresearch" ]]; then
    BASELINE_MODEL="hosted_vllm/${MODEL}"
else
    BASELINE_MODEL="$MODEL"
fi

BASELINE_CMD=(
    python -m "baselines.${BASELINE}"
    --benchmark "$BENCHMARK"
    --model "$BASELINE_MODEL"
    --api_base "$API_BASE"
    --api_key "$API_KEY"
    --max_workers "$MAX_WORKERS"
)

if [[ "$BENCHMARK" == "phantomwiki" ]]; then
    BASELINE_CMD+=(--pw_size "$PW_SIZE" --pw_seed "$PW_SEED")
fi

if [[ "$BENCHMARK" == "oolong" || "$BENCHMARK" == "deepresearchqa" ]]; then
    BASELINE_CMD+=(--seed "$SEED")
    [[ -n "$LIMIT" ]] && BASELINE_CMD+=(--limit "$LIMIT")
fi

[[ "$NO_THINKING" == true ]] && BASELINE_CMD+=(--no-thinking)

# ── run baseline ──────────────────────────────────────────────────────────────
echo "[run_baseline] Running: ${BASELINE_CMD[*]}"
echo "──────────────────────────────────────────────────"
"${BASELINE_CMD[@]}"
echo "──────────────────────────────────────────────────"
echo "[run_baseline] Done."
