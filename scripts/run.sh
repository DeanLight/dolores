#!/usr/bin/env bash
#
# Run one unified run config — a paper baseline, a token-matched cell or a Deep
# Reasoner run — against a managed local vLLM server. Starts vLLM with the model
# and flags the config names, waits for /health, runs
# `python -m dolores.unified run`, and stops vLLM on exit (even on failure).
#
# Usage:
#   bash scripts/run.sh --config configs/token_matched/phantomwiki_react_try_hard.yaml
#
# Required:
#   --config YAML          A config under configs/{baselines,token_matched,deep_reasoner}/.
#                          The YAML owns the agent, model, benchmark, prompt and
#                          token-matching keys.
#
# Optional:
#   --limit N              Run only the first N tasks (sanity slice)
#   --max-workers N        Override the config's max_workers
#   --token-budget N       Override the config's token_budget (smoke tests)
#   --samples N            Override the config's samples (smoke tests)
#   --tensor-parallel N    vLLM tensor-parallel-size (default: the config's
#                          slurm.tensor_parallel, else auto from the model)
#   --gpu-mem F            gpu_memory_utilization (default: config vllm_gpu_mem, else 0.9)
#   --vllm-wait N          Seconds to wait for vLLM /health (default: 600)
#   --port N               vLLM port (default: a free port)
#   --api-key KEY          Key vLLM serves with and the client sends (default: your_secret)
#   --extra-vllm-args ARGS Extra args appended to `vllm serve` (quoted string)
#
# vLLM flags: a config with `vllm_args:` (the Deep Reasoner configs) is served with
# exactly those flags, as the paper's runs were. Otherwise the model-family
# defaults below apply (the flags the paper's baseline runs used).

set -euo pipefail

# ── environment ───────────────────────────────────────────────────────────────
# Load Lmod modules only if the cluster needs them (set CLUSTER_MODULES in .envrc;
# leave it unset on clusters without Lmod). Word-split is intentional.
if [[ -n "${CLUSTER_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load ${CLUSTER_MODULES}
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# ── parse args ────────────────────────────────────────────────────────────────
CONFIG=""
LIMIT=""
MAX_WORKERS=""
TOKEN_BUDGET=""
SAMPLES=""
TENSOR_PARALLEL=""
GPU_MEM=""
VLLM_WAIT=600
PORT=""
API_KEY="your_secret"
EXTRA_VLLM_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)           CONFIG="$2";            shift 2 ;;
        --limit)            LIMIT="$2";             shift 2 ;;
        --max-workers)      MAX_WORKERS="$2";       shift 2 ;;
        --token-budget)     TOKEN_BUDGET="$2";      shift 2 ;;
        --samples)          SAMPLES="$2";           shift 2 ;;
        --tensor-parallel)  TENSOR_PARALLEL="$2";   shift 2 ;;
        --gpu-mem)          GPU_MEM="$2";           shift 2 ;;
        --vllm-wait)        VLLM_WAIT="$2";         shift 2 ;;
        --port)             PORT="$2";              shift 2 ;;
        --api-key)          API_KEY="$2";           shift 2 ;;
        --extra-vllm-args)  EXTRA_VLLM_ARGS="$2";   shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${CONFIG:?--config is required}"
[[ -f "$CONFIG" ]] || { echo "No such config: $CONFIG" >&2; exit 1; }

# ── read the config (resolving _compose) ──────────────────────────────────────
eval "$(uv run python - "$CONFIG" <<'PYEOF'
import shlex, sys
from dolores.unified.cli import load_config

c = load_config(sys.argv[1])
def emit(var, val):
    print(f"{var}={shlex.quote('' if val is None else str(val))}")
emit("MODEL", c["model"])
emit("CFG_TP", (c.get("slurm") or {}).get("tensor_parallel"))
emit("CFG_GPU_MEM", c.get("vllm_gpu_mem"))
args = c.get("vllm_args")
emit("HAS_VLLM_ARGS", "1" if args is not None else "")
print("CFG_VLLM_ARGS=(" + " ".join(shlex.quote(str(a)) for a in (args or [])) + ")")
PYEOF
)"

[[ -z "$TENSOR_PARALLEL" ]] && TENSOR_PARALLEL="$CFG_TP"
[[ -z "$GPU_MEM" ]] && GPU_MEM="${CFG_GPU_MEM:-0.9}"

# Auto-detect tensor-parallel-size from the model name if neither flag nor config set it.
if [[ -z "$TENSOR_PARALLEL" ]]; then
    case "$MODEL" in
        *70B*|*72B*) TENSOR_PARALLEL=4 ;;
        *)           TENSOR_PARALLEL=2 ;;
    esac
fi

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

# ── vllm command ──────────────────────────────────────────────────────────────
VLLM_CMD=(
    uv run vllm serve "$MODEL"
    --port "$PORT"
    --api-key "$API_KEY"
    --gpu-memory-utilization "$GPU_MEM"
)
if [[ -n "$HAS_VLLM_ARGS" ]]; then
    # The config pins its own serving flags (tensor parallelism included).
    VLLM_CMD+=("${CFG_VLLM_ARGS[@]}")
else
    VLLM_CMD+=(--enable-prefix-caching --seed 42 --tensor-parallel-size "$TENSOR_PARALLEL")
    case "$MODEL" in
        *Qwen3*|*qwen3*)
            VLLM_CMD+=(--reasoning-parser qwen3 --tool-call-parser hermes --enable-auto-tool-choice) ;;
        *Llama-3.3*|*llama-3.3*)
            VLLM_CMD+=(--tool-call-parser llama3_json --enable-auto-tool-choice) ;;
    esac
fi
# shellcheck disable=SC2206
[[ -n "$EXTRA_VLLM_ARGS" ]] && VLLM_CMD+=($EXTRA_VLLM_ARGS)

# Graceful fallback on nodes without a CUDA toolkit. No nvcc (and no CUDA_HOME)
# means vLLM's torch.compile/inductor path dies with "Permission denied: 'nvcc'",
# so serve eagerly and disable torch.compile (at TP>1 vLLM still compiles
# per-layer ops otherwise). When the deep_gemm kernels aren't installed, vLLM's
# FP8 warmup probe crashes even for BF16 models, so disable it. Both are no-ops
# on a provisioned node.
if ! command -v nvcc >/dev/null 2>&1 && [[ -z "${CUDA_HOME:-}" ]]; then
    echo "[run] no nvcc/CUDA_HOME — adding --enforce-eager + TORCH_COMPILE_DISABLE=1"
    VLLM_CMD+=(--enforce-eager)
    export TORCH_COMPILE_DISABLE=1
fi
if ! uv run python -c 'import deep_gemm' >/dev/null 2>&1; then
    echo "[run] deep_gemm not installed — setting VLLM_USE_DEEP_GEMM=0"
    export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
fi

# ── start vllm ────────────────────────────────────────────────────────────────
echo "[run] Starting vllm: ${VLLM_CMD[*]}"
"${VLLM_CMD[@]}" &
VLLM_PID=$!
trap 'echo "[run] Stopping vllm (pid=${VLLM_PID})"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# ── wait for /health ──────────────────────────────────────────────────────────
echo "[run] Waiting for vllm at port ${PORT} (timeout ${VLLM_WAIT}s)..."
DEADLINE=$(( $(date +%s) + VLLM_WAIT ))
LAST_PRINT=0
ATTEMPT=0
while true; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[run] vllm ready (attempt ${ATTEMPT})"
        break
    fi
    NOW=$(date +%s)
    if (( NOW >= DEADLINE )); then
        echo "[run] vllm did not become healthy within ${VLLM_WAIT}s" >&2
        exit 1
    fi
    if (( NOW - LAST_PRINT >= 30 )); then
        echo "[run] still waiting... elapsed=$(( NOW - (DEADLINE - VLLM_WAIT) ))s attempt=${ATTEMPT}"
        LAST_PRINT=$NOW
    fi
    sleep 5
done

# ── run ───────────────────────────────────────────────────────────────────────
# Point the client (and its children) at the port vLLM actually came up on,
# overriding the config's placeholder api_base, so jobs can share a node.
RUN_CMD=(
    uv run python -m dolores.unified run
    --config "$CONFIG"
    --api-base "$API_BASE"
    --api-key "$API_KEY"
)
[[ -n "$MAX_WORKERS"  ]] && RUN_CMD+=(--max-workers "$MAX_WORKERS")
[[ -n "$LIMIT"        ]] && RUN_CMD+=(--limit "$LIMIT")
[[ -n "$TOKEN_BUDGET" ]] && RUN_CMD+=(--token-budget "$TOKEN_BUDGET")
[[ -n "$SAMPLES"      ]] && RUN_CMD+=(--samples "$SAMPLES")

echo "[run] Running: ${RUN_CMD[*]}"
echo "──────────────────────────────────────────────────"
"${RUN_CMD[@]}"
echo "──────────────────────────────────────────────────"
echo "[run] Done."
