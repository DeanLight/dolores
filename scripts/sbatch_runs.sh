#!/usr/bin/env bash
#
# Submit one SLURM job per unified run config. Each job runs scripts/run.sh on
# its config (vLLM up, run, vLLM down).
#
# Usage:
#   bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_32b/*.yaml' [OPTIONS]
#
# Required:
#   --configs GLOB         Quoted glob (or a single path) of configs to submit, e.g.
#                            'configs/baselines/qwen3_32b/*.yaml'        paper baselines
#                            'configs/token_matched/*_try_hard.yaml'     token matched
#                            'configs/deep_reasoner/qwen3_32b/*.yaml'    Deep Reasoner
#
# Per-job resources come from each config's `slurm:` block (gpus, tensor_parallel,
# cpus_per_task, mem, time, qos). A config without one gets tensor_parallel from the
# model (4 for 70B, else 2), gpus = tensor_parallel, and the defaults below.
#
# Options:
#   --limit N              Run only the first N tasks of each config (sanity slice)
#   --max-workers N        Override every config's max_workers
#   --token-budget N       Override token_budget (smoke tests)
#   --samples N            Override samples (smoke tests)
#   --gpus N  --cpus N  --mem MEM  --time HH:MM:SS  --qos QOS
#                          Override the resources of every submitted job
#   --dry-run              Print the jobs (with task / attempt counts) and exit
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIGS_GLOB=""
LIMIT=""
MAX_WORKERS=""
TOKEN_BUDGET=""
SAMPLES=""
OVR_GPUS=""; OVR_CPUS=""; OVR_MEM=""; OVR_TIME=""; OVR_QOS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --configs)       CONFIGS_GLOB="$2"; shift 2 ;;
        --limit)         LIMIT="$2";        shift 2 ;;
        --max-workers)   MAX_WORKERS="$2";  shift 2 ;;
        --token-budget)  TOKEN_BUDGET="$2"; shift 2 ;;
        --samples)       SAMPLES="$2";      shift 2 ;;
        --gpus)          OVR_GPUS="$2";     shift 2 ;;
        --cpus)          OVR_CPUS="$2";     shift 2 ;;
        --mem)           OVR_MEM="$2";      shift 2 ;;
        --time)          OVR_TIME="$2";     shift 2 ;;
        --qos)           OVR_QOS="$2";      shift 2 ;;
        --dry-run)       DRY_RUN=true;      shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

: "${CONFIGS_GLOB:?--configs is required}"
# shellcheck disable=SC2206  # deliberate glob expansion
CONFIGS=($CONFIGS_GLOB)
if [[ ${#CONFIGS[@]} -eq 0 || ! -f "${CONFIGS[0]}" ]]; then
    echo "No configs match ${CONFIGS_GLOB}" >&2
    exit 1
fi

# ── resolve every config once: resources, sidecar fields, task counts ─────────
# One TSV line per config. Task counts load the benchmark datasets; where that is
# not possible (no HF cache / network) the count shows as "?".
PLAN=$(uv run python - "$DRY_RUN" "${LIMIT:-}" "${SAMPLES:-}" "${CONFIGS[@]}" <<'PYEOF'
import os, sys
from unittest.mock import patch
from dolores.unified.cli import load_config

dry, limit, samples_ovr = sys.argv[1] == "true", sys.argv[2], sys.argv[3]
configs = sys.argv[4:]
_counts = {}

def n_tasks(bench: dict):
    from dolores.unified.benchmarks import get_benchmark
    b = dict(bench)
    name = b.pop("name")
    key = (name, tuple(sorted(b.items())))
    if key not in _counts:
        try:
            # DeepResearchQA refuses to construct without a key; listing ids needs none.
            with patch.dict(os.environ, {"SERPER_API_KEY": os.environ.get("SERPER_API_KEY") or "x"}):
                _counts[key] = len(get_benchmark(name, **b).list_task_ids())
        except Exception:
            _counts[key] = None
    return _counts[key]

for path in configs:
    c = load_config(path)
    s = c.get("slurm") or {}
    model = c["model"]
    tp = s.get("tensor_parallel") or (4 if ("70B" in model or "72B" in model) else 2)
    gpus = s.get("gpus") or tp
    k = int(samples_ovr or c.get("samples", 1) or 1)
    n = n_tasks(c["benchmark"]) if dry else None
    if n is not None and limit:
        n = min(n, int(limit))
    tree = "{}/{}/{}/{}".format(c.get("log_dir") or "logs", c["benchmark"]["name"], c["agent"],
                                model + ("-nothink" if c.get("no_thinking") else ""))
    print("\t".join(str(x) for x in [
        path, c["agent"], c["benchmark"]["name"], model, gpus, tp,
        s.get("cpus_per_task", 16), s.get("mem", "200G"), s.get("time", "16:00:00"),
        s.get("qos", "normal"), k, "?" if n is None else n,
        "?" if n is None else n * k, tree,
    ]))
PYEOF
)

echo "[sbatch_runs] ${#CONFIGS[@]} config(s):"
ROW="  %-52s %-13s %-15s %-34s %4s %3s %4s %7s %9s\n"
# shellcheck disable=SC2059
printf "$ROW" "CONFIG" "AGENT" "BENCHMARK" "MODEL" "GPUS" "TP" "K" "TASKS" "ATTEMPTS"
while IFS=$'\t' read -r cfg agent bench model gpus tp cpus mem time qos k n att tree; do
    # shellcheck disable=SC2059
    printf "$ROW" "$cfg" "$agent" "$bench" "$model" "${OVR_GPUS:-$gpus}" "$tp" "$k" "$n" "$att"
done <<< "$PLAN"

[[ "$DRY_RUN" == true ]] && echo "[sbatch_runs] DRY RUN — no jobs submitted." && exit 0

# ── submit ────────────────────────────────────────────────────────────────────
mkdir -p logs/slurm
N=0
while IFS=$'\t' read -r cfg agent bench model gpus tp cpus mem time qos k n att tree; do
    JOB_NAME="$(basename "$(dirname "$cfg")")_$(basename "${cfg%.yaml}")"
    EXPORT_VARS="ALL,CONFIG=${cfg},TENSOR_PARALLEL=${tp}"
    [[ -n "$LIMIT"        ]] && EXPORT_VARS+=",LIMIT=${LIMIT}"
    [[ -n "$MAX_WORKERS"  ]] && EXPORT_VARS+=",MAX_WORKERS=${MAX_WORKERS}"
    [[ -n "$TOKEN_BUDGET" ]] && EXPORT_VARS+=",TOKEN_BUDGET=${TOKEN_BUDGET}"
    [[ -n "$SAMPLES"      ]] && EXPORT_VARS+=",SAMPLES=${SAMPLES}"

    JOB_GPUS="${OVR_GPUS:-$gpus}"; JOB_CPUS="${OVR_CPUS:-$cpus}"
    JOB_MEM="${OVR_MEM:-$mem}";    JOB_TIME="${OVR_TIME:-$time}"; JOB_QOS="${OVR_QOS:-$qos}"
    echo "[sbatch_runs] Submitting: ${JOB_NAME} gpus=${JOB_GPUS} tp=${tp} cpus=${JOB_CPUS} mem=${JOB_MEM} time=${JOB_TIME}"
    JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --gres="gpu:${JOB_GPUS}" \
        --time="${JOB_TIME}" \
        --mem="${JOB_MEM}" \
        --cpus-per-task="${JOB_CPUS}" \
        --qos="${JOB_QOS}" \
        --export="${EXPORT_VARS}" \
        "${SCRIPT_DIR}/slurm_run.sh")
    echo "[sbatch_runs] Submitted job ${JOB_ID} (${JOB_NAME})"

    # Sidecar for watch_experiments.sh: the config and the run tree it writes to.
    printf '{"job_id": "%s", "agent": "%s", "config": "%s", "log_dir": "%s", "model": "%s"}\n' \
        "$JOB_ID" "$agent" "$cfg" "$tree" "$model" > "logs/slurm/${JOB_ID}.json"
    N=$((N + 1))
done <<< "$PLAN"

echo "[sbatch_runs] Done. Submitted ${N} job(s)."
