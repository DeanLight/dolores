#!/usr/bin/env bash
#
# Submit one sbatch job per main config file.
#
# Usage:
#   bash scripts/sbatch_eval.sh [--dry-run] 'configs/main/llama3*.yaml' [config2.yaml ...]
#
# Arguments may be explicit paths or glob patterns (quoted to prevent premature
# shell expansion — the script expands them itself so they work from any cwd).
#
# --dry-run  Validate and preview each config. No jobs are submitted.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── parse args ────────────────────────────────────────────────────────────────
DRY_RUN=false
RAW_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        -*)        echo "Unknown flag: $1" >&2; exit 1 ;;
        *)         RAW_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#RAW_ARGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--dry-run] 'glob/or/path.yaml' [more...]" >&2
    exit 1
fi

# ── expand glob patterns into actual file list ────────────────────────────────
CONFIGS=()
for arg in "${RAW_ARGS[@]}"; do
    # Use bash glob expansion; fail if pattern matches nothing
    matches=( $arg )
    if [[ ${#matches[@]} -eq 0 || ! -f "${matches[0]}" ]]; then
        echo "No files matched: $arg" >&2
        exit 1
    fi
    for f in "${matches[@]}"; do
        CONFIGS+=("$f")
    done
done

echo "[sbatch_eval] ${#CONFIGS[@]} config(s): ${CONFIGS[*]}"

# ── validate each config individually ────────────────────────────────────────
echo "[sbatch_eval] Validating configs..."
for CONFIG in "${CONFIGS[@]}"; do
    python -m dolores.experiment preview "$CONFIG" >/dev/null
    echo "[sbatch_eval] Valid: $CONFIG"
done
echo "[sbatch_eval] All configs valid."

# ── dry-run: preview first config in full, diffs for the rest, then plan table ─
if [[ "$DRY_RUN" == true ]]; then
    python -m dolores.experiment preview "${CONFIGS[@]}"
    echo
    python -m dolores.experiment plan "${CONFIGS[@]}" 2>/dev/null
    exit 0
fi

# ── submit one sbatch job per config ─────────────────────────────────────────
echo "[sbatch_eval] Submitting ${#CONFIGS[@]} job(s)..."
for CONFIG in "${CONFIGS[@]}"; do
    # Read slurm: settings — resolve _compose first so inherited slurm keys work
    read -r SLURM_GPUS SLURM_TIME SLURM_MEM SLURM_CPUS SLURM_QOS < <(
        python - "$CONFIG" <<'PYEOF'
import sys, yaml
from pathlib import Path

def resolve(path):
    data = yaml.safe_load(open(path)) or {}
    result = {}
    for p in data.get("_compose", []):
        result.update(resolve(p).get("slurm", {}))
    result.update(data.get("slurm", {}))
    return {"slurm": result}

slurm = resolve(sys.argv[1])["slurm"]
print(
    slurm.get("gpus", 1),
    slurm.get("time", "04:00:00"),
    slurm.get("mem", "200G"),
    slurm.get("cpus_per_task", 8),
    slurm.get("qos", "normal"),
)
PYEOF
    )

    read -r LOG_DIR N_TASKS < <(python -c "
import yaml, sys
d = yaml.safe_load(open(sys.argv[1])) or {}

# Resolve _compose to find log_dir and key_limit
def resolve(path, seen=None):
    seen = seen or set()
    if path in seen: return {}
    seen.add(path)
    data = yaml.safe_load(open(path)) or {}
    result = {}
    for p in data.get('_compose', []):
        result.update(resolve(p, seen))
    result.update(data)
    return result

merged = resolve(sys.argv[1])
log_dir = merged.get('log_dir', 'logs')
kl = merged.get('benchmark', {}).get('key_limit', '')
print(log_dir, kl if kl != '' else '')
" "$CONFIG")
    JOB_NAME=$(basename "$LOG_DIR")

    echo "[sbatch_eval] Submitting: config=${CONFIG} gpus=${SLURM_GPUS} time=${SLURM_TIME} mem=${SLURM_MEM} cpus=${SLURM_CPUS} qos=${SLURM_QOS} name=${JOB_NAME}"
    JOB_ID=$(sbatch --parsable \
        --job-name="${JOB_NAME}" \
        --gres="gpu:${SLURM_GPUS}" \
        --time="${SLURM_TIME}" \
        --mem="${SLURM_MEM}" \
        --cpus-per-task="${SLURM_CPUS}" \
        --qos="${SLURM_QOS}" \
        "${SCRIPT_DIR}/slurm_eval.sh" \
        "$CONFIG")
    echo "[sbatch_eval] Submitted job ${JOB_ID}"

    # Write sidecar so watch_experiments.sh can map job → log_dir
    mkdir -p logs/slurm
    printf '{"job_id": "%s", "config": "%s", "log_dir": "%s", "n_tasks": "%s"}\n' \
        "$JOB_ID" "$CONFIG" "$LOG_DIR" "$N_TASKS" > "logs/slurm/${JOB_ID}.json"
done
echo "[sbatch_eval] Done. Submitted ${#CONFIGS[@]} job(s)."
