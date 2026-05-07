#!/usr/bin/env bash
# Show running slurm eval jobs and their task progress.
#
# Usage:
#   ./scripts/watch_experiments.sh

SIDECAR_DIR="logs/slurm"

jobs=$(squeue -u "$USER" --noheader --format="%i %T %M" 2>/dev/null)

if [[ -z "$jobs" ]]; then
    echo "No running jobs."
    exit 0
fi

# One Python call per log_dir: count done tasks and read n_tasks from run_metadata.json
count_progress() {
    python3 - "$1" <<'PYEOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.is_dir():
    print("0/?")
    sys.exit()

# Total from run_metadata.json written at experiment start
n_tasks = "?"
meta = p / "run_metadata.json"
try:
    v = json.loads(meta.read_text()).get("n_tasks")
    if v is not None:
        n_tasks = str(v)
except Exception:
    pass

done = 0
for d in p.iterdir():
    if not d.is_dir():
        continue
    qa = d / "qa.json"
    try:
        if qa.exists() and json.loads(qa.read_text()).get("answer") is not None:
            done += 1
    except Exception:
        pass

print(f"{done}/{n_tasks}")
PYEOF
}

WIDTH=$(tput cols 2>/dev/null || echo 140)
SEP=$(python3 -c "print('─' * $WIDTH)")
echo "$SEP"
printf "%-10s %-12s %-8s %-12s %-35s %-30s %s\n" "JOB_ID" "STATE" "TIME" "DONE/TOTAL" "CONFIG" "LOG_DIR" "SLURM_LOG"
echo "$SEP"

while read -r job_id state elapsed; do
    sidecar="${SIDECAR_DIR}/${job_id}.json"
    slurm_log="${SIDECAR_DIR}/${job_id}.log"
    if [[ -f "$sidecar" ]]; then
        config_short=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
import os; print(os.path.basename(d.get('config', '?')))
" "$sidecar" 2>/dev/null)
        log_dir=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get('log_dir', '?'))
" "$sidecar" 2>/dev/null)
        progress=$(count_progress "$log_dir")
    else
        config_short="(no sidecar)"
        log_dir="(not found)"
        progress="-"
    fi
    printf "%-10s %-12s %-8s %-12s %-35s %-30s %s\n" \
        "$job_id" "$state" "$elapsed" "$progress" "$config_short" "$log_dir" "$slurm_log"
done <<< "$jobs"

echo "$SEP"
