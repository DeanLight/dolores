#!/usr/bin/env bash
# Check whether LLM calls ran in parallel, for a single task or across many runs.
#
# Usage:
#   bash scripts/check_parallel.sh <path>
#
# <path> can be:
#   - a task dir   (contains llm_calls.jsonl)  → detailed view for that task
#   - a run dir    (contains task subdirs)      → aggregate across all tasks in the run
#   - a bench dir  (contains run subdirs)       → aggregate across all runs and tasks

set -euo pipefail

TARGET="${1:-${LOG_DIR:-}}"
if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 <task_dir | run_dir | bench_dir>" >&2
  exit 1
fi

python3 - "$TARGET" <<'PYEOF'
import json, sys
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

def ts_to_sec(s):
    """Parse HH:MM:SS → seconds since midnight.  Returns None on failure."""
    try:
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception:
        return None

def load_calls(jsonl_path):
    """Load llm.call events; derive _end_sec and _start_sec (end - duration_s).

    Timestamps are HH:MM:SS with no date, so a run that crosses midnight will
    have later log entries with a smaller numeric value than earlier ones.
    We fix this by walking the file in log order (which is chronological) and
    adding 86400s whenever a timestamp drops by more than 12 hours.
    """
    calls = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("event") == "llm.call":
            calls.append(obj)

    # Apply midnight rollover fix in file order (= log order = chronological).
    offset = 0
    prev_raw = None
    for c in calls:
        raw = ts_to_sec(c.get("timestamp", ""))
        if raw is None:
            c["_end_sec"] = None
            c["_start_sec"] = None
            continue
        if prev_raw is not None and raw < prev_raw - 43200:  # dropped >12 h → midnight
            offset += 86400
        prev_raw = raw
        end = raw + offset
        dur = c.get("duration_s")
        c["_end_sec"]   = end
        c["_start_sec"] = (end - dur) if dur is not None else end

    return [c for c in calls if c["_end_sec"] is not None]

def max_concurrent(calls):
    """Return the peak number of calls in-flight at the same time (sweep line)."""
    events = []
    for c in calls:
        events.append((c["_start_sec"], +1))
        events.append((c["_end_sec"],   -1))
    # sort by time; break ties by putting -1 before +1 so a call ending exactly
    # when another starts is NOT counted as concurrent
    events.sort(key=lambda e: (e[0], e[1]))
    peak = cur = 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return peak

def analyse_task(calls):
    if not calls:
        return None
    calls = sorted(calls, key=lambda c: c["_start_sec"])
    t0 = calls[0]["_start_sec"]

    n_nodes   = len(set(c.get("node_id") for c in calls))
    total_tok = sum((c.get("usage") or {}).get("completion_tokens", 0) for c in calls)
    serial_s  = sum(c["duration_s"] for c in calls if c.get("duration_s") is not None)
    wall_s    = calls[-1]["_end_sec"] - calls[0]["_start_sec"]
    peak      = max_concurrent(calls)

    return {
        "n_calls":      len(calls),
        "n_nodes":      n_nodes,
        "total_tokens": total_tok,
        "serial_s":     round(serial_s, 2),
        "wall_s":       round(wall_s, 2),
        "speedup":      round(serial_s / wall_s, 2) if wall_s > 0 else 1.0,
        "peak_concurrent": peak,
        "is_parallel":  peak > 1,
        "calls":        calls,
        "t0":           t0,
    }

def print_task_detail(label, st):
    calls = st["calls"]
    t0    = st["t0"]
    print(f"\nTask: {label}")
    print(f"  {'#':>3}  {'start':>7}  {'end':>7}  {'dur':>6}  {'node':>5}  {'d':>2}  {'tokens':>6}  ancestry")
    print("  " + "─" * 68)
    for i, c in enumerate(calls):
        s   = round(c["_start_sec"] - t0, 1)
        e   = round(c["_end_sec"]   - t0, 1)
        dur = c.get("duration_s", "?")
        tok = (c.get("usage") or {}).get("completion_tokens", "?")
        anc = c.get("ancestry", "")
        print(f"  {i+1:>3}  {s:>6.1f}s  {e:>6.1f}s  {str(dur):>5}s  "
              f"{str(c.get('node_id','?')):>5}  {str(c.get('depth','?')):>2}  "
              f"{str(tok):>6}  {anc}")

    has_dur = st["serial_s"] > 0
    print(f"\n  {st['n_calls']} calls · {st['n_nodes']} nodes · {st['total_tokens']} tokens")
    if has_dur:
        print(f"  wall: {st['wall_s']}s · serial sum: {st['serial_s']}s · speedup: {st['speedup']}x")
    else:
        print(f"  wall: {st['wall_s']}s  (no duration_s in logs — re-run to get speedup)")

    peak = st["peak_concurrent"]
    if peak > 1:
        print(f"  ✓  PARALLEL — peak {peak} calls in-flight at once")
    else:
        print(f"  ✗  SEQUENTIAL — never more than 1 call in-flight at once")

    print("\n  Responses:")
    for i, c in enumerate(calls):
        snippet = (c.get("response") or "").replace("\n", " ").strip()[:100]
        print(f"  #{i+1:>2} node={c.get('node_id')} d{c.get('depth')}  {snippet}")

def print_aggregate(task_stats):
    n        = len(task_stats)
    n_par    = sum(1 for _, s in task_stats if s["is_parallel"])
    has_dur  = any(s["serial_s"] > 0 for _, s in task_stats)
    avg_su   = (sum(s["speedup"]  for _, s in task_stats if s["serial_s"] > 0)
                / sum(1 for _, s in task_stats if s["serial_s"] > 0)) if has_dur else None
    avg_calls = sum(s["n_calls"] for _, s in task_stats) / n

    W = 48
    print(f"\n{'═'*(W+42)}")
    print(f"AGGREGATE  ({n} tasks)")
    print(f"{'─'*(W+42)}")
    print(f"  Parallel:        {n_par}/{n}  ({100*n_par//n if n else 0}%)")
    if avg_su is not None:
        print(f"  Avg speedup:     {avg_su:.2f}x")
    print(f"  Avg calls/task:  {avg_calls:.1f}")
    print()
    avg_peak = sum(s["peak_concurrent"] for _, s in task_stats) / n
    print(f"  Avg peak concurrent: {avg_peak:.1f}")
    hdr = f"  {'task':<{W}}  {'calls':>5}  {'wall':>6}  {'serial':>7}  {'speedup':>7}  {'peak':>4}  par"
    print()
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for label, s in sorted(task_stats, key=lambda x: str(x[0])):
        par = "✓" if s["is_parallel"] else "✗"
        su  = f"{s['speedup']:.2f}x" if s["serial_s"] > 0 else "  n/a"
        print(f"  {str(label):<{W}}  {s['n_calls']:>5}  {s['wall_s']:>5}s  "
              f"{s['serial_s']:>6}s  {su:>7}  {s['peak_concurrent']:>4}  {par}")

# ── main ──────────────────────────────────────────────────────────────────────

root = Path(sys.argv[1])
jsonl_files = sorted(root.rglob("llm_calls.jsonl"))

if not jsonl_files:
    print(f"No llm_calls.jsonl files found under {root}")
    sys.exit(1)

# Single task dir → detailed view
if (root / "llm_calls.jsonl").exists():
    calls = load_calls(root / "llm_calls.jsonl")
    stats = analyse_task(calls)
    if stats:
        print_task_detail(root, stats)
    else:
        print("No llm.call events found.")
    sys.exit(0)

# One file found but not directly in root → still do detailed view
if len(jsonl_files) == 1:
    calls = load_calls(jsonl_files[0])
    stats = analyse_task(calls)
    if stats:
        print_task_detail(jsonl_files[0].parent, stats)
    else:
        print("No llm.call events found.")
    sys.exit(0)

# Multiple tasks → aggregate
task_stats = []
for jf in jsonl_files:
    calls = load_calls(jf)
    stats = analyse_task(calls)
    if stats:
        task_stats.append((jf.parent.relative_to(root), stats))

if not task_stats:
    print("No tasks with llm.call events found.")
    sys.exit(1)

print_aggregate(task_stats)
PYEOF
