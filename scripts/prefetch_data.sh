#!/usr/bin/env bash
# Download every benchmark dataset into the HF cache ONCE, up front, so the
# concurrent jobs reuse the shared cache instead of each re-downloading (which
# races on the cache and can hit HF rate limits). Datasets land in
# $HF_DATASETS_CACHE (set in .envrc). Run to completion before submitting jobs.
set -euo pipefail
echo "[prefetch] datasets -> ${HF_DATASETS_CACHE:-<default HF cache>}"
uv run python - <<'PY'
from datasets import load_dataset
# (repo, config, split) for every benchmark the runs touch.
SPECS = [
    ("kilian-group/phantom-wiki-v1", "text-corpus",     None),    # phantomwiki
    ("kilian-group/phantom-wiki-v1", "question-answer", None),    # phantomwiki
    ("kenqgu/SynthWorlds",           "qa-sm",           "test"),  # synthworld
    ("oolongbench/oolong-real",      "dnd",             "test"),  # oolong
    ("google/deepsearchqa",          None,              "eval"),  # deepsearchqa
]
for repo, cfg, split in SPECS:
    label = repo + (f" [{cfg}]" if cfg else "") + (f" ({split})" if split else "")
    print(f"[prefetch] {label} ...", flush=True)
    load_dataset(repo, cfg, split=split)
print("[prefetch] all benchmark datasets cached")
PY
echo "[prefetch] done. Datasets are cached; submit jobs now (they reuse this cache)."
