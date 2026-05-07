# Reproducing results

Run everything from the **repository root** after installing the same stack as the paper experiments. Use **`python -m …`** with `src/` on your `PYTHONPATH`, or an editable install, so `dolores`, `baselines`, and shared modules resolve.

## Dry run

```bash
# DeepReasoner — prints merged config + task plan, no tasks executed
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_phantomwiki_size500.yaml --dry-run
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_oolong.yaml --dry-run
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_deepsearchqa.yaml --dry-run
```

## Sanity check (5 examples)

```bash
# DeepReasoner — 5 tasks per benchmark
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_phantomwiki_size500.yaml --set benchmark.key_limit=5
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_oolong.yaml --set benchmark.key_limit=5
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_deepsearchqa.yaml --set benchmark.key_limit=5

# Smolagents baselines — --limit 5 for oolong/deepresearchqa; --max-workers 1 for phantomwiki/synthworlds
bash scripts/run_baseline.sh --baseline react --benchmark phantomwiki --model Qwen/Qwen3-32B --pw-size 500 --pw-seed 1 --max-workers 1
bash scripts/run_baseline.sh --baseline react --benchmark deepresearchqa --model Qwen/Qwen3-32B --limit 5
bash scripts/run_baseline.sh --baseline codeact --benchmark oolong --model Qwen/Qwen3-32B --limit 5
bash scripts/run_baseline.sh --baseline rlm --benchmark synthworlds --model Qwen/Qwen3-32B --max-workers 1
bash scripts/run_baseline.sh --baseline deepresearch --benchmark deepresearchqa --model Qwen/Qwen3-32B --limit 5
```

## DeepReasoner — all runs

```bash
# Qwen3-32B
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_phantomwiki_size500.yaml
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_synthworld.yaml
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_oolong.yaml
python -m dolores.experiment configs/main/qwen3_32b/qwen3_32b_deepsearchqa.yaml

# Qwen3-32B — no model (ablation: decomposition only, no LLM)
python -m dolores.experiment configs/main/qwen3_32b_nomodel/qwen3_32b_phantomwiki_size500_nomodel.yaml
python -m dolores.experiment configs/main/qwen3_32b_nomodel/qwen3_32b_synthworld_nomodel.yaml
python -m dolores.experiment configs/main/qwen3_32b_nomodel/qwen3_32b_oolong_nomodel.yaml
python -m dolores.experiment configs/main/qwen3_32b_nomodel/qwen3_32b_deepsearchqa_nomodel.yaml

# Qwen3-32B — decomp (ablation: decomposition variant)
python -m dolores.experiment configs/main/qwen3_32b_decomp/qwen3_32b_phantomwiki_size500_decomp.yaml
python -m dolores.experiment configs/main/qwen3_32b_decomp/qwen3_32b_synthworld_decomp.yaml
python -m dolores.experiment configs/main/qwen3_32b_decomp/qwen3_32b_oolong_decomp.yaml
python -m dolores.experiment configs/main/qwen3_32b_decomp/qwen3_32b_deepsearchqa_decomp.yaml

# Qwen3-8B
python -m dolores.experiment configs/main/qwen3_8b/qwen3_8b_phantomwiki_size500.yaml
python -m dolores.experiment configs/main/qwen3_8b/qwen3_8b_synthworld.yaml
python -m dolores.experiment configs/main/qwen3_8b/qwen3_8b_oolong.yaml
python -m dolores.experiment configs/main/qwen3_8b/qwen3_8b_deepsearchqa.yaml

# Llama-3.3-70B
python -m dolores.experiment configs/main/llama3_70b/llama3_70b_phantomwiki_size500.yaml
python -m dolores.experiment configs/main/llama3_70b/llama3_70b_synthworld.yaml
python -m dolores.experiment configs/main/llama3_70b/llama3_70b_oolong.yaml
python -m dolores.experiment configs/main/llama3_70b/llama3_70b_deepsearchqa.yaml
```

## Smolagents baselines — all runs

```bash
# react — Qwen3-32B
bash scripts/run_baseline.sh --baseline react --benchmark phantomwiki --model Qwen/Qwen3-32B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline react --benchmark synthworlds --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline react --benchmark deepresearchqa --model Qwen/Qwen3-32B

# react — Qwen3-8B
bash scripts/run_baseline.sh --baseline react --benchmark phantomwiki --model Qwen/Qwen3-8B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline react --benchmark synthworlds --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline react --benchmark deepresearchqa --model Qwen/Qwen3-8B

# react — Llama-3.3-70B
bash scripts/run_baseline.sh --baseline react --benchmark phantomwiki --model meta-llama/Llama-3.3-70B-Instruct --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline react --benchmark synthworlds --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline react --benchmark deepresearchqa --model meta-llama/Llama-3.3-70B-Instruct

# codeact — Qwen3-32B
bash scripts/run_baseline.sh --baseline codeact --benchmark phantomwiki --model Qwen/Qwen3-32B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline codeact --benchmark synthworlds --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline codeact --benchmark oolong --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline codeact --benchmark deepresearchqa --model Qwen/Qwen3-32B

# codeact — Qwen3-8B
bash scripts/run_baseline.sh --baseline codeact --benchmark phantomwiki --model Qwen/Qwen3-8B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline codeact --benchmark synthworlds --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline codeact --benchmark oolong --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline codeact --benchmark deepresearchqa --model Qwen/Qwen3-8B

# codeact — Llama-3.3-70B
bash scripts/run_baseline.sh --baseline codeact --benchmark phantomwiki --model meta-llama/Llama-3.3-70B-Instruct --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline codeact --benchmark synthworlds --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline codeact --benchmark oolong --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline codeact --benchmark deepresearchqa --model meta-llama/Llama-3.3-70B-Instruct

# rlm — Qwen3-32B
bash scripts/run_baseline.sh --baseline rlm --benchmark phantomwiki --model Qwen/Qwen3-32B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline rlm --benchmark synthworlds --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline rlm --benchmark oolong --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline rlm --benchmark deepresearchqa --model Qwen/Qwen3-32B

# rlm — Qwen3-8B
bash scripts/run_baseline.sh --baseline rlm --benchmark phantomwiki --model Qwen/Qwen3-8B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline rlm --benchmark synthworlds --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline rlm --benchmark oolong --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline rlm --benchmark deepresearchqa --model Qwen/Qwen3-8B

# rlm — Llama-3.3-70B
bash scripts/run_baseline.sh --baseline rlm --benchmark phantomwiki --model meta-llama/Llama-3.3-70B-Instruct --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline rlm --benchmark synthworlds --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline rlm --benchmark oolong --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline rlm --benchmark deepresearchqa --model meta-llama/Llama-3.3-70B-Instruct

# deepresearch — Qwen3-32B
bash scripts/run_baseline.sh --baseline deepresearch --benchmark phantomwiki --model Qwen/Qwen3-32B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline deepresearch --benchmark synthworlds --model Qwen/Qwen3-32B
bash scripts/run_baseline.sh --baseline deepresearch --benchmark deepresearchqa --model Qwen/Qwen3-32B

# deepresearch — Qwen3-8B
bash scripts/run_baseline.sh --baseline deepresearch --benchmark phantomwiki --model Qwen/Qwen3-8B --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline deepresearch --benchmark synthworlds --model Qwen/Qwen3-8B
bash scripts/run_baseline.sh --baseline deepresearch --benchmark deepresearchqa --model Qwen/Qwen3-8B

# deepresearch — Llama-3.3-70B
bash scripts/run_baseline.sh --baseline deepresearch --benchmark phantomwiki --model meta-llama/Llama-3.3-70B-Instruct --pw-size 500 --pw-seed 1
bash scripts/run_baseline.sh --baseline deepresearch --benchmark synthworlds --model meta-llama/Llama-3.3-70B-Instruct
bash scripts/run_baseline.sh --baseline deepresearch --benchmark deepresearchqa --model meta-llama/Llama-3.3-70B-Instruct
```

## Figures and tables

Paper-style aggregation, LaTeX-style rows, and the section-1 teaser figure are in **`analysis/results.py`** (Jupytext `py:percent`; pair with `.ipynb` via juplit if you use notebooks). Committed **`analysis/*.csv`** snapshots hold cached per-task aggregates so you can reload and plot without re-scoring everything; overwrite those CSVs (or delete them) after you run the scoring cells to refresh from **`logs/`**. **`analysis/dr_analysis.py`** adds helpers for walking experiment run directories into DataFrames; **`analysis/rlm_analysis.py`** token stats from saved RLM JSON (install dev deps so **`transformers`** is available for tokenizers).

Run from the repo root with an editable install or **`PYTHONPATH=src`** so **`benchmarks`** and **`config`** resolve.
