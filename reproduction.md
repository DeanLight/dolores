# Reproducing results

Every run in the paper goes through one interface: a YAML run config, executed by

```bash
python -m dolores.unified run --config <config.yaml>
```

The config names the agent (`react`, `codeact`, `rlm`, `deepresearch` or
`deep_reasoner`), the model, the benchmark slice and everything agent-specific
(prompt, planner config, token budget, sample count). The configs are:

| family | configs | what it is |
|---|---|---|
| Deep Reasoner | `configs/deep_reasoner/<model>/*.yaml` | the paper's Deep Reasoner runs, 3 models |
| Deep Reasoner ablations | `configs/deep_reasoner/qwen3_32b_{decomp,nomodel}/*.yaml` | planner with decomposition instructions only / without mental models |
| Baselines | `configs/baselines/<model>/<benchmark>_<baseline>.yaml` | ReAct, CodeAct, RLM, Deep Research: 14 cells × 3 models |
| Token-matched baselines | `configs/token_matched/<benchmark>_<baseline>_{try_hard,samplek}.yaml` | 14 cells × 2 arms, Qwen3-32B ([README](configs/token_matched/README.md)) |

`<model>` is one of `qwen3_8b`, `qwen3_32b`, `llama3_70b`. ReAct and Deep Research
have no Oolong cell: its documents exceed their context window.

Each Deep Reasoner model directory holds one config per benchmark for the main
table (`phantomwiki_size500`, `synthworlds`, `oolong`, `deepresearchqa`) plus
`phantomwiki_size50` / `phantomwiki_size5000` (PhantomWiki universe-size results)
and `synthworlds_v2` (a second planner version; the paper reports `synthworlds`,
planner v1). Each Deep Reasoner config pins its planner YAML
(`configs/agents/`), benchmark slice and the exact vLLM flags it is served with.

## Setup

```bash
uv sync                                   # installs dolores, the vendored deep-reasoner and vLLM
```

Put the keys in `.envrc` (gitignored) and load it:

```bash
export OPENAI_API_KEY=...     # Oolong answer parsing, DeepSearchQA judge, SynthWorlds embeddings
export SERPER_API_KEY=...     # web search (DeepSearchQA, Deep Research)
export HF_TOKEN=...           # dataset + model downloads
# export CLUSTER_MODULES="cuda/12.4 gcc/12"   # only on clusters that need `module load`
```

Then, once:

```bash
bash scripts/preflight.sh                        # checks OpenAI / HuggingFace / Serper are reachable
bash scripts/prefetch_data.sh                    # caches every benchmark dataset
uv run python scripts/build_synthworlds_embeddings.py   # SynthWorlds retrieval index (gitignored blobs)
```

## How a run executes

- **`scripts/run.sh --config <yaml>`** starts a vLLM server for the config's model
  (on a free port, with the config's `vllm_args` when it has them, else the
  model-family defaults), runs the config against it, and stops vLLM on exit.
- **`scripts/sbatch_runs.sh --configs '<glob>'`** submits one SLURM job per config
  (resources from the config's `slurm:` block) that calls `run.sh`. `--dry-run`
  prints the jobs with their task and attempt counts and submits nothing.
- A run fans out one child process per task (per attempt, for sample-k) and writes
  `<log_dir>/<benchmark>/<agent>/<model>/<run_id>/qa.json` plus `calls.jsonl`.
  Re-running a config resumes: tasks that already have a completed `qa.json` are
  skipped.
- Every script takes `--limit N` (run only the first N tasks, the same N every
  time) and `--max-workers N`.

To run a single task locally against a server you already have, point the client
at it:

```bash
python -m dolores.unified run --config configs/debug_helloworld.yaml \
    --api-base http://localhost:8000/v1 --task-id fib_7
```

## 1. Paper runs — Deep Reasoner and baselines

### Dry run

```bash
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_32b/*.yaml' --dry-run
bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_32b/*.yaml'     --dry-run
```

### Sanity check (5 tasks per config)

```bash
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_32b/*.yaml' --limit 5
bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_32b/*.yaml'     --limit 5
```

### Full

```bash
# Deep Reasoner, one model at a time (every config; see above for which feed the main table)
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_8b/*.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_32b/*.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/llama3_70b/*.yaml'

# Deep Reasoner ablations (Qwen3-32B)
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_32b_decomp/*.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/deep_reasoner/qwen3_32b_nomodel/*.yaml'

# Baselines, one model at a time
bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_8b/*.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_32b/*.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/baselines/llama3_70b/*.yaml'
```

To submit only the main-table Deep Reasoner runs for a model, name them
(`--configs` takes several paths or globs):

```bash
bash scripts/sbatch_runs.sh --configs configs/deep_reasoner/qwen3_32b/{phantomwiki_size500,synthworlds,oolong,deepresearchqa}.yaml
```

Without SLURM, run a config directly: `bash scripts/run.sh --config configs/deep_reasoner/qwen3_32b/oolong.yaml`.

## 2. Token-matched baseline runs

Each baseline cell gets as many tokens as Deep Reasoner spent on that benchmark,
in two separate arms (see [configs/token_matched/README.md](configs/token_matched/README.md)
for where the numbers come from):

- **try hard** (`*_try_hard.yaml`): one run per task with a `2 × k_b` token budget
  and a don't-give-up instruction; a run that spends its budget returns its best
  answer so far.
- **sample-k** (`*_samplek.yaml`): `⌈k_b⌉` seeded attempts per task, combined
  afterwards into majority@k and best@k.

### Dry run

```bash
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_try_hard.yaml' --dry-run
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_samplek.yaml'  --dry-run
```

### Sanity check

`--token-budget` makes the budget bind in minutes and `--samples` cuts the attempt
count; both override the config without editing it.

```bash
# One cell, locally, against a server you already have
python -m dolores.unified run --config configs/token_matched/phantomwiki_react_try_hard.yaml \
    --api-base http://localhost:8000/v1 --limit 1 --token-budget 20000
python -m dolores.unified run --config configs/token_matched/phantomwiki_react_samplek.yaml \
    --api-base http://localhost:8000/v1 --limit 1 --samples 2

# Every cell, 5 tasks each, through SLURM
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_try_hard.yaml' --limit 5
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_samplek.yaml'  --limit 5 --samples 2
```

### Full

```bash
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_try_hard.yaml'
bash scripts/sbatch_runs.sh --configs 'configs/token_matched/*_samplek.yaml'
```

### Aggregate

Majority@k, best@k, accuracy-vs-k and the budget report for one cell, from its
saved attempts (no GPU):

```bash
python -m analysis.token_matched -r 'logs/token_matched/samplek/phantomwiki/react/Qwen/Qwen3-32B' --set size=500
python -m analysis.token_matched -r 'logs/token_matched/try_hard/oolong/codeact/Qwen/Qwen3-32B'
```

Tables are written as CSVs under `logs/analysis/token_matched/`.

## Figures and tables

- **`analysis/results.py`** (Jupytext `py:percent`) rebuilds the paper tables from
  the runs above: it finds each config's run tree (`analysis/runs.py`), rescores
  the recorded answers and prints the LaTeX rows (Deep Reasoner SynthWorlds from
  `synthworlds.yaml`, planner v1). The Oolong answer parse and the
  DeepSearchQA judge call OpenAI once per run and cache the result in its `qa.json`.
- **`analysis/rlm_analysis.py`** — RLM token accounting from the `result.json`
  each RLM run saves (needs `transformers` for the tokenizers).
- **`analysis/browse_benchmarks.py`** — look at tasks from each benchmark.

Monitoring: `bash scripts/watch_experiments.sh` shows submitted jobs with
done/total per run; `bash scripts/check_parallel.sh <run dir>` shows how many LLM
calls were in flight at once.
