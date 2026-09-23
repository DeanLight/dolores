# Dolores

Publication evaluation code, benchmarks, and experiment harness used in the paper.

🚧 🏗️ 👷 This repo is the **publication** home for the deep reasoning project — a user-facing release is on the way. 🛠️ ✨

### Quick start

```bash
uv sync
# One run config — Deep Reasoner, a paper baseline or a token-matched cell — with a managed vLLM server:
bash scripts/run.sh --config configs/deep_reasoner/qwen3_32b/phantomwiki_size500.yaml --limit 5
# Many configs as SLURM jobs:
bash scripts/sbatch_runs.sh --configs 'configs/baselines/qwen3_32b/*.yaml' --dry-run
```

Every run is a YAML config under `configs/` executed by `python -m dolores.unified run --config <yaml>`;
the scripts wrap that with vLLM and SLURM. Setup (API keys, dataset prefetch, the SynthWorlds
index) and the full command list per run family are in **[reproduction.md](reproduction.md)**.

### Layout

```text
src/
  dolores/
    unified/       # The run interface: `python -m dolores.unified run --config <yaml>`
      agents/      #   react, codeact, rlm, deepresearch, cot, deep_reasoner
      benchmarks/  #   phantomwiki, synthworlds, oolong, deepresearchqa, hello_world (+ majority@k / best@k)
      cli.py runner.py obs.py inference.py token_matched.py
  config.py        # Paths, settings
  core.py          # attempt counting, subprocess pool
  prompts.py       # baseline prompts, verbatim, with their provenance table
deep-reasoner/     # raw implementation of dolores and additional utilities (vendored runtime)
configs/
  deep_reasoner/   # Deep Reasoner runs (per model, plus the decomp / nomodel ablations)
  baselines/       # Baseline runs: benchmark x baseline, per model (composed from base/)
  base/            # Model / benchmark / prompt fragments the baseline configs compose
  token_matched/   # Token-matched baseline runs: try-hard and sample-k arms (see its README)
  agents/          # Deep Reasoner planner configs (system prompt, mental models)
  debug_helloworld.yaml  # dataset-free smoke config
scripts/           # run.sh (one config, managed vLLM), sbatch_runs.sh + slurm_run.sh (SLURM),
                   # preflight / prefetch / SynthWorlds index build, watch_experiments, check_parallel
analysis/          # Jupytext notebooks: paper tables (`results.py` via `runs.py`),
                   # token-matched aggregation (`token_matched.py`), RLM token accounting, benchmark browser
```

Until the old-vs-new comparison has run, the legacy runners (`src/baselines/`, `src/benchmarks/`,
`src/dolores/experiment.py` and `*_cli.py`, `configs/main/`, `configs/experiment_base/`,
`scripts/run_baseline.sh`, `sbatch_baselines.sh`, `sbatch_eval.sh`) are still in the tree; they are
being removed. Use the commands above.
