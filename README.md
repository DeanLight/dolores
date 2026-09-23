# Dolores

Publication evaluation code, benchmarks, and experiment harness used in the paper.

🚧 🏗️ 👷 This repo is the **publication** home for the deep reasoning project — a user-facing release is on the way. 🛠️ ✨

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
  prompts.py       # baseline prompts, verbatim
deep-reasoner/     # raw implementation of dolores and additional utilities
configs/
  deep_reasoner/   # Deep Reasoner runs (per model, plus the decomp / nomodel ablations)
  baselines/       # Baseline runs: benchmark x baseline, per model (composed from base/)
  token_matched/   # Token-matched baseline runs: try-hard and sample-k arms
  agents/          # Deep Reasoner planner configs
scripts/           # run.sh (one config, managed vLLM), sbatch_runs.sh (SLURM), monitoring
analysis/          # Jupytext notebooks: paper tables (`results.py`), token-matched aggregation, …
```

Until the old-vs-new comparison has run, the legacy runners (`src/baselines/`, `src/benchmarks/`, `dolores.experiment`, `configs/main/`) are still in the tree; they are being removed.

For concrete commands for reproducing runs in the paper (configs + scripts), see **[reproduction.md](reproduction.md)**.
