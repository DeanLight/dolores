# Dolores

Publication evaluation code, benchmarks, and experiment harness used in the paper.

### Layout

```text
src/
  dolores/         # YAML-driven experiment runner (`experiment.py`) and per-benchmark *_cli modules
  baselines/       # Baseline agents (react, codeact, cot, rlm, deepresearch, …)
  benchmarks/      # Benchmark implementations (synthworlds, phantomwiki, oolong, deepresearchqa, hello_world, …)
  config.py        # Paths, RunContext, settings
  core.py
  prompts.py
  vllm_utils.py
deep-reasoner/     # raw implementation of dolores and additional utilities
configs/
  agents/          # Agent YAML
  experiment_base/ # Shared includes for experiments
  main/            # Model / experiment matrix YAML
scripts/           # Run helpers (e.g. baselines, Slurm / sbatch)
analysis/          # Jupytext notebooks: aggregated results, tables, paper figures (`results.py`, …)
```

For concrete commands for reproducing runs in the paper (configs + scripts), see **[reproduction.md](reproduction.md)**.
