# Plan: move every baseline run to the unified interface and add the token-matched runs

> Planning document for this PR. It is deleted in the final commit, once the port lands
> and `reproduction.md` is rewritten.

## Goal

Dolores gets **one** interface for baseline runs: the unified agent CLI. It covers
both:

1. **The paper's baseline runs.** These are ReAct, CodeAct, RLM and Deep Research on
   PhantomWiki, SynthWorlds, Oolong and DeepSearchQA, for Qwen3-32B, Qwen3-8B and
   Llama-3.3-70B, with one attempt per task and no budget. That is 14 runnable cells
   × 3 models = 42 configs.
2. **The token-matched runs** on Qwen3-32B, 14 cells × 2 arms = 28 configs:
   - **try hard.** One run per task with a `2 × k_b` token budget and an instruction
     not to give up. When the budget runs out, the run returns its best answer so far.
   - **sample-k.** `⌈k_b⌉` independent seeded attempts per task. These are combined
     after the runs into majority@k and best@k.

`k_b` is Deep Reasoner's total tokens divided by the baseline's total tokens, measured
per cell. ReAct and Deep Research can't run Oolong because its documents are longer
than their context window. PhantomWiki runs at size 500 only.

The legacy per-baseline scripts in `src/baselines/*.py` are **deleted**. Deep Reasoner
itself stays on `python -m dolores.experiment` (see Open questions).

## Where things stand

Dolores's `src/` is a snapshot of the research code from **before** its unified-agent
refactor. It has one standalone script per baseline, and each script builds its own
client and loop. Upstream, the token-matched feature is built on the unified path:

| Needs | Provided by (upstream, not in Dolores) |
|---|---|
| One place to count and cap tokens | `obs.py` patches every litellm/openai completion call |
| Seeded, repeatable attempts per task | `obs.py` hook + `token_matched.RunBudget` |
| Budget handling that stops the run, then scoring and writing `qa.json` | `agents/base.Agent.run` (template method) |
| Prompts inlined in each config (`instructions:` override) | `agents/base.Agent._instructions` |
| Composing configs from shared fragments | `cli.load_config` (`_compose`) |
| Planning and resuming by `(task, attempt)` | `runner.plan_work` + `core.count_attempts` |
| majority@k / best@k per benchmark | `benchmarks/base.Benchmark.cluster_answers` + benchmark classes |
| vLLM backend startup | `inference.py` |

Upstream already has unified configs for the paper's baseline runs:
`configs/baselines/<model>/<benchmark>_<baseline>.yaml`, composed from `configs/base/`
fragments and produced by the same code path as the token-matched configs. So both
halves of the goal come from the same pinned commit.

The benchmark prompt text is identical between the two repos. Upstream `prompts.py`
differs from `src/prompts.py` only by a provenance markdown cell.

## Source of truth

Everything is ported from upstream `big_runs` at merge commit `ed05708`. That commit
contains the token-matched work and the smoke-test and SLURM fixes.

A separate upstream branch also moves everything to the unified path. It is still open
and branches from before the token-matched work, so it conflicts with `ed05708` in
`cli.py`, `obs.py` and `runner.py`, and its configs use a different layout. **It is not
the source.** The only thing to borrow from it is its analysis port from the legacy log
layout to `qa.json` (§5), and that is borrowed as a reference, not merged.

## Proposed layout

```text
src/
  dolores/
    unified/                   # NEW — the baseline interface
      __init__.py
      cli.py                   # `python -m dolores.unified.cli run --config ...`
                               #   (--dry-run, --limit, --attempt, --token-budget, --samples, --api-base)
      runner.py                # plan_work -> (task_id, attempt) pairs; dispatch
      obs.py                   # completion-call patches + budget/seed seam
      inference.py             # vLLM / OpenAI backends
      token_matched.py         # BudgetExhausted, ContextWindowExceeded, RunBudget, attempt_seed
      agents/
        __init__.py base.py _helpers.py react.py codeact.py rlm.py deepresearch.py cot.py
      benchmarks/
        __init__.py base.py benchmarks.py deepresearch_agent.py open_deep_research/
        data/{oolong,deepresearchqa}_prompts.yaml
    experiment.py *_cli.py     # UNCHANGED — Deep Reasoner runs
  baselines/                   # DELETED — superseded by dolores.unified.agents
  benchmarks/                  # KEPT — Deep Reasoner's experiment + analysis still import it
  config.py                    # MODIFIED — add `Paths`; RunContext stays while analysis/DR use it
  core.py                      # MODIFIED — add count_attempts, run_subprocess_pool
  prompts.py                   # UNCHANGED
  vllm_utils.py                # UNCHANGED — used by dolores.experiment
configs/
  baselines/<model>/           # NEW — 42 paper baseline runs (unified)
  base/                        # NEW — 15 fragments the 42 compose (models, benchmarks, prompts)
  token_matched/               # NEW — 28 token-matched runs + README
  agents/ experiment_base/ main/   # UNCHANGED — Deep Reasoner
analysis/
  token_matched.py             # NEW — majority@k, best@k, accuracy_vs_k, budget_report, token_summary
  dr_vs_baseline_token_ratios.csv   # NEW — the measured k_b every token-matched config cites
  results.py rlm_analysis.py browse_benchmarks.py   # MODIFIED — read baseline qa.json logs
scripts/
  run_baseline.sh              # REWRITTEN — config-driven only: --config <yaml>
  sbatch_baselines.sh          # REWRITTEN — --configs <glob>, per-config slurm: resources
  slurm_baselines.sh           # MODIFIED — runs `python -m dolores.unified.cli run --config`
  check_parallel.sh watch_experiments.sh   # MODIFIED — unified log layout
  build_synthworlds_embeddings.py   # NEW — SynthWorlds retrieval index
```

Port scope:

- **Skip** `agents/deep_reasoner.py`, since Deep Reasoner stays on `dolores.experiment`.
  Remove it from `agents/__init__.py`.
- **Move** `deepresearch_agent.py` and `open_deep_research/` under
  `dolores/unified/benchmarks/`, since the new path owns them. Then delete the legacy
  copies in `src/benchmarks/deepresearchqa/` if the only thing importing them is
  `dolores/deepsearchqa_cli.py`, pointing that import at the new location. Otherwise
  keep one copy and import it from both places. Never keep two copies.
- **Skip** the big-runs configs (Qwen3-Coder-480B), log-sync scripts, and design, plan
  and help docs.

## Work items

### 1. Shared modules
- `config.py`: add `Paths`. Keep `RunContext` until §5 and the Deep Reasoner CLIs no
  longer need it.
- `core.py`: add `count_attempts` and `run_subprocess_pool`. Leave `find_tested_ids`
  unchanged.

### 2. `src/dolores/unified/` port
- Copy the files listed above from `ed05708` and rewrite the imports:
  `deepreasoner_baselines.{agents,benchmarks,cli,runner,obs,inference,token_matched}`
  → `dolores.unified.…`, and `deepreasoner_baselines.{config,core,prompts}` →
  `{config,core,prompts}`.
- Keep the files byte-identical otherwise, so an import-normalised diff against
  upstream is empty. Put that diff command in the PR description.
- Replace the upstream console-script name `dr-baselines` in docstrings and config
  headers with `python -m dolores.unified.cli`.
- `obs.py` needs `deep_reasoner.core.configure_structlog_fixture`, which the vendored
  `deep-reasoner/` already provides.

### 3. Configs
- Copy `configs/baselines/` (42), `configs/base/` (15) and `configs/token_matched/`
  (28 plus README).
- `_compose` paths are repo-relative (`configs/base/...`), so they work unchanged.
- Check every config: `api_base`, ports and `slurm:` blocks must use defaults or env
  vars, never cluster paths.

### 4. Scripts: one config-driven entry point
- Rewrite `run_baseline.sh` around upstream's `--token-matched-config` branch, renamed
  to `--config <yaml>`. It must accept any unified config, matched or not. Keep its
  vLLM startup, the port and key read from the config, the eager/`deep_gemm` fallback
  and the torch.compile-off-at-TP>1 behaviour. Drop the legacy
  `--baseline/--benchmark/--pw-size` mode.
- Rewrite `sbatch_baselines.sh` around the `--token-matched <glob>` branch, renamed to
  `--configs <glob>`, with per-config `slurm:` resources and the dry-run task/attempt
  breakdown. Drop the model × baseline × benchmark loop.
- `slurm_baselines.sh`, `check_parallel.sh` and `watch_experiments.sh`: update to the
  new command and the new log layout
  (`logs/<benchmark>/<agent>/<model>/<run_id>/qa.json`, or the config's `log_dir`).

### 5. Analysis
- Add `analysis/token_matched.py` and `dr_vs_baseline_token_ratios.csv`.
- Port the baseline readers in `results.py`, `rlm_analysis.py` and
  `browse_benchmarks.py` from the legacy `RunContext` layout
  (`logs/<benchmark>/<method>/<model>/<stem>.json`) to unified `qa.json`. Use the open
  upstream branch's analysis port as a reference. Deep Reasoner result reading stays
  as it is.
- Code only, no result files. Every table and figure is regenerated from the user's
  own logs.

### 6. Delete the legacy baseline path
- `src/baselines/` (all five scripts), and remove `baselines` from
  `[tool.setuptools] packages`.
- Every reference to `python -m baselines.*` in scripts and docs.
- Do this as its own commit, after §2–§5, so `git log` shows the replacement landing
  before the removal.

### 7. Dependencies (`pyproject.toml`)
- Add `dolores.unified` and its subpackages to `packages`.
- Add `conflit`, `parse` and `coolname`.
- Declare `smolagents`, `rlms`, `transformers` and `huggingface-hub`, which the agents
  import but which aren't declared today.

### 8. Docs
- `reproduction.md`: rewrite the baselines section as the same three steps for both
  run families: dry run, `--limit 5`, then full, each through
  `sbatch_baselines.sh --configs '<glob>'`. Families:
  - paper baselines, one glob per model: `configs/baselines/qwen3_32b/*.yaml`, and so on
  - token matched, one glob per arm: `configs/token_matched/*_try_hard.yaml` and
    `configs/token_matched/*_samplek.yaml`

  Then add the post-hoc `python -m analysis.token_matched` step. The Deep Reasoner
  section is unchanged.
- `README.md`: update the layout tree.

### 9. Tests and verification (no GPU)
- `pytest src analysis`. The ported tests include the ones that check each
  token-matched config against the ratio CSV, and each inlined prompt against
  `prompts.py`.
- `python -m dolores.unified.cli run --config <cfg> --dry-run` for every config in
  `configs/baselines/` and `configs/token_matched/`.
- `sbatch_baselines.sh --configs '<glob>' --dry-run` for each glob in the reproduction
  doc. It must list 42 jobs for the baselines (14 per model) and 28 for token matched,
  with the expected attempt counts.
- `grep -r "baselines\.\(react\|codeact\|rlm\|deepresearch\|cot\)\|RunContext"`: the
  only hits left are in the Deep Reasoner code.

### 10. Numbers check (needs a GPU; for the author)
The paper's baseline numbers were produced by the legacy scripts. After this PR they
are reproduced by the unified path instead. Before merging, run one 5-task cell per
baseline through `configs/baselines/qwen3_32b/` and compare it with the published
per-task answers, so that any drift is known and documented.

### 11. Anonymisation pass (blocking, since this repo is public)
Before every push, grep the diff for organisation or user names, emails, cluster
hostnames, scratch paths, internal doc links, and upstream repo or PR URLs. Upstream
`RUNS.md` and `scripts/` contain several of these.

## Commit sequence

1. shared modules + deps
2. `dolores.unified` port (imports rewritten, otherwise verbatim)
3. configs: `baselines/`, `base/`, `token_matched/` + ratio CSV
4. scripts: config-driven `run_baseline.sh` / `sbatch_baselines.sh` / `slurm_baselines.sh`
5. analysis: `token_matched.py` + port the baseline readers to `qa.json`
6. delete `src/baselines/` and the legacy script mode
7. `reproduction.md` + README; delete this plan

## Decisions

1. **Source:** upstream `big_runs` @ `ed05708`.
2. **Package:** `src/dolores/unified/`, run as `python -m dolores.unified.cli`.
3. **One interface:** both the paper baseline runs and the token-matched runs go
   through `dolores.unified`, and the legacy `src/baselines/` is deleted.
4. **Results:** code to regenerate only; no result CSVs or figures.

## Open questions

1. **Deep Reasoner.** Should it stay on `dolores.experiment`, or also move behind the
   unified CLI? Upstream has a `DeepReasonerAgent`, but it wraps the upstream
   `deep_reasoner` package, while Dolores vendors its own. The plan assumes it stays.
2. **The competing upstream branch.** Once this lands, should the open upstream
   branch be closed, or rebased onto `big_runs`?
