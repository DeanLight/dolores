# Plan: move every run to the unified interface and add the token-matched runs

> Planning document for this PR. It is deleted in the final commit, once the port lands
> and `reproduction.md` is rewritten.

## Status

Implemented on this branch (commits 1–5 and 7 of the sequence below):

- [x] shared modules + deps
- [x] `dolores.unified` port — import-normalised diff against upstream `ed05708` is empty
- [x] Deep Reasoner parity (§2b), per-task timeout
- [x] configs: `baselines/` (42), `base/`, `token_matched/` (28), `deep_reasoner/` (29) + ratio CSV
- [x] scripts: `run.sh`, `sbatch_runs.sh`, `slurm_run.sh` (new names; the legacy scripts stay until §6)
- [x] analysis: `token_matched.py`, `runs.py`, `results.py` / `browse_benchmarks.py` / `rlm_analysis.py` on qa.json
- [x] `reproduction.md`, README
- [ ] **§10 numbers check — author, needs a GPU**
- [ ] §6 delete the legacy paths (after §10), plus `analysis/dr_analysis.py`, which only reads
      `dolores.experiment` output; then delete this plan

Found while implementing:

- The legacy Deep Reasoner runners (`dolores/*_cli.py`) do not run against the vendored
  `deep-reasoner` (`run_cli()` signature mismatch — `TypeError` before any task). §10
  therefore compares Deep Reasoner's unified runs against the published per-task logs,
  not against a live legacy run.
- Entry point is `python -m dolores.unified` (a package `__main__`): running `cli.py`
  as `__main__` executed its test blocks in every parent and child process.
- The paper's SynthWorlds Deep Reasoner result is planner v1 (`synthworlds.yaml`, confirmed);
  `synthworlds_v2.yaml` is kept as a config but left out of the paper table.
- `max_workers` now comes from each config (upstream's submit script forced 32).

## Goal

Dolores gets **one** interface for every run, Deep Reasoner included: the unified
agent CLI. It covers three run families:

1. **The paper's baseline runs.** These are ReAct, CodeAct, RLM and Deep Research on
   PhantomWiki, SynthWorlds, Oolong and DeepSearchQA, for Qwen3-32B, Qwen3-8B and
   Llama-3.3-70B, with one attempt per task and no budget. That is 14 runnable cells
   × 3 models = 42 configs.
2. **The token-matched runs** on Qwen3-32B, 14 cells × 2 arms = 28 configs:
   - **try hard.** One run per task with a `2 × k_b` token budget and an instruction
     not to give up. When the budget runs out, the run returns its best answer so far.
   - **sample-k.** `⌈k_b⌉` independent seeded attempts per task. These are combined
     after the runs into majority@k and best@k.
3. **Deep Reasoner's runs:** the main runs for 3 models, plus the Qwen3-32B
   `decomp` and `nomodel` ablations. Today these are `configs/main/**`, run by
   `dolores.experiment`.

`k_b` is Deep Reasoner's total tokens divided by the baseline's total tokens, measured
per cell. ReAct and Deep Research can't run Oolong because its documents are longer
than their context window. PhantomWiki runs at size 500 only.

The legacy per-baseline scripts in `src/baselines/*.py` are **deleted**. So are
`dolores.experiment` and the per-benchmark `dolores/*_cli.py`, but only after the Deep
Reasoner parity check in §10.

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

A separate upstream branch also moved everything to the unified path. It has now been
closed. It branched from before the token-matched work, so it conflicts with `ed05708` in
`cli.py`, `obs.py` and `runner.py`, and its configs use a different layout. **It is not
the source.** Two things are borrowed from it as references, not merged: its analysis port to
`qa.json` (§5) and the shape of its `configs/deep_reasoner/` (§3).

## Proposed layout

```text
src/
  dolores/
    unified/                   # NEW — the one run interface
      __init__.py
      cli.py                   # `python -m dolores.unified.cli run --config ...`
                               #   (--dry-run, --limit, --attempt, --token-budget, --samples, --api-base)
      runner.py                # plan_work -> (task_id, attempt) pairs; dispatch
      obs.py                   # completion-call patches + budget/seed seam
      inference.py             # vLLM / OpenAI backends
      token_matched.py         # BudgetExhausted, ContextWindowExceeded, RunBudget, attempt_seed
      agents/
        __init__.py base.py _helpers.py react.py codeact.py rlm.py deepresearch.py cot.py
        deep_reasoner.py       # DeepReasonerAgent, extended for parity (§2b)
      benchmarks/
        __init__.py base.py benchmarks.py deepresearch_agent.py open_deep_research/
        data/{oolong,deepresearchqa}_prompts.yaml
    experiment.py *_cli.py     # DELETED after parity (§10) — superseded by agents/deep_reasoner.py
  baselines/                   # DELETED — superseded by dolores.unified.agents
  benchmarks/                  # DELETED once nothing imports it (analysis ported in §5)
  config.py                    # MODIFIED — add `Paths`; drop RunContext at the end
  core.py                      # MODIFIED — add count_attempts, run_subprocess_pool
  prompts.py                   # UNCHANGED
  vllm_utils.py                # DELETED with dolores.experiment, unless something else still imports it
configs/
  baselines/<model>/           # NEW — 42 paper baseline runs (unified)
  base/                        # NEW — 15 fragments the 42 compose (models, benchmarks, prompts)
  token_matched/               # NEW — 28 token-matched runs + README
  agents/                      # KEPT — Deep Reasoner planner configs (agent_config points here)
  deep_reasoner/<variant>/     # NEW — unified Deep Reasoner configs, 1:1 with configs/main/**
  experiment_base/ main/       # DELETED — replaced by configs/deep_reasoner/
analysis/
  token_matched.py             # NEW — majority@k, best@k, accuracy_vs_k, budget_report, token_summary
  dr_vs_baseline_token_ratios.csv   # NEW — the measured k_b every token-matched config cites
  results.py rlm_analysis.py browse_benchmarks.py   # MODIFIED — read unified qa.json logs
scripts/
  run_baseline.sh              # REWRITTEN — config-driven only: --config <yaml>
  sbatch_baselines.sh          # REWRITTEN — --configs <glob>, per-config slurm: resources
  slurm_baselines.sh           # MODIFIED — runs `python -m dolores.unified.cli run --config`
  check_parallel.sh watch_experiments.sh   # MODIFIED — unified log layout
  build_synthworlds_embeddings.py   # NEW — SynthWorlds retrieval index
```

Port scope:

- **Port** `agents/deep_reasoner.py`. It already works with the vendored
  `deep-reasoner/`, which provides `cli_base` and `cli_utils`. It is not yet at parity
  with the per-benchmark runners (§2b).
- **Move** `deepresearch_agent.py` and `open_deep_research/` under
  `dolores/unified/benchmarks/`, since the new path owns them. Delete the legacy copies
  in `src/benchmarks/deepresearchqa/` together with `dolores/deepsearchqa_cli.py`.
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
  upstream is empty. The exceptions are the §2b parity changes, which land as their
  own commit so they can be reviewed separately. Put that diff command in the PR description.
- Replace the upstream console-script name `dr-baselines` in docstrings and config
  headers with `python -m dolores.unified.cli`.
- `obs.py` needs `deep_reasoner.core.configure_structlog_fixture`, which the vendored
  `deep-reasoner/` already provides.

### 2b. Deep Reasoner parity (the real work in this PR)
Upstream's `DeepReasonerAgent` runs one generic `PlanExec` pass, using the tools the
benchmark class provides. The per-benchmark runners in dolores differ from it:

| Benchmark | `dolores/*_cli.py` today | Unified path today | Fix |
|---|---|---|---|
| DeepSearchQA | `search` = smolagents web-search **sub-agent** on a separate search model (`search_model_id`, from `model_search_tool_*.yaml`); calls `agent._run_async`, not `PlanExec` | `web_search`, a plain Serper function; `PlanExec` | add the search-model sub-agent as the DR tool, keeping the tool name `search`; config keys `search_model_id`/`search_api_base`; match the call path |
| SynthWorlds | `search` = `create_retriever_tool` (OpenAI embeddings) | `retrieve_top_5` (local SM index) | expose the retriever that the paper's DR runs used, under the name `search` |
| PhantomWiki | `retrieve_article`, `search`; `_parse_phantom_output` | same tools; `_parse_answer` | diff the two parsers; keep dolores's |
| Oolong | `document` var, `PlanExec` | `document` var, `PlanExec` | check that the kwargs and namespaces match |
| all | `agent.configs` list (layered planner YAML), `version`, 600 s `task_timeout` per task, `sub_json` conversion | a single `agent_config`; no per-task timeout | accept a list for `agent_config`; add `task_timeout` to the runner; keep `version` in qa.json metadata |

The planner prompts name their tools (`search`, and so on), so tool names are part
of the behaviour and must match what the prompts expect. Make these DR-specific tool
choices a DR-agent option (`dr_tools:` in the config), so the baseline agents'
benchmark tools stay unchanged. Write an `if test():` block for each row.

### 3. Configs
- Copy `configs/baselines/` (42), `configs/base/` (15) and `configs/token_matched/`
  (28 plus README).
- Write `configs/deep_reasoner/<variant>/<benchmark>.yaml` 1:1 from each non-debug
  `configs/main/**` file, with the same planner YAML (`configs/agents/*.yaml`), model,
  inference, benchmark slice, `max_steps` and log dir. The closed upstream branch's
  `configs/deep_reasoner/` is a reference for the shape. Keep `configs/agents/`.
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
- Port `results.py`, `rlm_analysis.py` and `browse_benchmarks.py` so they read
  unified `qa.json` for both baselines and Deep Reasoner. They currently read the legacy
  `RunContext` layout and the `dolores.experiment` run dirs, and import the legacy
  `benchmarks` modules. Point them at `dolores.unified.benchmarks`. Use the closed
  upstream branch's analysis port as a reference.
- Code only, no result files. Every table and figure is regenerated from the user's
  own logs.

### 6. Delete the legacy paths (after §10 passes)
- `src/baselines/` (all five scripts).
- `src/dolores/experiment.py` and `*_cli.py`, `configs/main/`, `configs/experiment_base/`,
  `scripts/sbatch_eval.sh` and `slurm_eval.sh`.
- `src/benchmarks/`, `RunContext` and `vllm_utils.py`, once nothing imports them.
- The matching entries in `[tool.setuptools] packages` / `py-modules`, and every
  `python -m baselines.*` / `dolores.experiment` reference in scripts and docs.
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

  - Deep Reasoner, one glob per variant: `configs/deep_reasoner/qwen3_32b/*.yaml`, and so on

  Then add the post-hoc `python -m analysis.token_matched` step.
- `README.md`: update the layout tree.

### 9. Tests and verification (no GPU)
- `pytest src analysis`. The ported tests include the ones that check each
  token-matched config against the ratio CSV, and each inlined prompt against
  `prompts.py`.
- `python -m dolores.unified.cli run --config <cfg> --dry-run` for every config in
  `configs/baselines/`, `configs/token_matched/` and `configs/deep_reasoner/`.
- `sbatch_baselines.sh --configs '<glob>' --dry-run` for each glob in the reproduction
  doc. It must list 42 jobs for the baselines (14 per model), 28 for token matched
  with the expected attempt counts, and one per Deep Reasoner config.
- `grep -rE "baselines\.(react|codeact|rlm|deepresearch|cot)|RunContext|dolores\.experiment"`:
  no hits.

### 10. Numbers check (needs a GPU; for the author; **gates the deletions in §6**)
The paper's numbers were produced by the legacy scripts and `dolores.experiment`.
After this PR they are reproduced by the unified path instead. Before the legacy code
is deleted, run a 5-task slice of each Qwen3-32B cell through both paths and compare
the per-task answers and scores, so that any drift is known and documented:
- every baseline (`configs/baselines/qwen3_32b/`)
- Deep Reasoner on all 4 benchmarks, plus one `decomp` and one `nomodel` ablation

Deep Reasoner matters most here, because the headline numbers are its own.

### 11. Anonymisation pass (blocking, since this repo is public)
Before every push, grep the diff for organisation or user names, emails, cluster
hostnames, scratch paths, internal doc links, and upstream repo or PR URLs. Upstream
`RUNS.md` and `scripts/` contain several of these.

## Commit sequence

1. shared modules + deps
2. `dolores.unified` port (imports rewritten, otherwise verbatim)
2b. Deep Reasoner parity in `agents/deep_reasoner.py` (+ `task_timeout` in the runner)
3. configs: `baselines/`, `base/`, `token_matched/`, `deep_reasoner/` + ratio CSV
4. scripts: config-driven `run_baseline.sh` / `sbatch_baselines.sh` / `slurm_baselines.sh`
5. analysis: `token_matched.py` + port the baseline readers to `qa.json`
— author runs the §10 numbers check —
6. delete the legacy paths
7. `reproduction.md` + README; delete this plan

## Decisions

1. **Source:** upstream `big_runs` @ `ed05708`.
2. **Package:** `src/dolores/unified/`, run as `python -m dolores.unified.cli`.
3. **One interface:** baselines, token-matched runs and Deep Reasoner all go through
   `dolores.unified`. The legacy scripts and `dolores.experiment` are deleted once the
   parity check passes.
4. **Results:** code to regenerate only; no result CSVs or figures.
5. **Upstream:** the competing branch that also moved everything to the unified
   path has been closed; this PR supersedes it.
