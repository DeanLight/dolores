# Plan: add the token-matched baseline runs to Dolores

> Planning document for this PR. It is deleted in the final commit, once the port lands
> and `reproduction.md` has a token-matched chapter.

## Goal

Anyone with this repo can re-run the **token-matched baseline comparisons** from the
paper, using the same code that produced the reported numbers:

- **try hard.** One run per task with a token budget of `2 × k_b` and an instruction
  not to give up. When the budget runs out, the run returns its best answer so far.
- **sample-k.** `⌈k_b⌉` independent seeded attempts per task. These are combined
  after the runs into majority@k and best@k.

`k_b` is Deep Reasoner's total tokens divided by the baseline's total tokens, measured
for each benchmark × baseline cell on Qwen3-32B. There are 14 runnable cells × 2 arms,
which gives 28 run configs. ReAct and Deep Research can't run Oolong because its
documents are longer than their context window. PhantomWiki runs at size 500 only.

## Where things stand

Dolores's `src/` is a snapshot of the research code from **before** its
unified-agent refactor. It has one standalone script per baseline in `src/baselines/*.py`,
and each script builds its own client and loop.

Upstream, the token-matched feature is built on that refactor and needs it to work:

| Needs | Provided by (upstream, not in Dolores) |
|---|---|
| One place to count and cap tokens | `obs.py` patches every litellm/openai completion call |
| Seeded, repeatable attempts per task | `obs.py` hook + `token_matched.RunBudget` |
| Budget handling that stops the run, then scoring and writing `qa.json` | `agents/base.Agent.run` (template method) |
| Prompts inlined in each config (`instructions:` override) | `agents/base.Agent._instructions` + `cli.load_config` (`_compose`) |
| Planning and resuming by `(task, attempt)` | `runner.plan_work` + `core.count_attempts` |
| majority@k / best@k per benchmark | `benchmarks/base.Benchmark.cluster_answers` + benchmark classes |
| vLLM backend startup | `inference.py` |

For that reason the plan is to **port the unified path alongside the legacy scripts**
rather than retrofit budgets into `src/baselines/*.py`. A retrofit would mean four
separate budget and seed implementations. It would also produce code that differs from
the code that generated the numbers, which defeats the point of a reproduction repo.

The benchmark prompt text is already identical between the two repos. Upstream
`prompts.py` differs from `src/prompts.py` only by a provenance markdown cell. So the
inlined prompts in the token-matched configs will match Dolores's `prompts.py`
character for character.

## Source of truth

Port from the upstream token-matched work including the smoke-test and SLURM fixes,
pinned at merge commit `ed05708` (tree identical to fix head `e9efa29`). Every ported
file is copied from that commit.

## Proposed layout

A new subpackage `src/dolores/unified/` holds the unified path, with imports
rewritten from `deepreasoner_baselines.X` to `dolores.unified.X`. The shared top-level modules
(`config`, `core`, `prompts`) get the upstream additions merged in so that nothing is
duplicated.

```text
src/
  dolores/unified/             # NEW — unified agent path (used by token-matched runs)
    __init__.py
    cli.py                     # `python -m dolores.unified.cli run --config ...` (+ --attempt, --limit,
                               #   --token-budget, --samples, --api-base overrides, _compose loader)
    runner.py                  # plan_work -> (task_id, attempt) pairs; dispatch
    obs.py                     # completion-call patches + budget/seed seam
    inference.py               # vLLM / OpenAI backends
    token_matched.py           # BudgetExhausted, ContextWindowExceeded, RunBudget, attempt_seed
    agents/
      __init__.py base.py _helpers.py react.py codeact.py rlm.py deepresearch.py cot.py
    benchmarks/
      __init__.py base.py benchmarks.py   # Benchmark + cluster_answers/majority_at_k/best_at_k
      data/{oolong,deepresearchqa}_prompts.yaml
  config.py                    # MODIFIED — add `Paths` next to RunContext/settings
  core.py                      # MODIFIED — add count_attempts, run_subprocess_pool
  prompts.py                   # UNCHANGED (optionally add the provenance cell)
  baselines/ benchmarks/ dolores/*.py   # UNCHANGED — paper-run path stays as-is
configs/
  token_matched/               # NEW — 28 run configs + README (numbers and prompts inlined)
analysis/
  token_matched.py             # NEW — load_attempts, aggregate, accuracy_vs_k, budget_report, token_summary
  dr_vs_baseline_token_ratios.csv   # NEW — the measured k_b per cell every config cites
scripts/
  run_baseline.sh              # MODIFIED — --token-matched-config <yaml>
  sbatch_baselines.sh          # MODIFIED — --token-matched <glob>, per-config resources
  build_synthworlds_embeddings.py   # NEW — needed by SynthWorlds cells
```

Port only what the token-matched path uses:

- **Skip** `agents/deep_reasoner.py`. Deep Reasoner's reference numbers come from the
  existing `dolores.experiment` runs, and the ratios CSV already records them. Remove
  it from `agents/__init__.py` so importing the registry doesn't require the upstream
  wrapper around `deep_reasoner.cli_base`.
- **Reuse, don't duplicate**, `src/benchmarks/deepresearchqa/deepresearch_agent.py` and
  `open_deep_research/` if they match upstream once the `hosted_vllm/` litellm prefix
  fix is applied. Otherwise port the upstream copy under `dolores/unified/benchmarks/`.
- **Skip** the big-runs (Qwen3-Coder-480B) configs, `sync_*_logs.sh`, and any design,
  plan or help docs.

## Work items

### 1. Shared modules (small, do first)
- `config.py`: add `Paths` from upstream. Keep `RunContext` so the legacy scripts still
  work.
- `core.py`: add `count_attempts` and `run_subprocess_pool`. Leave `find_tested_ids`
  unchanged.
- Test: `pytest src/baselines src/dolores` still passes, and a legacy
  `run_baseline.sh --dry-run` still produces the same commands as before.

### 2. `src/dolores/unified/` port
- Copy the files listed above from the pinned commit. Rewrite imports with `sed`
  (`deepreasoner_baselines.` → `dolores.unified.`, `deepreasoner_baselines.config` →
  `config`, `deepreasoner_baselines.core` → `core`, `deepreasoner_baselines.prompts` →
  `prompts`).
- Keep the files byte-identical otherwise, so an import-normalised diff against
  upstream is empty. Record that diff command in this PR's description.
- `obs.py` imports `deep_reasoner.core.configure_structlog_fixture`. The vendored
  `deep-reasoner/` already provides it, so no change is needed there.
- Keep the jupytext `py:percent` headers and `if test():` blocks, as in the rest of
  `src/`.

### 3. Configs
- Copy `configs/token_matched/` (28 files plus README). Each file is self-contained,
  with prompts and numbers inlined and no `_compose`.
- Leave out upstream `configs/baselines/`, the 42 unmatched configs that re-run the
  original baselines through the unified path, and `configs/base/`, the fragments that
  only those configs compose. Sample-k's attempt 0 (`accuracy_vs_k` at j=1) already gives
  that single-attempt number on the same code path. The `_compose` loader stays in
  `cli.py` as part of the verbatim port.
- Check each config: model paths, `api_base`/port and SLURM resources must use env
  vars or defaults, never cluster paths.

### 4. Scripts
- Port the `--token-matched-config` branch of `run_baseline.sh`: auto-assigned vLLM
  port, API key from config, eager/`deep_gemm` fallback, and torch.compile disabled at
  TP>1.
- Port the `--token-matched <glob>` branch of `sbatch_baselines.sh`, including the
  per-config resources and the dry-run task/attempt breakdown.
- Remove site-specific module loads and scratch paths. Keep `CLUSTER_MODULES` as an
  optional, empty-by-default env var.

### 5. Analysis
- Add `analysis/token_matched.py` (imports → `dolores.unified.benchmarks`, `config`)
  and `dr_vs_baseline_token_ratios.csv`, the input every config's numbers cite.
- Code only, no result files: `python -m analysis.token_matched -r '<log glob>'`
  regenerates the majority@k / best@k / budget tables from a user's own run logs.

### 6. Dependencies (`pyproject.toml`)
- Add `conflit`, `parse` and `coolname` (`coolname` currently arrives only
  transitively, through `deep-reasoner`).
- Add `dolores.unified` and its subpackages to `[tool.setuptools] packages`.
- Pre-existing gap worth fixing in the same PR: `smolagents`, `rlms`, `transformers`
  and `huggingface-hub` are imported by `src/baselines/` but not declared.

### 7. Docs
- `reproduction.md`: add a **Token-matched runs** chapter in the same shape as the
  others (dry run, then 5-task limit, then full). Cover both arms, the `--token-budget`
  and `--samples` smoke overrides, and the post-hoc
  `python -m analysis.token_matched` step.
- `README.md`: add `src/dolores/unified/` and `configs/token_matched/` to the layout tree.

### 8. Tests and verification (no GPU needed)
- `pytest` over `src/` and `analysis/`. The ported tests include the ones that check
  every token-matched config against the ratio CSV, and every inlined prompt against
  `prompts.py`.
- `python -m dolores.unified.cli run --config configs/token_matched/phantomwiki_react_try_hard.yaml --dry-run`
  for one config per benchmark.
- `bash scripts/sbatch_baselines.sh --token-matched 'configs/token_matched/*.yaml' --dry-run`
  must list 28 jobs and the expected attempt counts.
- Legacy-path regression: the reproduction.md dry runs for `dolores.experiment` and
  `run_baseline.sh` behave the same as before.

### 9. Anonymisation pass (blocking, since this repo is public)
Before pushing each step, grep the diff for organisation or user names, emails, cluster
hostnames, scratch paths, internal doc links, and upstream repo or PR URLs. Upstream
`RUNS.md` and `scripts/` contain several of these. Configs and code comments should be
checked too.

## Suggested commit sequence

1. shared-module additions (`config`, `core`) + deps
2. `src/dolores/unified/` port (imports rewritten, otherwise verbatim)
3. `configs/token_matched/` + ratio CSV
4. scripts
5. `analysis/token_matched.py`
6. `reproduction.md` + README; delete this plan

## Decisions

1. **Source:** upstream merge commit `ed05708` (token-matched work + smoke/SLURM fixes).
2. **Package:** `src/dolores/unified/`, run as `python -m dolores.unified.cli`.
3. **Unmatched reference configs** (`configs/baselines/`, `configs/base/`): not ported (see §3).
4. **Results:** code to regenerate only; no result CSVs or figures.
5. **Legacy baselines:** `src/baselines/*.py` stays; `reproduction.md` says which paper
   result each path produces.
