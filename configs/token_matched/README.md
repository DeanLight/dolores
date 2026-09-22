# Token-matched run configs (Qwen3-32B)

Every runnable cell, both arms: **14 cells × 2 arms = 28 configs**, named
`<benchmark>_<baseline>_<arm>.yaml` so the arm is obvious from the filename.

Two of the 16 grid squares are missing on purpose: **ReAct and Deep Research
cannot run Oolong**, whose long-context documents exceed their context window.

PhantomWiki is **size 500 only** — the size-50 and size-5000 variants are no
longer used for evals, so they get no token-matched configs.

| arm | filename suffix | what it does |
|---|---|---|
| **try hard** | `_try_hard` | One run per task with a `2 × k_b` token budget, turns effectively uncapped, and a don't-give-up instruction. Ends when the budget is spent, returning its best answer so far. |
| **sample-k** | `_samplek` | `⌈k_b⌉` independent attempts per task, all kept, aggregated post hoc into majority@k and best@k. |

`k_b` = Deep Reasoner's total tokens ÷ this baseline's total tokens on this
benchmark (Qwen3-32B, from `memory_thread_io_simple_summary`). It is per-cell —
one square of the benchmark × baseline grid — not one global multiplier.

The two arms are **separate runs of the same cell**; a config sets one or the
other, never both. Budgeting a run *and* sampling it k times would spend
`k × 2 × k_b`, which is a match with nothing.

## The numbers

Both come from [`analysis/dr_vs_baseline_token_ratios.csv`](../../analysis/dr_vs_baseline_token_ratios.csv),
the measured per-cell token ratios on Qwen3-32B. **Every cell has one, so all 28
configs are ready to run** — nothing is gated any more.

`token_budget` is twice Deep Reasoner's own mean tokens per task on that
benchmark (733,155 PhantomWiki · 88,421 SynthWorlds · 2,775,366 Oolong ·
205,014 DeepSearchQA) — the same quantity as `2 × k_b ×` the baseline's mean, so
it is one figure per benchmark regardless of which baseline is budgeted.

`samples` is `⌈k_b⌉`, rounded up so a cell never spends *less* than Deep Reasoner.

A test asserts every config still matches the CSV, so the configs cannot drift
from the measurement they cite.

## Prompts

Each config inlines the **whole prompt** that cell runs with under
`instructions:` — the benchmark instructions verbatim from `prompts.py`, plus the
don't-give-up paragraph for `_try_hard`. `cat` the file and you see exactly what
ran; a test asserts the inlined text still matches `prompts.py`.

The 6 runnable cells on **Oolong and DeepSearchQA carry no instructions, by
design** — per the author's provenance table (reproduced in `prompts.py`), those
cells run on their framework's default prompt. That is a deliberate choice, not a
gap, and the configs say which default they fall back to.

**Deep Research configs carry two prompts.** `instructions` is the short
orchestrator one; `subagent_instructions` is the ReAct translation the sub-agent
runs, which is the substantive prompt for that cell. **RLM configs carry the
benchmark half only** — rlm's own `RLM_SYSTEM_PROMPT` still leads, exactly as it
does without a config.

## The unmatched baselines, for comparison

`configs/baselines/<model>/<benchmark>_<baseline>.yaml` — 42 configs (the same 14
cells × 3 models) mirroring the pre-token-matched runs: no budget, one attempt
per task, default turn cap. Composed from `configs/base/` fragments via
`_compose`, so each prompt is written once.
