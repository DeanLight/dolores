# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RLM cognitive-load analysis (per memory thread)
#
# Designed to run with **only the saved RLM logs** + the model's HF tokenizer. No live model needed.
#
# For each saved experiment (one JSON file = one root `rlm.completion(...)` call), this notebook emits a `pandas.DataFrame` with **one row per memory thread**. Each row carries that thread's output tokens, visible tokens, and an estimate of its reasoning tokens. Aggregate however you want from there.

# %%
from juplit import test

# %% [markdown]
# ## Definitions
#
# ### Memory thread
#
# A **memory thread** is one discrete model-context. There are exactly two kinds:
#
# 1. **`rlm_instance`** — one RLM agent (root or any sub-RLM spawned via `rlm_query`). Its context is the accumulated transcript across the agent's iteration loop. Cognitive load = output tokens emitted across those iteration LLM calls **only** (excludes any `llm_query` calls the agent issued — those are their own threads).
#    - `depth = 0` for the root, `+1` for each `rlm_query` recursion level.
# 2. **`llm_query`** — one `llm_query(...)` or `llm_query_batched([...])` call. Each prompt the LLM sees is a fresh one-shot context with no agent state. Each prompt = its own thread. **Standalone — no depth.**
#
# So an RLM agent's cognitive load and the cognitive load of any `llm_query` it issues are bookkept **separately**, never bundled.
#
# ### Recipe (from saved JSON)
#
# For each `rlm_instance` thread (a node with `metadata is not None`):
#
#     in_iter        = thread.usage_summary.total_input_tokens  − Σ (llm_query records' total_input_tokens)
#     out_iter       = thread.usage_summary.total_output_tokens − Σ (llm_query records' total_output_tokens)
#     visible_iter   = Σ tokenize(metadata.iterations[i].response)
#     reasoning_iter = max(0, out_iter − visible_iter)
#
# For each `llm_query` thread (one record in `code_blocks[j].result.rlm_calls[k]` with `metadata is None`):
#
#     in        = record.usage_summary.total_input_tokens
#     out       = record.usage_summary.total_output_tokens
#     visible   = tokenize(record.response)
#     reasoning = max(0, out − visible)
#
# `rlm_query` records (same slot, `metadata is not None`) recurse — each becomes its own `rlm_instance` row at depth `+1`.
#
# ### Why this works
#
# - Each RLM agent runs with its own `OpenAIClient` (`rlm/core/rlm.py:_spawn_completion_context`), so its `usage_summary` aggregates only its own LLM calls. Subtracting its `llm_query` records isolates the agent's own iteration tokens.
# - vLLM `--reasoning-parser qwen3` puts `<think>` content in `message.reasoning_content` (dropped by the rlm client) but counts it in `usage.completion_tokens`. The asymmetry is what `out − visible` recovers.
#
# ### Caveats
#
# - **`llm_query_batched` race-stamping**: the same `usage_summary` object is stamped on every record in a batch (`rlm/core/lm_handler.py:_handle_batched`), so per-record `total_*_tokens` is the last-finisher's value, not per-prompt truth. We emit one row per record anyway, accepting per-row bias.
# - **Negative `rlm_instance` rows are normal.** When a batch's stamp values sum to MORE than the agent's true client total (e.g., a 50-prompt batch stamped at the slowest finisher), the subtraction `thread_out − Σ records` goes negative. **We don't clip** — the negative value is the mathematical adjustment that keeps conservation holding (`Σ df.in_tokens` and `Σ df.out_tokens` always equal the experiment's true totals; see Sanity check 3). Treat a negative rlm_instance as a flag: this experiment had heavy batched-stamping bias.
# - `reasoning_est` IS clipped at `max(0, …)` because negative reasoning is meaningless. Per-row reasoning numbers may be biased by the same race; they balance out at the experiment level for `out_tokens` but not for the reasoning estimate (since visible isn't subtracted from llm_query rows back into rlm_instance rows).
# - Small per-call over-estimate of reasoning by ~2–6 tok: `<think>` / `</think>` tags and EOS tokens are counted by vLLM but absent from saved text.

# %%
import json, glob, random
from pathlib import Path
from functools import lru_cache
import pandas as pd
from transformers import AutoTokenizer
from config import Paths

from config import Paths

# Point at any directory containing RLM saved JSONs.
# Flat layout (oolong / phantomwiki / synthworlds): <stem>.json
# Folder layout (deepresearchqa):                    <stem>/result.json
LOG_DIR = Paths.LOGS_DIR / "oolong-real" / "rlm" / "Qwen-Qwen3-32B"

def find_rlm_files(d: Path) -> list[Path]:
    return sorted([Path(p) for p in glob.glob(str(d / "*.json"))]) \
        or sorted([Path(p) for p in glob.glob(str(d / "*/result.json"))])

# %%
if test():
    files = find_rlm_files(LOG_DIR)
    print(f"{len(files)} RLM files in {LOG_DIR}")

# %%
@lru_cache(maxsize=8)
def get_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def visible(text: str, tokenizer) -> int:
    """Tokenized length of saved visible text (no chat-template wrap)."""
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def experiment_threads_df(data: dict) -> pd.DataFrame:
    """One row per memory thread for a single experiment (saved RLMChatCompletion dict).

    Columns:
      kind            'rlm_instance' or 'llm_query'
      depth           int for rlm_instance (root=0); pd.NA for llm_query (standalone)
      n_calls         iter-count for rlm_instance; 1 for llm_query
      in_tokens       input tokens attributable to this thread only
      out_tokens      output tokens (incl. thinking) attributable to this thread only
      visible_tokens  Σ tokenize of the thread's saved response text(s)
      reasoning_est   max(0, out_tokens - visible_tokens)

    NOTE: rlm_instance.in_tokens / .out_tokens may be NEGATIVE when llm_query_batched
    over-stamps per-record totals (each record in a batch carries the last-finisher's
    token count, so Σ records can exceed the agent's true client total). We do NOT
    clip them — that way Σ df.in_tokens / Σ df.out_tokens always equal the
    experiment's true totals (Sanity check 3). A negative rlm_instance row is the
    diagnostic signal that batched stamping affected this experiment.

    Reasoning_est IS clipped at 0 since negative reasoning is meaningless.
    """
    tok = get_tokenizer(data["model"])
    rows = []

    def visit(node, depth):
        ms = node.get("usage_summary", {}).get("model_usage_summaries", {})
        thread_in  = sum(s["total_input_tokens"]  for s in ms.values())
        thread_out = sum(s["total_output_tokens"] for s in ms.values())

        llm_query_in_sum  = 0
        llm_query_out_sum = 0
        iter_visible_sum = 0
        n_iter = 0
        md = node.get("metadata") or {}

        for it in md.get("iterations", []):
            n_iter += 1
            iter_visible_sum += visible(it.get("response", ""), tok)
            for cb in it.get("code_blocks", []):
                for sub in cb.get("result", {}).get("rlm_calls", []):
                    if not sub.get("metadata"):
                        # llm_query / llm_query_batched record — its own thread
                        sub_ms = sub.get("usage_summary", {}).get("model_usage_summaries", {})
                        sub_in  = sum(s["total_input_tokens"]  for s in sub_ms.values())
                        sub_out = sum(s["total_output_tokens"] for s in sub_ms.values())
                        sub_vis = visible(sub.get("response", ""), tok)
                        rows.append({
                            "kind":           "llm_query",
                            "depth":          pd.NA,
                            "n_calls":        1,
                            "in_tokens":      sub_in,
                            "out_tokens":     sub_out,
                            "visible_tokens": sub_vis,
                            "reasoning_est":  max(0, sub_out - sub_vis),
                        })
                        llm_query_in_sum  += sub_in
                        llm_query_out_sum += sub_out
                    else:
                        # sub-RLM (rlm_query) — recurse, becomes its own rlm_instance row
                        visit(sub, depth + 1)

        # The RLM-instance row for *this* node: only its own iteration cognitive load.
        # NOT clipped at 0 (see docstring). Conservation requires the raw signed value.
        instance_in  = thread_in  - llm_query_in_sum
        instance_out = thread_out - llm_query_out_sum
        rows.append({
            "kind":           "rlm_instance",
            "depth":          depth,
            "n_calls":        n_iter,
            "in_tokens":      instance_in,
            "out_tokens":     instance_out,
            "visible_tokens": iter_visible_sum,
            "reasoning_est":  max(0, instance_out - iter_visible_sum),
        })

    visit(data, 0)
    return pd.DataFrame(rows)

# %% [markdown]
# ## Sanity check 1 — random single file
#
# Load one random experiment, build its memory-thread dataframe, print it. Most files will be one row (root rlm_instance only — the model didn't call `rlm_query` and made no `llm_query` calls). That's fine; we look at richer ones in Sanity 2.

# %%
if test():
    path = random.choice(files)
    data = json.load(open(path))
    df = experiment_threads_df(data)

    print(f"file:  {path.name}")
    print(f"model: {data['model']}\n")
    df

# %% [markdown]
# ## Sanity check 2 — multi-thread example
#
# Find a file where the root agent actually called `rlm_query` (so we get sub-RLM rows at `depth >= 1`). Print its full dataframe and a summary by `kind`.

# %%
def has_subrlm(d):
    md = d.get("metadata") or {}
    for it in md.get("iterations", []):
        for cb in it.get("code_blocks", []):
            for sub in cb.get("result", {}).get("rlm_calls", []):
                if sub.get("metadata"):
                    return True
    return False

# %%
if test():
    multi = next((p for p in files if has_subrlm(json.load(open(p)))), None)
    print(f"multi-thread file: {multi.name if multi else 'NONE FOUND'}")

    if multi is not None:
        data = json.load(open(multi))
        df = experiment_threads_df(data)
        print(f"\n# rows: {len(df)}")
        print(f"\n--- by kind ---")
        print(df.groupby("kind")[["out_tokens", "visible_tokens", "reasoning_est"]].agg(["count", "mean", "sum"]).round(0))
        print(f"\n--- full df ---")
        df

# %% [markdown]
# ## Sanity check 3 — token conservation
#
# The `rlm_instance` rows + the `llm_query` rows should partition all output tokens with **no overlap and no gap**:
#
#     Σ df.out_tokens  ==  Σ (every nested rlm_instance node's usage_summary.total_output_tokens)
#
# This is true even when `llm_query_batched` was used: per-record stamps may be biased, but the bias inside each rlm_instance's subtraction is exactly cancelled by the same biased records being emitted as `llm_query` rows (we add back what we subtracted).

# %%
def all_instance_outs(node, key):
    """Σ usage_summary[key] across the root node and every sub-RLM (recursive)."""
    ms = node.get("usage_summary", {}).get("model_usage_summaries", {})
    total = sum(s[key] for s in ms.values())
    md = node.get("metadata") or {}
    for it in md.get("iterations", []):
        for cb in it.get("code_blocks", []):
            for sub in cb.get("result", {}).get("rlm_calls", []):
                if sub.get("metadata"):
                    total += all_instance_outs(sub, key)
    return total

# %%
if test():
    # Check on a wide sample. Conservation must hold for BOTH inputs and outputs.
    random.seed(42)
    candidates = random.sample(files, min(20, len(files)))
    if multi and multi not in candidates:
        candidates = [multi] + candidates

    mismatches = 0
    neg_rows = 0
    for path in candidates:
        data = json.load(open(path))
        df = experiment_threads_df(data)
        gi, ei = int(df["in_tokens"].sum()),  all_instance_outs(data, "total_input_tokens")
        go, eo = int(df["out_tokens"].sum()), all_instance_outs(data, "total_output_tokens")
        in_ok  = gi == ei
        out_ok = go == eo
        has_neg = ((df["in_tokens"] < 0) | (df["out_tokens"] < 0)).any()
        if not (in_ok and out_ok): mismatches += 1
        if has_neg: neg_rows += 1
        flag_neg = " (NEG: heavy batched)" if has_neg else ""
        print(f"{path.name:<40s}  rows={len(df):>3d}  Σin={gi:>8d}={ei:>8d} {'OK' if in_ok else 'X'}   "
              f"Σout={go:>8d}={eo:>8d} {'OK' if out_ok else 'X'}{flag_neg}")

    print(f"\n{mismatches} conservation mismatch / {len(candidates)} files.")
    print(f"{neg_rows} files with negative rlm_instance rows (batched-stamping diagnostic).")

# %% [markdown]
# ## Using the dataframe
#
# `experiment_threads_df(data)` is the building block. Aggregate however you want for a single experiment:
#
# ```python
# df = experiment_threads_df(data)
#
# # Average reasoning tokens per thread (mixed kinds)
# df["reasoning_est"].mean()
#
# # Per-kind means
# df.groupby("kind")["reasoning_est"].mean()
#
# # Just the RLM agents (root + sub-RLMs), broken down by depth
# df[df["kind"] == "rlm_instance"].groupby("depth")["reasoning_est"].mean()
# ```
#
# To aggregate across many experiments, concat them with an experiment id:
#
# ```python
# import pandas as pd
# parts = []
# for p in files:
#     d = json.load(open(p))
#     sub = experiment_threads_df(d)
#     sub["experiment"] = p.stem
#     parts.append(sub)
# all_df = pd.concat(parts, ignore_index=True)
# ```

# %%
if test():
    # Example: per-kind reasoning summary on the multi-thread file (or a random one)
    data = json.load(open(multi if multi else random.choice(files)))
    df = experiment_threads_df(data)
    df.groupby("kind")["reasoning_est"].agg(["count", "mean", "median", "sum"]).round(1)
