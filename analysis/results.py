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
# # Results Viz
#
# Let's makle things prettyyyy!

# %%
from juplit import test

# %%
# If True, append a "score" field to each result JSON as it's scored.
save = False

# %%
import glob
import json
import pandas as pd
from benchmarks import phantomwiki
from config import RunContext, Paths

# %%
# Paper-table column order; missing entries render as "-".

def latex_rows(df, metric, digits=3, extra=()):
    """Print one LaTeX-formatted row per (model, *extra) combo, methods in ORDER."""
    keys = ["model", *extra]
    table = df.groupby([*keys, "method"])[metric].mean()
    combos = df[keys].drop_duplicates().sort_values(keys).itertuples(index=False, name=None)
    for combo in combos:
        print(f"% {' / '.join(f'{k}={v}' for k, v in zip(keys, combo))}")
        row = []
        for m in ORDER:
            try:
                row.append(f"{table.loc[(*combo, m)]:.{digits}f}")
            except KeyError:
                row.append("-")
        print(" & ".join(row))

# %% [markdown]
# ## PhantomWiki

# %%
models = [
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
]

configs = [(50, 1), (500, 1), (5000, 1)]

# %% [markdown]
# ### Load scores

# %%
methods = {
    # label          slug             glob pattern         models
    "ReAct":        ("react",        "*.json",        models),
    "CodeAct":      ("codeact",      "*.json",        models),
    "Deep Research": ("deepresearch", "*/result.json", ["hosted_vllm/"+model for model in models]),
    "Deep Reasoner (ours)": ("deepreasoner", "*/qa.json", models),
    "RLM":          ("rlm",           "*.json", models),
}

# %%
if test():
    rows = []
    for method_label, (method_slug, glob_pat, method_models) in methods.items():
        for model in method_models:
            for size, seed in configs:
                benchmark = f"phantomwiki_{size}_{seed}"
                log_dir = RunContext.get_log_dir(benchmark, method_slug, model)
                for file in glob.glob(f"{log_dir}/{glob_pat}"):
                    with open(file) as f:
                        data = json.load(f)
                    test_id = data.get("test_id") or data.get("example_id") or data.get("task_id")
                    answer = data.get("parsed") or data.get("parsed_output") or []
                    f1, em = phantomwiki.score(size, seed, test_id, answer)

                    model_str = model.split("/")[-1]
                    if data.get("no_thinking") and not model_str.endswith("-nothink"):
                        model_str += "-nothink"

                    if save:
                        data["score"] = f1
                        with open(file, "w") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)

                    rows.append({
                        "method":  method_label,
                        "model":   model_str,
                        "size":    size,
                        "seed":    seed,
                        "test_id": test_id,
                        "f1":      f1,
                        "em":      em,
                    })

    df_pw = pd.DataFrame(rows)
    print(f"{len(df_pw)} scored examples across {df_pw['method'].nunique()} methods, {df_pw['model'].nunique()} models")
    df_pw.head()

# %%
if test():
    len(df_pw)

# %% [markdown]
# # Oolong

# %%
import re
from benchmarks import oolong
from config import RunContext
import glob
import json
from functools import partial
import pandas as pd

# %%
if test():
    ds = oolong._load_dataset()

# %% [markdown]
# ## Load scores

# %%
benchmark = "oolong-real"

# List of tuples (not a dict!) so the same method label can appear with multiple models.
oolong_methods = [
    # label           slug             glob pattern  model
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "meta-llama/Llama-3.3-70B-Instruct"),
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "Qwen/Qwen3-32B"),
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "Qwen/Qwen3-8B"),
    ("RLM",          "rlm",           "*.json",     "meta-llama/Llama-3.3-70B-Instruct"),
    ("RLM",          "rlm",           "*.json",     "Qwen/Qwen3-32B"),
    # ("RLM",          "rlm",           "*.json",     "Qwen/Qwen3-32B-nothink"),
    ("RLM",          "rlm",           "*.json",     "Qwen/Qwen3-8B"),
    ("CodeAct",     "codeact",       "*.json",     "Qwen/Qwen3-8B"),
    ("CodeAct",      "codeact",       "*.json",     "Qwen/Qwen3-32B"),
    ("CodeAct",      "codeact",       "*.json",     "meta-llama/Llama-3.3-70B-Instruct"),
]

# %%
if test():
    rows = []
    for method_label, method_slug, glob_pat, model in oolong_methods:
        log_dir = RunContext.get_log_dir(benchmark, method_slug, model)
        for file in glob.glob(f"{log_dir}/{glob_pat}"):
            with open(file) as f:
                data = json.load(f)
            ex_id = data.get("test_id") or data.get("example_id") or data.get("ex_id") or data.get("task_id")
            ex = ds.get(ex_id, {})
            episodes = ex.get("episodes") or []
            answer = data.get("parsed") or data.get("parsed_output")

            model_str = model.split("/")[-1]
            if data.get("no_thinking") and not model_str.endswith("-nothink"):
                model_str += "-nothink"

            if save:
                s = oolong.score(ex_id, answer, scorer=oolong.relaxed_accuracy)
                data["score"] = s
                with open(file, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            rows.append({
                "method":        method_label,
                "model":         model_str,
                "id":            ex_id,
                "question_type": ex.get("question_type"),
                "num_episodes":  len(episodes),
                "prediction":    answer,
                "gold_answer":   ex.get("answer"),
            })

    df_oo = pd.DataFrame(rows)
    print(f"{len(df_oo)} rows across {df_oo['method'].nunique()} methods, {df_oo['model'].nunique()} models")
    df_oo.groupby(["method", "model"]).size()

# %%
def score(row, scorer=None):
    if row["prediction"] is None:
        return None
    return oolong.score(row["id"], row["prediction"], numeric_scorer=scorer)

# %%
if test():
    df_oo["score"]         = df_oo.apply(lambda r: score(r, scorer=None),                    axis=1)
    df_oo["score_relaxed"] = df_oo.apply(lambda r: score(r, scorer=oolong.relaxed_accuracy), axis=1)

# %%
if test():
    len(df_oo)

# %% [markdown]
# # SynthWorlds

# %%
from benchmarks import synthworlds

benchmark = "synthworlds"

# %%
if test():
    qa_index = synthworlds._load_qa_index()

# %% [markdown]
# ## Load scores

# %%
# List of tuples so the same method label can appear with multiple models.
sw_methods = [
    # label           slug             glob pattern      model
    ("ReAct",        "react",         "*.json",         "Qwen/Qwen3-8B"),
    ("ReAct",        "react",         "*.json",         "Qwen/Qwen3-32B"),
    ("ReAct",        "react",         "*.json",         "meta-llama/Llama-3.3-70B-Instruct"),
    ("CodeAct",      "codeact",       "*.json",         "Qwen/Qwen3-8B"),
    ("CodeAct",      "codeact",       "*.json",         "Qwen/Qwen3-32B"),
    # ("CodeAct",      "codeact",       "*.json",         "Qwen/Qwen3-32B-nothink"),
    ("CodeAct",      "codeact",       "*.json",         "meta-llama/Llama-3.3-70B-Instruct"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/Qwen/Qwen3-8B"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/Qwen/Qwen3-32B"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"),
    ("RLM", "rlm",  "*.json",  "Qwen/Qwen3-8B"),
    ("RLM", "rlm",  "*.json",  "Qwen/Qwen3-32B"),
    ("RLM", "rlm",  "*.json",  "meta-llama/Llama-3.3-70B-Instruct"),
    ("Deep Reasoner (ours)", "deepreasoner", "*/qa.json", "Qwen/Qwen3-8B"),
    ("Deep Reasoner (ours)", "deepreasoner", "*/qa.json", "Qwen/Qwen3-32B"),
    ("Deep Reasoner (ours)", "deepreasoner", "*/qa.json", "meta-llama/Llama-3.3-70B-Instruct"),
]

# %%
if test():
    rows = []
    for method_label, method_slug, glob_pat, model in sw_methods:
        log_dir = RunContext.get_log_dir(benchmark, method_slug, model)
        for file in glob.glob(f"{log_dir}/{glob_pat}"):
            with open(file) as f:
                data = json.load(f)
            test_id = data.get("test_id") or data.get("example_id") or data.get("task_id")
            answer = str(data.get("answer", ""))
            #answer = data.get("answer") or ''
            f1, em = synthworlds.score(test_id, answer)

            model_str = model.split("/")[-1]
            if data.get("no_thinking") and not model_str.endswith("-nothink"):
                model_str += "-nothink"

            if save:
                data["score"] = f1
                with open(file, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            rows.append({
                "method":     method_label,
                "model":      model_str,
                "test_id":    test_id,
                "answer":     answer,
                "graph_type": qa_index[test_id]["graph_type"],
                "f1":         f1,
                "em":         em,
            })

    df_sw = pd.DataFrame(rows)
    print(f"{len(df_sw)} scored examples across {df_sw['method'].nunique()} methods, {df_sw['model'].nunique()} models")
    df_sw.groupby(["method", "model"]).size()

# %% [markdown]
# # DeepSearchQA

# %%
from benchmarks import deepresearchqa

benchmark = "DeepResearchQA"

# %% [markdown]
# ## Load scores

# %%
# List of tuples so the same method label can appear with multiple models.
drqa_methods = [
    # label           slug             glob pattern      model
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "meta-llama/Llama-3.3-70B-Instruct"),
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "Qwen/Qwen3-32B"),
    ("Deep Reasoner (ours)", "deepreasoner",  "*/qa.json",   "Qwen/Qwen3-8B"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/Qwen/Qwen3-8B"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/Qwen/Qwen3-32B"),
    # ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/Qwen/Qwen3-32B-nothink"),
    ("Deep Research", "deepresearch",  "*/result.json",  "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"),
    ("RLM", "rlm",  "*/result.json",  "Qwen/Qwen3-8B"),
    ("RLM", "rlm",  "*/result.json",  "Qwen/Qwen3-32B"),
    ("RLM", "rlm",  "*/result.json",  "meta-llama/Llama-3.3-70B-Instruct"),
    ("ReAct", "react",  "*/result.json",  "Qwen/Qwen3-8B"),
    ("ReAct", "react",  "*/result.json",  "Qwen/Qwen3-32B"),
    ("ReAct", "react",  "*/result.json",  "meta-llama/Llama-3.3-70B-Instruct"),
    ("CodeAct", "codeact",  "*/result.json",  "Qwen/Qwen3-8B"),
    ("CodeAct", "codeact",  "*/result.json",  "Qwen/Qwen3-32B"),
    ("CodeAct", "codeact",  "*/result.json",  "meta-llama/Llama-3.3-70B-Instruct"),
]

# %%
if test():
    # Judge (once): for each (method, model), find files missing judge_score,
    # batch-judge them, write judge_score + judge_reasoning back to disk.
    # Skips files that already have judge_score (idempotent).
    for method_label, method_slug, glob_pat, model in drqa_methods:
        log_dir = RunContext.get_log_dir(benchmark, method_slug, model)

        pending = []  # (path, data, test_id, answer)
        for path in glob.glob(f"{log_dir}/{glob_pat}"):
            with open(path) as f:
                data = json.load(f)
            if "judge_score" in data:
                continue
            test_id = data.get("test_id") or data.get("example_id") or data.get("task_id")
            answer = data.get("answer", "")
            pending.append((path, data, test_id, answer))

        tag = f"[{method_label} / {model.split('/')[-1]}]"
        if not pending:
            print(f"{tag} nothing to judge")
            continue

        print(f"{tag} judging {len(pending)} files...")
        pairs = [(tid, ans) for _, _, tid, ans in pending]
        results = deepresearchqa.score_judge_batch(pairs)
        by_id = {tid: (score, reasoning) for tid, score, reasoning in results}

        for path, data, tid, _ in pending:
            score, reasoning = by_id[tid]
            data["judge_score"] = score
            data["judge_reasoning"] = reasoning
            with open(path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"{tag} wrote {len(pending)} files")

# %%
def em_score(answer, gold):
    return 1.0 if str(answer).strip().lower() == str(gold).strip().lower() else 0.0

# %%
if test():
    rows = []
    for method_label, method_slug, glob_pat, model in drqa_methods:
        log_dir = RunContext.get_log_dir(benchmark, method_slug, model)
        tag = f"[{method_label} / {model.split('/')[-1]}]"
        for file in glob.glob(f"{log_dir}/{glob_pat}"):
            with open(file) as f:
                data = json.load(f)
            test_id = data.get("test_id") or data.get("example_id") or data.get("task_id")
            pred = data.get("answer", "")
            gold = deepresearchqa.get_answer(test_id)
            judge_score = data.get("judge_score")
            judge_reasoning = data.get("judge_reasoning")

            model_str = model.split("/")[-1]
            if data.get("no_thinking") and not model_str.endswith("-nothink"):
                model_str += "-nothink"

            if save:
                data["score"] = judge_score
                with open(file, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            rows.append({
                "method":       method_label,
                "model":        model_str,
                "test_id":      test_id,
                "prediction":   pred,
                "gold_answer":  gold,
                "judge_score":  judge_score,
                "em":           em_score(pred, gold),
            })

    df_drqa = pd.DataFrame(rows)
    df_drqa = df_drqa.drop_duplicates(subset=["method", "model", "test_id"], keep="last")

    print(f"\n{len(df_drqa)} scored examples across {df_drqa['method'].nunique()} methods, {df_drqa['model'].nunique()} models")
    df_drqa.groupby(["method", "model"]).size()

# %% [markdown]
# ## Save

# %%
if test():
    # Save dataframes for quick plotting later
    analysis_dir = Paths.ANALYSIS_DIR
    df_pw.to_csv(f"{analysis_dir}/phantomwiki_results.csv", index=False)
    df_oo.to_csv(f"{analysis_dir}/oolong_results.csv", index=False)
    df_sw.to_csv(f"{analysis_dir}/synthworlds_results.csv", index=False)
    df_drqa.to_csv(f"{analysis_dir}/deepsearchqa_results.csv", index=False)

# %% [markdown]
# # Analysis

# %%
import glob
import json
import pandas as pd
from benchmarks import phantomwiki
from config import RunContext, Paths

# %%
if test():
    #analysis_dir = Paths.ANALYSIS_DIR
    analysis_dir = Paths.ANALYSIS_DIR

    # Optional: Run these lines to load them without rerunning everything above
    df_pw = pd.read_csv(f"{analysis_dir}/phantomwiki_results.csv")
    df_oo = pd.read_csv(f"{analysis_dir}/oolong_results.csv")
    df_sw = pd.read_csv(f"{analysis_dir}/synthworlds_results.csv")
    df_drqa = pd.read_csv(f"{analysis_dir}/deepsearchqa_results.csv")

# %%
ORDER = ["ReAct", "CodeAct", "Deep Research", "RLM", "Deep Reasoner (ours)"]

# %%
# Paper-table column order; missing entries render as "-".

def latex_rows(df, metric, digits=3, extra=()):
    """Print one LaTeX-formatted row per (model, *extra) combo, methods in ORDER."""
    keys = ["model", *extra]
    table = df.groupby([*keys, "method"])[metric].mean()
    combos = df[keys].drop_duplicates().sort_values(keys).itertuples(index=False, name=None)
    for combo in combos:
        print(f"% {' / '.join(f'{k}={v}' for k, v in zip(keys, combo))}")
        row = []
        for m in ORDER:
            try:
                row.append(f"{table.loc[(*combo, m)]:.{digits}f}")
            except KeyError:
                row.append("-")
        print(" & ".join(row))

# %% [markdown]
# # Tables

# %%
if test():
    # PhantomWiki
    display(df_pw.groupby(["model", "method"]).size())

    summary = (
        df_pw.groupby(["method", "model", "size", "seed"])
          .agg(N=("f1", "size"), F1=("f1", "mean"), EM=("em", "mean"))
          .reset_index()
          .sort_values(["model", "size", "seed", "method"])
    )
    # display(summary[(summary['model'] == "Qwen3-32B") & (summary['size'] == 500) & (summary['seed'] == 1)][['method', 'F1']])
    # display(summary[(summary['model'] == "Llama-3.3-70B-Instruct") & (summary['size'] == 500) & (summary['seed'] == 1)][['method', 'F1']])
    latex_rows(df_pw, "f1", extra=("size", "seed"))

# %%
if test():
    # Oolong
    display(df_oo.groupby(["model", "method"]).size())

    # display(df_oo.groupby(["method", "model"])[["score", "score_relaxed"]].agg(["mean"]).round(5))
    # display(df_oo.groupby(["num_episodes", "method", "model"])["score"].mean().unstack(["method", "model"]))
    latex_rows(df_oo, "score_relaxed")

# %%
if test():
    # SynthWorlds
    display(df_sw.groupby(["model", "method"]).size())

    result = (df_sw.groupby(["method", "model"])[["f1", "em"]].mean()).round(3)
    # display(result.swaplevel().sort_index())

    summary_sw = df_sw.groupby(["method", "model", "graph_type"])[["f1", "em"]].mean().unstack("graph_type") * 100
    summary_sw[("f1", "Average")] = df_sw.groupby(["method", "model"])["f1"].mean() * 100
    summary_sw[("em", "Average")] = df_sw.groupby(["method", "model"])["em"].mean() * 100
    # display(summary_sw.round(1))

    latex_rows(df_sw, "f1")

# %%
if test():
    # DeepSearchQA
    display(df_drqa.groupby(["model", "method"]).size())

    # display(df_drqa.groupby(["model", "method"])["judge_score"].mean().round(3))
    # display(df_drqa.groupby(["method", "model"])[["judge_score", "em"]].agg(["mean", "count"]).round(4))
    latex_rows(df_drqa, "judge_score")

# %% [markdown]
# # Paper Table
#
# Combined LaTeX table: per model, one row per benchmark (SynthWorlds → PhantomWiki → DeepResearchQA → Oolong); columns are methods in `ORDER`.

# %%
# Paper table: rows = (model, benchmark), columns = methods (in ORDER).
# with_sem=True appends "$\pm$ <sem>" to each cell. Models grouped; \midrule between models.

def paper_table_rows(digits=3, sem_digits=3, with_sem=False, pw_size=500, pw_seed=1):
    model_order = ["Qwen3-8B", "Qwen3-32B", "Llama-3.3-70B-Instruct"]
    benchmarks = [
        ("SynthWorlds",    df_sw,                                                          "f1"),
        ("PhantomWiki",    df_pw[(df_pw["size"] == pw_size) & (df_pw["seed"] == pw_seed)], "f1"),
        ("DeepResearchQA", df_drqa,                                                        "judge_score"),
        ("Oolong",         df_oo,                                                          "score_relaxed"),
    ]
    for mi, model in enumerate(model_order):
        if mi > 0:
            print(r"\midrule")
        print(f"% ===== {model} =====")
        for bench_name, df, metric in benchmarks:
            sub = df[df["model"] == model]
            grp = sub.groupby("method")[metric]
            means, sems = grp.mean(), grp.sem()
            cells = []
            for m in ORDER:
                if m in means.index and pd.notna(means[m]):
                    mu = f"{means[m]:.{digits}f}"
                    if with_sem:
                        se = sems[m] if (m in sems.index and pd.notna(sems[m])) else 0.0
                        cells.append(f"{mu} $\\pm$ {se:.{sem_digits}f}")
                    else:
                        cells.append(mu)
                else:
                    cells.append("-")
            print(f"{bench_name:14s} & " + " & ".join(cells) + r" \\")

# %%
if test():
    paper_table_rows()

# %%
if test():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter

    # Set font sizes to 10 everywhere (except smaller y-ticks) and family
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'sans-serif'],
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 8,
        'hatch.linewidth': 0.5,  # Make the lines of the inside shapes thinner
    })

    # Or alternatively you can uncomment this to go full XKCD style mode:
    # plt.xkcd()

    # Professional Colorbrewer Set1 palette + denser hatches to make shapes smaller.
    COLORS = {
        "ReAct":                "#377eb8",
        "CodeAct":              "#ff7f00",
        "RLM":                  "#4daf4a",
        "Deep Research":        "#984ea3",
        "Deep Reasoner (ours)": "#e41a1c",
    }
    HATCHES = {
        "ReAct":                "////",
        "CodeAct":              "\\\\\\\\",
        "RLM":                  "xxxx",
        "Deep Research":        "....",
        "Deep Reasoner (ours)": "", # Solid fill to stand out (removed the "++" squares)
    }

    MODEL = "Qwen3-32B"

    panels = [
        ("PhantomWiki",            df_pw[(df_pw["model"] == MODEL) & (df_pw["size"] == 500)], "f1"),
        ("Oolong (real)",                 df_oo[df_oo["model"] == MODEL],                   "score_relaxed"),
        ("SynthWorlds",            df_sw[df_sw["model"] == MODEL],                   "f1"),
        ("DeepSearchQA",         df_drqa[df_drqa["model"] == MODEL],               "judge_score"),
    ]

    # NeurIPS text width is exactly 5.5 inches.
    # Reducing height to shorten the bars.
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.35))

    for idx, (ax, (title, data, metric)) in enumerate(zip(axes, panels)):
        means = data.groupby("method")[metric].mean()
        sems = data.groupby("method")[metric].sem()
        # Explicitly plot for every method in the exact defined ORDER
        for i, m in enumerate(ORDER):
            if m in means.index and pd.notna(means[m]):
                hatch = HATCHES[m] if HATCHES[m] else None
                err = sems[m] if (m in sems.index and pd.notna(sems[m])) else 0
                ax.bar(i * 1.3, means[m], width=0.65, yerr=err,
                       error_kw={'capsize': 1.5, 'elinewidth': 0.4, 'capthick': 0.4, 'ecolor': '#555555'},
                       color=COLORS[m], alpha=0.65, hatch=hatch,
                       edgecolor="black", linewidth=0.4)
            else:
                # Draw N/A text for missing methods at y=0.05
                ax.text(i * 1.3, 0.05, "N/A", ha="center", va="bottom", rotation=90,
                        color="gray", fontsize=10, transform=ax.get_xaxis_transform())

        # Ensure axes limits correctly encompass all methods
        ax.set_xticks([i * 1.3 for i in range(len(ORDER))])
        ax.set_xlim(-0.8, (len(ORDER) - 1) * 1.3 + 0.8)
        ax.set_xticklabels([])
        ax.set_title(title, pad=4)

        # Format y-axis to always show 2 decimal places
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Only set a unified Y-label on the very first, leftmost plot
        if idx == 0:
            ax.set_ylabel("Score", labelpad=2, y=0.4)

        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1,
                             facecolor=COLORS[m], alpha=0.65, hatch=HATCHES[m] if HATCHES[m] else None,
                             edgecolor="black", linewidth=0.4)
               for m in ORDER]

    # Add a single shared x-axis label centering over all subplots
    fig.supxlabel("Scaffolds", fontsize=10)

    # Legend split into 2 rows (ncol=3) to fit cleanly in 5.5 inches
    legend = fig.legend(handles, ORDER,
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.53, 1.42), frameon=False, columnspacing=1.0)

    # Bold the target method in the legend
    for text in legend.get_texts():
        if "Deep Reasoner (ours)" in text.get_text():
            text.set_weight("bold")

    # Auto-adjust without arbitrary bottom margins
    fig.tight_layout(rect=[0, 0, 1, 1.0], pad=0.2, w_pad=0.5)
    plt.savefig(f"{analysis_dir}/qwen3-32b-section1-teaser.pdf", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## End
