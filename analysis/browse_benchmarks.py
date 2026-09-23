# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from juplit import test

# %% [markdown]
# # Browse Benchmarks
#
# Browse and inspect tasks from each benchmark: Oolong, PhantomWiki, and SynthWorlds.
# Useful for understanding task format, difficulty, and expected output before running experiments.

# %%
import random

from datasets import load_dataset

from dolores.unified.benchmarks import DeepResearchQA, Oolong, PhantomWiki, SynthWorlds

# %% [markdown]
# ## Oolong
#
# Aggregation QA over long DnD game transcripts. Each task provides a long document
# (transcript, the `document` var) and a question about game statistics.

# %%
oolong = Oolong(limit=500, seed=42)
oolong_ids = oolong.list_task_ids()
print(f"Oolong tasks: {len(oolong_ids)}")
print(f"Sample IDs: {oolong_ids[:5]}")

# %%
# Browse 3 example tasks
rng = random.Random(0)
for task_id in rng.sample(oolong_ids, 3):
    task = oolong.get_task(task_id)
    print(f"task_id: {task_id}")
    print(f"question: {task.question}")
    print(f"expected: {task.gold}")
    print(f"document length: {len(task.vars['document']):,} chars")
    print()

# %% [markdown]
# ## PhantomWiki
#
# Multi-hop entity QA over a synthetic Wikipedia-like corpus. Tasks require chaining
# article lookups via `retrieve_article` and `search` tools. Answers are scored with F1 + EM.

# %%
SIZE, SEED = 500, 1
phantom = PhantomWiki(size=SIZE, seed=SEED)
phantom_ids = phantom.list_task_ids()
print(f"PhantomWiki tasks (size={SIZE}, seed={SEED}): {len(phantom_ids)}")
print(f"Sample IDs: {phantom_ids[:5]}")

# %%
# Browse 3 example tasks
for task_id in rng.sample(phantom_ids, 3):
    task = phantom.get_task(task_id)
    print(f"task_id: {task_id}")
    print(f"question: {task.question}")
    print(f"expected: {task.gold}")
    print(f"tools: {sorted(task.tools)}")
    print()

# %% [markdown]
# ## SynthWorlds
#
# Multi-hop QA over a synthetic world corpus of 6,290 documents. Tasks require
# iterative dense retrieval (`retrieve_top_5(query)`; Deep Reasoner's `search`).
# Answers are scored with F1 + EM. The retriever embeds queries with OpenAI
# `text-embedding-3-small` against the index built by
# `scripts/build_synthworlds_embeddings.py`.

# %%
synth_index = SynthWorlds._qa_index()
synth_ids = list(synth_index)
print(f"SynthWorlds tasks: {len(synth_ids)}")
print(f"Sample IDs: {synth_ids[:5]}")

# %%
# Browse 3 example tasks (read from the QA index, so no OpenAI key is needed)
for task_id in rng.sample(synth_ids, 3):
    row = synth_index[task_id]
    print(f"task_id: {task_id}")
    print(f"question: {row['query']}")
    print(f"expected: {row['gold_answers'][0]}")
    print()

# %%
# Peek at the document corpus
docs = [row["doc"] for row in load_dataset("kenqgu/SynthWorlds", "qa-sm-docs", split="test")]
print(f"Total documents: {len(docs)}")
print(f"\nSample document:\n{docs[0][:500]}")

# %% [markdown]
# ## DeepResearchQA
#
# Open-domain research QA; full runs use the judge path (`score_judge` / `results.py`).
# Here we only peek at a task. (Read from the index: constructing `DeepResearchQA`
# requires `SERPER_API_KEY`, which browsing does not need.)

# %%
dr_index = DeepResearchQA._index()
dr_ids = list(dr_index)
random.Random(42).shuffle(dr_ids)
print(f"DeepResearchQA tasks: {len(dr_ids)}")
test_id = dr_ids[0]
question = dr_index[test_id]["question"]
gold = dr_index[test_id]["answer"]
print(f"task_id: {test_id}")
print(f"question: {question[:800]}…" if len(question) > 800 else f"question: {question}")
print(f"gold: {gold}")
