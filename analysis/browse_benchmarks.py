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
from benchmarks import oolong, phantomwiki, synthworlds

# %% [markdown]
# ## Oolong
#
# Multi-hop QA over DnD game transcripts. Each task provides a long document (transcript)
# and a question about game statistics. Answers are scored with token-level F1.

# %%
oolong_ids = oolong.list_test_ids(limit=500, seed=42)
print(f"Oolong tasks: {len(oolong_ids)}")
print(f"Sample IDs: {oolong_ids[:5]}")

# %%
# Browse 3 example tasks
rng = random.Random(0)
for task_id in rng.sample(oolong_ids, 3):
    document, question = oolong.get_task(task_id)
    expected = oolong.get_answer(task_id)
    print(f"task_id: {task_id}")
    print(f"question: {question}")
    print(f"expected: {expected}")
    print(f"document length: {len(document):,} chars")
    print()

# %% [markdown]
# ## PhantomWiki
#
# Multi-hop entity QA over a synthetic Wikipedia-like corpus. Tasks require chaining
# article lookups via `retrieve_article` and `search` tools. Answers are scored with F1 + EM.

# %%
SIZE, SEED = 50, 1
phantom_ids = phantomwiki.list_test_ids(SIZE, SEED)
print(f"PhantomWiki tasks (size={SIZE}, seed={SEED}): {len(phantom_ids)}")
print(f"Sample IDs: {phantom_ids[:5]}")

# %%
# Browse 3 example tasks
qa_index = phantomwiki._load_qa_index(SIZE, SEED)
for task_id in rng.sample(phantom_ids, 3):
    question, _, _ = phantomwiki.get_task(SIZE, SEED, task_id)
    expected = qa_index[task_id]["answer"]
    print(f"task_id: {task_id}")
    print(f"question: {question}")
    print(f"expected: {expected}")
    print()

# %% [markdown]
# ## SynthWorlds
#
# Multi-hop QA over a synthetic world corpus of 6,290 documents. Tasks require
# iterative dense retrieval via `retrieve_top_5(query)`. Answers are scored with F1 + EM.
# The retriever uses OpenAI `text-embedding-3-small` — set `OPENAI_API_KEY` to use it.

# %%
synth_ids = synthworlds.list_test_ids()
print(f"SynthWorlds tasks: {len(synth_ids)}")
print(f"Sample IDs: {synth_ids[:5]}")

# %%
# Browse 3 example tasks
for task_id in rng.sample(synth_ids, 3):
    question = synthworlds.get_task(task_id)
    expected = synthworlds.get_answer(task_id)
    print(f"task_id: {task_id}")
    print(f"question: {question}")
    print(f"expected: {expected}")
    print()

# %%
# Peek at the document corpus
docs = synthworlds.load_qa_docs()
print(f"Total documents: {len(docs)}")
print(f"\nSample document:\n{docs[0][:500]}")

# %% [markdown]
# ## DeepResearchQA
#
# Open-domain research QA; full runs use the judge path (`score_judge` / `results.py`). Here we only peek at a task.

# %%
from benchmarks import deepresearchqa

dr_ids = deepresearchqa.list_test_ids(limit=500, seed=42)
print(f"DeepResearchQA tasks (capped sample): {len(dr_ids)}")
test_id = dr_ids[0]
question = deepresearchqa.get_task(test_id)
gold = deepresearchqa.get_answer(test_id)
print(f"task_id: {test_id}")
print(f"question: {question[:800]}…" if len(question) > 800 else f"question: {question}")
print(f"gold: {gold}")
