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
# # DeepResearchQA Benchmark
#
# Hooks for running the [DeepResearchQA](https://arxiv.org/pdf/2601.20975) benchmark (Gupta et al., 2026).

# %%
from juplit import test

# %%
import hashlib
import random
from functools import cache

from datasets import load_dataset
import os


@cache
def _load_qa_index() -> dict[str, dict]:
    """Load the DeepResearchQA eval split, keep only ``Single Answer`` rows,
    and index by a stable sha256(problem) id. Cached after first call."""
    ds = load_dataset("google/deepsearchqa")["eval"]
    index = {}
    for row in ds:
        if row["answer_type"] != "Single Answer":
            continue
        tid = hashlib.sha256(row["problem"].encode()).hexdigest()
        index[tid] = {
            "id":               tid,
            "question":         row["problem"],
            "answer":           row["answer"],
            "problem_category": row["problem_category"],
        }
    return index

# %%
def list_test_ids(limit: int | None, seed: int) -> list[str]:
    """Return test IDs for the DeepResearchQA Single-Answer rows.

    Args:
        limit:  Max IDs to return (None for all). A deterministic random subsample
                is taken (using `seed`) so the same IDs are returned every time.
        seed:   RNG seed for the subsample.
    """
    ids = list(_load_qa_index().keys())
    random.Random(seed).shuffle(ids)
    return ids[:limit]

# %%
def get_task(test_id: str) -> str:
    """Get the question for a specific test instance.

    Args:
        test_id: A test ID from list_test_ids().

    Returns:
        The question string.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in DeepResearchQA")
    return index[test_id]["question"]

# %%
def score(test_id: str, output: str) -> float:
    """Score a model output against the gold Single Answer.

    Args:
        test_id: A test ID from list_test_ids().
        output:  Model prediction string.

    Returns:
        1.0 if output matches the gold answer (case-insensitive, stripped); 0.0 otherwise.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in DeepResearchQA")

    o = str(output).strip().lower()
    a = index[test_id]["answer"].strip().lower()

    if a == o:
        return 1.0

    return 0.0

# %%
def get_answer(test_id: str) -> str:
    """Return the gold answer for a given test ID.

    Args:
        test_id: A test ID from list_test_ids().

    Returns:
        The gold answer string.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in DeepResearchQA")
    return index[test_id]["answer"]

# %%
from pydantic import BaseModel

from ..parsing import _parse


class Judgment(BaseModel):
    reasoning: str
    correct: bool


_JUDGE_SYSTEM_PROMPT = """\
You are judging whether an agent answered a factual question correctly.

The agent's answers may be long, include intermediate reasoning, calculations, caveats, or extra context. Your job is not to judge style, confidence, or completeness of the writeup. Your job is to decide whether the agent ultimately arrived at the same answer as the gold answer.

Read the response holistically and determine what answer the agent actually lands on. Then compare that resolved answer to the gold answer by meaning, not surface form. A response should be marked correct when a reasonable reader would come away believing that the agent's answer is the same fact as the gold answer, even if the response is verbose, hedged, or wrapped in explanation.

Examples:

---

Question: Which place is cheaper to buy the bag?
Gold Answer: Europe
Agent's Answer: Based on the reported prices, it seems Europe would be cheaper.

Correct: true

---

Question: Which CBSA in Arizona matched the condition?
Gold Answer: Tucson
Agent's Answer: The answer is in Arizona, likely around Tucson.

Correct: true

---

Question: How many cases were reported?
Gold Answer: 412341
Agent's Answer: Using the WHO data, the corresponding figure is approximately 412,341.

Correct: true

---

Question: Who is the most recurring Hugo Award winner from 2016 to 2022 in the Allegheny County library catalogues?
Gold Answer: N.K. Jemisin
Agent's Answer: The most recurring Hugo Award winner during that period is Nora Jemisin.

Correct: true

---

Question: Which CBSA in Arizona matched the condition?
Gold Answer: Tucson
Agent's Answer: The answer is somewhere in Arizona, but I can't narrow it down further.

Correct: false

---

Question: Which place is cheaper to buy the bag?
Gold Answer: Europe
Agent's Answer: Europe seems possible, but the US might also be right.

Correct: false

---

Question: Which legislator authored the bills?
Gold Answer: L. Nelson Cowles
Agent's Answer: I couldn't determine the legislator from the sources I found.

Correct: false

---

Question: On which day was the fish caught?
Gold Answer: July 3, 2001
Agent's Answer: It was caught sometime in 2001.

Correct: false

---

Return JSON matching the schema.
In `reasoning`, briefly explain your reasoning.
In `correct`, return true or false.
"""


def score_judge(test_id: str, output: str, model: str = 'gpt-5-nano') -> tuple[str, float, str]:
    """Score an agent's answer against the gold answer using an LLM-as-judge.

    Args:
        test_id: A test ID from list_test_ids().
        output:  The agent's answer string.
        model:   OpenAI model name for the judge.

    Returns:
        (test_id, score, reasoning) where score is 1.0 if the judge deems the answer
        correct and 0.0 otherwise, and reasoning is the judge's brief reasoning string.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in DeepResearchQA")

    row = index[test_id]
    user_prompt = (
        f"Question: {row['question']}\n"
        f"Gold Answer: {row['answer']}\n"
        f"Agent's Answer: {output}\n\n"
        f"Did the agent find the correct answer?"
    )
    judgment = _parse(
        user_prompt=user_prompt,
        system_prompt=_JUDGE_SYSTEM_PROMPT,
        schema=Judgment,
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    score = 1.0 if judgment["correct"] else 0.0
    reasoning = judgment["reasoning"]
    return test_id, score, reasoning

# %%
from concurrent.futures import ThreadPoolExecutor


def score_judge_batch(
    pairs: list[tuple[str, str]],
    model: str = 'gpt-5-nano',
    max_workers: int = 50,
) -> list[tuple[str, float, str]]:
    """Score many `(test_id, output)` pairs in parallel via a thread pool.

    Args:
        pairs:       List of `(test_id, agent_output)` pairs.
        model:       OpenAI model name for the judge.
        max_workers: Number of concurrent OpenAI requests.

    Returns:
        List of `(test_id, score, reasoning)` tuples in the same order as `pairs`.
    """
    def _one(pair: tuple[str, str]) -> tuple[str, float, str]:
        test_id, output = pair
        return score_judge(test_id, output, model=model)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, pairs))
