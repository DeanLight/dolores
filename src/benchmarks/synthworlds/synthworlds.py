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
# # SynthWorlds Benchmark
#
# Hooks for running the [Synthworlds](https://arxiv.org/abs/2510.24427) benchmark — **SM (Synth-Mapped) world only** (Gu et al., 2026, ICLR).

# %%
from juplit import test

# %%
from functools import cache

from datasets import load_dataset
from .eval_utils import normalize_answer, compute_f1, score_instance

# %%
@cache
def _load_qa_index() -> dict[str, dict]:
    """Load SM QA split and index by instance_id. Cached after first call."""
    ds = load_dataset("kenqgu/SynthWorlds", "qa-sm", split="test")
    index = {}
    for inst in ds:
        row = {
            "instance_id":             inst["instance_id"],
            "query":                   inst["query"],
            "gold_answers":            list(inst["gold_answers"]),
            "gold_docs":               list(inst["gold_docs"]),
            "gold_qids":               list(inst["gold_qids"]),
            "graph_type":              inst["question_graph_type"],
            "expected_output_is_time": bool(inst["expected_output_is_time"]),
        }
        index[row["instance_id"]] = row
    return index


def list_test_ids() -> list[str]:
    """Return all 1,200 test IDs for SynthWorlds SM multi-hop QA."""
    return list(_load_qa_index().keys())

# %%
def get_task(test_id: str) -> str:
    """Get the question for a specific test ID.

    Args:
        test_id: An instance_id from list_test_ids().

    Returns:
        The multi-hop question string.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in SynthWorlds SM")
    return index[test_id]["query"]

# %%
@cache
def load_qa_docs() -> tuple[str, ...]:
    """Load the SynthWorlds SM document corpus (6,290 documents).

    Returns:
        Tuple of 6,290 document strings.
    """
    ds = load_dataset("kenqgu/SynthWorlds", "qa-sm-docs", split="test")
    return tuple(row["doc"] for row in ds)

# %%
import json
import numpy as np
from pathlib import Path
from openai import OpenAI

_DATA_DIR = Path(__file__).resolve().parent / "data"


@cache
def _load_sm_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load pre-computed SM embeddings and documents from package data."""
    embs = np.load(_DATA_DIR / "sm.npy")
    with open(_DATA_DIR / "sm_docs.json") as f:
        docs = json.load(f)
    return embs, docs


def create_retriever_tool(api_key: str) -> callable:
    """Return a retrieve_top_5 function with the OpenAI API key baked in.

    Args:
        api_key: OpenAI API key for embedding queries.

    Returns:
        A callable ``retrieve_top_5(query: str) -> list[str]`` that returns
        the 5 most similar documents ranked by cosine similarity.
    """
    client = OpenAI(api_key=api_key)

    def retrieve_top_5(query: str) -> list[str]:
        """Retrieve the 5 documents most similar to a query.

        Args:
            query: The question string to embed.

        Returns:
            List of 5 document strings, ranked by cosine similarity (highest first).
        """
        resp = client.embeddings.create(input=query, model="text-embedding-3-small")
        query_emb = np.array(resp.data[0].embedding, dtype=np.float32)

        embs, docs = _load_sm_embeddings()
        scores = embs @ query_emb
        top_k_idx = np.argsort(scores)[-5:][::-1]
        return [docs[i] for i in top_k_idx]

    return retrieve_top_5

# %%
def get_answer(test_id: str) -> str:
    """Return the gold answer for a given test ID.

    Args:
        test_id: An instance_id from list_test_ids().

    Returns:
        The gold answer string.
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in SynthWorlds SM")
    return index[test_id]["gold_answers"][0]

# %%
def score(test_id: str, prediction: str) -> tuple[float, float]:
    """Score a parsed answer string against the gold answer.

    Args:
        test_id:    The instance_id from list_test_ids().
        prediction: The parsed answer string (e.g. "Charles", "28 July 1896").

    Returns:
        (f1, em) — both floats in [0, 1].
    """
    index = _load_qa_index()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in SynthWorlds SM")
    row = index[test_id]
    em, f1 = score_instance(prediction, row["gold_answers"], expected_output_is_time=row["expected_output_is_time"])
    return f1, em
