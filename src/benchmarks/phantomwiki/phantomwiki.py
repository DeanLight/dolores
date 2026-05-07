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
# # PhantomWiki Benchmark
#
# Hooks for running the [PhantomWiki](https://arxiv.org/abs/2502.20377) benchmark (Gong et al., 2025, ICML).

# %%
from juplit import test

# %%
from functools import cache

from datasets import load_dataset
from .score import exact_match, f1
from .tools import make_tools

_VALID_SIZES = (50, 500, 5000)
_VALID_SEEDS = (1, 2, 3)


def _validate(size: int, seed: int):
    if size not in _VALID_SIZES:
        raise ValueError(f"size must be one of {_VALID_SIZES}, got {size}")
    if seed not in _VALID_SEEDS:
        raise ValueError(f"seed must be one of {_VALID_SEEDS}, got {seed}")


@cache
def _load_qa_index(size: int, seed: int) -> dict[str, dict]:
    """Load full QA split and index by id. Cached after first call."""
    _validate(size, seed)
    split = f"depth_20_size_{size}_seed_{seed}"
    ds = load_dataset("kilian-group/phantom-wiki-v1", "question-answer")[split]
    index = {}
    for row in ds:
        r = {
            "id":         row["id"],
            "question":   row["question"],
            "answer":     list(row["answer"]),
            "difficulty": int(row["difficulty"]),
            "size":       size,
            "seed":       seed,
        }
        index[r["id"]] = r
    return index


def _load_articles(size: int, seed: int) -> list[dict]:
    """Load the article corpus for a given universe."""
    _validate(size, seed)
    split = f"depth_20_size_{size}_seed_{seed}"
    ds = load_dataset("kilian-group/phantom-wiki-v1", "text-corpus")[split]
    return [{"title": row["title"], "article": row["article"]} for row in ds]

# %%
def list_test_ids(size: int, seed: int, **kwargs) -> list[str]:
    """Return all test IDs for a PhantomWiki split.

    Args:
        size: Universe size — 50, 500, or 5000.
        seed: Dataset seed — 1, 2, or 3.

    Returns:
        List of 500 test ID strings.
    """
    return list(_load_qa_index(size, seed).keys())

# %%
def get_task(size: int, seed: int, test_id: str) -> tuple[str, callable, callable]:
    """Get the question and tools for a specific test instance.

    Args:
        size:    Universe size — 50, 500, or 5000.
        seed:    Dataset seed — 1, 2, or 3.
        test_id: A test ID from list_test_ids().

    Returns:
        (question, retrieve_article, search) — the question string and two tool callables
        bound to the article corpus of this (size, seed) universe.
    """
    index = _load_qa_index(size, seed)
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in PhantomWiki (size={size}, seed={seed})")
    question = index[test_id]["question"]

    articles = _load_articles(size, seed)
    retrieve_article, search = make_tools(articles)

    return question, retrieve_article, search

# %%
def score(size: int, seed: int, test_id: str, output: list[str]) -> tuple[float, float]:
    """Score a model output against the gold answer list.

    Args:
        size:    Universe size — must match the size the test_id came from.
        seed:    Dataset seed — must match the seed the test_id came from.
        test_id: Row ID from list_test_ids().
        output:  Model predictions as a list of name strings.

    Returns:
        (f1, em) — both floats in [0, 1].
    """
    index = _load_qa_index(size, seed)
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in PhantomWiki (size={size}, seed={seed})")
    row = index[test_id]
    pred = ",".join(output)
    true = ",".join(row["answer"])
    return f1(pred, true), float(exact_match(pred, true))

# %%
def get_answer(test_id: str, size: int, seed: int) -> list[str]:
    """Return the gold answer for a given test ID.

    Args:
        test_id: A test ID from list_test_ids().
        size:    Universe size — 50, 500, or 5000.
        seed:    Dataset seed — 1, 2, or 3.

    Returns:
        list[str] — the gold answer names.
    """
    index = _load_qa_index(size, seed)
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in PhantomWiki (size={size}, seed={seed})")
    return index[test_id]["answer"]
