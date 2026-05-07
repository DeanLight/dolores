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
# # Oolong Benchmark
#
# Hooks for running the [Oolong](https://arxiv.org/abs/2511.02817) long-context aggregation benchmark (Bertsch et al., 2025).

# %%
from juplit import test

# %%
from functools import cache

from datasets import load_dataset
from .eval_helpers import dnd_process_response, dnd_parse_answer, dnd_score


@cache
def _load_dataset() -> dict[str, dict]:
    """Load the full Oolong real split and index by ID. Cached after first call.
    Drops context_window_text to save memory — only metadata needed for scoring."""
    ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]
    index = {}
    for row in ds:
        row = {k: v for k, v in row.items() if k != "context_window_text"}
        index[row["id"]] = row
    return index

# %%
def list_test_ids(limit: int | None, seed: int) -> list[str]:
    """Return test IDs for the Oolong real split.

    Args:
        limit:       Max IDs to return (None for all). A deterministic random subsample
                     is taken (using `seed`) so the same IDs are returned every time.
        seed:        RNG seed for the subsample.
    """
    ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]

    # Deterministic shuffle before filtering so the subsample is stable
    ds = ds.shuffle(seed=seed)

    ids = []
    for row in ds:
        ids.append(row["id"])
        if limit is not None and len(ids) >= limit:
            break
    return ids

# %%
def get_task(test_id: str) -> tuple[str, str]:
    """Get the document and question for a specific test instance.

    Args:
        test_id: A test ID from list_test_ids().

    Returns:
        (document, question) — the D&D transcript and the aggregation question.
    """
    ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]
    for row in ds:
        if row["id"] == test_id:
            return row["context_window_text"], row["question"]
    raise KeyError(f"test_id {repr(test_id)} not found in Oolong dataset")

# %%
def get_answer(test_id: str) -> int | str | list[str]:
    """Return the gold answer for a given test ID.

    Args:
        test_id: A test ID from list_test_ids().

    Returns:
        int | str | list[str] — the parsed gold answer.
    """
    index = _load_dataset()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in Oolong dataset")
    return dnd_parse_answer(index[test_id]["answer"])

# %%
def relaxed_accuracy(y, yhat, tau=0.05):
    if y == 0:
        return 1.0 if yhat == 0 else 0.0
    return 1.0 if abs(y - yhat) / abs(y) <= tau else 0.0

# %%
def score(test_id: str, pred: int | str | list, numeric_scorer: callable = relaxed_accuracy, **kwargs) -> float:
    """Score a parsed prediction against the gold answer.

    Args:
        test_id:        Row ID from list_test_ids().
        pred:           Already-parsed prediction (int, str, or list[str]) — from parse().
        numeric_scorer: Optional callable(gold, pred) -> float to replace the default
                        0.75^|error| numeric penalty.

    Returns:
        float — 0.0 to 1.0.
    """
    index = _load_dataset()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in Oolong dataset")
    gold = dnd_parse_answer(index[test_id]["answer"])
    return dnd_score(gold, pred, numeric_scorer=numeric_scorer)

# %%
from pydantic import BaseModel
from ..parsing import _parse

# --- Schemas ---

class ExtractedInt(BaseModel):
    answer: int

class ExtractedStr(BaseModel):
    answer: str

class ExtractedList(BaseModel):
    answer: list[str]


# --- Prompts ---

INT_PROMPT = """You are an answer extractor.

You will receive:
- a question
- the raw output of a language model that attempted to answer it

Extract the final integer answer from the model output.

The model output may contain the final answer inside a \\boxed{} expression. If so, extract only the value inside \\boxed{} and do not include the \\boxed{} markup.

Return only the integer.

Examples:

Question: How many rolls occurred in the episode?
Model output: Let me count each roll... I found 17 natural 20s and 8 natural 1s, giving a total of \\boxed{114} rolls.
Answer: 114

Question: How many cantrips were cast?
Model output: After reviewing all episodes, the total cantrip count is approximately 25. Final answer: 25
Answer: 25
"""

STR_PROMPT = """You are an answer extractor.

You will receive:
- a question
- the raw output of a language model that attempted to answer it

Extract the final answer string from the model output.

The model output may contain the final answer inside a \\boxed{} expression. If so, extract only the value inside \\boxed{} and do not include the \\boxed{} markup.

Return only the string.

Examples:

Question: What is the most frequently cast spell?
Model output: Looking through the transcript, the most frequently cast spell is \\boxed{Fire Bolt} with 12 casts.
Answer: Fire Bolt

Question: Which spell appears most often?
Model output: Based on my analysis, the answer is Dispel Magic. It appears 7 times.
Answer: Dispel Magic
"""


LIST_PROMPT = """You are an answer extractor.

You will receive:
- a question
- the raw output of a language model that attempted to answer it

Extract the list of answer strings from the model output.

The model output may contain the final answer inside a \\boxed{} expression. If so, extract only the values inside \\boxed{} and do not include the \\boxed{} markup.

Return a list of strings.

Examples:

Question: Which spells were cast?
Model output: The unique spells cast are \\boxed{Dispel Magic, Fire Bolt, Healing Word}
Answer: ["Dispel Magic", "Fire Bolt", "Healing Word"]

Question: Which skills were used?
Model output: The transcript shows several rolls, including Perception, Stealth, and Athletics.
Answer: ["Perception", "Stealth", "Athletics"]
"""


# --- User message template ---

USER_TEMPLATE = """Question: {question}

Model output: {prediction}"""

# %%
_TYPE_CONFIG = {
    int:  (INT_PROMPT, ExtractedInt),
    str:  (STR_PROMPT, ExtractedStr),
    list: (LIST_PROMPT, ExtractedList),
}


def parse(test_id: str, prediction: str, api_key: str, model: str = 'gpt-5-nano') -> int | str | list[str]:
    """Parse a raw model prediction into the correct answer type for scoring.

    Looks up the gold answer for `test_id`, determines its type (int, str, or
    list[str]) via `dnd_parse_answer`, and calls `_parse` with the appropriate
    prompt and schema. The question and prediction are passed as the user message.

    Args:
        test_id:    Row ID from list_test_ids().
        prediction: The raw string output from your model.
        model:      OpenAI model name for parsing (e.g. "gpt-4o-mini").
        api_key:    OpenAI API key.

    Returns:
        int | str | list[str] — parsed answer, ready to pass to score().
    """
    index = _load_dataset()
    if test_id not in index:
        raise KeyError(f"test_id {repr(test_id)} not found in Oolong dataset")
    row = index[test_id]

    gold = dnd_parse_answer(row["answer"])
    gold_type = type(gold)

    system_prompt, schema = _TYPE_CONFIG[gold_type]
    user_prompt = USER_TEMPLATE.format(question=row["question"], prediction=prediction)

    result = _parse(user_prompt=user_prompt, system_prompt=system_prompt, schema=schema, model=model, api_key=api_key)
    return result["answer"]
