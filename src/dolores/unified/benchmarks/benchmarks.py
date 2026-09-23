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
# # Benchmark class implementations
#
# Benchmark loading, task shaping, tool creation, and scoring for all benchmark classes.

# %%
from juplit import test

# %%
import hashlib
import json
import os
import random
import re
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
import urllib.request

import numpy as np
import parse as _parselib
import structlog
import yaml
from dateutil import parser as date_parser
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel

from config import settings
from .base import Benchmark, Task, register_benchmark

logger = structlog.get_logger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"
_OOLONG_PROMPTS = yaml.safe_load((_DATA_DIR / "oolong_prompts.yaml").read_text())
_DEEPRESEARCHQA_PROMPTS = yaml.safe_load((_DATA_DIR / "deepresearchqa_prompts.yaml").read_text())

# %% [markdown]
# ## Shared helpers

# %%
def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _parse(user_prompt: str, system_prompt: str, schema: type[BaseModel], model: str, api_key: str) -> dict:
    """Parse a raw string into a dict using OpenAI structured outputs."""
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=schema,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Model refused: {message.refusal}")
    if not message.parsed:
        raise ValueError("Could not parse response")
    return message.parsed.model_dump()


# %% [markdown]
# ## HelloWorld
#
# Simplest benchmark — no external data, no tools except pure Python functions.
# Use as a reference implementation and for quick smoke-testing of agent harnesses.

# %%
def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number using the standard definition.

    Use this tool when the task asks for a **Fibonacci** number. Do not use it for
    "shmibonacci" or any sequence whose first two values are both 1.

    **Definition**
    Let F be the Fibonacci sequence. Then F(0) = 0, F(1) = 1, and for every
    integer k ≥ 2, F(k) = F(k - 1) + F(k - 2). This function returns F(n).

    **Examples**
    - n = 0 → 0
    - n = 1 → 1
    - n = 2 → 1
    - n = 7 → 13

    Args:
        n: Non-negative index into the sequence (0 means F(0), 1 means F(1), etc.).

    Returns:
        The integer F(n).

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def shmibonacci(n: int) -> int:
    """Compute the n-th **shmibonacci** number: same recurrence as Fibonacci, different starting values.

    Use this tool only when the task explicitly asks for **shmibonacci** (not ordinary Fibonacci).
    The classic Fibonacci sequence starts with 0 and 1; shmibonacci starts with 1 and 1, then
    uses the same "add the previous two terms" rule.

    **Definition**
    Let S be the shmibonacci sequence. Then S(0) = 1, S(1) = 1, and for every
    integer k ≥ 2, S(k) = S(k - 1) + S(k - 2). This function returns S(n).

    **How this differs from ``fibonacci``**
    - Fibonacci: first terms are 0, 1 (standard F(0), F(1)).
    - Shmibonacci: first terms are 1, 1, so the sequence begins 1, 1, 2, 3, 5, …

    **Examples**
    - n = 0 → 1
    - n = 1 → 1
    - n = 2 → 2
    - n = 5 → 8

    Args:
        n: Non-negative index into the shmibonacci sequence (0 means S(0), etc.).

    Returns:
        The integer S(n).

    Raises:
        ValueError: If ``n`` is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@register_benchmark
class HelloWorld(Benchmark):
    def __init__(self, seed: int = 0):
        self.seed = seed

    @classmethod
    def benchmark_name(cls) -> str:
        return "hello_world"

    @staticmethod
    def _parse_task_id(task_id: str) -> tuple[str, int]:
        result = _parselib.parse("{kind}_{n:d}", task_id.lower())
        if result is None or result["kind"] not in ("fib", "shmib"):
            raise KeyError(f"invalid test_id {task_id!r}; expected like fib_7 or shmib_4")
        return result["kind"], result["n"]

    @staticmethod
    def _gold(task_id: str) -> int:
        kind, n = HelloWorld._parse_task_id(task_id)
        if kind == "fib":
            return fibonacci(n)
        return shmibonacci(n)

    def list_task_ids(self) -> list[str]:
        _ = self.seed
        return ["fib_7", "shmib_5", "fib_3"]

    def get_task(self, task_id: str) -> Task:
        kind, n = HelloWorld._parse_task_id(task_id)
        if kind == "fib":
            question = (
                f"Using the tools, what is the Fibonacci number F({n}) "
                f"(with F(0)=0, F(1)=1)? Reply with the integer only."
            )
        else:
            question = (
                f"Using the tools, what is the shmibonacci number S({n}) "
                f"(with S(0)=1, S(1)=1, same recurrence as Fibonacci)? Reply with the integer only."
            )
        return Task(
            task_id=task_id,
            question=question,
            tools={"fibonacci": fibonacci, "shmibonacci": shmibonacci},
            vars={},
            gold=HelloWorld._gold(task_id),
        )

    def score(self, task_id: str, answer, **kwargs) -> tuple[float, float]:
        _ = kwargs
        gold = HelloWorld._gold(task_id)
        pred = str(answer).strip()
        ok = pred == str(gold)
        return (1.0 if ok else 0.0, 1.0 if ok else 0.0)


# %% [markdown]
# ## PhantomWiki

# %%
@register_benchmark
class PhantomWiki(Benchmark):
    _VALID_SIZES = (50, 500, 5000)
    _VALID_SEEDS = (1, 2, 3)
    _ANSWER_SEP = ","

    def __init__(self, size: int = 50, seed: int = 1):
        self.size = size
        self.seed = seed

    @classmethod
    def benchmark_name(cls) -> str:
        return "phantomwiki"

    @staticmethod
    def _validate(size: int, seed: int):
        if size not in PhantomWiki._VALID_SIZES:
            raise ValueError(f"size must be one of {PhantomWiki._VALID_SIZES}, got {size}")
        if seed not in PhantomWiki._VALID_SEEDS:
            raise ValueError(f"seed must be one of {PhantomWiki._VALID_SEEDS}, got {seed}")

    @staticmethod
    @cache
    def _qa_index(size: int, seed: int) -> dict[str, dict]:
        PhantomWiki._validate(size, seed)
        split = f"depth_20_size_{size}_seed_{seed}"
        ds = load_dataset("kilian-group/phantom-wiki-v1", "question-answer")[split]
        index = {}
        for row in ds:
            index[row["id"]] = {
                "id": row["id"],
                "question": row["question"],
                "answer": list(row["answer"]),
            }
        return index

    @staticmethod
    @cache
    def _articles(size: int, seed: int) -> tuple[tuple[str, str], ...]:
        PhantomWiki._validate(size, seed)
        split = f"depth_20_size_{size}_seed_{seed}"
        ds = load_dataset("kilian-group/phantom-wiki-v1", "text-corpus")[split]
        return tuple((row["title"], row["article"]) for row in ds)

    @staticmethod
    def _normalize_pred(pred: str, sep: str = ",") -> set[str]:
        return set(map(str.lower, map(str.strip, pred.split(sep))))

    @staticmethod
    def _exact_match(pred: str, true: str, sep: str = ",") -> bool:
        return PhantomWiki._normalize_pred(pred, sep) == PhantomWiki._normalize_pred(true, sep)

    @staticmethod
    def _precision(pred: str, true: str, sep: str = ",") -> float:
        normalized_preds = PhantomWiki._normalize_pred(pred, sep)
        normalized_trues = PhantomWiki._normalize_pred(true, sep)
        count = sum(word in normalized_trues for word in normalized_preds)
        return _safe_divide(count, len(normalized_preds))

    @staticmethod
    def _recall(pred: str, true: str, sep: str = ",") -> float:
        normalized_preds = PhantomWiki._normalize_pred(pred, sep)
        normalized_trues = PhantomWiki._normalize_pred(true, sep)
        count = sum(word in normalized_preds for word in normalized_trues)
        return _safe_divide(count, len(normalized_trues))

    @staticmethod
    def _f1(pred: str, true: str, sep: str = ",") -> float:
        precision = PhantomWiki._precision(pred, true, sep)
        recall = PhantomWiki._recall(pred, true, sep)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _make_tools(articles: list[dict]) -> tuple:
        """Bind retrieve_article and search tools to a fixed article corpus."""

        # Docstring verbatim from ReactLLMPrompt.REACT_INSTRUCTION (phantom_eval/prompts.py)
        def retrieve_article(entity: str) -> str:
            """Retrieve the article about the given entity, if it exists.

            Args:
                entity: The name of the entity to retrieve the article for.

            Returns:
                The full article text, or an error string if no article with that title exists.
            """
            for a in articles:
                if a["title"].lower() == entity.lower():
                    return a["article"]
            return (
                "No article exists for the requested entity. "
                "Please try retrieving article for another entity."
            )

        # Docstring verbatim from ReactLLMPrompt.REACT_INSTRUCTION (phantom_eval/prompts.py)
        def search(attribute: str) -> str:
            """Search the database for the given attribute and retrieve all articles that contain it.

            Args:
                attribute: The keyword or phrase to search for (e.g. a hobby, occupation, or name).

            Returns:
                A numbered list of matching article titles (e.g. "(1) Alice Smith\\n\\n(2) Bob Jones"),
                or an error string if no articles contain the attribute.
            """
            matching_titles = [
                a["title"] for a in articles
                if attribute.lower() in a["article"].lower()
            ]
            if not matching_titles:
                return (
                    "No articles contain the requested attribute. "
                    "Please try searching for another attribute."
                )
            return "\n\n".join(f"({i + 1}) {title}" for i, title in enumerate(matching_titles))

        return retrieve_article, search

    def list_task_ids(self) -> list[str]:
        return list(PhantomWiki._qa_index(self.size, self.seed).keys())

    def get_task(self, task_id: str) -> Task:
        index = PhantomWiki._qa_index(self.size, self.seed)
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in PhantomWiki (size={self.size}, seed={self.seed})")
        row = index[task_id]

        articles = [{"title": title, "article": article} for title, article in PhantomWiki._articles(self.size, self.seed)]
        retrieve_article, search = PhantomWiki._make_tools(articles)

        return Task(
            task_id=task_id,
            question=row["question"],
            tools={"retrieve_article": retrieve_article, "search": search},
            vars={},
            gold=row["answer"],
        )

    def cluster_answers(self, task_id: str, answers: list) -> list[list[int]]:
        """Two answers are the same when they name the same set of entities."""
        groups: dict = {}
        for i, answer in enumerate(answers):
            pred = ",".join(str(x) for x in answer) if isinstance(answer, list) else str(answer)
            groups.setdefault(frozenset(PhantomWiki._normalize_pred(pred)), []).append(i)
        return sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    def score(self, task_id: str, answer, **kwargs) -> tuple[float, float]:
        _ = kwargs
        index = PhantomWiki._qa_index(self.size, self.seed)
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in PhantomWiki (size={self.size}, seed={self.seed})")

        pred = ",".join(str(x) for x in answer) if isinstance(answer, list) else str(answer)
        true = ",".join(index[task_id]["answer"])
        return PhantomWiki._f1(pred, true), float(PhantomWiki._exact_match(pred, true))


# %% [markdown]
# ## SynthWorlds

# %%
@dataclass(frozen=True)
class _SynthworldsTimestampInfo:
    sign: str
    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    second: int | None = None

    def to_string(self) -> str:
        months = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        sign_str = "BCE" if self.sign == "-" else "CE"
        if self.year is not None and self.year <= 1000:
            return f"{self.year} {sign_str}"
        if self.month is None:
            return f"{self.year}"
        if self.day is None:
            return f"{months[self.month - 1]}, {self.year}"
        return f"{months[self.month - 1]} {self.day}, {self.year}"


@register_benchmark
class SynthWorlds(Benchmark):
    @classmethod
    def benchmark_name(cls) -> str:
        return "synthworlds"

    @staticmethod
    @cache
    def _qa_index() -> dict[str, dict]:
        ds = load_dataset("kenqgu/SynthWorlds", "qa-sm", split="test")
        index = {}
        for inst in ds:
            index[inst["instance_id"]] = {
                "instance_id": inst["instance_id"],
                "query": inst["query"],
                "gold_answers": list(inst["gold_answers"]),
                "expected_output_is_time": bool(inst["expected_output_is_time"]),
            }
        return index

    @staticmethod
    def _normalize_answer(answer: str) -> str:
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the|is)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punctuation(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_articles(remove_punctuation(answer.lower())))

    @staticmethod
    def _compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = SynthWorlds._normalize_answer(gold).split()
        predicted_tokens = SynthWorlds._normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = _safe_divide(num_same, len(predicted_tokens))
        recall = _safe_divide(num_same, len(gold_tokens))
        return _safe_divide(2 * precision * recall, precision + recall)

    @staticmethod
    def _parse_date_string(date_str: str) -> _SynthworldsTimestampInfo:
        date_str = date_str.strip()
        sign = "+"
        if "BCE" in date_str:
            sign = "-"
            date_str = date_str.replace("BCE", "").strip()
            year = int(date_str)
            return _SynthworldsTimestampInfo(sign=sign, year=year)
        if "CE" in date_str:
            date_str = date_str.replace("CE", "").strip()
            year = int(date_str)
            return _SynthworldsTimestampInfo(sign=sign, year=year)
        if "(year)" in date_str:
            date_str = date_str.replace("(year)", "").strip()
            year = int(date_str)
            return _SynthworldsTimestampInfo(sign=sign, year=year)

        year_month_match = re.match(r"^(\d{4})-(\d{1,2})$", date_str)
        iso_datetime_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z?$", date_str)
        iso_date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
        if year_month_match:
            year, month = map(int, year_month_match.groups())
            return _SynthworldsTimestampInfo(sign=sign, year=year, month=month)
        if iso_date_match:
            year, month, day = map(int, iso_date_match.groups())
            return _SynthworldsTimestampInfo(sign=sign, year=year, month=month, day=day)
        if iso_datetime_match:
            year, month, day, hour, minute, second = map(int, iso_datetime_match.groups())
            return _SynthworldsTimestampInfo(
                sign=sign, year=year, month=month, day=day, hour=hour, minute=minute, second=second
            )
        raise ValueError(f"Could not parse date string: {date_str}")

    @staticmethod
    def _parse_date(date_text: str) -> _SynthworldsTimestampInfo | None:
        try:
            dt = date_parser.parse(date_text)
            return _SynthworldsTimestampInfo(
                sign="+",
                year=dt.year,
                month=dt.month,
                day=dt.day,
                hour=dt.hour,
                minute=dt.minute,
                second=dt.second,
            )
        except Exception:
            pass

        try:
            return SynthWorlds._parse_date_string(date_text)
        except Exception:
            return None

    @staticmethod
    def _score_instance(
        pred: str,
        gold_answers: list[str],
        expected_output_is_time: bool = False,
    ) -> tuple[float, float]:
        em_scores = [
            1.0 if SynthWorlds._normalize_answer(pred) == SynthWorlds._normalize_answer(gold) else 0.0
            for gold in gold_answers
        ]

        agent_date_output: str | None = None
        gold_date_outputs: list[str] = []

        if expected_output_is_time:
            pred_date = SynthWorlds._parse_date(pred)
            if pred_date is not None:
                agent_date_output = pred_date.to_string()
            if agent_date_output is not None:
                for gold in gold_answers:
                    gold_date = SynthWorlds._parse_date(gold)
                    if gold_date is not None:
                        gold_as_string = gold_date.to_string()
                        gold_date_outputs.append(gold_as_string)
                        em_scores.append(1.0 if agent_date_output == gold_as_string else 0.0)

        em_score = float(np.max(em_scores))
        f1_scores = [SynthWorlds._compute_f1(gold, pred) for gold in gold_answers]
        if em_score == 1.0:
            f1_scores.append(1.0)
        if agent_date_output is not None:
            for gold_date in gold_date_outputs:
                f1_scores.append(SynthWorlds._compute_f1(gold_date, agent_date_output))
        f1_score = float(np.max(f1_scores))
        return em_score, f1_score

    @staticmethod
    @cache
    def _load_sm_embeddings() -> tuple[np.ndarray, list[str]]:
        """Load pre-computed SM embeddings and documents from package data."""
        embs = np.load(_DATA_DIR / "sm.npy")
        with open(_DATA_DIR / "sm_docs.json") as f:
            docs = json.load(f)
        return embs, docs

    @staticmethod
    def _create_retriever_tool(api_key: str):
        """Return a retrieve_top_5 function with the OpenAI API key baked in."""
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

            embs, docs = SynthWorlds._load_sm_embeddings()
            scores = embs @ query_emb
            top_k_idx = np.argsort(scores)[-5:][::-1]
            return [docs[i] for i in top_k_idx]

        return retrieve_top_5

    def list_task_ids(self) -> list[str]:
        return list(SynthWorlds._qa_index().keys())

    def get_task(self, task_id: str) -> Task:
        index = SynthWorlds._qa_index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in SynthWorlds SM")
        row = index[task_id]
        retrieve_top_5 = SynthWorlds._create_retriever_tool(api_key=settings.openai_api_key)
        return Task(
            task_id=task_id,
            question=row["query"],
            tools={"retrieve_top_5": retrieve_top_5},
            vars={},
            gold=row["gold_answers"][0],
        )

    @staticmethod
    def create_retriever_tool(api_key: str):
        return SynthWorlds._create_retriever_tool(api_key=api_key)

    def cluster_answers(self, task_id: str, answers: list) -> list[list[int]]:
        """Cluster on the same token normalisation the F1 score uses."""
        groups: dict = {}
        for i, answer in enumerate(answers):
            groups.setdefault(SynthWorlds._normalize_answer(str(answer)), []).append(i)
        return sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    def score(self, task_id: str, answer, **kwargs) -> tuple[float, float]:
        _ = kwargs
        index = SynthWorlds._qa_index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in SynthWorlds SM")
        row = index[task_id]
        em, f1 = SynthWorlds._score_instance(
            str(answer),
            row["gold_answers"],
            expected_output_is_time=row["expected_output_is_time"],
        )
        return f1, em


# %% [markdown]
# ## Oolong

# %%
class _OolongExtractedInt(BaseModel):
    answer: int

class _OolongExtractedStr(BaseModel):
    answer: str

class _OolongExtractedList(BaseModel):
    answer: list[str]


@register_benchmark
class Oolong(Benchmark):
    _INT_PROMPT = _OOLONG_PROMPTS["int_prompt"]
    _STR_PROMPT = _OOLONG_PROMPTS["str_prompt"]
    _LIST_PROMPT = _OOLONG_PROMPTS["list_prompt"]
    _USER_TEMPLATE = _OOLONG_PROMPTS["user_template"]
    _TYPE_CONFIG = {
        int:  (_INT_PROMPT, _OolongExtractedInt),
        str:  (_STR_PROMPT, _OolongExtractedStr),
        list: (_LIST_PROMPT, _OolongExtractedList),
    }

    def __init__(self, limit: int | None = None, seed: int = 0):
        self.limit = limit
        self.seed = seed

    @classmethod
    def benchmark_name(cls) -> str:
        return "oolong"

    @staticmethod
    @cache
    def _score_index() -> dict[str, dict]:
        ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        index = {}
        for row in ds:
            index[row["id"]] = {k: v for k, v in row.items() if k != "context_window_text"}
        return index

    @staticmethod
    def _document_and_question(task_id: str) -> tuple[str, str]:
        ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        for row in ds:
            if row["id"] == task_id:
                return row["context_window_text"], row["question"]
        raise KeyError(f"test_id {task_id!r} not found in Oolong dataset")

    @staticmethod
    def _parse_answer(answer: str) -> int | str | list[str]:
        try:
            return int(answer)
        except ValueError:
            pass
        if "," in answer:
            return [item.strip() for item in answer.split(",") if item.strip()]
        return answer

    @staticmethod
    def _relaxed_accuracy(y: int, yhat: int, tau: float = 0.05) -> float:
        if y == 0:
            return 1.0 if yhat == 0 else 0.0
        return 1.0 if abs(y - yhat) / abs(y) <= tau else 0.0

    @staticmethod
    def _score_answer(gold: int | str | list, pred: int | str | list, numeric_scorer=None) -> float:
        if isinstance(gold, int) and isinstance(pred, int):
            return numeric_scorer(gold, pred) if numeric_scorer else Oolong._relaxed_accuracy(gold, pred)
        if isinstance(gold, str) and isinstance(pred, str):
            return float(gold.strip().lower() == pred.strip().lower())
        if isinstance(gold, list) and isinstance(pred, list):
            overlap = set(gold) & set(pred)
            return _safe_divide(len(overlap), len(gold))
        return 0.0

    def list_task_ids(self) -> list[str]:
        ds = load_dataset("oolongbench/oolong-real", "dnd")["test"]
        ds = ds.shuffle(seed=self.seed)
        ids = []
        for row in ds:
            ids.append(row["id"])
            if self.limit is not None and len(ids) >= self.limit:
                break
        return ids

    def get_task(self, task_id: str) -> Task:
        document, question = Oolong._document_and_question(task_id)
        score_index = Oolong._score_index()
        if task_id not in score_index:
            raise KeyError(f"test_id {task_id!r} not found in Oolong dataset")
        gold = Oolong._parse_answer(score_index[task_id]["answer"])
        return Task(
            task_id=task_id,
            question=question,
            tools={},
            vars={"document": document},
            gold=gold,
        )

    def cluster_answers(self, task_id: str, answers: list) -> list[list[int]]:
        """Cluster on the parsed answer, so "7" and " 7 " are one vote."""
        groups: dict = {}
        for i, answer in enumerate(answers):
            parsed = Oolong._parse_answer(str(answer))
            key = tuple(parsed) if isinstance(parsed, list) else parsed
            groups.setdefault((type(key).__name__, key), []).append(i)
        return sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    def score(self, task_id: str, answer, **kwargs) -> float:
        numeric_scorer = kwargs.get("numeric_scorer")
        index = Oolong._score_index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in Oolong dataset")
        gold = Oolong._parse_answer(index[task_id]["answer"])
        return Oolong._score_answer(gold, answer, numeric_scorer=numeric_scorer)

    def parse(self, task_id: str, prediction: str, api_key: str, model: str = "gpt-5-nano") -> int | str | list[str]:
        """Parse a raw model prediction into the correct answer type for scoring."""
        index = Oolong._score_index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in Oolong dataset")
        row = index[task_id]

        gold = Oolong._parse_answer(row["answer"])
        gold_type = type(gold)

        system_prompt, schema = Oolong._TYPE_CONFIG[gold_type]
        user_prompt = Oolong._USER_TEMPLATE.format(question=row["question"], prediction=prediction)

        result = _parse(user_prompt=user_prompt, system_prompt=system_prompt, schema=schema, model=model, api_key=api_key)
        return result["answer"]


# %% [markdown]
# ## DeepResearchQA

# %%
class _DeepResearchQAClusters(BaseModel):
    """Judge output for majority voting: a partition of the candidate answers."""

    groups: list[list[int]]


class _DeepResearchQAJudgment(BaseModel):
    reasoning: str
    correct: bool


@register_benchmark
class DeepResearchQA(Benchmark):
    _JUDGE_SYSTEM_PROMPT = _DEEPRESEARCHQA_PROMPTS["judge_system_prompt"]
    _CLUSTER_SYSTEM_PROMPT = (
        "You group answers to the same question by meaning. Given a question and a "
        "numbered list of candidate answers, partition every index into groups whose "
        "answers say the same thing. Ignore wording, formatting and verbosity; two "
        "answers belong together only if a reader would call them the same answer. "
        "Every index must appear in exactly one group. You never see the correct "
        "answer and must not guess which group is right."
    )

    def __init__(self, limit: int | None = None, seed: int = 0):
        self.limit = limit
        self.seed = seed
        if not os.environ.get("SERPER_API_KEY"):
            raise ValueError(
                "DeepResearchQA requires SERPER_API_KEY environment variable for web search."
            )

    @classmethod
    def benchmark_name(cls) -> str:
        return "deepresearchqa"

    @staticmethod
    @cache
    def _index() -> dict[str, dict]:
        ds = load_dataset("google/deepsearchqa")["eval"]
        index = {}
        for row in ds:
            if row["answer_type"] != "Single Answer":
                continue
            tid = hashlib.sha256(row["problem"].encode()).hexdigest()
            index[tid] = {
                "id": tid,
                "question": row["problem"],
                "answer": row["answer"],
            }
        return index

    @staticmethod
    def _web_search():
        api_key = os.environ["SERPER_API_KEY"]

        def web_search(query: str) -> str:
            """Search the web for information relevant to the query using the Serper API.

            Args:
                query: The search query or question to research.

            Returns:
                Formatted search results with titles, snippets, and source URLs.
            """
            payload = json.dumps({"q": query, "num": 10}).encode()
            req = urllib.request.Request(
                "https://google.serper.dev/search",
                data=payload,
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            results = []
            for item in data.get("organic", [])[:5]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append(f"{title}\n{snippet}\nURL: {link}")
            return "\n\n".join(results) if results else "No results found."

        return web_search

    def list_task_ids(self) -> list[str]:
        ids = list(DeepResearchQA._index().keys())
        random.Random(self.seed).shuffle(ids)
        return ids[:self.limit]

    def get_task(self, task_id: str) -> Task:
        index = DeepResearchQA._index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in DeepResearchQA")
        row = index[task_id]
        return Task(
            task_id=task_id,
            question=row["question"],
            tools={"web_search": DeepResearchQA._web_search()},
            vars={},
            gold=row["answer"],
        )

    def score(self, task_id: str, answer, **kwargs) -> float:
        _ = kwargs
        index = DeepResearchQA._index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in DeepResearchQA")
        pred = str(answer).strip().lower()
        gold = index[task_id]["answer"].strip().lower()
        return 1.0 if pred == gold else 0.0

    def cluster_answers(self, task_id: str, answers: list) -> list[list[int]]:
        """Ask the judge to partition the answers into equivalence groups.

        Free-text answers cannot be normalised one at a time — deciding whether
        two answers say the same thing needs the other answers in hand. On any
        judge failure we fall back to the default string clustering and say so,
        so a run without a judge still aggregates (and the caller can see which
        tasks were clustered which way in the warning log).
        """
        texts = [str(a).strip() for a in answers]
        listing = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
        try:
            judgment = _parse(
                user_prompt=f"Question: {DeepResearchQA._index()[task_id]['question']}\n"
                            f"Candidate answers:\n{listing}",
                system_prompt=DeepResearchQA._CLUSTER_SYSTEM_PROMPT,
                schema=_DeepResearchQAClusters,
                model="gpt-5-nano",
                api_key=os.environ["OPENAI_API_KEY"],
            )
            groups = [sorted(set(g)) for g in judgment["groups"] if g]
            covered = {i for g in groups for i in g}
            if covered != set(range(len(texts))):
                raise ValueError(f"judge partition covered {covered}, expected all indices")
            return sorted(groups, key=lambda idx: (-len(idx), idx[0]))
        except Exception as exc:
            logger.warning("deepresearchqa.cluster_fallback", task_id=task_id, error=str(exc))
            return super().cluster_answers(task_id, answers)

    def score_judge(self, task_id: str, output: str, model: str = "gpt-5-nano") -> tuple[str, float, str]:
        """Score an agent's answer against the gold answer using an LLM-as-judge."""
        index = DeepResearchQA._index()
        if task_id not in index:
            raise KeyError(f"test_id {task_id!r} not found in DeepResearchQA")

        row = index[task_id]
        user_prompt = (
            f"Question: {row['question']}\n"
            f"Gold Answer: {row['answer']}\n"
            f"Agent's Answer: {output}\n\n"
            f"Did the agent find the correct answer?"
        )
        judgment = _parse(
            user_prompt=user_prompt,
            system_prompt=DeepResearchQA._JUDGE_SYSTEM_PROMPT,
            schema=_DeepResearchQAJudgment,
            model=model,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        score = 1.0 if judgment["correct"] else 0.0
        reasoning = judgment["reasoning"]
        return task_id, score, reasoning

    def score_judge_batch(
        self,
        pairs: list[tuple[str, str]],
        model: str = "gpt-5-nano",
        max_workers: int = 50,
    ) -> list[tuple[str, float, str]]:
        """Score many (task_id, output) pairs in parallel via a thread pool."""
        def _one(pair: tuple[str, str]) -> tuple[str, float, str]:
            task_id, output = pair
            return self.score_judge(task_id, output, model=model)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, pairs))


# %% [markdown]
# ## Smoke tests

# %%
def test_cluster_per_benchmark():
    """Each benchmark votes with its own notion of "the same answer"."""
    # PhantomWiki: an entity set, order- and case-insensitive.
    pw = PhantomWiki()
    assert pw.cluster_answers("t", ["Bob, alice", "Alice,Bob", "carol"]) == [[0, 1], [2]]
    assert pw.cluster_answers("t", [["Bob", "Alice"], "alice, bob"]) == [[0, 1]]

    # SynthWorlds: the token normalisation its F1 already uses.
    sw = SynthWorlds()
    assert sw.cluster_answers("t", ["The dog.", "dog", "a cat"]) == [[0, 1], [2]]

    # Oolong: the parsed answer, so "7" and " 7 " are one vote and "7" != "seven".
    ool = Oolong()
    assert ool.cluster_answers("t", ["7", " 7 ", "seven"]) == [[0, 1], [2]]
    assert ool.cluster_answers("t", ["a, b", "b,a "])[0] != [0, 1]   # list order is meaningful

    # Largest class first; ties fall back to the earliest attempt.
    assert sw.cluster_answers("t", ["x", "y", "y"]) == [[1, 2], [0]]


def test_deepresearchqa_cluster_falls_back_without_judge():
    """A judge failure degrades to string clustering instead of losing the task."""
    import os
    from unittest.mock import patch

    from dolores.unified.benchmarks import benchmarks as benchmarks_module

    with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
        bench = DeepResearchQA()

    with patch.object(benchmarks_module, "_parse", side_effect=RuntimeError("no judge")):
        groups = bench.cluster_answers("t", ["Paris", "paris ", "Lyon"])
    assert groups == [[0, 1], [2]]


def test_deepresearchqa_cluster_uses_judge_partition():
    import os
    from unittest.mock import patch

    from dolores.unified.benchmarks import benchmarks as benchmarks_module

    with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
        bench = DeepResearchQA()

    answers = ["It is Paris.", "The capital is Paris", "Lyon"]
    fake_index = {"t": {"question": "capital of France?", "answer": "Paris"}}
    with patch.object(DeepResearchQA, "_index", staticmethod(lambda: fake_index)), \
            patch.dict(os.environ, {"OPENAI_API_KEY": "k"}), \
            patch.object(benchmarks_module, "_parse", return_value={"groups": [[0, 1], [2]]}):
        assert bench.cluster_answers("t", answers) == [[0, 1], [2]]

        # A partition that drops an answer is rejected -> string fallback.
        with patch.object(benchmarks_module, "_parse", return_value={"groups": [[0, 1]]}):
            assert bench.cluster_answers("t", answers) == [[0], [1], [2]]


def test_hello_world_smoke():
    bench = HelloWorld()
    task = bench.get_task("fib_7")
    assert task.gold == 13
    assert bench.score("fib_7", "13") == (1.0, 1.0)


def test_phantomwiki_class_smoke():
    bench = PhantomWiki()
    task_ids = bench.list_task_ids()
    assert task_ids
    task = bench.get_task(task_ids[0])
    assert isinstance(task, Task)
    assert set(task.tools.keys()) == {"retrieve_article", "search"}
