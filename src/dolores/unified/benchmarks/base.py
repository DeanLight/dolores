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
# # Benchmark base classes and registry

# %%
from juplit import test

# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class Task:
    task_id: str
    question: str
    tools: dict[str, Callable] = field(default_factory=dict)
    vars: dict[str, Any] = field(default_factory=dict)
    gold: Any = None


class Benchmark(ABC):
    @classmethod
    def benchmark_name(cls) -> str:
        return cls.__name__.lower()

    def prepare(self) -> None:
        return None

    @abstractmethod
    def list_task_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_task(self, task_id: str) -> Task:
        raise NotImplementedError

    @abstractmethod
    def score(self, task_id: str, answer: Any, **kwargs) -> dict:
        raise NotImplementedError

    # ── aggregation over k sampled attempts ───────────────────────────────────
    # Equivalence is a property of the answer *set*, not of one answer in
    # isolation: a judge-scored benchmark can only decide "same answer" with all
    # the answers in hand. Each benchmark therefore owns `cluster_answers`, and
    # the two aggregation rules below are implemented here, once, on top of it.

    def cluster_answers(self, task_id: str, answers: list[Any]) -> list[list[int]]:
        """Group answer indices into equivalence classes, largest class first.

        Default: exact match on a casefolded, whitespace-collapsed string.
        Benchmarks override this with their own notion of "the same answer".
        """
        groups: dict[Any, list[int]] = {}
        for i, answer in enumerate(answers):
            groups.setdefault(" ".join(str(answer).casefold().split()), []).append(i)
        return sorted(groups.values(), key=lambda idx: (-len(idx), idx[0]))

    def majority_at_k(self, task_id: str, answers: list[Any]) -> tuple[Any, float, Any]:
        """FAIR rule: the answer most attempts agree on. Never looks at the gold.

        Returns ``(answer, vote_share, score)``. Ties break on the earliest
        attempt, so the result does not depend on dict or file ordering.
        """
        if not answers:
            raise ValueError(f"no attempts to aggregate for task {task_id!r}")
        winner = self.cluster_answers(task_id, answers)[0]
        answer = answers[winner[0]]
        return answer, len(winner) / len(answers), self.score(task_id, answer)

    def best_at_k(self, task_id: str, answers: list[Any], gold: Any) -> tuple[Any, float]:
        """ORACLE rule: the best-scoring attempt. **Requires the gold answer.**

        `gold` is a required argument rather than something looked up internally,
        so that every call site reads as what it is: a rule that sees the answer
        key. That makes this an upper bound, never a fair comparison.
        """
        if not answers:
            raise ValueError(f"no attempts to aggregate for task {task_id!r}")
        if gold is None:
            raise ValueError(
                f"best_at_k is an oracle rule and needs the gold answer for task {task_id!r}"
            )
        scored = [(_scalar(self.score(task_id, a)), a) for a in answers]
        best_score, best_answer = max(scored, key=lambda pair: pair[0])
        return best_answer, best_score


def _scalar(score: Any) -> float:
    """Compare scores uniformly: benchmarks return either a float or an (f1, em)."""
    if isinstance(score, (tuple, list)):
        return float(score[0]) if score else 0.0
    return float(score)


_BENCHMARK_REGISTRY: dict[str, type[Benchmark]] = {}


def register_benchmark(cls: type[Benchmark]) -> type[Benchmark]:
    name = cls.benchmark_name().strip().lower()
    if not name:
        raise ValueError("benchmark_name() must return a non-empty value")
    _BENCHMARK_REGISTRY[name] = cls
    return cls


def get_benchmark(name: str, **kwargs) -> Benchmark:
    key = name.strip().lower()
    if key not in _BENCHMARK_REGISTRY:
        known = ", ".join(sorted(_BENCHMARK_REGISTRY))
        raise KeyError(f"Unknown benchmark {name!r}. Available: [{known}]")
    return _BENCHMARK_REGISTRY[key](**kwargs)


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARK_REGISTRY)


# %%
def test_default_majority_and_best():
    from dolores.unified.benchmarks import HelloWorld

    bench = HelloWorld()

    # 3 of 5 agree -> that answer wins with a 0.6 vote share, and it is right.
    answers = ["13", "12", "13", "13", "8"]
    answer, share, score = bench.majority_at_k("fib_7", answers)
    assert answer == "13"
    assert share == 0.6
    assert _scalar(score) == 1.0

    # All distinct -> the earliest attempt wins, deterministically.
    distinct = ["8", "12", "13"]
    answer, share, _ = bench.majority_at_k("fib_7", distinct)
    assert answer == "8"
    assert share == 1 / 3

    # Whitespace/case differences are the same vote.
    answer, share, _ = bench.majority_at_k("fib_7", ["13", " 13 ", "12"])
    assert answer == "13" and share == 2 / 3

    # The oracle finds the right answer even when it is in the minority — and it
    # has to be handed the gold answer to do it.
    best, best_score = bench.best_at_k("fib_7", ["8", "12", "13"], gold=13)
    assert best == "13"
    assert best_score == 1.0

    try:
        bench.best_at_k("fib_7", ["8"], gold=None)
        assert False, "an oracle without the gold answer should raise"
    except ValueError:
        pass

    try:
        bench.majority_at_k("fib_7", [])
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        bench.best_at_k("fib_7", [], gold=13)
        assert False, "should have raised"
    except ValueError:
        pass


def test_scalar_handles_both_score_shapes():
    assert _scalar(0.5) == 0.5
    assert _scalar((0.75, 1.0)) == 0.75
    assert _scalar([0.25, 0.0]) == 0.25


def test_registry_round_trip():
    import os
    from unittest.mock import patch

    from dolores.unified.benchmarks import (
        DeepResearchQA,
        HelloWorld,
        Oolong,
        PhantomWiki,
        SynthWorlds,
    )

    expected = {"deepresearchqa", "hello_world", "oolong", "phantomwiki", "synthworlds"}
    assert set(list_benchmarks()) == expected

    # Smoke construction through the registry.
    assert isinstance(get_benchmark("phantomwiki"), PhantomWiki)
    assert isinstance(get_benchmark("synthworlds"), SynthWorlds)
    assert isinstance(get_benchmark("oolong"), Oolong)
    with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
        assert isinstance(get_benchmark("deepresearchqa"), DeepResearchQA)
    assert isinstance(get_benchmark("hello_world"), HelloWorld)

