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
# # Hello World Benchmark
#
# Hooks for running hello world bench, a simple tiny benchmark just for testing purposes.

# %%
from juplit import test

# %%
def list_test_ids(seed: int, **kwargs) -> list[str]:
    """Return all test IDs for the hello-world split.

    Args:
        seed: Ignored for this toy benchmark (kept for a stable API).

    Returns:
        Small list of task ids like ``fib_7`` / ``shmib_4`` (kind + index).
    """
    _ = seed
    return ['fib_7', 'shmib_5', 'fib_3']

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
        raise ValueError('n must be non-negative')
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
        raise ValueError('n must be non-negative')
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _parse_task_id(test_id: str) -> tuple[str, int]:
    parts = test_id.lower().split('_', 1)
    if len(parts) != 2:
        raise KeyError(f'invalid test_id {test_id!r}; expected like fib_7 or shmib_4')
    kind, ns = parts
    try:
        n = int(ns)
    except ValueError as e:
        raise KeyError(f'invalid test_id {test_id!r}; second part must be an int') from e
    if kind not in ('fib', 'shmib'):
        raise KeyError(f'unknown sequence {kind!r} in {test_id!r}')
    return kind, n


def get_task(test_id: str, seed: int, **kwargs) -> tuple[str, dict[str, callable]]:
    """Get the question and tool dict for a specific test instance.

    Args:
        test_id: From ``list_test_ids()``, e.g. ``fib_7`` or ``shmib_4``.
        seed:    Ignored (API compatibility).
        **kwargs: Ignored (e.g. legacy ``size=``).

    Returns:
        (question, tools) where ``tools`` maps names to ``fibonacci`` and ``shmibonacci``.
    """
    _ = seed, kwargs
    kind, n = _parse_task_id(test_id)
    if kind == 'fib':
        question = (
            f'Using the tools, what is the Fibonacci number F({n}) '
            f'(with F(0)=0, F(1)=1)? Reply with the integer only.'
        )
    else:
        question = (
            f'Using the tools, what is the shmibonacci number S({n}) '
            f'(with S(0)=1, S(1)=1, same recurrence as Fibonacci)? Reply with the integer only.'
        )
    tools = {'fibonacci': fibonacci, 'shmibonacci': shmibonacci}
    return question, tools

# %%
def _gold_for_task(test_id: str) -> int:
    kind, n = _parse_task_id(test_id)
    if kind == 'fib':
        return fibonacci(n)
    return shmibonacci(n)


def score(test_id: str, output: str | int, seed: int = 0, **kwargs) -> tuple[float, float]:
    """Score a model output against gold computed with ``fibonacci`` / ``shmibonacci``.

    Args:
        test_id: Row ID from ``list_test_ids()``.
        output:  Model answer (string or int).
        seed:    Ignored (API compatibility).
        **kwargs: Ignored (e.g. legacy ``size=``).

    Returns:
        (f1, em) — both 1.0 on exact match, else 0.0.
    """
    _ = seed, kwargs
    gold = _gold_for_task(test_id)
    if isinstance(output, str):
        pred = output.strip()
    else:
        pred = str(int(output))
    ok = pred == str(gold)
    return (1.0 if ok else 0.0, 1.0 if ok else 0.0)
