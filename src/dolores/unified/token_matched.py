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
# # token_matched — per-attempt token budget and sampling
#
# Both arms of the token-fair comparison hang off one seam: `obs.py` already
# patches every litellm/openai completion call, so a single `RunBudget` installed
# for the duration of one attempt can
#
# 1. **inject** `seed` / `temperature` on the way in — Arm B's *different but
#    reproducible* attempts; and
# 2. **count** `total_tokens` on the way out, raising `BudgetExhausted` when the
#    attempt has spent its allowance — Arm A's compute budget.
#
# One attempt is one subprocess (the runner fans out that way), so a process-wide
# budget is exactly the right scope.

# %%
from juplit import test

# %%
import hashlib
import threading
from contextlib import contextmanager

# %% [markdown]
# ## BudgetExhausted


# %%
class BudgetExhausted(BaseException):
    """Raised inside a patched completion call when the attempt's budget is spent.

    Subclasses ``BaseException`` on purpose: agent frameworks wrap each step in
    ``except Exception``, so an ``Exception`` here would be swallowed and the run
    would keep spending past its budget.
    """


class ContextWindowExceeded(BaseException):
    """Raised inside a patched completion call when the prompt overshot the model's
    context window (the backend returns a context-length 400).

    Subclasses ``BaseException`` for the same reason as ``BudgetExhausted``: agent
    frameworks' ``except Exception`` must not swallow it and retry the same
    oversized prompt forever. The attempt is ended cleanly instead, so the work
    pool records a result and does not respawn the child indefinitely.
    """


# %% [markdown]
# ## attempt_seed


# %%
# 2**31 - 1: vLLM/OpenAI reject seeds outside signed 32-bit.
_SEED_MODULUS = 2**31 - 1


def attempt_seed(task_id: str, attempt: int) -> int:
    """Deterministic sampling seed for one attempt of one task.

    Uses ``hashlib`` rather than ``hash()`` so the value is stable across
    processes and machines — re-running or retrying attempt *i* regenerates that
    attempt instead of drawing a fresh one.
    """
    digest = hashlib.sha256(f"{task_id}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


# %%
if test():
    assert attempt_seed("t1", 0) == attempt_seed("t1", 0)
    assert attempt_seed("t1", 0) != attempt_seed("t1", 1)
    assert attempt_seed("t1", 0) != attempt_seed("t2", 0)
    assert 0 <= attempt_seed("t1", 7) < _SEED_MODULUS

# %% [markdown]
# ## RunBudget


# %%
def _assistant_text(response) -> str | None:
    """The assistant message text of a completion response, if it has one."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return None
    text = str(content).strip() if content else ""
    return text or None


class RunBudget:
    """Token budget and sampling parameters for a single attempt."""

    def __init__(
        self,
        budget_tokens: int | None = None,
        seed: int | None = None,
        temperature: float | None = None,
    ):
        if budget_tokens is not None and budget_tokens <= 0:
            raise ValueError(f"token_budget must be positive, got {budget_tokens}")
        self.budget_tokens = budget_tokens
        self.seed = seed
        self.temperature = temperature
        self.spent = 0
        self.last_text: str | None = None
        self.exhausted = False

    def before_call(self, kwargs: dict) -> dict:
        """Inject sampling parameters the caller did not set. Runs on every call."""
        if self.exhausted:
            raise BudgetExhausted(f"budget of {self.budget_tokens} tokens already spent")
        if self.seed is not None and kwargs.get("seed") is None:
            kwargs["seed"] = self.seed
        if self.temperature is not None and kwargs.get("temperature") is None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def after_call(self, response, usage: dict) -> None:
        """Account one call's tokens; raise once the budget is gone."""
        self.spent += usage.get("total_tokens") or 0
        text = _assistant_text(response)
        if text:
            self.last_text = text
        if self.budget_tokens is not None and self.spent >= self.budget_tokens:
            self.exhausted = True
            raise BudgetExhausted(
                f"spent {self.spent} tokens of a {self.budget_tokens} token budget"
            )


# %% [markdown]
# ## Installing a budget for one attempt

# %%
_local = threading.local()


def active_budget() -> RunBudget | None:
    """The budget installed for the current attempt, or None when neither arm is on."""
    return getattr(_local, "budget", None)


@contextmanager
def run_budget(budget_tokens=None, seed=None, temperature=None):
    """Install a `RunBudget` for the duration of one attempt, then clear it."""
    previous = active_budget()
    budget = RunBudget(budget_tokens=budget_tokens, seed=seed, temperature=temperature)
    _local.budget = budget
    try:
        yield budget
    finally:
        _local.budget = previous


# %% [markdown]
# ## Tests


# %%
class _FakeResponse:
    """Minimal stand-in for an OpenAI completion response."""

    def __init__(self, text: str | None):
        message = type("_M", (), {"content": text})()
        self.choices = [type("_C", (), {"message": message})()]


def test_before_call_injects_sampling_params():
    budget = RunBudget(seed=42, temperature=0.7)
    assert budget.before_call({}) == {"seed": 42, "temperature": 0.7}

    # A caller that set its own values keeps them.
    kwargs = budget.before_call({"seed": 1, "temperature": 0.1})
    assert kwargs == {"seed": 1, "temperature": 0.1}

    # Nothing configured -> nothing injected.
    assert RunBudget().before_call({}) == {}


def test_after_call_accumulates_and_raises():
    budget = RunBudget(budget_tokens=100)
    budget.after_call(_FakeResponse("partial answer"), {"total_tokens": 60})
    assert budget.spent == 60
    assert budget.last_text == "partial answer"
    assert not budget.exhausted

    try:
        budget.after_call(_FakeResponse("later answer"), {"total_tokens": 50})
        assert False, "should have raised"
    except BudgetExhausted:
        pass
    assert budget.exhausted
    assert budget.spent == 110
    assert budget.last_text == "later answer"

    # Every subsequent call fails fast, before spending anything more.
    try:
        budget.before_call({})
        assert False, "should have raised"
    except BudgetExhausted:
        pass


def test_no_budget_never_raises():
    budget = RunBudget()
    for _ in range(5):
        budget.after_call(_FakeResponse("x"), {"total_tokens": 10_000})
    assert budget.spent == 50_000
    assert not budget.exhausted


def test_rejects_nonpositive_budget():
    for bad in (0, -1):
        try:
            RunBudget(budget_tokens=bad)
            assert False, "should have raised"
        except ValueError:
            pass


def test_run_budget_installs_and_clears():
    assert active_budget() is None
    with run_budget(budget_tokens=10, seed=3) as budget:
        assert active_budget() is budget
        assert budget.seed == 3
    assert active_budget() is None


def test_assistant_text_handles_empty_response():
    assert _assistant_text(_FakeResponse(None)) is None
    assert _assistant_text(_FakeResponse("  ")) is None
    assert _assistant_text(_FakeResponse(" hi ")) == "hi"


# %%
if test():
    test_before_call_injects_sampling_params()
    test_after_call_accumulates_and_raises()
    test_no_budget_never_raises()
    test_rejects_nonpositive_budget()
    test_run_budget_installs_and_clears()
    test_assistant_text_handles_empty_response()
    print("token_matched tests passed")

# %% [markdown]
# ## End
