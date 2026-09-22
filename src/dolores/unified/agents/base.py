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
# # Agent framework — `AgentCfg`, `Agent`, `AgentResult`, registry
#
# The uniform interface every baseline is driven through. One `AgentCfg` carries
# the connection (`inference`), the `benchmark`, the `model`, and shared knobs;
# agent-specific extras flow through `extra="allow"`. Each agent subclasses
# `Agent`, sets a `name`, and implements `_answer(task, log_dir) -> answer`. The
# `run()` template method owns all the boilerplate every old `run_single`
# repeated: run id, log dir, `install_patches`/`install_run_logger`,
# `write_initial_qa`, the `agent_context` frame, scoring, and the shared
# `qa.json` schema (`AgentResult`).

# %%
from juplit import test

# %%
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from dolores.unified.benchmarks import Benchmark, Task
from dolores.unified.inference import InferenceBackend
from dolores.unified.token_matched import (
    BudgetExhausted,
    ContextWindowExceeded,
    attempt_seed,
    run_budget,
)
from dolores.unified.obs import (
    agent_context,
    install_patches,
    install_run_logger,
    make_run_id,
    run_log_dir,
    write_initial_qa,
)

# %% [markdown]
# ## AgentResult — the shared qa.json schema

# %%
class AgentResult(BaseModel):
    """The shared ``qa.json`` payload every agent writes."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str
    question: str
    expected_answer: Any = None
    answer: Any = None
    parsed: Any = None
    scores: Any = None
    attempt: int = 0
    stop_reason: Literal["finished", "out_of_budget", "context_exceeded"] = "finished"
    tokens_spent: int = 0
    seed: int | None = None


# %% [markdown]
# ## AgentCfg

# %%
class AgentCfg(BaseModel):
    """Config passed to every agent.

    Typed fields are shared by all agents; agent-specific extras (e.g.
    deep_reasoner's ``agent_config`` or rlm's ``max_depth``) flow through
    ``extra="allow"`` and are read with ``cfg.get("key", default)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    model: str
    inference: InferenceBackend
    benchmark: Benchmark
    max_steps: int = 75
    no_thinking: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        """Read an agent-specific extra (or typed field), with a default."""
        return getattr(self, key, default)


# %% [markdown]
# ## Agent ABC

# %%
class Agent(ABC):
    """Base class for all agents. Subclasses set ``name`` and implement ``_answer``."""

    name: ClassVar[str] = ""

    def __init__(self, cfg: AgentCfg):
        self.cfg = cfg

    @abstractmethod
    def _answer(self, task: Task, log_dir: Path) -> Any:
        """Run the agent on *task* and return the raw answer (stored + scored)."""
        ...

    def _parsed(self, answer: Any) -> Any:
        """Post-process the answer for the ``parsed`` field. Default: phantomwiki split."""
        from dolores.unified.agents._helpers import _parsed_for_phantomwiki

        return _parsed_for_phantomwiki(self.cfg.benchmark, answer)

    def _instructions(self, default: str | None) -> str | None:
        """The run config's inlined prompt when it supplies one, else *default*."""
        return self.cfg.get("instructions") or default

    def _sampling(self, task: Task, attempt: int) -> tuple[int | None, float | None]:
        """``(seed, temperature)`` for this attempt — both None unless sampling is on."""
        cfg = self.cfg
        temperature = cfg.get("temperature")
        if cfg.get("samples", 1) > 1:
            if temperature is None:
                raise ValueError(
                    "samples > 1 needs a temperature — k greedy attempts would be identical"
                )
            return attempt_seed(task.task_id, attempt), temperature
        return (None, temperature)

    def run(self, task: Task, attempt: int = 0) -> AgentResult:
        """Template method: set up the run, call ``_answer``, score, write ``qa.json``."""
        cfg = self.cfg
        bench = cfg.benchmark
        run_id = make_run_id()
        log_dir = run_log_dir(
            bench.benchmark_name(), self.name, cfg.model, run_id,
            no_thinking=cfg.no_thinking,
            base=Path(cfg.get("log_dir")) if cfg.get("log_dir") else None,
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        write_initial_qa(
            log_dir, task_id=task.task_id, question=task.question,
            expected_answer=task.gold,
        )
        install_patches()
        install_run_logger()

        seed, temperature = self._sampling(task, attempt)
        with agent_context.bind(agent=self.name, run_id=run_id, log_dir=str(log_dir)), \
                run_budget(cfg.get("token_budget"), seed, temperature) as budget:
            try:
                answer = self._answer(task, log_dir)
                stop_reason = "finished"
            except BudgetExhausted:
                # Report the best answer seen so far rather than nothing.
                answer = budget.last_text or ""
                stop_reason = "out_of_budget"
            except ContextWindowExceeded:
                # The prompt overshot the model's context window. End the attempt
                # cleanly with the best answer so far so the work pool records a
                # result instead of respawning the same oversized prompt forever.
                answer = budget.last_text or ""
                stop_reason = "context_exceeded"

        parsed = self._parsed(answer)
        scores = bench.score(task.task_id, answer)
        result = AgentResult(
            task_id=task.task_id,
            question=task.question,
            expected_answer=task.gold,
            answer=answer,
            parsed=parsed,
            scores=scores,
            attempt=attempt,
            stop_reason=stop_reason,
            tokens_spent=budget.spent,
            seed=seed,
        )
        (log_dir / "qa.json").write_text(
            json.dumps(result.model_dump(), indent=2, ensure_ascii=False, default=str)
        )
        return result


# %% [markdown]
# ## Registry

# %%
_AGENT_REGISTRY: dict[str, type[Agent]] = {}


def register_agent(cls: type[Agent]) -> type[Agent]:
    """Class decorator: register an Agent subclass under its ``name``."""
    name = (cls.name or "").strip().lower()
    if not name:
        raise ValueError(f"{cls.__name__}.name must be a non-empty string")
    _AGENT_REGISTRY[name] = cls
    return cls


def get_agent_cls(name: str) -> type[Agent]:
    key = name.strip().lower()
    if key not in _AGENT_REGISTRY:
        known = ", ".join(sorted(_AGENT_REGISTRY))
        raise KeyError(f"Unknown agent {name!r}. Available: [{known}]")
    return _AGENT_REGISTRY[key]


def list_agents() -> list[str]:
    return sorted(_AGENT_REGISTRY)


# %% [markdown]
# ## Tests

# %%
def test_agent_cfg_typed_and_extras():
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    cfg = AgentCfg(
        model="stub",
        inference=OpenAIBackend(api_base="http://x", api_key="k"),
        benchmark=HelloWorld(),
        max_steps=10,
        agent_config="configs/agents/debug.yaml",  # extra
    )
    assert cfg.model == "stub"
    assert cfg.max_steps == 10
    assert cfg.no_thinking is False
    assert cfg.get("agent_config") == "configs/agents/debug.yaml"
    assert cfg.get("missing", 7) == 7
    assert cfg.inference.client_kwargs()["base_url"] == "http://x"


def test_registry_round_trip():
    @register_agent
    class _Dummy(Agent):
        name = "dummy_test_agent"

        def _answer(self, task, log_dir):
            return "ok"

    assert "dummy_test_agent" in list_agents()
    assert get_agent_cls("DUMMY_TEST_AGENT") is _Dummy
    try:
        get_agent_cls("nope")
        assert False, "should raise"
    except KeyError:
        pass
    del _AGENT_REGISTRY["dummy_test_agent"]


def test_register_requires_name():
    try:
        @register_agent
        class _NoName(Agent):
            def _answer(self, task, log_dir):
                return ""
        assert False, "should raise"
    except ValueError:
        pass


def test_run_template_writes_qa_and_binds_context():
    import tempfile
    from unittest.mock import patch

    from dolores.unified.benchmarks import HelloWorld
    from config import Paths
    from dolores.unified.inference import OpenAIBackend

    seen = {}

    class _Echo(Agent):
        name = "echo_test_agent"

        def _answer(self, task, log_dir):
            seen["agent"] = agent_context.agent.get()
            seen["log_dir"] = agent_context.log_dir.get()
            return "13"

    bench = HelloWorld()
    cfg = AgentCfg(model="stub", inference=OpenAIBackend(api_base="http://x", api_key="k"),
                   benchmark=bench)
    with tempfile.TemporaryDirectory() as tmp, patch.object(Paths, "LOGS_DIR", Path(tmp)):
        result = _Echo(cfg).run(bench.get_task("fib_7"))
        assert result.answer == "13"
        assert result.scores == (1.0, 1.0)
        assert seen["agent"] == "echo_test_agent"
        qa_files = list(Path(tmp).rglob("qa.json"))
        assert qa_files
        qa = json.loads(qa_files[0].read_text())
        assert qa["answer"] == "13"
        assert qa["scores"] == [1.0, 1.0]
        # Defaults leave a plain run looking exactly like it did before budgets existed.
        assert qa["attempt"] == 0
        assert qa["stop_reason"] == "finished"
        assert qa["tokens_spent"] == 0
        assert qa["seed"] is None


def test_instructions_prefers_config():
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    class _Stub(Agent):
        name = "instructions_test_agent"

        def _answer(self, task, log_dir):
            return ""

    def _cfg(**extra):
        return AgentCfg(model="stub", benchmark=HelloWorld(),
                        inference=OpenAIBackend(api_base="http://x", api_key="k"), **extra)

    assert _Stub(_cfg())._instructions("from constant") == "from constant"
    assert _Stub(_cfg(instructions="inlined"))._instructions("from constant") == "inlined"
    assert _Stub(_cfg())._instructions(None) is None


def test_run_reports_budget_exhaustion():
    import tempfile
    from unittest.mock import patch

    from dolores.unified.benchmarks import HelloWorld
    from config import Paths
    from dolores.unified.inference import OpenAIBackend
    from dolores.unified.token_matched import active_budget

    class _Spender(Agent):
        """Burns its budget mid-run, having produced one partial answer."""
        name = "budget_test_agent"

        def _answer(self, task, log_dir):
            budget = active_budget()
            budget.last_text = "partial 13"
            raise BudgetExhausted("out of tokens")

    class _Silent(Agent):
        """Exhausts the budget before any assistant text exists."""
        name = "silent_budget_test_agent"

        def _answer(self, task, log_dir):
            raise BudgetExhausted("out of tokens")

    bench = HelloWorld()
    cfg = AgentCfg(model="stub", inference=OpenAIBackend(api_base="http://x", api_key="k"),
                   benchmark=bench, token_budget=100)
    with tempfile.TemporaryDirectory() as tmp, patch.object(Paths, "LOGS_DIR", Path(tmp)):
        result = _Spender(cfg).run(bench.get_task("fib_7"))
        assert result.stop_reason == "out_of_budget"
        assert result.answer == "partial 13"
        assert result.scores is not None

        blank = _Silent(cfg).run(bench.get_task("fib_7"))
        assert blank.stop_reason == "out_of_budget"
        assert blank.answer == ""
        assert blank.scores is not None

    for name in ("budget_test_agent", "silent_budget_test_agent"):
        _AGENT_REGISTRY.pop(name, None)


def test_sampling_requires_temperature():
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    class _Stub(Agent):
        name = "sampling_test_agent"

        def _answer(self, task, log_dir):
            return ""

    bench = HelloWorld()
    task = bench.get_task("fib_7")
    backend = OpenAIBackend(api_base="http://x", api_key="k")

    hot = _Stub(AgentCfg(model="stub", inference=backend, benchmark=bench,
                         samples=5, temperature=0.8))
    seed, temperature = hot._sampling(task, 3)
    assert seed == attempt_seed("fib_7", 3)
    assert temperature == 0.8

    cold = _Stub(AgentCfg(model="stub", inference=backend, benchmark=bench))
    assert cold._sampling(task, 0) == (None, None)

    greedy = _Stub(AgentCfg(model="stub", inference=backend, benchmark=bench, samples=5))
    try:
        greedy._sampling(task, 0)
        assert False, "should have raised"
    except ValueError:
        pass


# %%
if test():
    test_agent_cfg_typed_and_extras()
    test_registry_round_trip()
    test_register_requires_name()
    test_run_template_writes_qa_and_binds_context()
    test_instructions_prefers_config()
    test_run_reports_budget_exhaustion()
    test_sampling_requires_temperature()
    print("agents.base tests passed")

# %% [markdown]
# ## End
