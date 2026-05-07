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
#     display_name: deep-reasoner
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DeepReasoner
#
# An iterative symbolic reasoning LM agent

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import inspect
import re
from copy import deepcopy
from typing import Any, Callable, Optional

import coolname
import structlog

from deep_reasoner.code_exec import ExecutionContext
from deep_reasoner.core import ContextGroup, TrackedVar
import asyncio as _asyncio

from deep_reasoner.llm import AsyncCaller, _run_sync, assistant, jinja_render, make_llm, system, user
from juplit import test

logger = structlog.get_logger(__name__)


def _live_agent_integration_tests() -> bool:
    """Notebook-style checks that call a real API (opt-in: set DEEP_REASONER_LIVE_TESTS=1)."""
    import os

    return os.environ.get("DEEP_REASONER_LIVE_TESTS", "").strip().lower() in ("1", "true", "yes")

# %% [markdown]
# ## Code extraction

# %%
START_REPL = "<repl>"
END_REPL = "</repl>"


def get_code(plan_text: str) -> tuple[str | None, bool]:
    """Extract code from the **last** ``<repl>`` … ``</repl>`` span in *plan_text*.

    Uses a greedy ``.*`` prefix so that any earlier ``<repl>`` mentions in the
    reasoning trace are skipped — only the final block is captured.

    The closing ``</repl>`` may be missing when generation ends at EOS. Then
    *code* is the text after the last ``<repl>`` and *ended_with_eos* is ``True``.
    """
    esc_start = re.escape(START_REPL)
    esc_end = re.escape(END_REPL)
    match = re.search(rf".*{esc_start}(.*?)({esc_end}|$)", plan_text, re.DOTALL)
    if not match:
        return None, False
    code = match.group(1).strip()
    ended_with_eos = match.group(2) == ""
    return (code or None), ended_with_eos

# %%
if test():
    # ── Test 1: Normal case — single <repl> block
    t1 = f"I know the answer.\n\n{START_REPL}\nFinalAnswer(1969)\n{END_REPL}"
    assert get_code(t1) == ("FinalAnswer(1969)", False)

    # ── Test 2: <repl> mentioned in reasoning, then actual block
    t2 = f"I need to open a {START_REPL} block.\n\n{START_REPL}\nx = 42\nFinalAnswer(x)\n{END_REPL}"
    assert get_code(t2) == ("x = 42\nFinalAnswer(x)", False)

    # ── Test 3: No <repl> at all
    t3 = "I think the answer is 42 but I forgot to write code."
    assert get_code(t3) == (None, False)

    # ── Test 4: Unclosed block — EOS without literal </repl>
    t4 = f"Let me compute this.\n\n{START_REPL}\nFinalAnswer(42)"
    assert get_code(t4) == ("FinalAnswer(42)", True)

    # ── Test 5: Full fake <repl>...</repl> in reasoning + real block
    t5 = f'Like {START_REPL}print("hello"){END_REPL} but different:\n\n{START_REPL}\nFinalAnswer("world")\n{END_REPL}'
    assert get_code(t5) == ('FinalAnswer("world")', False)

    print("All get_code tests passed ✓")

# %% [markdown]
# ## Var and Func wrappers

# %%
class Var:
    def __init__(self, value: Any, description: str = ""):
        self.value = value
        self.description = description

    @property
    def type(self) -> str:
        return type(self.value).__qualname__

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"value={repr(self.value)}, type={self.type}, "
            f"description={repr(self.description)})"
        )


class Func:
    def __init__(self, value: Callable, description: Optional[str] = None):
        self.value = value
        self.description = (
            description if description is not None else (inspect.getdoc(value) or "")
        )

    @property
    def signature(self) -> str:
        name = getattr(self.value, "__name__", "func")
        try:
            return f"{name}{inspect.signature(self.value)}"
        except (TypeError, ValueError):
            return f"{name}(...)"

    def __repr__(self) -> str:
        fn_name = getattr(self.value, "__name__", repr(self.value))
        return (
            f"{self.__class__.__name__}("
            f"value={fn_name}, signature={self.signature}, "
            f"description={repr(self.description)})"
        )

# %%
if test():
    v = Var(42, "the answer")
    assert v.type == "int"
    assert "42" in repr(v)
    assert "the answer" in repr(v)
    print(repr(v))

    def _add(a, b):
        """Add two numbers."""
        return a + b

    f = Func(_add)
    assert f.description == "Add two numbers."
    assert "_add" in f.signature
    assert "description" in repr(f)
    print(repr(f))

    f2 = Func(_add, description="custom desc")
    assert f2.description == "custom desc"
    print(repr(f2))

# %% [markdown]
# ## Agent context tracking

# %%
def _fresh_name() -> str:
    return coolname.generate_slug(2)


agent_context = ContextGroup(
    # global node counter — shared across all nested contexts
    TrackedVar(
        "node_id",
        default=0,
        shared=True,
        derive=lambda parent, _: parent + 1,
    ),
    # fresh readable name per context
    TrackedVar(
        "node_name",
        default="root",
        derive=lambda parent, _: _fresh_name(),
    ),
    # depth
    TrackedVar(
        "depth",
        default=0,
        derive=lambda parent, _: parent + 1,
    ),
    # ancestry: tuple of node_ids from root to current
    TrackedVar(
        "ancestry",
        default=(),
        derive=lambda parent, inputs: parent + (inputs["node_id"],),
    ),
)

# %% [markdown]
# ## LLM call logging wrapper

# %%
class _LoggedCaller:
    """Drop-in for :class:`AsyncCaller` that logs every call as its own agent node.

    Both single calls (``llm(msg)``) and each member of a batch
    (``llm.batch([m1, m2, …])``) bind a fresh ``agent_context`` entry and emit
    an ``agent.loop`` debug event, so the ``agent_log_processor`` writes a YAML
    file per call — identical in format to ``plan_exec`` sub-agent files.
    """

    def __init__(self, caller: AsyncCaller):
        self._caller = caller

    async def _logged_call(self, messages, **kwargs) -> str:
        with agent_context.bind():
            msgs = [user(messages)] if isinstance(messages, str) else list(messages)
            response = await self._caller._fn(msgs, **kwargs)
            logger.debug("agent.loop", messages=msgs + [assistant(response)])
            return response

    def __call__(self, messages, **kwargs) -> str:
        return _run_sync(self._logged_call(messages, **kwargs))

    async def call_async(self, messages, **kwargs) -> str:
        return await self._logged_call(messages, **kwargs)

    def batch(self, messages_list: list, **kwargs) -> list:
        """Run each message concurrently; each produces its own agent node."""
        return list(_run_sync(_asyncio.gather(
            *[self._logged_call(m, **kwargs) for m in messages_list]
        )))

# %% [markdown]
# ## PlanExec

# %%
class PlanExec:
    """Run a deep reasoning sub-agent.

    Available as ``plan_exec`` in the REPL. Supports two usage styles:

    **Single task** (runs immediately, returns the result):

        result = plan_exec("Describe the task in natural language",
                           var1=Var(data, "description"),
                           func1=Func(fn, "description"))

    **Batch** (queues tasks and runs them concurrently, returns list in order):

        plan_exec.add_task("task A", var1=Var(...))
        plan_exec.add_task("task B", var2=Var(...))
        result_a, result_b = plan_exec.run_all()

    Kwargs can be plain values or wrapped in ``Var()``/``Func()``. Unwrapped
    values are auto-wrapped (callables → Func, everything else → Var). Wrap
    explicitly to attach a description the sub-agent will see.
    Use ``namespaces=[...]`` to restrict which mental models the sub-agent sees.
    """

    __name__ = "plan_exec"

    def __init__(self, deepreasoner: "DeepReasoner"):
        self._dr = deepreasoner
        self._pending: list[tuple] = []  # (task, kwargs, sub_namespaces)

    def _autowrap(self, kwargs) -> dict:
        """Auto-wrap unwrapped kwargs: callables → Func, everything else → Var."""
        wrapped = {}
        for name, val in kwargs.items():
            if isinstance(val, (Var, Func)):
                wrapped[name] = val
            elif callable(val):
                wrapped[name] = Func(val)
            else:
                wrapped[name] = Var(val)
        return wrapped

    def _resolve(self, namespaces, kwargs) -> list:
        """Filter self._dr.models by namespace regex patterns."""
        requested = [str(n) for n in (namespaces or [])]
        if requested:
            return [
                m for m in self._dr.models
                if any(
                    re.search(pat, m.get("name", "") if isinstance(m, dict) else getattr(m, "name", ""))
                    for pat in requested
                )
            ]
        return self._dr.models

    def __call__(self, task: str, namespaces: Optional[list] = None, **kwargs) -> Any:
        """Run a single sub-agent task immediately (sync)."""
        kwargs = self._autowrap(kwargs)
        sub_models = self._resolve(namespaces, kwargs)
        return _run_sync(self._dr._run_async(task, vars=kwargs, mental_models=sub_models))

    async def call_async(self, task: str, namespaces: Optional[list] = None, **kwargs) -> Any:
        """Run a single sub-agent task asynchronously."""
        kwargs = self._autowrap(kwargs)
        sub_models = self._resolve(namespaces, kwargs)
        return await self._dr._run_async(task, vars=kwargs, mental_models=sub_models)

    def add_task(self, task: str, namespaces: Optional[list] = None, **kwargs) -> None:
        """Queue a sub-agent task for concurrent execution via :meth:`run_all`."""
        kwargs = self._autowrap(kwargs)
        sub_models = self._resolve(namespaces, kwargs)
        self._pending.append((task, kwargs, sub_models))

    def run_all(self) -> list:
        """Run all queued tasks concurrently and return results in submission order.

        The pending queue is cleared after each call.
        """
        tasks, self._pending = self._pending, []
        return list(_run_sync(_asyncio.gather(*(
            self._dr._run_async(t, vars=kw, mental_models=sm)
            for t, kw, sm in tasks
        ))))

# %%
if test():
    # PlanExec docstring should appear as the Func description when no explicit
    # description is passed — so the planner sees usage instructions automatically.
    import inspect
    _pe = PlanExec.__new__(PlanExec)  # don't call __init__; just test docstring path

    class _FakeFunc:
        def __init__(self, value, description=None):
            self.description = description if description is not None else (inspect.getdoc(value) or "")

    f = _FakeFunc(_pe)
    assert "Single task" in f.description, f"Expected PlanExec docstring in Func.description, got: {f.description!r}"
    assert "plan_exec.add_task" in f.description
    assert "Var()" in f.description
    print("PlanExec docstring propagates to Func.description ✓")
    print(f.description[:200])


# %% [markdown]
# ## DeepReasoner

# %%
class DeepReasoner:
    """An iterative symbolic reasoning LM agent (CodeAct loop).

    The planner writes reasoning text followed by a ``<repl>...</repl>`` code
    block. Code executes in a persistent REPL; captured stdout (and any
    traceback) is fed back as an ``<observation>`` message. The loop ends when
    the code calls ``FinalAnswer(value)`` — that value is returned.

    Parameters
    ----------
    planner_llm:
        An :class:`~deep_reasoner.llm.AsyncCaller` instance (returned by
        :func:`~deep_reasoner.llm.make_llm`).  Must expose a ``call_async``
        coroutine method.
    system_prompt:
        Jinja2 template for the system message.
    init_vars:
        Variables and functions injected into the REPL and listed in the system
        prompt. Must be wrapped in ``Var`` or ``Func`` classes.
    max_iter:
        Maximum planning iterations before giving up.
    **kwargs:
        Additional keyword arguments used to render the system prompt.
    """

    def __init__(
        self,
        planner_llm: Callable,
        system_prompt: str,
        init_vars: dict[str, Any] | None = None,
        max_iter: int = 30,
        **kwargs: Any,
    ):
        self._planner_llm = planner_llm
        self.history = [system(system_prompt)]
        self.init_vars = init_vars or {}
        self.template_vars = kwargs or {}
        self.models = self.template_vars.get("models", [])
        self.max_iter = max_iter

    def __call__(self, task: str, vars=None, mental_models=None, **kwargs) -> Any:
        """Run the agent synchronously (sync wrapper around :meth:`_run_async`)."""
        return _run_sync(self._run_async(task, vars=vars or {}, mental_models=mental_models, **kwargs))

    async def _run_async(self, task: str, vars=None, mental_models=None, **kwargs) -> Any:
        with agent_context.bind():
            final_value = {}

            def FinalAnswer(value: Any) -> None:
                """Terminate the deep reasoner and return *value* to the caller."""
                final_value["value"] = value

            plan_exec_obj = PlanExec(self)
            vars = self.init_vars | (vars or {}) | {
                "FinalAnswer": Func(FinalAnswer),
                "plan_exec": Func(plan_exec_obj),  # description from PlanExec.__doc__
            }
            # Wrap any AsyncCaller Funcs so their calls are logged as agent nodes.
            vars = {
                k: Func(_LoggedCaller(v.value), description=v.description)
                if isinstance(v, Func) and isinstance(v.value, AsyncCaller)
                else v
                for k, v in vars.items()
            }
            logger.debug("vars", vars=vars)

            vars_values = {var_name: var.value for var_name, var in vars.items()}
            repl = ExecutionContext(**vars_values)

            template_vars = {
                **self.template_vars,
                **(kwargs or {}),
                "funcs": {k: f for k, f in vars.items() if isinstance(f, Func)},
                "vars": {k: v for k, v in vars.items() if isinstance(v, Var)},
            }
            if mental_models is not None:
                template_vars["models"] = mental_models

            messages = [
                {**msg, "content": jinja_render(msg["content"], template_vars)}
                for msg in deepcopy(self.history)
            ]
            messages.append(user(task))

            result = f"Agent failed to produce a final answer within {self.max_iter} iterations."
            for _ in range(self.max_iter):
                plan_text = await self._planner_llm.call_async(messages)
                code, ended_with_eos = get_code(plan_text)

                if code is not None and ended_with_eos:
                    plan_text = plan_text.rstrip() + "\n" + END_REPL

                messages.append(assistant(plan_text))
                logger.debug("agent.loop", messages=messages)

                if code is None:
                    messages.append(
                        user(
                            f"Your response did not contain a `{START_REPL}...{END_REPL}` block. "
                            "Please include one."
                        )
                    )
                    continue

                stdout, tb = repl.run_code(code)

                obs_parts = []
                if stdout.strip():
                    obs_parts.append(stdout.rstrip())
                elif stdout:  # ran but printed only whitespace — make it visible
                    obs_parts.append("(no output)")
                if tb is not None:
                    obs_parts.append(tb.rstrip())
                if obs_parts:
                    obs = "<observation>\n" + "\n".join(obs_parts) + "\n</observation>"
                    messages.append(user(obs))
                    logger.debug("agent.loop", messages=messages)

                if tb is None and "value" in final_value:
                    result = final_value["value"]
                    obs = f"<observation>\nFinalAnswer: {result!r}\n</observation>"
                    messages.append(user(obs))
                    logger.debug("agent.loop", messages=messages)
                    break

            return result


# %% [markdown]
# ### Testing

# %%
if test() and _live_agent_integration_tests():
    import logging
    import nest_asyncio

    import structlog

    from deep_reasoner.core import (
        configure_structlog_fixture,
        live_test_openai_client,
        load_live_test_dotenv,
    )

    nest_asyncio.apply()

    logger = structlog.getLogger(__name__)
    configure_structlog_fixture()
    load_live_test_dotenv()
    client = live_test_openai_client()

# %%
if test() and _live_agent_integration_tests():
    import os
    from structlog.contextvars import bound_contextvars
    from pathlib import Path
    import yaml

    from deep_reasoner.core import package_config_path

    _agent_yaml = package_config_path("debug_agent.yaml")
    agent_config = yaml.safe_load(_agent_yaml.read_text(encoding="utf-8"))
    agent_config.keys()


    agent = DeepReasoner(
        planner_llm=make_llm(client=client,model=os.getenv("MODEL")),
        max_iter=3,
        **agent_config
    )

# %%
if test() and _live_agent_integration_tests():
    doc = Var(""" 
Meeting Minutes - City Planning Commission
Date: 2023-03-15

Attendees:
- Commissioner A
- Commissioner B
- Commissioner C
- Commissioner D

Discussion:
The commission discussed the proposed solar project and its potential benefits for the community.

Votes:
- Commissioner A: Yes
- Commissioner B: Yes
- Commissioner C: No
- Commissioner D: Yes
    """,
        description="city planning commission meeting minutes"
    )

    task = (
        "You have city planning commission meeting minutes in `doc`. "
        "How many commissioners voted in favor of the solar project?"
    )

    with bound_contextvars(benchmark="debug",task_id="test",version="0.0.2"):
        agent_context.reset()
        with checkLogs(namespace='__main__'):
            with checkLogs(namespace='deep_reasoner'):
                result = agent(task,vars={"doc":doc})
                logger.debug("agent.result", answer=result,question=task)
    print(result)


# %%
if test() and _live_agent_integration_tests():
    # ── Test: LLM batch and plan_exec batch ──────────────────────────────────
    from pathlib import Path
    from structlog.contextvars import bound_contextvars
    from deep_reasoner.logging_utils import agent_log_processor, generate_run_id

    _batch_run_id = generate_run_id()
    configure_structlog_fixture(
        console=True,
        extra_processors=[agent_log_processor("../logs")],
    )
    print(f"Logs → ../logs/debug_0.0.2/{_batch_run_id}/batch-test/")

    # 1) LLM batch — two simple arithmetic questions in parallel
    llm = agent._planner_llm
    q1 = [{"role": "user", "content": "What is 3 + 4? Reply with just the number."}]
    q2 = [{"role": "user", "content": "What is 5 + 6? Reply with just the number."}]
    batch_results = llm.batch([q1, q2])
    assert len(batch_results) == 2
    assert "7" in batch_results[0]
    assert "11" in batch_results[1]
    print("LLM batch test passed ✓", batch_results)

    # 2) plan_exec batch — two sub-agent tasks run concurrently
    doc_a = Var("Commissioner A: Yes\nCommissioner B: No", description="votes doc A")
    doc_b = Var("Commissioner X: Yes\nCommissioner Y: Yes\nCommissioner Z: No", description="votes doc B")

    with bound_contextvars(benchmark="debug", task_id="batch-test", version="0.0.2", run_id=_batch_run_id):
        agent_context.reset()
        pe = PlanExec(agent)
        pe.add_task("Count how many commissioners voted Yes in `doc_a`. Call FinalAnswer(n).", doc_a=doc_a)
        pe.add_task("Count how many commissioners voted Yes in `doc_b`. Call FinalAnswer(n).", doc_b=doc_b)
        results = pe.run_all()

    assert len(results) == 2
    assert results[0] == 1, f"Expected 1, got {results[0]}"
    assert results[1] == 2, f"Expected 2, got {results[1]}"
    print("plan_exec batch test passed ✓", results)

# %%
if test():
    # ── Test: namespaces= filter is applied to child sub-agent system prompt ────
    #
    # Creates an agent with 2 mental models. A scripted LLM makes the root agent
    # call plan_exec(..., namespaces=["model alpha"]) — restricting the child to only
    # that model. We then inspect the YAML log files and assert:
    #   - Root's system prompt contains both "model alpha" and "model beta"
    #   - Child's system prompt contains "model alpha" but NOT "model beta"
    import shutil
    import tempfile
    from pathlib import Path as _Path
    from structlog.contextvars import bound_contextvars
    from deep_reasoner.core import configure_structlog_fixture, package_config_path
    from deep_reasoner.logging_utils import agent_log_processor, agent_run_log_dir, generate_run_id
    import yaml as _yaml

    _debug_yaml = package_config_path("debug_agent.yaml")
    agent_config = _yaml.safe_load(_debug_yaml.read_text(encoding="utf-8"))

    _model_alpha = {"name": "model alpha", "content": "Alpha: compute things directly."}
    _model_beta  = {"name": "model beta",  "content": "Beta: use llm.batch for classification."}

    # Scripted LLM — no real API call, deterministic
    _script_calls = [0]
    async def _scripted_fn(messages, **kwargs):
        _script_calls[0] += 1
        sys_content = messages[0].get("content", "") if isinstance(messages[0], dict) else ""
        if "model beta" in sys_content:
            # Root agent (sees both models) → delegate with only model_alpha
            return (
                "I will delegate to a sub-agent restricted to model alpha.\n"
                "<repl>\n"
                "result = plan_exec('complete the subtask', namespaces=['model alpha'])\n"
                "FinalAnswer(result)\n"
                "</repl>"
            )
        else:
            # Child agent (should only see model_alpha) → return immediately
            return "<repl>\nFinalAnswer('subtask done')\n</repl>"

    _scripted_llm = AsyncCaller(_scripted_fn)

    _log_dir = _Path(tempfile.mkdtemp())
    _run_id = generate_run_id()
    configure_structlog_fixture(
        console=False,
        extra_processors=[agent_log_processor(str(_log_dir))],
    )

    _filter_agent = DeepReasoner(
        planner_llm=_scripted_llm,
        max_iter=5,
        **{**agent_config, "models": [_model_alpha, _model_beta]},
    )

    with bound_contextvars(benchmark="debug", task_id="models-filter-test", version="test", run_id=_run_id):
        agent_context.reset()
        _filter_result = _filter_agent("Test that the namespaces= filter works.")

    assert _filter_result == "subtask done", f"Expected 'subtask done', got {_filter_result!r}"

    # Inspect log files: root (n_1) and child (n_2)
    _run_log_dir = agent_run_log_dir(_log_dir, "models-filter-test")
    _node_files = sorted(_run_log_dir.glob("n_*.yaml"))
    assert len(_node_files) == 2, f"Expected 2 node log files, got {len(_node_files)}: {_node_files}"

    _root_sys  = _yaml.safe_load(_node_files[0].read_text())["messages"][0]["system"]
    _child_sys = _yaml.safe_load(_node_files[1].read_text())["messages"][0]["system"]

    assert "model alpha" in _root_sys and "model beta" in _root_sys, \
        "Root agent should see both mental models in its system prompt"
    assert "model alpha" in _child_sys, \
        "Child agent should see 'model alpha' in its system prompt"
    assert "model beta" not in _child_sys, \
        f"Child agent should NOT see 'model beta', but found it in:\n{_child_sys[:500]}"

    shutil.rmtree(_log_dir, ignore_errors=True)
    print("namespaces= filter test passed ✓")
    print("  Root agent sees both models ✓")
    print("  Child agent sees only 'model alpha' ✓")

# %%
if test() and _live_agent_integration_tests():
    # ── Test: end-to-end parallel execution — plan_exec batch + llm.batch ────
    #
    # Two separate async wrappers distinguish call types:
    #   _plan_timed  — used as planner_llm  → records type='PE' (agent reasoning)
    #   _llm_timed   — injected as `llm` var → records type='LLM' (direct batch calls)
    #
    # Each record also captures structlog context (depth, node_id) so codes look
    # like PE_d1_01 (top-level reasoning), PE_d2_03 (sub-agent reasoning),
    # LLM_d2_04 (llm.batch call from sub-agent REPL).
    import time as _time
    import structlog.contextvars as _sctx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from pathlib import Path
    from structlog.contextvars import bound_contextvars
    from deep_reasoner.logging_utils import agent_log_processor, generate_run_id

    _records: list[dict] = []
    _seq = [0]
    _real_fn = agent._planner_llm._fn

    def _record(call_type, t0, t1, result):
        ctx = _sctx.get_contextvars()
        depth = ctx.get("depth", 0)
        node_id = ctx.get("node_id", 0)
        _seq[0] += 1
        code = f"{_seq[0]:02d}_{call_type}_d{depth}"
        _records.append({"code": code, "type": call_type, "depth": depth,
                          "node_id": node_id, "start": t0, "end": t1, "content": result})

    async def _plan_timed(messages, **kwargs):
        t0 = _time.monotonic()
        result = await _real_fn(messages, **kwargs)
        _record("PE", t0, _time.monotonic(), result)
        return result

    async def _llm_timed(messages, **kwargs):
        t0 = _time.monotonic()
        result = await _real_fn(messages, **kwargs)
        _record("LLM", t0, _time.monotonic(), result)
        return result

    _plan_llm  = AsyncCaller(_plan_timed)
    _direct_llm = AsyncCaller(_llm_timed)

    _group_a = ["What is 3 + 4? Reply with just the number.",
                "What is 5 + 6? Reply with just the number."]
    _group_b = ["What is 100 - 37? Reply with just the number.",
                "What is 2 ** 8? Reply with just the number."]

    _timed_agent = DeepReasoner(
        planner_llm=_plan_llm,
        init_vars={
            "llm":  Func(_direct_llm,
                         "Call the LLM. llm(prompt) for one call; "
                         "llm.batch([p1, p2, ...]) for concurrent calls."),
            "Var":  Func(Var,  "Wrap a value: Var(value, description)"),
            "Func": Func(Func, "Wrap a callable: Func(fn, description)"),
            "group_a": Var(_group_a, "arithmetic questions for group A"),
            "group_b": Var(_group_b, "arithmetic questions for group B"),
        },
        max_iter=5,
        **agent_config,
    )

    _task = """\
You have two groups of arithmetic questions pre-loaded as `group_a` and `group_b`.
Start by printing both to confirm their contents.

Your job:
1. Launch one plan_exec sub-task per group using plan_exec.add_task / plan_exec.run_all()
   so both groups are processed concurrently.
2. Inside each sub-task, use llm.batch(questions) to answer all questions in that group
   concurrently. FinalAnswer the list of raw LLM replies.
"""

    _run_id = generate_run_id()
    configure_structlog_fixture(
        console=True,
        extra_processors=[agent_log_processor("../logs")],
    )
    print(f"Logs → ../logs/debug_0.0.2/{_run_id}/parallel-batch-test/")

    _records.clear()
    _seq[0] = 0
    agent_context.reset()
    _wall_t0 = _time.monotonic()
    with bound_contextvars(
        benchmark="debug", task_id="parallel-batch-test", version="0.0.2", run_id=_run_id
    ):
        _result = _timed_agent(_task)
    _wall_time = _time.monotonic() - _wall_t0

    # ── Parallelism assertion ─────────────────────────────────────────────────
    assert len(_records) >= 4, f"Expected ≥4 LLM calls, got {len(_records)}"

    def _overlaps(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]

    _overlapping = [(i, j) for i in range(len(_records))
                    for j in range(i + 1, len(_records))
                    if _overlaps(_records[i], _records[j])]
    assert _overlapping, (
        "No overlapping LLM call intervals — calls were fully sequential.\n"
        + "\n".join(f"  {r['code']}  {r['start']:.3f}–{r['end']:.3f}" for r in _records)
    )
    _serial_time = sum(r["end"] - r["start"] for r in _records)
    _speedup = _serial_time / _wall_time
    print(f"Parallelism confirmed ✓  {len(_records)} calls, "
          f"{len(_overlapping)} overlapping pairs\n"
          f"  Wall time:        {_wall_time:.2f}s\n"
          f"  Sum of LLM calls: {_serial_time:.2f}s\n"
          f"  Speedup:          {_speedup:.1f}x\n"
          f"  Result: {_result}")

    # ── Interval plot ─────────────────────────────────────────────────────────
    _t0 = min(r["start"] for r in _records)
    _colors = {"PE": "#4C72B0", "LLM": "#DD8452"}
    _depth_alpha = {0: 1.0, 1: 1.0, 2: 0.65, 3: 0.4}

    _fig, _ax = plt.subplots(figsize=(10, max(3, 0.55 * len(_records) + 1)))
    for _r in _records:
        _ax.barh(
            _r["code"],
            _r["end"] - _r["start"],
            left=_r["start"] - _t0,
            color=_colors[_r["type"]],
            alpha=_depth_alpha.get(_r["depth"], 0.4),
            height=0.6,
            edgecolor="white",
        )
    _ax.set_xlabel("Time (s, relative to first call)")
    _ax.set_title("LLM completion intervals  (PE = agent reasoning · LLM = llm.batch)")
    _ax.legend(handles=[mpatches.Patch(color=c, label=t) for t, c in _colors.items()])
    plt.tight_layout()
    plt.show()

    # ── Completions dict ──────────────────────────────────────────────────────
    _completions = {r["code"]: r["content"] for r in sorted(_records, key=lambda r: r["code"])}
    from pprint import pprint
    pprint(_completions, width=120)

# %%
# # ! poe sync
