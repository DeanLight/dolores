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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Code Execution
#
# Thread-safe REPL context for executing Python code with captured stdout/stderr.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import io
import traceback as _traceback
from typing import Any, Callable

import structlog

from juplit import test

logger = structlog.get_logger(__name__)

if test():
    import pytest

# %% [markdown]
# ## Partial

# %%
def Partial(fn: Callable, **default_kwargs) -> Callable:
    """Wrap any Callable with some default kwargs, returning a new Callable
    with those kwargs pre-filled.

    Parameters
    ----------
    fn:
        Any callable
    **default_kwargs:
        Parameters forwarded to *fn* on every call (e.g. ``temperature=0``).
    """
    def wrapped(*args, **override_kwargs) -> Any:
        return fn(*args, **{**default_kwargs, **override_kwargs})
    return wrapped

# %%
def _mock_llm(content: str, fruit_of_choice: str) -> str:
    """Keyword-based deterministic mock — swappable with any real LLM."""
    content = content.lower()
    if 'fruit'  in content: return fruit_of_choice
    if 'car'    in content: return 'ferrari'
    if 'animal' in content: return 'rabbit'
    if 'hello'  in content: return 'Hello! How can I help you today?'
    return 'mock response'

if test():
    mock_llm = Partial(_mock_llm, fruit_of_choice='banana')
    print(mock_llm("Hello, how are you?"))


# %% [markdown]
# ## exec_and_capture

# %%
def exec_and_capture(code: str, env: dict) -> tuple[str, str | None]:
    """Execute *code* in *env*, returning ``(stdout, traceback_or_None)``.

    Thread-safe: a per-call ``print`` that writes to a local buffer is
    injected into *env*, so ``sys.stdout`` is never touched.

    Always returns — never raises.
    """
    buf = io.StringIO()
    env["print"] = lambda *a, **kw: print(*a, **kw, file=buf)
    tb = None
    try:
        exec(code, env)
    except Exception:
        tb = _traceback.format_exc()
    return buf.getvalue(), tb

# %%
if test():
    env = {'llm': mock_llm}
    exec_and_capture("result = llm('hello')", env)
    print(env['result'])

# %% [markdown]
# ## ExecutionContext

# %%
class ExecutionContext:
    def __init__(self, **kwargs):
        self.env: dict = kwargs or {}

    def bind(self, **kwargs: Any) -> None:
        self.env.update(kwargs)

    def run_code(self, code: str) -> str:
        """Execute code and return stdout. Raises ``RuntimeError`` on failure."""
        stdout, tb = exec_and_capture(code, self.env)
        return stdout, tb

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self.env)

# %%
if test():
    e = ExecutionContext(llm= mock_llm)
    e.run_code("""result = llm("Hello, how are you?")""")
    assert e.eval_code("result") == 'Hello! How can I help you today?'

# %%
if test():
    fruit_selection_code = """
r_dict = {}
for p in [
    "Name a single fruit (just the fruit name, nothing else)",
    "Name a single car (just the car name, nothing else)",
    "Name a single animal (just the animal name, nothing else)",
]:
    result = llm(p).strip()
    r_dict[result] = result.lower().count('r')

total_rs = sum(r_dict.values())
    """

    e = ExecutionContext(llm=Partial(_mock_llm, fruit_of_choice='strawberry'))
    out,tb = e.run_code(fruit_selection_code)
    assert tb is None, f"fruit selection code raised exception:\n{tb}"

    r_dict   = e.eval_code("r_dict")
    total_rs = e.eval_code("total_rs")

    assert r_dict == {'strawberry': 3, 'ferrari': 3, 'rabbit': 1}, r_dict
    assert total_rs == 7

    print(f"{r_dict}, {total_rs}")

# %% [markdown]
# ## Hooking functions from the outside
#
# `bind` lets you inject (or swap) any variable/func/class to the repl env.

# %%
if test():
    ctx = ExecutionContext()

    ctx.bind(double=lambda x: x * 2)
    ctx.run_code('result = double(21)')
    assert ctx.eval_code('result') == 42

    ctx.bind(double=lambda x: x * 3)
    stdout, tb = ctx.run_code('''
print("Doubling 7...")
result = double(7)
    ''')
    assert stdout.strip() == "Doubling 7..."
    assert tb is None
    assert ctx.eval_code('result') == 21

    ctx.bind(offset=10)
    ctx.run_code('shifted = double(5) + offset')
    assert ctx.eval_code('shifted') == 25  # 5*3 + 10

# %%
if test():
    ## Assignment propagation — all target forms work

    ctx = ExecutionContext()
    ctx.run_code("a = 1")
    assert ctx.eval_code("a") == 1

    ctx = ExecutionContext()
    ctx.run_code("a, b = 10, 20")
    assert ctx.eval_code("a") == 10
    assert ctx.eval_code("b") == 20

    ctx = ExecutionContext()
    ctx.run_code("(x, (y, z)) = (1, (2, 3))")
    assert ctx.eval_code("x") == 1
    assert ctx.eval_code("y") == 2
    assert ctx.eval_code("z") == 3

    ctx = ExecutionContext()
    ctx.run_code("first, *rest = [1, 2, 3, 4]")
    assert ctx.eval_code("first") == 1
    assert ctx.eval_code("rest") == [2, 3, 4]

    ctx = ExecutionContext()
    ctx.run_code("for i in range(3): pass")
    assert ctx.eval_code("i") == 2

    ctx = ExecutionContext()
    ctx.run_code("import io\nwith io.StringIO() as buf: buf.write('hi')")
    assert ctx.eval_code("buf") is not None

    print("all assertions passed")

# %% [markdown]
# ### More tests

# %%
if test():
    # ── 1. print output is captured ──────────────────────────────────────────────
    stdout, tb = exec_and_capture("print('hello')", {})
    assert stdout == "hello\n"
    assert tb is None
    print(f"stdout={stdout!r}  tb={tb!r}")
    print("-"*80)
    # ── 2. no print → empty stdout ───────────────────────────────────────────────
    stdout, tb = exec_and_capture("x = 42", {})
    assert stdout == ""
    assert tb is None
    print(f"stdout={stdout!r}  tb={tb!r}")
    print("-"*80)
    # ── 3. exception → tb contains full traceback, stdout preserved ──────────────
    stdout, tb = exec_and_capture("print('before')\ny=1\nx = 1 / 0", {})
    assert stdout == "before\n"
    assert tb is not None
    assert "ZeroDivisionError" in tb
    print(f"stdout={stdout!r}")
    print(f"tb=\n{tb}")
    print("-"*80)
    # ── 4. assignments persist in env ────────────────────────────────────────────
    env = {}
    exec_and_capture("x = 10", env)
    exec_and_capture("y = x + 5", env)
    assert env['y'] == 15
    print(f"y={env['y']}")
    print("-"*80)
    # ── 5. injected callable available in REPL code ──────────────────────────────
    env = {'double': lambda x: x * 2}
    exec_and_capture("result = double(21)", env)
    assert env['result'] == 42
    print(f"result={env['result']}")
    # ── 6. multi-turn state persists across run_code calls ───────────────────────
    ctx = ExecutionContext()
    ctx.run_code("acc = 0")
    ctx.run_code("acc += 10")
    ctx.run_code("acc += 5")
    assert ctx.eval_code("acc") == 15
    print(f"acc={ctx.eval_code('acc')}")
    print("-"*80)
    print("all proof assertions passed")

# %%
if test():
    # ── 7. thread-safety: concurrent exec_and_capture never leaks across threads ─
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(thread_id):
        """Each thread prints its own id 100 times; output must never contain another thread's id."""
        ctx = ExecutionContext()
        code = f"""
for _ in range(100):
    print("thread-{thread_id}")
    """
        stdout, tb = ctx.run_code(code)
        assert tb is None, f"thread-{thread_id} raised: {tb}"
        lines = stdout.strip().splitlines()
        assert all(line == f"thread-{thread_id}" for line in lines), (
            f"thread-{thread_id} got foreign output: {set(lines)}"
        )
        return thread_id

    n_threads = 32
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(n_threads)]
        for f in as_completed(futures):
            f.result()  # raises if any assertion failed

    print(f"thread-safety test passed ✓  ({n_threads} threads, 100 prints each)")
