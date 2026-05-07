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
# # Core
#
# Logging helpers, disk caching, and context-variable tracking primitives.

# %%
# notebook edit
# edited in notebook
# edited
# edited in notebook
# edited in notebook
from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import joblib
import structlog
from structlog.contextvars import bound_contextvars

from juplit import test

T = TypeVar("T")


def package_root() -> Path:
    """Directory containing the installed ``deep_reasoner`` package."""
    return Path(__file__).resolve().parent


def package_config_path(*parts: str) -> Path:
    """Path under ``deep_reasoner/config`` (shipped YAML snippets)."""
    return package_root() / "config" / Path(*parts)


def load_live_test_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Populate ``os.environ`` from a dotenv file (defaults to ``<repo>/.envrc`` in dev)."""
    from dotenv import load_dotenv

    load_dotenv(Path(path) if path is not None else package_root().parent / ".envrc")


def live_test_openai_client(
    *,
    base_url_env: str = "OPENAI_BASE_URL",
    api_key_env: str = "OPENAI_API_KEY",
    max_retries: int = 5,
):
    """Build ``AsyncOpenAI`` from OpenAI-compatible env vars (``OPENAI_BASE_URL``, ``OPENAI_API_KEY``; used by optional live tests)."""
    import httpx
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=os.getenv(base_url_env),
        api_key=os.getenv(api_key_env),
        max_retries=max_retries,
        timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=None),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=500,
                max_keepalive_connections=100,
            )
        ),
    )

# %% [markdown]
# ## Logging

# %%
def _structlog_console_renderer_colors() -> bool:
    """True for real TTYs and Jupyter; False for file redirects and plain pipes."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip() or os.environ.get(
        "CLICOLOR_FORCE", ""
    ).strip():
        return True
    if sys.stdout.isatty():
        return True
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    return ip is not None and getattr(ip, "kernel", None) is not None


# %%
def configure_structlog_fixture(
    console=True, extra_processors=None, default_level=logging.WARNING
):
    root = logging.getLogger()
    root.handlers.clear()
    h = logging.StreamHandler()
    h.setLevel(logging.NOTSET)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    root.setLevel(default_level)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%H:%M:%S"),
    ]
    if extra_processors:
        processors.extend(extra_processors)
    processors.append(structlog.stdlib.filter_by_level)
    if console:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=_structlog_console_renderer_colors())
        )
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


# %%
@contextmanager
def checkLogs(namespace: str = "__main__", level: int = logging.DEBUG):
    logger = logging.getLogger(namespace)
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield logger
    finally:
        logger.setLevel(old_level)


# %%
if test():
    configure_structlog_fixture()

    log = structlog.get_logger("my_app.service")
    log.warning("warning message",task_id=1) # Logs normally

    with bound_contextvars(task_id=2):
        with checkLogs("my_app.service", logging.INFO):
            log.debug("hidden info") # Will not log
            log.info("visible info",extra="hi") # Will log

# %%
# define your own log routers and configure them
if test():
    def log_router(logger, method_name, event_dict):
        # placeholder for future routing; keep the processor chain valid
        print(f"intercepted_print {logger} {method_name} {event_dict}",flush=True)
        return event_dict

    configure_structlog_fixture(console=True,extra_processors=[log_router])
    log = structlog.get_logger("my_app.service")
    second_logger = structlog.get_logger('__main__')
    log.warning("warning message",task_id=1) # Logs normally

    with bound_contextvars(task_id=2):
        with checkLogs():
            with checkLogs("my_app.service", logging.INFO):
                log.debug("hidden info") # Will not log
                log.info("visible info",extra="hi") # Will log
            second_logger.debug('visible since i am another logger')

# %% [markdown]
# ## Disk cache

# %%
_memory = joblib.Memory(location=None, verbose=0)  # disabled by default
_cache_registry: dict = {}


def set_cache_dir(path: str = None):
    global _memory
    _memory = joblib.Memory(location=path, verbose=0)
    if path:
        os.makedirs(path, exist_ok=True)
    _cache_registry.clear()


def disk_cache(func=None, *, ignore: list[str] | None = None):
    """Decorator that lazily binds to the current _memory at call time.

    Usage::

        @disk_cache
        def foo(x, y): ...

        @disk_cache(ignore=["verbose"])
        def bar(x, y, verbose=False): ...
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            key = (id(_memory), f, tuple(ignore or []))
            if key not in _cache_registry:
                _cache_registry[key] = _memory.cache(f, ignore=ignore)
            return _cache_registry[key](*args, **kwargs)
        wrapper.__wrapped__ = f
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator

# %%
if test():
    import tempfile, os, shutil, time

    @disk_cache
    def slow_add(x, y):
        time.sleep(0.5)
        return x + y


    t0 = time.perf_counter()
    slow_add(1, 2)
    t1 = time.perf_counter()
    slow_add(1, 2)
    t2 = time.perf_counter()

    assert (t2 - t1) >= 0.5, "Second call should not be cached when no cache dir is set"

    # --- With cache dir: second call should be near-instant ---
    dir1 = tempfile.mkdtemp(prefix="cache_test_1_")
    dir2 = tempfile.mkdtemp(prefix="cache_test_2_")

    try:
        set_cache_dir(dir1)
        slow_add(1, 2)      # cold call
        t3 = time.perf_counter()
        slow_add(1, 2)      # should hit cache
        t4 = time.perf_counter()

        assert (t4 - t3) < 0.1, f"Second call should be cached, took {t4 - t3:.3f}s"
        assert any(os.scandir(dir1)), f"Expected cache files in {dir1}"

        # Switch to second cache dir
        set_cache_dir(dir2)
        t5 = time.perf_counter()
        slow_add(1, 2)      # cold again — new dir, empty cache
        t6 = time.perf_counter()

        assert (t6 - t5) >= 0.5, f"First call on new cache dir should be slow, took {t6 - t5:.3f}s"
        assert any(os.scandir(dir2)), f"Expected cache files in {dir2}"
        assert any(os.scandir(dir1)), "dir1 should be untouched after switching"

    finally:
        shutil.rmtree(dir1, ignore_errors=True)
        shutil.rmtree(dir2, ignore_errors=True)
        assert not os.path.exists(dir1), "dir1 not cleaned up"
        assert not os.path.exists(dir2), "dir2 not cleaned up"
    print("all tests pass")

# %%
if test():
    call_count = 0

    def counting_add(x, y, verbose=False):
        global call_count
        call_count += 1
        return x + y

    cached_counting_add = disk_cache(counting_add, ignore=["verbose"])

    dir1 = tempfile.mkdtemp(prefix="cache_test_ignore_")

    try:
        set_cache_dir(dir1)

        # Cold call
        cached_counting_add(1, 2, verbose=False)
        assert call_count == 1, "First call should execute the function"

        # Different verbose — should hit cache, not increment count
        cached_counting_add(1, 2, verbose=True)
        assert call_count == 1, "verbose=True should hit cache, not re-execute"

        # Different x,y — should miss cache
        cached_counting_add(9, 9, verbose=False)
        assert call_count == 2, "Different args should re-execute"

    finally:
        shutil.rmtree(dir1, ignore_errors=True)
        assert not os.path.exists(dir1), "dir1 not cleaned up"
    print("all tests pass")


# %% [markdown]
# ## Context tracking

# %%
class TrackedVar(Generic[T]):
    """A ContextVar that knows how to derive its child value from its parent value.

    Args:
        name:    Used as the ContextVar name and as the structlog key.
        default: Initial value before any context is entered.
        derive:  ``(parent_value, sibling_new_values) -> child_value``.
                 Defaults to identity (inherit parent value unchanged).
        shared:  If True, all contexts share one mutable box. Mutations are
                 permanent and visible to parent + siblings. reset() is a no-op.
    """

    def __init__(
        self,
        name: str,
        default: T,
        derive: Callable[[T, dict[str, Any]], T] | None = None,
        shared: bool = False,
    ):
        self.name = name
        self.default = default
        self.shared = shared
        self.derive = derive or (lambda parent, _: parent)

        if shared:
            self._box: list[T] = [default]
            self.var: ContextVar[T] | None = None
        else:
            self.var = ContextVar(name, default=default)

    def get(self) -> T:
        if self.shared:
            return self._box[0]
        return self.var.get()

    def set(self, value: T) -> Any:
        if self.shared:
            self._box[0] = value
            return None
        return self.var.set(value)

    def reset(self, token: Any) -> None:
        if self.shared or token is None:
            return
        self.var.reset(token)


# %%
class ContextGroup:
    """A group of TrackedVars with a shared bind() context manager.

    Vars are derived in declaration order, so put dependencies before dependents.
    Each var's derive() receives ``(parent_value, inputs)`` where inputs is the
    union of caller-supplied kwargs and already-derived sibling new values.
    """

    def __init__(self, *vars: TrackedVar, structlog_keys: list[str] | None = None):
        self.vars = vars
        self.structlog_keys = structlog_keys

    def reset(self) -> None:
        for tracked in self.vars:
            tracked.set(tracked.default)

    @contextmanager
    def bind(self, **inputs):
        raw_tokens: dict[str, Any] = {}
        new_values: dict[str, Any] = {}

        for tracked in self.vars:
            parent_val = tracked.get()
            new_val = tracked.derive(parent_val, {**inputs, **new_values})
            new_values[tracked.name] = new_val

        for tracked in self.vars:
            raw_tokens[tracked.name] = tracked.set(new_values[tracked.name])

        structlog_tokens: dict[str, Any] = {}
        if self.structlog_keys is None:
            sync_names = {t.name for t in self.vars}
        else:
            sync_names = set(self.structlog_keys)

        if sync_names:
            structlog_tokens = structlog.contextvars.bind_contextvars(
                **{t.name: t.get() for t in self.vars if t.name in sync_names}
            )

        try:
            yield new_values
        finally:
            for tracked in self.vars:
                tracked.reset(raw_tokens[tracked.name])
            if structlog_tokens:
                structlog.contextvars.reset_contextvars(**structlog_tokens)

# %%
if test():
    cg = ContextGroup(
        TrackedVar('depth', default=0, derive=lambda parent, _: parent + 1),
    )

    # Top-level implicit bind starts from default/root state.
    with cg.bind() as top:
        assert top['depth'] == 1

        # Nested implicit bind derives from the active bound group.
        with cg.bind() as nested:
            assert nested['depth'] == 2

    # After exit, state is restored.
    assert cg.vars[0].get() == 0


# %%
if test():
    import random
    import coolname
    def fresh_name() -> str:
        return coolname.generate_slug(2)


    agent_context = ContextGroup(
        # global node counter
        TrackedVar('node_id', 
            default=0,
            shared=True,
            derive=lambda parent, _: parent + 1,
        ),
        # fresh readable name
        TrackedVar(
            'node_name',
            default='root',
            derive=lambda parent,_: fresh_name(),
        ),
        # depth
        TrackedVar('depth',
            default=0,
            derive=lambda parent, _: parent + 1,
        ),
        # ancestry, tuple of ids marking the path from root node to you
        TrackedVar('ancestry',
            default=(),
            derive=lambda parent, inputs: parent + (inputs['node_id'],),
        ),
    )

# %%
if test():
    import itertools
    log = structlog.get_logger()

    random_numbers = [7,11,18,15,9]

    rng_iter = itertools.cycle(random_numbers)

    def process(node_type:str,remaining_depth: int) -> int:
        with agent_context.bind(node_type=node_type) as ctx:
            own_value = next(rng_iter)

            log.info(
                "node started",
                own_value=own_value,
            )

            children_sum = 0
            if remaining_depth > 0:
                child_type = "beta" if node_type == "alpha" else "alpha"
                for _ in range(2):
                    children_sum += process(child_type, remaining_depth - 1)

            total = own_value + children_sum

            log.info(
                "node finished",
                own_value=own_value,
                children_sum=children_sum,
                total=total,
            )

            return total

    from collections import defaultdict
    log_dict = defaultdict(list)

    def tree_collector_processor(logger, method, event_dict: dict[str, Any]) -> dict[str, Any]:
        global log_dict
        event_dict['severity']=str(method)
        log_dict[event_dict.get('node_id')].append(event_dict)
        return event_dict

# %%
if test():
    configure_structlog_fixture(extra_processors=[tree_collector_processor])
    print(process(node_type='gamma',remaining_depth=2))

# %%
if test():
    for key, events in log_dict.items():
        print (key)
        for event in events:
            print(' '*4 + str(event))

# %% [markdown]
# ## Semchunk

# %%
# semchunk test — run this cell standalone to evaluate chunking behaviour
# pip install semchunk   (if not yet installed)
if test():
    import semchunk

    sample = "\n\n".join([
        f"Paragraph {i}: " + ("This is some DnD transcript dialogue. " * 20)
        for i in range(20)
    ])

    chunks = semchunk.chunk(sample, chunk_size=5000,token_counter=len)
    print(f"Total chars: {len(sample)}")
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  chunk {i}: {len(c)} chars | starts: {c[:60]!r}")

# %% [markdown]
# ## Config diff and pretty-print utilities

# %%
def diff_configs(base: dict, other: dict, *, _prefix: str = "") -> dict[str, tuple]:
    """Return a flat dotted-key diff between two config dicts.

    Returns ``{dotted.key: (base_value, other_value)}`` for every key where
    the value changed, was added (base_value is None), or was removed
    (other_value is None).  Dicts are recursed into; all other types are
    compared with ``==``.
    """
    result: dict[str, tuple] = {}
    all_keys = set(base) | set(other)
    for k in sorted(all_keys):
        full_key = f"{_prefix}{k}" if _prefix else k
        if k not in base:
            result[full_key] = (None, other[k])
        elif k not in other:
            result[full_key] = (base[k], None)
        elif isinstance(base[k], dict) and isinstance(other[k], dict):
            result.update(diff_configs(base[k], other[k], _prefix=f"{full_key}."))
        elif base[k] != other[k]:
            result[full_key] = (base[k], other[k])
    return result


def pretty_print_config(cfg: dict, *, title: str | None = None) -> None:
    """Pretty-print a config dict to stdout using rich."""
    from rich import print as rprint
    from rich.panel import Panel
    from rich.pretty import Pretty

    content = Pretty(cfg, expand_all=True)
    if title:
        rprint(Panel(content, title=title, border_style="blue"))
    else:
        rprint(content)


def pretty_print_config_diff(diff: dict[str, tuple], *, title: str | None = None) -> None:
    """Pretty-print a diff dict (output of diff_configs) using rich."""
    from rich import print as rprint
    from rich.panel import Panel
    from rich.text import Text

    lines = Text()
    for key, (old, new) in diff.items():
        if old is None:
            lines.append(f"  + {key}: ", style="bold green")
            lines.append(repr(new) + "\n", style="green")
        elif new is None:
            lines.append(f"  - {key}: ", style="bold red")
            lines.append(repr(old) + "\n", style="red")
        else:
            lines.append(f"  ~ {key}: ", style="bold yellow")
            lines.append(repr(old), style="red")
            lines.append(" → ", style="dim")
            lines.append(repr(new) + "\n", style="green")

    if not diff:
        lines.append("  (no differences)\n", style="dim")

    label = title or "diff"
    rprint(Panel(lines, title=label, border_style="yellow"))


# %%
if test():
    assert diff_configs({"a": 1}, {"a": 1}) == {}
    assert diff_configs({"a": 1}, {"a": 2}) == {"a": (1, 2)}
    assert diff_configs({"a": 1}, {"b": 2}) == {"a": (1, None), "b": (None, 2)}
    assert diff_configs({"a": {"x": 1, "y": 2}}, {"a": {"x": 1, "y": 9}}) == {"a.y": (2, 9)}
    assert diff_configs({}, {"a": 1}) == {"a": (None, 1)}

# %%
# # ! poe sync
