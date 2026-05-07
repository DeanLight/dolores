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
# # LLM Interface
#
# OpenAI-compatible LLM client with Jinja2 system prompt templating.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable

import nest_asyncio
import logging
import structlog
from structlog.contextvars import bound_contextvars
from jinja2 import Environment as _JinjaEnv
from jinja2 import meta as _jinja_meta
from openai import AsyncOpenAI

from deep_reasoner.code_exec import Partial
from deep_reasoner.core import configure_structlog_fixture, disk_cache, set_cache_dir
from juplit import test

logger = structlog.get_logger(__name__)


# %%
if test():
    configure_structlog_fixture()
    from pathlib import Path

    _disk_cache = Path(__file__).resolve().parent.parent / "data" / "disk_cache"
    _disk_cache.mkdir(parents=True, exist_ok=True)
    set_cache_dir(str(_disk_cache))



# %% [markdown]
# ## Jinja2 rendering

# %%
def jinja_render(template: str, template_vars: dict[str, Any]) -> str:
    _env = _JinjaEnv()
    _required = _jinja_meta.find_undeclared_variables(_env.parse(template))
    if _required:
        _missing = _required - set(template_vars)
        if _missing:
            raise ValueError(
                f"The template has undefined template variables: {_missing}\n"
                f"template: {template}\nvars: {template_vars}"
            )
    return _env.from_string(template).render(**template_vars)

# %%
if test():
    import pytest

    assert jinja_render("Hello {{ name }}!", {"name": "World"}) == "Hello World!"
    assert jinja_render("no vars", {}) == "no vars"

    try:
        jinja_render("Hello {{ name }}!", {})
        assert False, "should have raised"
    except ValueError as e:
        assert "name" in str(e)

# %% [markdown]
# ## Message builders

# %%
def user(content: str, **kwargs) -> dict:
    """Create a user message dict, rendering *content* as Jinja2 when kwargs provided."""
    return {"role": "user", "content": jinja_render(content, kwargs) if kwargs else content}


def system(content: str, **kwargs) -> dict:
    """Create a system message dict, rendering *content* as Jinja2 when kwargs provided."""
    return {"role": "system", "content": jinja_render(content, kwargs) if kwargs else content}


def assistant(content: str, **kwargs) -> dict:
    """Create an assistant message dict, rendering *content* as Jinja2 when kwargs provided."""
    return {"role": "assistant", "content": jinja_render(content, kwargs) if kwargs else content}

# %%
if test():
    assert user("hello")== {"role": "user", "content": "hello"}
    assert system("hello") == {"role": "system", "content": "hello"}
    assert assistant("hello") == {"role": "assistant", "content": "hello"}
    assert user("Hi {{ name }}", name="Alice")["content"] == "Hi Alice"
    assert system("You are {{ role }}", role="helpful")["content"] == "You are helpful"

# %% [markdown]
# ## LLM client

# %%
@disk_cache(ignore=["client"])
async def openai_llm(
    messages: str | list[dict],
    *,
    client: AsyncOpenAI,
    model: str,
    **kwargs,
) -> str:
    """Call an LLM via the OpenAI client (compatible with vLLM, Ollama, etc.).

    *messages* can be a plain string (auto-wrapped as a ``user`` message) or
    a list of message dicts. Extra *kwargs* (``temperature``, ``max_tokens``, …)
    are forwarded to ``client.chat.completions.create``.
    """
    msgs = [user(messages)] if isinstance(messages, str) else messages

    import time as _time
    _t0 = _time.monotonic()

    completion = await client.chat.completions.create(
        model=model,
        messages=msgs,
        **kwargs,
    )

    response_text = completion.choices[0].message.content
    usage = completion.usage.to_dict()
    _duration_s = round(_time.monotonic() - _t0, 2)
    logger.debug("llm.call", usage=usage, response=response_text, model=model, messages=msgs,
                 duration_s=_duration_s)
    logger.info("llm.usage", usage=usage, model=model, duration_s=_duration_s)
    if response_text is None:
        logger.warning("No response text from OpenAI")
        return ""

    return response_text

# %% [markdown]
# ## Sync wrapper and async batcher

# %%
def _run_sync(coro):
    """Run a coroutine synchronously, re-using the running loop when inside one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        nest_asyncio.apply()
        return loop.run_until_complete(coro)


def make_sync(fn):
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if not inspect.isawaitable(out):
            return out
        return _run_sync(out)
    return wrapper


# %%
class AsyncCaller:
    """Wraps an async callable to expose sync, async, and concurrent-batch interfaces.

    Instances created by :func:`make_llm` hold a pre-bound async function and
    expose three calling styles:

    * ``caller(messages)`` — single call, sync (blocks until done).
    * ``caller.batch([m1, m2, …])`` — concurrent calls, sync (all in-flight at once).
    * ``await caller.call_async(messages)`` — single call, async (for use in coroutines).
    """

    def __init__(self, async_fn: Callable):
        self._fn = async_fn

    async def call_async(self, messages, **kwargs) -> str:
        """Single async call — used internally by :class:`DeepReasoner._run_async`."""
        return await self._fn(messages, **kwargs)

    def __call__(self, messages, **kwargs) -> str:
        """Single call (sync)."""
        return _run_sync(self._fn(messages, **kwargs))

    def batch(self, messages_list: list, **kwargs) -> list:
        """Run multiple calls concurrently and return results in order (sync).

        Each element of *messages_list* is a plain string or list of message
        dicts — the same formats accepted by a single call.
        """
        return list(_run_sync(asyncio.gather(
            *[self._fn(m, **kwargs) for m in messages_list]
        )))


# %%
def make_llm(**kwargs) -> AsyncCaller:
    """Return an :class:`AsyncCaller` wrapping :func:`openai_llm` with pre-bound config.

    Pre-bound *kwargs* (``client``, ``model``, …) act as defaults and can be
    overridden at call time.

    The returned object supports single calls (``llm(prompt)``),
    concurrent batches (``llm.batch([p1, p2, …])``), and
    async use (``await llm.call_async(prompt)``).
    """
    return AsyncCaller(Partial(openai_llm, **kwargs))


# %% [markdown]
# ## Examples
#
# Use `make_llm` to create a sync callable with infrastructure pre-bound. Pass `messages` at call time — either a plain string (becomes a bare user message) or an explicit list built with `user()` / `system()` / `assistant()`.

# %% [markdown]
# ### Tests

# %%
if test():
    from dotenv import load_dotenv
    from deep_reasoner.core import checkLogs
    from openai import AsyncOpenAI
    import httpx
    import os

    configure_structlog_fixture()
    print(load_dotenv('../.envrc'))

# %%
if test():
    client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=5,
        timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=None),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100)
        ))

    llm = make_llm(client=client, model=os.getenv("MODEL"))

# %%
if test():
    print(llm([system("You are a helpful assistant."), user("What is the capital of France?")]))

    with checkLogs():
        with bound_contextvars(benchmark="capital_qa", method="deepreasoner", model=os.getenv("MODEL")):
            print(llm([system("You are a helpful assistant."),
                user("What is the capital of France?")]))

# %% [markdown]
# #### Test Error

# %% [markdown]
# Explicit list — add a system prompt with system()

# %%
if test():
    with checkLogs():
        answer = llm([
            system("You are a geography expert. Answer with a few sentences. Be concise and precise and very polite."),
            user("What is the largest country by area?"),
        ])

    with checkLogs():
        answer2 = llm([
            system("You are a geography expert. Answer with a few sentences. Be concise and precise and very polite."),
            user("What is the largest country by area?"),
            assistant(f"{answer}"),
            user("What is the capital of Russia?"),
        ])

    with checkLogs():
        answer3 = llm([
            system("You are a geography expert. Answer with a few sentences. Be concise and precise and very polite."),
            user("What is the largest country by area?"),
            assistant(f"{answer}"),
            user("What is the capital of Russia?"),
            assistant(f"{answer2}"),
            user("What is the population of Russia?"),
        ])

# %% [markdown]
# Jinja does not render stuff we dont want to explicitely render
#

# %%
if test():
    with checkLogs():
        answer = llm([
            system("You are a geography expert {{ HI }}. Answer with just the name, nothing else."),
            user("What is the largest country by area?"),
        ])
    print(answer)  # -> 'Russia'

# %%
if test():
    # Explicit list with Jinja — render the system prompt at construction time
    with checkLogs():
        answer = llm([
            system("You are a {{ role }}. Answer in one sentence.", role="history expert"),
            user("When did the Berlin Wall fall?"),
        ])
    print(answer)


# %%
# ! poe sync
