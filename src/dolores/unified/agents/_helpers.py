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
# # Agent helpers
#
# Shared bits factored out of the smolagents baselines so each agent module
# stays a thin `_answer`. Connection details come from the `InferenceBackend`
# on the cfg; smolagents is imported lazily so non-smolagents agents (cot, rlm,
# deep_reasoner) don't pull it in.

# %%
from juplit import test

# %%
from typing import Any

from dolores.unified.benchmarks import Benchmark, Task

# %% [markdown]
# ## Connection + thinking knobs


# %%
def _conn(cfg) -> tuple[str, str]:
    """(base_url, api_key) from the cfg's InferenceBackend."""
    ck = cfg.inference.client_kwargs()
    return ck.get("base_url"), ck.get("api_key")


def _no_thinking_extra_body(cfg) -> dict:
    """The vLLM ``enable_thinking=False`` extra_body block, or ``{}``."""
    if cfg.no_thinking:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


# %% [markdown]
# ## Answer parsing


# %%
def _parsed_for_phantomwiki(benchmark: Benchmark, answer: Any) -> Any:
    """phantomwiki wants a ``list[str]`` (comma-split); everything else passes through."""
    if benchmark.benchmark_name() == "phantomwiki":
        return [s.strip() for s in str(answer).split(",") if s.strip()]
    return answer


# %% [markdown]
# ## smolagents builders (lazy import)


# %%
def _smolagent_tools(task: Task) -> list:
    """Wrap each ``task.tools`` callable as a smolagents tool."""
    from smolagents import tool

    return [tool(fn) for fn in task.tools.values()]


def _openai_model(cfg, **extra):
    """Build a smolagents ``OpenAIModel`` from the cfg (model + connection + no_thinking)."""
    from smolagents import OpenAIModel

    base, key = _conn(cfg)
    return OpenAIModel(
        model_id=cfg.model,
        api_base=base,
        api_key=key,
        **_no_thinking_extra_body(cfg),
        **extra,
    )


# %% [markdown]
# ## Tests


# %%
def test_no_thinking_extra_body():
    from types import SimpleNamespace

    assert _no_thinking_extra_body(SimpleNamespace(no_thinking=False)) == {}
    out = _no_thinking_extra_body(SimpleNamespace(no_thinking=True))
    assert out["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_conn():
    from dolores.unified.inference import OpenAIBackend
    from types import SimpleNamespace

    cfg = SimpleNamespace(inference=OpenAIBackend(api_base="http://h/v1", api_key="k"))
    assert _conn(cfg) == ("http://h/v1", "k")


def test_parsed_for_phantomwiki():
    from dolores.unified.benchmarks import HelloWorld

    class _PW:
        @staticmethod
        def benchmark_name():
            return "phantomwiki"

    assert _parsed_for_phantomwiki(_PW(), "Alice, Bob ,") == ["Alice", "Bob"]
    assert _parsed_for_phantomwiki(HelloWorld(), "13") == "13"


# %%
if test():
    test_no_thinking_extra_body()
    test_conn()
    test_parsed_for_phantomwiki()
    print("agents._helpers tests passed")

# %% [markdown]
# ## End
