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
# # DeepReasoner agent
#
# Wraps `deep_reasoner.DeepReasoner` (a CodeAct REPL planner). The planner config
# (system prompt, mental models, stop tokens) loads from this repo's
# `configs/agents/debug.yaml` by default; override with the `agent_config` cfg
# extra. `cfg.max_steps` maps to the planner's `max_iter`. The model / connection
# / no_thinking come from the shared `AgentCfg`. Logging rides the repo's
# Agent Context (set up by `Agent.run`): DeepReasoner drives the same patched
# `AsyncOpenAI` client, so its calls land in `calls.jsonl`.

# %%
from juplit import test

# %%
import asyncio
import os
from pathlib import Path
from typing import Any

import yaml

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.benchmarks import Benchmark, Task
from config import Paths

# %%
_DEFAULT_AGENT_CONFIG = Paths.ROOT / "configs" / "agents" / "debug.yaml"


# %%
def _apply_no_thinking(llm_kwargs: dict) -> dict:
    """Set extra_body.chat_template_kwargs.enable_thinking=False, preserving other keys."""
    extra_body = dict(llm_kwargs.get("extra_body", {}))
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    return {**llm_kwargs, "extra_body": extra_body}


def _load_planner_cfg(agent_config, model_name, api_base, api_key, no_thinking, max_iter):
    """Parse the repo planner YAML into deep_reasoner's MainConfig, applying cfg overrides."""
    from deep_reasoner.cli_base import MainConfig

    cfg = MainConfig(**(yaml.safe_load(Path(agent_config).read_text()) or {}))
    cfg.model = model_name
    cfg.client.base_url = api_base
    cfg.max_iter = max_iter
    if api_key:
        # build_client reads the key from os.environ[cfg.client.api_key_env]; let api_key win.
        os.environ[cfg.client.api_key_env] = api_key
    if no_thinking:
        cfg.llm_kwargs = _apply_no_thinking(cfg.llm_kwargs)
    return cfg


def _task_tools(task):
    from deep_reasoner.deepreasoner import Func

    return {name: Func(fn) for name, fn in task.tools.items()}


def _task_vars(task):
    from deep_reasoner.deepreasoner import Var

    return {name: Var(value, description=name) for name, value in task.vars.items()}


def _parse_answer(benchmark: Benchmark, answer) -> Any:
    """phantomwiki wants list[str]; everything else passes the answer through."""
    if benchmark.benchmark_name() != "phantomwiki":
        return answer
    if isinstance(answer, list):
        return [str(x).strip() for x in answer if str(x).strip()]
    return [s.strip() for s in str(answer).split(",") if s.strip()]


async def _run_agent_async(cfg_main, task):
    """Build a DeepReasoner from the planner cfg + task and run one pass."""
    from deep_reasoner.cli_base import make_base_tools
    from deep_reasoner.cli_utils import build_client, make_agent
    from deep_reasoner.deepreasoner import PlanExec

    client = build_client(cfg_main.client)
    try:
        tools = {**make_base_tools(cfg_main, client), **_task_tools(task)}
        agent = make_agent(cfg_main, tools, client)
        repl_vars = _task_vars(task)
        return await PlanExec(agent).call_async(
            task.question, namespaces=cfg_main.root_namespaces, **repl_vars
        )
    finally:
        await client.close()


# %%
@register_agent
class DeepReasonerAgent(Agent):
    name = "deep_reasoner"

    def _answer(self, task: Task, log_dir: Path) -> Any:
        cfg = self.cfg
        ck = cfg.inference.client_kwargs()
        cfg_main = _load_planner_cfg(
            cfg.get("agent_config", _DEFAULT_AGENT_CONFIG),
            cfg.model, ck.get("base_url"), ck.get("api_key"),
            cfg.no_thinking, cfg.max_steps,
        )
        return asyncio.run(_run_agent_async(cfg_main, task))

    def _parsed(self, answer: Any) -> Any:
        return _parse_answer(self.cfg.benchmark, answer)


# %% [markdown]
# ## Tests


# %%
def test_load_planner_cfg_max_iter_and_overrides():
    cfg = _load_planner_cfg(
        _DEFAULT_AGENT_CONFIG, "my-model", "http://localhost:9/v1", "",
        no_thinking=True, max_iter=42,
    )
    assert cfg.model == "my-model"
    assert cfg.client.base_url == "http://localhost:9/v1"
    assert cfg.max_iter == 42
    assert cfg.llm_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_deep_reasoner_smoke_scripted_llm():
    """HelloWorld fib_7 with a scripted planner that calls FinalAnswer(13) (no network)."""
    import json
    import tempfile
    from unittest.mock import patch

    import deep_reasoner.cli_base as dr_cli_base
    import deep_reasoner.llm as dr_llm
    from deep_reasoner.llm import AsyncCaller

    from dolores.unified.agents.base import AgentCfg
    from dolores.unified.benchmarks import get_benchmark
    from dolores.unified.inference import OpenAIBackend

    async def _scripted(messages, **kwargs):
        return "<repl>\nFinalAnswer(13)\n</repl>"

    def _fake_make_llm(**kwargs):
        return AsyncCaller(_scripted)

    bench = get_benchmark("hello_world")
    cfg = AgentCfg(model="scripted-model",
                   inference=OpenAIBackend(api_base="http://localhost:0/v1", api_key=""),
                   benchmark=bench)
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(dr_llm, "make_llm", _fake_make_llm), \
            patch.object(dr_cli_base, "make_llm", _fake_make_llm), \
            patch.object(Paths, "LOGS_DIR", Path(tmp)):
        result = DeepReasonerAgent(cfg).run(bench.get_task("fib_7"))
        assert result.answer == 13
        assert result.scores == (1.0, 1.0)
        qa = json.loads(next(Path(tmp).rglob("qa.json")).read_text())
        assert qa["answer"] == 13


# %%
if test():
    test_load_planner_cfg_max_iter_and_overrides()
    test_deep_reasoner_smoke_scripted_llm()
    print("agents.deep_reasoner tests passed")

# %% [markdown]
# ## End
