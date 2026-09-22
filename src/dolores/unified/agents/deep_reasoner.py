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
# (system prompt, mental models, stop tokens) loads from `configs/agents/*.yaml`
# via the run config's `agent_config` extra (default `configs/agents/debug.yaml`),
# through deep_reasoner's own config loader. `cfg.max_steps` maps to the planner's
# `max_iter`. The model / connection / no_thinking come from the shared `AgentCfg`.
# Logging rides the repo's Agent Context (set up by `Agent.run`): DeepReasoner
# drives the same patched `AsyncOpenAI` client, so its calls land in `calls.jsonl`.
#
# ## Per-benchmark wiring
#
# The paper's Deep Reasoner runs went through one runner per benchmark. Each gave
# the planner specific tool names, which the planner prompts refer to by name, and
# parsed and scored the answer its own way. `_dr_wiring`, `_parse` and `_score`
# reproduce those runners:
#
# | benchmark | REPL tools / vars | `parsed` | `scores` (f1, em) |
# |---|---|---|---|
# | phantomwiki | `retrieve_article`, `search` | `_parse_phantom_output` | F1 / EM of the parsed list |
# | synthworlds | `search` (the dense retriever) | raw answer | F1 / EM |
# | oolong | `document` Var ("DnD game text") | LLM-parsed typed answer | score, score == 1 |
# | deepresearchqa | `search` = web-search sub-agent on the search model | raw answer | LLM judge, judge |
#
# A scoring or parsing failure records `None` (and a warning) instead of failing
# the attempt, as the per-benchmark runners did.

# %%
from juplit import test

# %%
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import structlog

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.benchmarks import Task
from config import Paths

logger = structlog.get_logger(__name__)

# %%
_DEFAULT_AGENT_CONFIG = Paths.ROOT / "configs" / "agents" / "debug.yaml"

# Tool description the DeepSearchQA runner gave its web-search sub-agent.
_DSQA_SEARCH_DESCRIPTION = (
    "Web search: search(query) -> str. "
    "Performs a Serper web search and returns a summary of the top results. "
    "Use to retrieve factual information from the web. "
    "Call multiple times with different queries to follow multi-hop chains."
)


# %%
def _apply_no_thinking(llm_kwargs: dict) -> dict:
    """Set extra_body.chat_template_kwargs.enable_thinking=False, preserving other keys."""
    extra_body = dict(llm_kwargs.get("extra_body", {}))
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
    chat_template_kwargs["enable_thinking"] = False
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    return {**llm_kwargs, "extra_body": extra_body}


def _load_planner_cfg(agent_config, model_name, api_base, api_key, no_thinking, max_iter):
    """Load the planner YAML into deep_reasoner's MainConfig, applying cfg overrides.

    Uses deep_reasoner's own loader (``_compose`` relative to the YAML, the same
    settings layer), exactly as the per-benchmark runners loaded it.
    """
    from deep_reasoner.cli_base import MainConfig
    from deep_reasoner.cli_utils import load_main_and_validate

    cfg = load_main_and_validate(MainConfig, Path(agent_config))
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


def _parse_phantom_output(raw_output: Any) -> list[str]:
    """Best-effort parse of model output into list[str] for phantomwiki scoring."""
    if isinstance(raw_output, list):
        return [str(x).strip() for x in raw_output if str(x).strip()]
    text = str(raw_output).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    items: list[str] = []
    for part in re.split(r"[\n,]+", text):
        cleaned = re.sub(r"^\s*[-*•\d\.\)\(]+\s*", "", part).strip()
        if cleaned:
            items.append(cleaned)
    return items


def _dsqa_search_tool(cfg, cfg_main, log_dir: Path):
    """The DeepSearchQA web-search sub-agent, on the search model.

    ``search_model_id`` (LiteLLM format) comes from the run config, else the planner
    YAML, else the planner model; ``search_api_base`` defaults to the run's inference
    endpoint; the key comes from ``search_api_key_env`` (default OPENAI_API_KEY).
    """
    from deep_reasoner.deepreasoner import Func

    from dolores.unified.benchmarks.deepresearch_agent import (
        create_web_search_agent_tool,
        litellm_model,
    )

    search_model_id = (cfg.get("search_model_id")
                       or getattr(cfg_main, "search_model_id", None)
                       or cfg_main.model)
    search_api_base = cfg.get("search_api_base") or cfg_main.client.base_url
    key_env = cfg.get("search_api_key_env", "OPENAI_API_KEY")
    search_api_key = os.getenv(key_env) or os.getenv("OPENAI_API_KEY") or ""
    lm = litellm_model(
        model_id=search_model_id,
        api_base=search_api_base,
        api_key=search_api_key,
        timeout=cfg_main.client.read_timeout,
    )
    search_agent = create_web_search_agent_tool(lm, run_logs_dir=str(log_dir))
    return Func(search_agent, description=_DSQA_SEARCH_DESCRIPTION)


def _dr_wiring(cfg, cfg_main, task: Task, log_dir: Path) -> tuple[dict, dict, bool]:
    """``(tools, repl_vars, direct)`` the paper's runner gave the planner on this benchmark.

    ``direct`` means the DeepSearchQA runner's entry point (``agent._run_async``)
    rather than ``PlanExec``.
    """
    from deep_reasoner.deepreasoner import Func, Var

    name = cfg.benchmark.benchmark_name()
    if name == "phantomwiki":
        return ({"retrieve_article": Func(task.tools["retrieve_article"]),
                 "search": Func(task.tools["search"])}, {}, False)
    if name == "synthworlds":
        return {"search": Func(task.tools["retrieve_top_5"])}, {}, False
    if name == "oolong":
        return {}, {"document": Var(task.vars["document"], "DnD game text")}, False
    if name == "deepresearchqa":
        return {"search": _dsqa_search_tool(cfg, cfg_main, log_dir)}, {}, True
    return _task_tools(task), _task_vars(task), False


async def _run_agent_async(cfg_main, question: str, tools: dict, repl_vars: dict, direct: bool):
    """Build a DeepReasoner from the planner cfg + wiring and run one pass."""
    from deep_reasoner.cli_base import make_base_tools
    from deep_reasoner.cli_utils import build_client, make_agent
    from deep_reasoner.deepreasoner import PlanExec

    client = build_client(cfg_main.client)
    try:
        agent = make_agent(cfg_main, {**make_base_tools(cfg_main, client), **tools}, client)
        if direct:
            return await agent._run_async(question, vars={})
        return await PlanExec(agent).call_async(
            question, namespaces=cfg_main.root_namespaces, **repl_vars
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
        tools, repl_vars, direct = _dr_wiring(cfg, cfg_main, task, log_dir)
        return asyncio.run(_run_agent_async(cfg_main, task.question, tools, repl_vars, direct))

    def _parse(self, task: Task, answer: Any) -> Any:
        bench = self.cfg.benchmark
        name = bench.benchmark_name()
        if name == "phantomwiki":
            return _parse_phantom_output(answer)
        if name == "oolong":
            key_env = self.cfg.get("parse_api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(key_env) or os.getenv("OPENAI_API_KEY")
            try:
                return bench.parse(task.task_id, answer, api_key=api_key)
            except Exception as exc:
                logger.warning("deep_reasoner.parse_failed", task_id=task.task_id, error=str(exc))
                return None
        return answer

    def _score(self, task: Task, answer: Any, parsed: Any) -> Any:
        bench = self.cfg.benchmark
        name = bench.benchmark_name()
        try:
            if name == "phantomwiki":
                return bench.score(task.task_id, parsed)
            if name == "synthworlds":
                return bench.score(task.task_id, str(answer))
            if name == "oolong":
                if parsed is None:
                    return None
                score = bench.score(task.task_id, parsed)
                return score, float(score == 1.0)
            if name == "deepresearchqa":
                _, score, _ = bench.score_judge(task.task_id, str(answer))
                return score, score
            return bench.score(task.task_id, answer)
        except Exception as exc:
            logger.warning("deep_reasoner.score_failed", task_id=task.task_id, error=str(exc))
            return None


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


def _stub_cfg(bench_name: str, **extras):
    """An AgentCfg whose benchmark reports *bench_name* but needs no dataset."""
    from dolores.unified.agents.base import AgentCfg
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    class _Stub(HelloWorld):
        @classmethod
        def benchmark_name(cls) -> str:
            return bench_name

    return AgentCfg(model="m", inference=OpenAIBackend(api_base="http://localhost:0/v1", api_key=""),
                    benchmark=_Stub(), **extras)


def test_parse_phantom_output():
    assert _parse_phantom_output(["Alice ", "", "Bob"]) == ["Alice", "Bob"]
    assert _parse_phantom_output('["Alice", "Bob"]') == ["Alice", "Bob"]
    assert _parse_phantom_output("1. Alice\n2. Bob, Carol") == ["Alice", "Bob", "Carol"]
    assert _parse_phantom_output("- Alice\n* Bob") == ["Alice", "Bob"]
    assert _parse_phantom_output("  ") == []


def test_dr_wiring_tool_names_match_the_planner_prompts():
    from deep_reasoner.deepreasoner import Func, Var

    def retrieve_article(entity: str) -> str: return entity
    def search(attribute: str) -> str: return attribute
    def retrieve_top_5(query: str) -> list[str]: return [query]

    cfg_main = _load_planner_cfg(_DEFAULT_AGENT_CONFIG, "m", "http://localhost:9/v1", "", False, 30)
    log_dir = Path("unused")

    pw = Task(task_id="t", question="q", tools={"retrieve_article": retrieve_article, "search": search})
    tools, repl_vars, direct = _dr_wiring(_stub_cfg("phantomwiki"), cfg_main, pw, log_dir)
    assert set(tools) == {"retrieve_article", "search"} and not repl_vars and not direct
    assert tools["search"].value is search and isinstance(tools["search"], Func)

    sw = Task(task_id="t", question="q", tools={"retrieve_top_5": retrieve_top_5})
    tools, repl_vars, direct = _dr_wiring(_stub_cfg("synthworlds"), cfg_main, sw, log_dir)
    assert set(tools) == {"search"} and tools["search"].value is retrieve_top_5 and not direct

    oo = Task(task_id="t", question="q", vars={"document": "the transcript"})
    tools, repl_vars, direct = _dr_wiring(_stub_cfg("oolong"), cfg_main, oo, log_dir)
    assert tools == {} and not direct
    assert isinstance(repl_vars["document"], Var)
    assert repl_vars["document"].value == "the transcript"
    assert repl_vars["document"].description == "DnD game text"


def test_dr_wiring_deepresearchqa_search_subagent():
    from unittest.mock import patch

    import dolores.unified.benchmarks.deepresearch_agent as dra

    seen = {}

    def fake_litellm_model(**kwargs):
        seen.update(kwargs)
        return "lm"

    def fake_tool(model, run_logs_dir):
        def search_agent(query: str) -> str:
            return query
        seen["run_logs_dir"] = run_logs_dir
        return search_agent

    cfg_main = _load_planner_cfg(_DEFAULT_AGENT_CONFIG, "planner-m", "http://localhost:9/v1", "", False, 30)
    task = Task(task_id="t", question="q", tools={"web_search": lambda query: query})
    with patch.object(dra, "litellm_model", fake_litellm_model), \
            patch.object(dra, "create_web_search_agent_tool", fake_tool):
        tools, repl_vars, direct = _dr_wiring(
            _stub_cfg("deepresearchqa", search_model_id="hosted_vllm/Qwen/Qwen3-32B"),
            cfg_main, task, Path("/tmp/run"))
        assert set(tools) == {"search"} and direct and not repl_vars
        assert tools["search"].description == _DSQA_SEARCH_DESCRIPTION
        assert seen["model_id"] == "hosted_vllm/Qwen/Qwen3-32B"
        assert seen["api_base"] == "http://localhost:9/v1"
        assert seen["run_logs_dir"] == "/tmp/run"

        # No search model in the run config -> fall back to the planner model.
        _dr_wiring(_stub_cfg("deepresearchqa"), cfg_main, task, Path("/tmp/run"))
        assert seen["model_id"] == "planner-m"


def test_parse_and_score_per_benchmark():
    from unittest.mock import patch

    # phantomwiki: the parsed list is what gets scored.
    agent = DeepReasonerAgent(_stub_cfg("phantomwiki"))
    task = Task(task_id="t", question="q")
    parsed = agent._parse(task, "1. Alice\n2. Bob")
    assert parsed == ["Alice", "Bob"]
    with patch.object(type(agent.cfg.benchmark), "score", lambda self, tid, ans: (ans, "em")):
        assert agent._score(task, "raw", parsed) == (["Alice", "Bob"], "em")

    # oolong: LLM parse, then (score, score == 1); a parse failure scores None.
    agent = DeepReasonerAgent(_stub_cfg("oolong"))
    bench_cls = type(agent.cfg.benchmark)
    with patch.object(bench_cls, "parse", lambda self, tid, ans, api_key: 7, create=True), \
            patch.object(bench_cls, "score", lambda self, tid, ans: 1.0):
        parsed = agent._parse(task, "seven")
        assert parsed == 7
        assert agent._score(task, "seven", parsed) == (1.0, 1.0)

    def boom(self, tid, ans, api_key):
        raise RuntimeError("no key")

    with patch.object(bench_cls, "parse", boom, create=True):
        assert agent._parse(task, "seven") is None
    assert agent._score(task, "seven", None) is None

    # deepresearchqa: judge score for both f1 and em.
    agent = DeepReasonerAgent(_stub_cfg("deepresearchqa"))
    bench_cls = type(agent.cfg.benchmark)
    with patch.object(bench_cls, "score_judge", lambda self, tid, out: (tid, 1.0, "ok"), create=True):
        assert agent._parse(task, "Paris") == "Paris"
        assert agent._score(task, "Paris", "Paris") == (1.0, 1.0)

    # A scorer that raises records None rather than failing the attempt.
    def score_boom(self, tid, ans):
        raise RuntimeError("scorer down")

    agent = DeepReasonerAgent(_stub_cfg("synthworlds"))
    with patch.object(type(agent.cfg.benchmark), "score", score_boom):
        assert agent._score(task, "x", "x") is None


# %%
if test():
    test_load_planner_cfg_max_iter_and_overrides()
    test_deep_reasoner_smoke_scripted_llm()
    test_parse_phantom_output()
    test_dr_wiring_tool_names_match_the_planner_prompts()
    test_dr_wiring_deepresearchqa_search_subagent()
    test_parse_and_score_per_benchmark()
    print("agents.deep_reasoner tests passed")

# %% [markdown]
# ## End
