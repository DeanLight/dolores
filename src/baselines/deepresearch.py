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
# # Deep Research
#
# Deep research implementation

# %%
from juplit import test

# %% [markdown]
# ## Running the open_deep_research agent
#
# Requires `HF_TOKEN`, `SERPER_API_KEY`, and (for default `o1`) `OPENAI_API_KEY` in `.env`.

# %%
import sys
import argparse
import glob
import json
import subprocess
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from smolagents import CodeAgent, LogLevel
from benchmarks import deepresearchqa, phantomwiki, synthworlds
from benchmarks.deepresearchqa.deepresearch_agent import (
    litellm_model,
    _create_web_search_agent,
    _create_search_agent,
    visualizer,
    TextInspectorTool,
)
from config import RunContext, settings
from vllm_utils import VllmConfig, vllm_server
from prompts import (
    SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS,
    SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS,
    PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS,
    PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS,
)

# %%
def create_logs(agent):
    """Extract logs from a smolagents agent run.

    Returns a dict with:
        - "total_usage": aggregate token counts from agent.monitor
        - "steps": list of per-step dicts with "usage" and "messages"

    Note: PlanningStep messages start with a ``user`` turn (built from the
    ``planning`` prompt templates), while ActionStep messages start with the
    agent ``system`` prompt — so the first role per step is not uniform.
    """
    steps_data = []

    for step in agent.memory.steps:
        if not hasattr(step, 'token_usage') or step.token_usage is None:
            continue  # skip TaskStep or steps with no LLM call

        messages = []
        if hasattr(step, 'model_input_messages') and step.model_input_messages is not None:
            for msg in step.model_input_messages:
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content[0]["text"] if isinstance(msg.content, list) else msg.content,
                })

        if hasattr(step, 'model_output') and step.model_output is not None:
            messages.append({
                "role": "assistant",
                "content": step.model_output,
            })

        # Read token usage from step.token_usage directly — PlanningStep has
        # token_usage populated but model_output_message.raw is None, so the
        # old `step.model_output_message.raw.usage.to_dict()` path crashed.
        tu = step.token_usage
        steps_data.append({
            "usage": {
                "prompt_tokens":     tu.input_tokens,
                "completion_tokens": tu.output_tokens,
                "total_tokens":      tu.total_tokens,
            },
            "messages": messages,
        })

    total_usage = agent.monitor.get_total_token_counts()
    return {
        "total_usage": {
            "prompt_tokens":     total_usage.input_tokens,
            "completion_tokens": total_usage.output_tokens,
            "total_tokens":      total_usage.total_tokens,
        },
        "steps": steps_data,
    }

# %%
def create_deep_research_agent(model, run_logs_dir: str, subagent_instr: str = None,
                               orch_instructions: str = None, manager_model=None):
    """Manager ``CodeAgent`` + managed web-search subagent. ``run_logs_dir``
    is forwarded to the subagent (one JSON log per delegation).

    If ``manager_model`` is given, the manager uses it while the subagent keeps
    using ``model`` (used to disable thinking only on the top agent)."""

    search_agent = _create_web_search_agent(model, run_logs_dir=run_logs_dir, instructions=subagent_instr)

    text_limit = 100000
    manager_agent = CodeAgent(
        model=manager_model if manager_model is not None else model,
        tools=[visualizer, TextInspectorTool(model, text_limit)],
        max_steps=75,
        verbosity_level=LogLevel.ERROR,
        instructions=orch_instructions,
        # NOTE: this is a clear improvement of base DR method because it was getting lots of timeouts due to subagent long-horizon
        executor_kwargs={"timeout_seconds": None},
        #verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[search_agent],
    )
    return manager_agent

# %%
def create_custom_deep_research_agent(model, tools: list, run_logs_dir: str,
                                      subagent_instr: str = None,
                                      orch_instructions: str = None,
                                      manager_model=None):
    """Like ``create_deep_research_agent`` but the subagent uses a
    caller-supplied ``tools`` list instead of web-browsing tools.

    If ``manager_model`` is given, the manager uses it while the subagent keeps
    using ``model`` (used to disable thinking only on the top agent)."""
    search_agent = _create_search_agent(model, tools=tools, run_logs_dir=run_logs_dir, instructions=subagent_instr)

    text_limit = 100000
    manager_agent = CodeAgent(
        model=manager_model if manager_model is not None else model,
        tools=[],
        max_steps=75,
        verbosity_level=LogLevel.ERROR,
        instructions=orch_instructions,
        # NOTE: this is a clear improvement of base DR method because it was getting lots of timeouts due to subagent long-horizon
        executor_kwargs={"timeout_seconds": None},
        #verbosity_level=2,
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[search_agent],
    )
    return manager_agent

# %% [markdown]
# ## DeepResearchQA

# %%
def find_tested_ids(log_dir):
    """Find test IDs that already have a saved result.json in a subfolder."""
    done = set()
    for path in glob.glob(f"{log_dir}/*/result.json"):
        with open(path) as f:
            data = json.load(f)
            tid = data.get("test_id")
            if tid is not None:
                done.add(tid)
    return done

# %%
def run_single_deepresearchqa(test_id, model_name, benchmark, method,
                              api_base, api_key, no_thinking=False):
    """Run one deep research agent on a DeepResearchQA example.

    ``no_thinking`` only affects the top (manager) agent. The web-search
    subagent keeps thinking enabled."""
    question = deepresearchqa.get_task(test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    exp_dir = ctx.log_dir() / ctx.stem
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = litellm_model(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    manager_model = None
    if no_thinking:
        manager_model = litellm_model(
            model_id=model_name,
            api_base=api_base,
            api_key=api_key,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    agent = create_deep_research_agent(model, run_logs_dir=str(exp_dir),
                                       manager_model=manager_model)

    result = agent.run(question)
    answer_str = str(result).strip()

    logs = create_logs(agent)
    data = {**ctx.identity, "answer": answer_str, **logs}
    result_path = exp_dir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return test_id, result

# %% [markdown]
# ## PhantomWiki

# %%
def run_single_phantomwiki(test_id, size, seed, model_name, benchmark, method,
                           api_base, api_key, no_thinking=False):
    """Run one deep research agent on a PhantomWiki example.

    ``no_thinking`` only affects the top (manager) agent. The custom-search
    subagent keeps thinking enabled."""
    question, retrieve_article_fn, search_fn = phantomwiki.get_task(size, seed, test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        test_id=test_id,
        size=size,
        seed=seed,
        no_thinking=no_thinking,
    )

    exp_dir = ctx.log_dir() / ctx.stem
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = litellm_model(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    manager_model = None
    if no_thinking:
        manager_model = litellm_model(
            model_id=model_name,
            api_base=api_base,
            api_key=api_key,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    # The DR subagent is ReAct and inherits all ReAct tools/instructions (e.g., in-context examples), the orchestrator
    # agent gets the main instructions of the Phantomwiki task.
    agent = create_custom_deep_research_agent(
        model, tools=[retrieve_article_fn, search_fn], run_logs_dir=str(exp_dir),
        subagent_instr=PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS,
        orch_instructions=PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS,
        manager_model=manager_model,
    )

    result = agent.run(question)

    answer_str = str(result).strip()
    parsed = [s.strip() for s in answer_str.split(",") if s.strip()] if answer_str else []

    logs = create_logs(agent)
    data = {**ctx.identity, "answer": answer_str, "parsed": parsed, **logs}
    result_path = exp_dir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return test_id, result

# %% [markdown]
# ## SynthWorlds

# %%
def run_single_synthworlds(test_id, model_name, benchmark, method,
                           api_base, api_key, no_thinking=False):
    """Run one deep research agent on a SynthWorlds example.

    ``no_thinking`` only affects the top (manager) agent. The custom-search
    subagent keeps thinking enabled."""
    question = synthworlds.get_task(test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    exp_dir = ctx.log_dir() / ctx.stem
    exp_dir.mkdir(parents=True, exist_ok=True)

    model = litellm_model(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
    )

    manager_model = None
    if no_thinking:
        manager_model = litellm_model(
            model_id=model_name,
            api_base=api_base,
            api_key=api_key,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    retrieve_top_5 = synthworlds.create_retriever_tool(api_key=settings.openai_api_key)

    # The DR subagent is ReAct and inherits all ReAct tools/instructions (e.g., in-context examples), the orchestrator
    # agent gets the main instructions of the SynthWorlds task.
    agent = create_custom_deep_research_agent(
        model, tools=[retrieve_top_5], run_logs_dir=str(exp_dir),
        subagent_instr=SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS,
        orch_instructions=SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS,
        manager_model=manager_model,
    )

    result = agent.run(question)
    answer_str = str(result).strip()

    logs = create_logs(agent)
    data = {**ctx.identity, "answer": answer_str, **logs}
    result_path = exp_dir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return test_id, result

# %% [markdown]
# ## Main Entry

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="hosted_vllm/Qwen/Qwen3-32B")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["deepresearchqa", "phantomwiki", "synthworlds"],
                        help="Benchmark to run")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--api_base", type=str, default="http://localhost:8555/v1")
    parser.add_argument("--api_key", type=str, default="your_secret")
    # Local vllm server — parent mode only; ignored in child/worker mode.
    parser.add_argument("--vllm_model", type=str, default=None,
                        help="If set, spin up a local vllm server for this model and point --api_base at it.")
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.9,
                        help="gpu_memory_utilization for the local vllm server (default 0.9).")
    parser.add_argument("--vllm_wait", type=int, default=600,
                        help="Seconds to wait for vllm /health before giving up (default 600).")
    # DeepResearchQA-specific
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # PhantomWiki-specific
    parser.add_argument("--pw_size", type=int, default=500, help="PhantomWiki corpus size (50, 500, 5000)")
    parser.add_argument("--pw_seed", type=int, default=1, help="PhantomWiki seed")
    # Disable Qwen3-style thinking on the top (manager) agent only — subagents
    # keep thinking enabled. Logs land in a separate `<model>-nothink/` folder.
    parser.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                        help="Disable thinking on the top manager agent (extra_body.chat_template_kwargs.enable_thinking=False) and route logs to <model>-nothink/. Subagents are unaffected.")
    # Worker mode — if set, run one example and exit.
    parser.add_argument("--test_id", type=str, default=None,
                        help="If set, run only this one example and exit (child/worker mode).")
    args = parser.parse_args()

    method = "deepresearch"

    # ------------------------------------------------------------------
    # CHILD MODE — run exactly one example and exit. No dispatcher work.
    # ------------------------------------------------------------------
    if args.test_id is not None:
        if args.benchmark == "deepresearchqa":
            run_single_deepresearchqa(
                test_id=args.test_id,
                model_name=args.model,
                benchmark="deepresearchqa",
                method=method,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "phantomwiki":
            run_single_phantomwiki(
                test_id=args.test_id,
                size=args.pw_size,
                seed=args.pw_seed,
                model_name=args.model,
                benchmark=f"phantomwiki_{args.pw_size}_{args.pw_seed}",
                method=method,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "synthworlds":
            run_single_synthworlds(
                test_id=args.test_id,
                model_name=args.model,
                benchmark="synthworlds",
                method=method,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        sys.exit(0)

    # ------------------------------------------------------------------
    # PARENT MODE — read test ids, filter, fan out subprocess workers.
    # Each child runs in its own OS process. If one OOMs or crashes,
    # only that child dies; siblings and the parent keep going.
    # ------------------------------------------------------------------
    if args.benchmark == "deepresearchqa":
        benchmark = "deepresearchqa"
        test_ids = deepresearchqa.list_test_ids(limit=args.limit, seed=args.seed)
    elif args.benchmark == "phantomwiki":
        benchmark = f"phantomwiki_{args.pw_size}_{args.pw_seed}"
        test_ids = phantomwiki.list_test_ids(args.pw_size, args.pw_seed)
        random.shuffle(test_ids)
    elif args.benchmark == "synthworlds":
        benchmark = "synthworlds"
        test_ids = synthworlds.list_test_ids()
        random.shuffle(test_ids)

    log_dir = RunContext.get_log_dir(benchmark, method, args.model,
                                     no_thinking=args.no_thinking)
    already_done = find_tested_ids(log_dir)
    work = [tid for tid in test_ids if tid not in already_done]

    print(f"Running {len(work)} / {len(test_ids)} examples -- (Total {len(already_done)} done)")

    def launch(tid):
        cmd = [
            sys.executable, "-m", "baselines.deepresearch",
            "--benchmark", args.benchmark,
            "--model", args.model,
            "--api_base", args.api_base,
            "--api_key", args.api_key,
            "--test_id", str(tid),
        ]
        if args.benchmark == "phantomwiki":
            cmd += ["--pw_size", str(args.pw_size), "--pw_seed", str(args.pw_seed)]
        if args.no_thinking:
            cmd += ["--no-thinking"]

        while True:
            r = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if r.returncode == 0:
                return r

            tail = (r.stderr or "").strip().splitlines()[-1:] if r.stderr else []
            msg = tail[0] if tail else f"exit {r.returncode}"
            print(f"↻ {tid} failed ({msg}) — Retrying indefinitely...")

    def _run_pool():
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {pool.submit(launch, tid): tid for tid in work}
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    future.result()
                    print(f"✓ {tid}")
                except Exception as e:
                    print(f"✗ {tid} — Unhandled Exception: {e}")

    if args.vllm_model:
        vllm_cfg = VllmConfig(
            model=args.vllm_model,
            gpu_memory_utilization=args.vllm_gpu_mem,
        )
        with vllm_server(vllm_cfg, wait_timeout=args.vllm_wait):
            args.api_base = f"http://localhost:{vllm_cfg.port}/v1"
            _run_pool()
    else:
        _run_pool()

# %%
if test():
    #   uv run python -m baselines.deepresearch --benchmark deepresearchqa --model "hosted_vllm/Qwen/Qwen3-32B" --max_workers 70
    #   uv run python -m baselines.deepresearch --benchmark phantomwiki --pw_size 50 --model "hosted_vllm/Qwen/Qwen3-32B" --max_workers 100
    #   uv run python -m baselines.deepresearch --benchmark synthworlds --model "hosted_vllm/Qwen/Qwen3-32B" --max_workers 100
    pass

# %% [markdown]
# ## End
