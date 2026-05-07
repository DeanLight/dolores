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
# # ReAct
#
# ReAct Baseline

# %%
from juplit import test

# %%
import sys
import argparse
import glob
import json
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from smolagents import ToolCallingAgent, tool, OpenAIModel, LogLevel
from config import RunContext, settings
from benchmarks import phantomwiki, synthworlds, deepresearchqa
from benchmarks.deepresearchqa.deepresearch_agent import litellm_model, create_web_search_agent_tool
from vllm_utils import VllmConfig, vllm_server
import prompts
import random

# %% [markdown]
# ## Logs Smolagents
#
# Log Structure:
# ```
# {
#     "answer": "Tokyo has the highest population density...",
#     "steps": [
#         {"usage": {...}, "messages": [...]},
#         {"usage": {...}, "messages": [...]},
#         ...
#     ],
#     "usage": {  # total across all steps
#         "prompt_tokens": 79562,
#         "completion_tokens": 7634,
#         "total_tokens": 87196,
#     },
# }
# ```

# %%
def create_logs(agent):
    """Extract logs from a smolagents agent run.

    Args:
        agent: The smolagents agent after .run() has completed.
        result: The return value of agent.run() (i.e. the final answer).

    Returns a dict with:
        - "answer": the final answer string
        - "steps": list of per-step dicts with "usage" and "messages"
        - "usage": total token usage across all steps
    """
    steps_data = []

    for step in agent.memory.steps:
        if not hasattr(step, 'token_usage') or step.token_usage is None:
            continue  # skip TaskStep or steps with no LLM call

        # --- messages: input + output ---
        messages = []

        if hasattr(step, 'model_input_messages') and step.model_input_messages is not None:
            for msg in step.model_input_messages:
                # try:
                #     content = msg.content[0]["text"] if isinstance(msg.content, list) else msg.content
                # except (TypeError, KeyError, IndexError):
                #     content = str(msg.content) if msg.content else ""
                # messages.append({
                #     "role": msg.role.value,
                #     "content": content,
                # })
                messages.append({
                    "role": msg.role.value,
                    "content": msg.content[0]["text"] if isinstance(msg.content, list) else msg.content,
                })

        # append the assistant output
        if hasattr(step, 'model_output') and step.model_output is not None:
            messages.append({
                "role": "assistant",
                "content": step.model_output,
            })

        steps_data.append({
            "usage": step.model_output_message.raw.usage.to_dict(),
            "messages": messages
        })

    # --- total usage ---
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
if test():
    # CodeAct vs ReAct
    # https://huggingface.co/docs/smolagents/en/conceptual_guides/intro_agents
    pass

# %% [markdown]
# ## Actual Implementation

# %% [markdown]
# ### Playing Around with ReAct

# %%
if test():
    @tool
    def get_capital(country: str) -> str:
        """Look up the capital city of a country.

        Args:
            country: The name of the country, e.g. "France".
        """
        data = {
            "france": "Paris",
            "japan": "Tokyo",
            "germany": "Berlin",
        }
        return data.get(country.lower(), f"No data found for '{country}'.")

    @tool
    def get_population(city: str) -> str:
        """Look up the population of a city.

        Args:
            city: The name of the city, e.g. "Paris".
        """
        data = {
            "paris": "2,102,650",
            "tokyo": "13,960,000",
            "berlin": "3,677,472",
        }
        return data.get(city.lower(), f"No data found for '{city}'.")

    task = "What is the population of the capital of Japan?"

# %%
if test():
    model = OpenAIModel(
        model_id="gpt-5-mini",
        api_key=settings.openai_api_key,
    )

    ctx = RunContext(
        model="gpt-5-mini",
        benchmark="testing-bench",
        method="react",
        max_steps=50,
    )

    # model = OpenAIModel(
    #     model_id=ctx.model,
    #     api_base="http://localhost:8000/v1",
    #     api_key="dummy",
    #     tool_choice="auto"
    # )

    agent = ToolCallingAgent(
        tools=[get_capital, get_population],
        model=model,
        max_steps=ctx.max_steps,
        instructions="WTF",
        verbosity_level=LogLevel.ERROR
    )

# %%
if test():
    print(agent.system_prompt)

# %%
if test():
    # result = agent.run(task)
    # print(f"\nagent answer: {result}")
    pass

# %%
if test():
    # logs = create_logs(agent)
    # logs = {**ctx.model_dump(), "answer": str(result), **logs}
    pass

# %%
if test():
    # ctx.save(logs)
    pass

# %%
if test():
    # logs['total_usage']
    pass

# %% [markdown]
# ### Extra info on token counts
#
# **smolagents does NOT pass reasoning tokens into the history for the next agent call.**

# %% [markdown]
# ## PhantomWiki

# %% [markdown]
# ### In-Context Examples
#
# Adapted from **`REACT_EXAMPLES_EASY`** in [`phantom_eval/prompts.py`](https://github.com/kilian-group/phantom-wiki/blob/main/src/phantom_eval/prompts.py) These are the **10 examples used in the paper's ReAct experiments**   (see §4.1 of *Gong et al., 2025*).
#
# *In the COT and REACT prompts, we include 10 QA exemplars and hand-written reasoning traces. We choose these exemplars from a dataset instance of size 25 that is not used for evaluation. In REACT, we limit LLMs to interact with the text corpus for up to 50 steps, which is sufficient to answer almost all questions in PhantomWiki instances.*
#
# The **reasoning traces are preserved verbatim**. The only modifications are formatting changes to match the `smolagents` tool-calling paradigm:
#
# | Original Format | Converted Format |
# |---|---|
# | `Action N: RetrieveArticle[entity]` | `{"name": "retrieve_article", "arguments": {"entity": "..."}}` |
# | `Action N: Search[attribute]` | `{"name": "search", "arguments": {"attribute": "..."}}` |
# | `Action N: Finish[answer]` | `{"name": "final_answer", "arguments": {"answer": "..."}}` |
#
# Additional formatting change:
#
# - `Thought N:` prefixes were **removed**.
# - The **free text before each `Action:` block** serves the same role in **smolagents**.

# %% [markdown]
# ### Keep track of tested IDs

# %%
def find_tested_ids(model, benchmark, method, no_thinking=False):

    log_dir = RunContext(model=model, benchmark=benchmark, method=method,
                         no_thinking=no_thinking).log_dir()

    tested_ids = set()
    for log_file in glob.glob(f"{log_dir}/*.json"):
        with open(log_file, "r") as f:
            data = json.load(f)
            tested_ids.add(data.get('test_id') or data.get('example_id'))

    return tested_ids

# %%
def find_tested_ids_deepresearchqa(model, benchmark, method, no_thinking=False):
    """Find test IDs that already have a saved result.json in a subfolder.

    Used for the deepresearchqa benchmark, which writes per-example subfolders
    (``<stem>/result.json`` plus the search-subagent's per-delegation JSONs)
    instead of the flat ``<stem>.json`` layout used by the other benchmarks."""
    log_dir = RunContext(model=model, benchmark=benchmark, method=method,
                         no_thinking=no_thinking).log_dir()

    tested_ids = set()
    for path in glob.glob(f"{log_dir}/*/result.json"):
        with open(path) as f:
            data = json.load(f)
            tid = data.get("test_id")
            if tid is not None:
                tested_ids.add(tid)
    return tested_ids

# %% [markdown]
# ### Running PhantomWiki
#
# Check `agent.system_prompt`. Basically, `instructions` go in a good place. nothing to worry about! Take a look at the prompt, it is nicely constructed!
#
# Basically, from the [paper](https://arxiv.org/pdf/2502.20377) we use:
#   - in context examples of the prompt
#   - 50 turns of ReAct

# %%
def run_single_phantomwiki(test_id, size, seed, model_name, benchmark, method,
                           max_steps, api_base, api_key, no_thinking=False):
    """Run one ReAct agent on a PhantomWiki example. Fully isolated — no shared state."""
    question, retrieve_article_fn, search_fn = phantomwiki.get_task(size, seed, test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        max_steps=max_steps,
        test_id=test_id,
        size=size,
        seed=seed,
        no_thinking=no_thinking,
    )

    extra_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if no_thinking else {}

    model = OpenAIModel(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
        tool_choice="auto",
        **extra_kwargs,
    )

    _retrieve_article = tool(retrieve_article_fn)
    _search = tool(search_fn)

    agent = ToolCallingAgent(
        tools=[_retrieve_article, _search],
        model=model,
        max_steps=max_steps,
        instructions=prompts.PHANTOM_WIKI_REACT_INSTRUCTIONS,
        verbosity_level=LogLevel.ERROR,
    )

    result = agent.run(question)

    answer_str = str(result).strip()
    parsed = [s.strip() for s in answer_str.split(",") if s.strip()] if answer_str else []

    logs = create_logs(agent)
    logs = {"answer": answer_str, "parsed": parsed, **logs}
    ctx.save(logs)

    return test_id, result

# %% [markdown]
# ## SynthWorlds
#
# SM (Synthetic-Mapped) only — 1,200 rows.
#
# Retrieval tool: `synthworlds.retrieve_top_5` (requires `OPENAI_API_KEY` in env).
#
# Scoring: F1 is per-token (partial answers ok). Parsing the model output is our responsibility.

# %%
def run_single_synthworlds(test_id, model_name, benchmark, method,
                           max_steps, api_base, api_key, no_thinking=False):
    """Run one ReAct agent on a SynthWorlds example. Fully isolated — no shared state."""
    question = synthworlds.get_task(test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        max_steps=max_steps,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    extra_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if no_thinking else {}

    model = OpenAIModel(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
        tool_choice="auto",
        **extra_kwargs,
    )

    retrieve_top_5 = synthworlds.create_retriever_tool(api_key=settings.openai_api_key)
    _retrieve_top_5 = tool(retrieve_top_5)

    agent = ToolCallingAgent(
        tools=[_retrieve_top_5],
        model=model,
        max_steps=max_steps,
        instructions=prompts.SYNTHWORLDS_REACT_INSTRUCTIONS,
        verbosity_level=LogLevel.ERROR,
    )

    result = agent.run(question)

    answer_str = str(result).strip()

    logs = create_logs(agent)
    logs = {"answer": answer_str, **logs}
    ctx.save(logs)

    return test_id, result

# %%
if test():
    retrieve_top_5 = synthworlds.create_retriever_tool(api_key=settings.openai_api_key)
    retrieve_top_5(query="Cultural Heritage Archives")

# %% [markdown]
# ## Running DeepResearchQA

# %%
def run_single_deepresearchqa(test_id, model_name, benchmark, method,
                              max_steps, api_base, api_key, no_thinking=False):
    """Run one ReAct agent on a DeepResearchQA example. Fully isolated — no shared state.

    The DeepResearch web-search subagent is wrapped as a single ``tool`` for the
    outer ToolCallingAgent. The subagent writes per-delegation JSON logs into
    ``exp_dir``, and the ReAct run's final output is saved alongside them as
    ``result.json`` (same layout as ``06_deepresearch.ipynb``).

    Note: ``no_thinking`` only affects the outer ToolCallingAgent's OpenAIModel.
    The inner search subagent uses ``litellm_model`` and is not toggled here."""
    question = deepresearchqa.get_task(test_id)

    ctx = RunContext(
        model=model_name,
        benchmark=benchmark,
        method=method,
        max_steps=max_steps,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    exp_dir = ctx.log_dir() / ctx.stem
    exp_dir.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if no_thinking else {}

    model = OpenAIModel(
        model_id=model_name,
        api_base=api_base,
        api_key=api_key,
        tool_choice="auto",
        **extra_kwargs,
    )

    lm = litellm_model(
        model_id=f"hosted_vllm/{model_name}",
        api_base=api_base,
        api_key=api_key,
    )
    search_agent_fn = create_web_search_agent_tool(lm, run_logs_dir=str(exp_dir))
    _search_agent = tool(search_agent_fn)

    agent = ToolCallingAgent(
        tools=[_search_agent],
        model=model,
        max_steps=max_steps,
        verbosity_level=LogLevel.ERROR,
    )

    result = agent.run(question)
    answer_str = str(result).strip()

    logs = create_logs(agent)
    data = {**ctx.model_dump(), "answer": answer_str, **logs}
    result_path = exp_dir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return test_id, result

# %% [markdown]
# ## Main Entry

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["phantomwiki", "synthworlds", "deepresearchqa"],
                        help="Benchmark to run (phantomwiki, synthworlds, or deepresearchqa)")
    parser.add_argument("--max_steps", type=int, default=75)
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
    # PhantomWiki-specific
    parser.add_argument("--pw_size", type=int, default=500, help="PhantomWiki corpus size (50, 500, 5000)")
    parser.add_argument("--pw_seed", type=int, default=1, help="PhantomWiki seed")
    # DeepResearchQA-specific
    parser.add_argument("--limit", type=int, default=None, help="DeepResearchQA: cap on # of test ids (None = all)")
    parser.add_argument("--seed", type=int, default=42, help="DeepResearchQA: sampling seed for list_test_ids")
    # Disable Qwen3-style thinking via vLLM chat_template_kwargs.
    # Logs land in a separate `<model>-nothink/` folder so they don't mix with thinking-on runs.
    parser.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                        help="Disable model thinking (extra_body.chat_template_kwargs.enable_thinking=False) and route logs to <model>-nothink/.")
    # Worker mode — if set, run one example and exit.
    parser.add_argument("--test_id", type=str, default=None,
                        help="If set, run only this one example and exit (child/worker mode).")
    args = parser.parse_args()

    method = "react"

    # ------------------------------------------------------------------
    # CHILD MODE — run exactly one example and exit. No dispatcher work.
    # ------------------------------------------------------------------
    if args.test_id is not None:
        if args.benchmark == "phantomwiki":
            run_single_phantomwiki(
                test_id=args.test_id,
                size=args.pw_size,
                seed=args.pw_seed,
                model_name=args.model,
                benchmark=f"phantomwiki_{args.pw_size}_{args.pw_seed}",
                method=method,
                max_steps=args.max_steps,
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
                max_steps=args.max_steps,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "deepresearchqa":
            run_single_deepresearchqa(
                test_id=args.test_id,
                model_name=args.model,
                benchmark="deepresearchqa",
                method=method,
                max_steps=args.max_steps,
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
    if args.benchmark == "phantomwiki":
        benchmark = f"phantomwiki_{args.pw_size}_{args.pw_seed}"
        test_ids = phantomwiki.list_test_ids(args.pw_size, args.pw_seed)
        random.shuffle(test_ids)
    elif args.benchmark == "synthworlds":
        benchmark = "synthworlds"
        test_ids = synthworlds.list_test_ids()
        random.shuffle(test_ids)
    elif args.benchmark == "deepresearchqa":
        benchmark = "deepresearchqa"
        test_ids = deepresearchqa.list_test_ids(limit=args.limit, seed=args.seed)

    # deepresearchqa writes per-example folders (<stem>/result.json); others write flat <stem>.json.
    if args.benchmark == "deepresearchqa":
        already_done = find_tested_ids_deepresearchqa(args.model, benchmark, method,
                                                     no_thinking=args.no_thinking)
    else:
        already_done = find_tested_ids(args.model, benchmark, method,
                                       no_thinking=args.no_thinking)
    work = [tid for tid in test_ids if tid not in already_done]

    print(f"Running {len(work)} / {len(test_ids)} examples -- (Total {len(already_done)} done)")

    def launch(tid):
        cmd = [
            sys.executable, "-m", "baselines.react",
            "--benchmark", args.benchmark,
            "--model", args.model,
            "--max_steps", str(args.max_steps),
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
    #   export HF_DATASETS_OFFLINE=1
    #   uv run python -m baselines.react --benchmark phantomwiki --pw_size 500 --model "Qwen/Qwen3-32B"
    #   uv run python -m baselines.react --benchmark synthworlds --model "Qwen/Qwen3-32B" --max_workers 70
    #   uv run python -m baselines.react --benchmark deepresearchqa --model "Qwen/Qwen3-32B" --max_workers 70

    #   uv run python -m baselines.react --benchmark phantomwiki --model "Qwen/Qwen3-235B-A22B" --max_workers 1
    #   uv run python -m baselines.react --benchmark phantomwiki --model "Qwen/Qwen3-Coder-Next" --max_workers 150
    pass

# %% [markdown]
# ## End
