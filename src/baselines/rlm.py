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

# Recursive Language Models (RLM) — upstream implementation and paper:
# https://github.com/alexzhang13/rlm (Alex Zhang et al.)
#
# This file is Dolores-specific integration only (benchmark runners, env wiring,
# optional OpenAI client patches for vLLM). Import ``from rlm import RLM`` via the
# ``rlms`` PyPI package, which implements the linked repository.

# %% [markdown]
# # RLM baseline
#
# Uses the **RLM** library from [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm).
# Everything below that is not in the upstream package is maintained in this repo.

# %%
from juplit import test

# %%
import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv
from config import RunContext, settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmarks import oolong, phantomwiki, synthworlds, deepresearchqa
from benchmarks.deepresearchqa.deepresearch_agent import litellm_model, create_web_search_agent_tool
from vllm_utils import VllmConfig, vllm_server
import glob
import json
from rlm.logger import RLMLogger
from rlm import RLM
import random
from prompts import SYNTHWORLDS_RLM_INSTRUCTIONS, PHANTOM_WIKI_RLM_INSTRUCTIONS
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

# %% [markdown]
# ## OpenAI client monkeypatch — extra_body passthrough
#
# Patches the **third-party** ``rlm`` package (see [alexzhang13/rlm](https://github.com/alexzhang13/rlm));
# not part of upstream. Keeps behavior aligned with the vendored ``rlm`` OpenAI client while merging
# ``extra_body`` from ``RLM`` constructor kwargs (e.g. for Qwen3 thinking toggles on vLLM).
#
# rlm's OpenAIClient.completion / acompletion build extra_body locally and
# don't expose a hook for user-supplied chat_template_kwargs (or any other
# extra_body field). We need that hook to disable Qwen3 thinking on the top
# RLM via vLLM's chat_template_kwargs.
#
# Strategy: replace both methods with versions that mirror the upstream body
# verbatim, plus one merge step that pulls self.kwargs.get("extra_body") into
# the request. BaseLM.__init__ already stores extra constructor kwargs in
# self.kwargs, so passing ``extra_body=...`` via ``backend_kwargs`` flows
# naturally without any further plumbing.
#
# Default behavior is byte-identical when ``extra_body`` is not passed — the
# merge is a no-op, _track_cost still runs, and the prime-intellect base_url
# special case is preserved.
#
# Pinned against rlms==0.1.1 (Python module ``rlm``). If rlms is bumped and
# the upstream OpenAIClient.completion / acompletion bodies change shape,
# this patch will silently get out of date — re-derive from the new source.

# %%
import rlm.clients.openai as _rlm_openai

_DEFAULT_PRIME_INTELLECT_BASE_URL = _rlm_openai.DEFAULT_PRIME_INTELLECT_BASE_URL


def _patched_completion(self, prompt, model=None):
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        messages = prompt
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    model = model or self.model_name
    if not model:
        raise ValueError("Model name is required for OpenAI client.")

    extra_body = {}
    if self.client.base_url == _DEFAULT_PRIME_INTELLECT_BASE_URL:
        extra_body["usage"] = {"include": True}

    user_extra = self.kwargs.get("extra_body")
    if user_extra:
        extra_body = {**extra_body, **user_extra}

    response = self.client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body
    )
    self._track_cost(response, model)
    return response.choices[0].message.content


async def _patched_acompletion(self, prompt, model=None):
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        messages = prompt
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")

    model = model or self.model_name
    if not model:
        raise ValueError("Model name is required for OpenAI client.")

    extra_body = {}
    if self.client.base_url == _DEFAULT_PRIME_INTELLECT_BASE_URL:
        extra_body["usage"] = {"include": True}

    user_extra = self.kwargs.get("extra_body")
    if user_extra:
        extra_body = {**extra_body, **user_extra}

    response = await self.async_client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body
    )
    self._track_cost(response, model)
    return response.choices[0].message.content


_rlm_openai.OpenAIClient.completion = _patched_completion
_rlm_openai.OpenAIClient.acompletion = _patched_acompletion

# %%
if test():
    question, retrieve_article_fn, search_fn = phantomwiki.get_task(50, 1, 'e4be6828-ec6e-41c9-b1ab-55b14cb8807b')

# %%
if test():
    retrieve_article_fn.__doc__

# %%
if test():
    from rlm.utils.prompts import RLM_SYSTEM_PROMPT

    # WITH_EXAMPLES_PROMPT = RLM_SYSTEM_PROMPT + """ my continuation """

    # RLM(..., custom_system_prompt=WITH_EXAMPLES_PROMPT, ...)

    # my_result = "michael"
    # FINAL_VAR(my_result)
    # # or FINAL_VAR("michael") in text only
    pass

# %%
def unquote(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s

# %% [markdown]
# ### Send

# %%
def run_single_oolong(test_id, model, benchmark, method,
                     max_depth, max_iterations, api_base, api_key, openai_api_key,
                     no_thinking=False):
    """Run one RLM agent on an Oolong example. Fully isolated — no shared state."""
    document, question = oolong.get_task(test_id)

    ctx = RunContext(
        model=model,
        benchmark=benchmark,
        method=method,
        max_depth=max_depth,
        max_iterations=max_iterations,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    logger = RLMLogger()

    backend_kwargs = {
        "model_name": model,
        "api_key": api_key,
        "base_url": api_base,
    }
    if no_thinking:
        backend_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    rlm = RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=logger,
    )

    result = rlm.completion(
        prompt=document,
        root_prompt=question,
    )

    # unquote: model writes FINAL("x"); RLM's FINAL parser (unlike FINAL_VAR) keeps
    # the quotes, which would unfairly fail exact-match.
    parsed = oolong.parse(test_id, unquote(result.response), api_key=openai_api_key)

    logs = result.to_dict()
    logs["parsed"] = parsed
    ctx.save(logs)

    return test_id, result

# %% [markdown]
# ## Running PhantomWiki

# %%
def run_single_phantomwiki(test_id, size, seed, model, benchmark, method,
                           max_depth, max_iterations, api_base, api_key,
                           no_thinking=False):
    """Run one RLM agent on a PhantomWiki example. Fully isolated — no shared state."""
    question, retrieve_article_fn, search_fn = phantomwiki.get_task(size, seed, test_id)

    ctx = RunContext(
        model=model,
        benchmark=benchmark,
        method=method,
        max_depth=max_depth,
        max_iterations=max_iterations,
        test_id=test_id,
        size=size,
        seed=seed,
        no_thinking=no_thinking,
    )

    logger = RLMLogger()

    backend_kwargs = {
        "model_name": model,
        "api_key": api_key,
        "base_url": api_base,
    }
    if no_thinking:
        backend_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    rlm = RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        # We give Phantomwiki-specific instructions + some examples of how to use the tools (as we do with other baselines)!
        custom_system_prompt=RLM_SYSTEM_PROMPT + "\n\n" + PHANTOM_WIKI_RLM_INSTRUCTIONS,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=logger,
        custom_tools={
            "retrieve_article": {
                "tool": retrieve_article_fn,
                "description": retrieve_article_fn.__doc__,
            },
            "search": {
                "tool": search_fn,
                "description": search_fn.__doc__,
            },
        },
    )

    result = rlm.completion(
        prompt="",
        root_prompt=question,
    )

    # unquote: model writes FINAL("x"); RLM's FINAL parser (unlike FINAL_VAR) keeps
    # the quotes, which would unfairly fail exact-match. Twice: list and items.
    answer_str = unquote(result.response)
    parsed = [unquote(s) for s in answer_str.split(",") if s.strip()] if answer_str else []

    logs = result.to_dict()
    logs["parsed"] = parsed
    ctx.save(logs)

    return test_id, result

# %% [markdown]
# ## Running SynthWorlds

# %%
def run_single_synthworlds(test_id, model, benchmark, method,
                           max_depth, max_iterations, api_base, api_key,
                           no_thinking=False):
    """Run one RLM agent on a SynthWorlds example. Fully isolated — no shared state."""
    question = synthworlds.get_task(test_id)

    ctx = RunContext(
        model=model,
        benchmark=benchmark,
        method=method,
        max_depth=max_depth,
        max_iterations=max_iterations,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    logger = RLMLogger()

    retrieve_top_5 = synthworlds.create_retriever_tool(api_key=settings.openai_api_key)

    backend_kwargs = {
        "model_name": model,
        "api_key": api_key,
        "base_url": api_base,
    }
    if no_thinking:
        backend_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    rlm = RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        environment="local",
        # We give Phantomwiki-specific instructions + some examples of how to use the tools (as we do with other baselines)!
        custom_system_prompt=RLM_SYSTEM_PROMPT + "\n\n" + SYNTHWORLDS_RLM_INSTRUCTIONS,
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=logger,
        custom_tools={
            "retrieve_top_5": {
                "tool": retrieve_top_5,
                "description": retrieve_top_5.__doc__,
            },
        },
    )

    result = rlm.completion(
        prompt="",
        root_prompt=question,
    )

    # unquote: model writes FINAL("x"); RLM's FINAL parser (unlike FINAL_VAR) keeps
    # the quotes, which would unfairly fail exact-match.
    answer_str = unquote(result.response)

    logs = result.to_dict()
    logs["answer"] = answer_str
    ctx.save(logs)

    return test_id, result

# %% [markdown]
# ## Running DeepResearchQA

# %%
def run_single_deepresearchqa(test_id, model, benchmark, method,
                              max_depth, max_iterations, api_base, api_key,
                              no_thinking=False):
    """Run one RLM agent on a DeepResearchQA example. Fully isolated — no shared state.

    Uses the DeepResearch web-search subagent as a single RLM tool. The subagent writes
    one JSON log per delegation into ``exp_dir``, and the RLM's final output is saved
    alongside them as ``result.json`` (same layout as ``06_deepresearch.ipynb``).

    ``no_thinking`` only affects the top RLM (via the rlm OpenAIClient monkeypatch
    forwarding ``extra_body.chat_template_kwargs``). The inner web-search subagent
    uses ``litellm_model`` (smolagents.LiteLLMModel — a different class) and is
    not toggled here."""
    question = deepresearchqa.get_task(test_id)

    ctx = RunContext(
        model=model,
        benchmark=benchmark,
        method=method,
        max_depth=max_depth,
        max_iterations=max_iterations,
        test_id=test_id,
        no_thinking=no_thinking,
    )

    exp_dir = ctx.log_dir() / ctx.stem
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = RLMLogger()

    lm = litellm_model(
        model_id=f"hosted_vllm/{model}",
        api_base=api_base,
        api_key=api_key,
    )
    search_agent = create_web_search_agent_tool(lm, run_logs_dir=str(exp_dir))

    backend_kwargs = {
        "model_name": model,
        "api_key": api_key,
        "base_url": api_base,
    }
    if no_thinking:
        backend_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    rlm = RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_depth=max_depth,
        max_iterations=max_iterations,
        logger=logger,
        custom_tools={
            "search_agent": {
                "tool": search_agent,
                "description": search_agent.__doc__,
            },
        },
    )

    result = rlm.completion(
        prompt="",
        root_prompt=question,
    )

    # unquote: model writes FINAL("x"); RLM's FINAL parser (unlike FINAL_VAR) keeps
    # the quotes, which would unfairly fail exact-match.
    answer_str = unquote(result.response)

    logs = result.to_dict()
    data = {**ctx.identity, "answer": answer_str, **logs}
    result_path = exp_dir / "result.json"
    result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    return test_id, result

# %%
def find_tested_ids(log_dir):
    """Find test IDs that already have a saved JSON."""
    done = set()
    for path in glob.glob(f"{log_dir}/*.json"):
        with open(path) as f:
            data = json.load(f)
            tid = data.get("test_id") or data.get("ex_id") or data.get("example_id")
            if tid is not None:
                done.add(tid)
    return done

# %%
def find_tested_ids_deepresearchqa(log_dir):
    """Find test IDs that already have a saved result.json in a subfolder.

    Used for the deepresearchqa benchmark, which writes per-example subfolders
    (``<stem>/result.json`` plus the search-subagent's per-delegation JSONs)
    instead of the flat ``<stem>.json`` layout used by the other benchmarks."""
    done = set()
    for path in glob.glob(f"{log_dir}/*/result.json"):
        with open(path) as f:
            data = json.load(f)
            tid = data.get("test_id")
            if tid is not None:
                done.add(tid)
    return done

# %% [markdown]
# ## Main Entry

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--benchmark", type=str, default="oolong",
                        choices=["oolong", "phantomwiki", "synthworlds", "deepresearchqa"],
                        help="Which benchmark to run")
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--max_iterations", type=int, default=75)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--api_base", type=str, default="http://localhost:8555/v1")
    parser.add_argument("--api_key", type=str, default="your_secret")
    # Local vllm server — parent mode only; ignored in child/worker mode.
    parser.add_argument("--vllm_model", type=str, default=None,
                        help="If set, spin up a local vllm server for this model and point --api_base at it.")
    parser.add_argument("--vllm_gpu_mem", type=float, default=0.9,
                        help="gpu_memory_utilization for the local vllm server (default 0.9).")
    parser.add_argument("--vllm_wait", type=int, default=600,
                        help="Seconds to wait for vllm /health before giving up (default 600).")
    # Oolong / DeepResearchQA-specific
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    # PhantomWiki-specific
    parser.add_argument("--pw_size", type=int, default=500, help="PhantomWiki corpus size (50, 500, 5000)")
    parser.add_argument("--pw_seed", type=int, default=1, help="PhantomWiki seed")
    # Disable Qwen3-style thinking on the top RLM only — DR's inner search subagent
    # uses smolagents.LiteLLMModel (a different class than rlm's OpenAIClient) and
    # is unaffected by the rlm monkeypatch. Logs land in `<model>-nothink/`.
    parser.add_argument("--no-thinking", dest="no_thinking", action="store_true",
                        help="Disable thinking on the top RLM (extra_body.chat_template_kwargs.enable_thinking=False via the rlm OpenAIClient monkeypatch) and route logs to <model>-nothink/. DR search subagent unaffected.")
    # Worker mode — if set, run one example and exit.
    parser.add_argument("--test_id", type=str, default=None,
                        help="If set, run only this one example and exit (child/worker mode).")
    args = parser.parse_args()

    method = "rlm"

    # ------------------------------------------------------------------
    # CHILD MODE — run exactly one example and exit. No dispatcher work.
    # ------------------------------------------------------------------
    if args.test_id is not None:
        if args.benchmark == "oolong":
            run_single_oolong(
                test_id=args.test_id,
                model=args.model,
                benchmark="oolong-real",
                method=method,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                api_base=args.api_base,
                api_key=args.api_key,
                openai_api_key=settings.openai_api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "phantomwiki":
            run_single_phantomwiki(
                test_id=args.test_id,
                size=args.pw_size,
                seed=args.pw_seed,
                model=args.model,
                benchmark=f"phantomwiki_{args.pw_size}_{args.pw_seed}",
                method=method,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "synthworlds":
            run_single_synthworlds(
                test_id=args.test_id,
                model=args.model,
                benchmark="synthworlds",
                method=method,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                api_base=args.api_base,
                api_key=args.api_key,
                no_thinking=args.no_thinking,
            )
        elif args.benchmark == "deepresearchqa":
            run_single_deepresearchqa(
                test_id=args.test_id,
                model=args.model,
                benchmark="deepresearchqa",
                method=method,
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
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
    if args.benchmark == "oolong":
        benchmark = "oolong-real"
        test_ids = oolong.list_test_ids(limit=args.limit, seed=args.seed)
    elif args.benchmark == "phantomwiki":
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

    log_dir = RunContext.get_log_dir(benchmark, method, args.model,
                                     no_thinking=args.no_thinking)
    # deepresearchqa writes per-example folders (<stem>/result.json); others write flat <stem>.json.
    if args.benchmark == "deepresearchqa":
        already_done = find_tested_ids_deepresearchqa(log_dir)
    else:
        already_done = find_tested_ids(log_dir)

    work = [tid for tid in test_ids if tid not in already_done]

    print(f"Running {len(work)} / {len(test_ids)} examples -- (Total {len(already_done)} done)")

    def launch(tid):
        cmd = [
            sys.executable, "-m", "baselines.rlm",
            "--benchmark", args.benchmark,
            "--model", args.model,
            "--max_depth", str(args.max_depth),
            "--max_iterations", str(args.max_iterations),
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
    len(find_tested_ids(RunContext.get_log_dir('oolong-real', 'rlm', 'Qwen/Qwen3-32B')))

# %%
if test():
    #   export HF_DATASETS_OFFLINE=1
    #   uv run python -m baselines.rlm --benchmark oolong --model "Qwen/Qwen3-32B" --max_workers 64
    #   uv run python -m baselines.rlm --benchmark phantomwiki --pw_size 50 --model "Qwen/Qwen3-32B" --max_workers 50
    #   uv run python -m baselines.rlm --benchmark phantomwiki --pw_size 500 --model "Qwen/Qwen3-32B" --max_workers 50
    #   uv run python -m baselines.rlm --benchmark phantomwiki --pw_size 5000 --model "Qwen/Qwen3-32B" --max_workers 50
    #   uv run python -m baselines.rlm --benchmark synthworlds --model "Qwen/Qwen3-32B" --max_workers 100
    #   uv run python -m baselines.rlm --benchmark deepresearchqa --model "Qwen/Qwen3-32B" --max_workers 70
    pass

# %%
if test():
    # model = litellm_model(
    #     model_id="hosted_vllm/Qwen/Qwen3-32B",
    #     api_base="http://localhost:8555/v1",
    #     api_key="your_secret",
    # )

    # search_agent_tool = create_web_search_agent_tool(model, run_logs_dir="./run_logs", instructions="Be concise, precise, and beautiful!")

    # search_agent_tool("What is the date today?")
    pass

# %% [markdown]
# ## End
