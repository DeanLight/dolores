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
# # RLM agent
#
# Recursive Language Model baseline. `cfg.max_steps` maps to RLM's
# `max_iterations`; `max_depth` is an agent-specific extra (default 10).

# %%
from juplit import test

# %%
import json
from pathlib import Path
from typing import Any

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.agents._helpers import _conn
from dolores.unified.benchmarks import Task
from prompts import SYNTHWORLDS_RLM_INSTRUCTIONS, PHANTOM_WIKI_RLM_INSTRUCTIONS
from dolores.unified.token_matched import ContextWindowExceeded

# Returned in place of a completion whose prompt overshot the context window. RLM
# fans many calls out through ``asyncio.gather`` with no ``return_exceptions``, so a
# ``ContextWindowExceeded`` (a BaseException) escaping a gathered task orphans its
# siblings and hangs the whole run. Absorbing the overshoot per-call keeps the
# gather intact: the oversized call contributes this marker and RLM keeps going.
_CONTEXT_OVERFLOW_NOTE = "[skipped: this input exceeded the model's context window]"

# %% [markdown]
# ## OpenAI client monkeypatch — extra_body passthrough
#
# rlm's OpenAIClient.completion / acompletion build extra_body locally and don't
# expose a hook for user-supplied chat_template_kwargs. We need that hook to
# disable Qwen3 thinking on the top RLM via vLLM's chat_template_kwargs. The
# patched methods mirror the upstream body verbatim plus one merge step that
# pulls self.kwargs.get("extra_body") into the request. Default behavior is
# byte-identical when extra_body is not passed. Pinned against rlms==0.1.1.
#
# Applied lazily (first ``_answer``) so importing this module doesn't require the
# ``rlm`` package to be installed.

# %%
_rlm_patched = False


def _ensure_rlm_patched() -> None:
    global _rlm_patched
    if _rlm_patched:
        return
    import rlm.clients.openai as _rlm_openai

    prime_base = _rlm_openai.DEFAULT_PRIME_INTELLECT_BASE_URL

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
        if self.client.base_url == prime_base:
            extra_body["usage"] = {"include": True}

        user_extra = self.kwargs.get("extra_body")
        if user_extra:
            extra_body = {**extra_body, **user_extra}

        try:
            response = self.client.chat.completions.create(
                model=model, messages=messages, extra_body=extra_body
            )
        except ContextWindowExceeded:
            return _CONTEXT_OVERFLOW_NOTE
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
        if self.client.base_url == prime_base:
            extra_body["usage"] = {"include": True}

        user_extra = self.kwargs.get("extra_body")
        if user_extra:
            extra_body = {**extra_body, **user_extra}

        try:
            response = await self.async_client.chat.completions.create(
                model=model, messages=messages, extra_body=extra_body
            )
        except ContextWindowExceeded:
            return _CONTEXT_OVERFLOW_NOTE
        self._track_cost(response, model)
        return response.choices[0].message.content

    _rlm_openai.OpenAIClient.completion = _patched_completion
    _rlm_openai.OpenAIClient.acompletion = _patched_acompletion
    _rlm_patched = True


# %%
def _unquote(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


# %%
@register_agent
class RlmAgent(Agent):
    name = "rlm"

    def _answer(self, task: Task, log_dir: Path) -> str:
        _ensure_rlm_patched()
        from rlm import RLM
        from rlm.logger import RLMLogger
        from rlm.utils.prompts import RLM_SYSTEM_PROMPT
        from dolores.unified.benchmarks.deepresearch_agent import (
            litellm_model, create_web_search_agent_tool,
        )

        cfg = self.cfg
        base, key = _conn(cfg)

        backend_kwargs: dict = {"model_name": cfg.model, "api_key": key, "base_url": base}
        if cfg.no_thinking:
            backend_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        bname = cfg.benchmark.benchmark_name()
        if bname == "phantomwiki":
            bench_instructions = PHANTOM_WIKI_RLM_INSTRUCTIONS
            custom_tools = {
                "retrieve_article": {"tool": task.tools["retrieve_article"],
                                     "description": task.tools["retrieve_article"].__doc__},
                "search": {"tool": task.tools["search"],
                           "description": task.tools["search"].__doc__},
            }
        elif bname == "synthworlds":
            bench_instructions = SYNTHWORLDS_RLM_INSTRUCTIONS
            custom_tools = {
                "retrieve_top_5": {"tool": task.tools["retrieve_top_5"],
                                   "description": task.tools["retrieve_top_5"].__doc__},
            }
        elif bname == "deepresearchqa":
            lm = litellm_model(model_id=f"hosted_vllm/{cfg.model}", api_base=base, api_key=key)
            search_agent = create_web_search_agent_tool(lm, run_logs_dir=str(log_dir))
            bench_instructions = None
            custom_tools = {"search_agent": {"tool": search_agent,
                                             "description": search_agent.__doc__}}
        else:
            bench_instructions = None
            custom_tools = {}

        # A config supplies the *benchmark* half only; rlm's own system prompt still
        # leads, exactly as it does without a config. No instructions at all means
        # rlm runs on its default prompt, which is what oolong/deepresearchqa do.
        bench_instructions = self._instructions(bench_instructions)
        custom_system_prompt = (
            RLM_SYSTEM_PROMPT + "\n\n" + bench_instructions if bench_instructions else None
        )

        rlm_kwargs: dict = dict(
            backend="vllm",
            backend_kwargs=backend_kwargs,
            environment="local",
            max_depth=cfg.get("max_depth", 10),
            max_iterations=cfg.max_steps,
            logger=RLMLogger(),
        )
        if custom_system_prompt is not None:
            rlm_kwargs["custom_system_prompt"] = custom_system_prompt
        if custom_tools:
            rlm_kwargs["custom_tools"] = custom_tools

        rlm = RLM(**rlm_kwargs)
        prompt = task.vars.get("document", "")
        result = rlm.completion(prompt=prompt, root_prompt=task.question)
        # The full completion record (per-thread usage, trajectories) for
        # analysis/rlm_analysis.py, as the paper's RLM runs saved it.
        (log_dir / "result.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str)
        )
        return _unquote(result.response)

    def _parsed(self, answer: Any) -> Any:
        if self.cfg.benchmark.benchmark_name() == "phantomwiki":
            return [_unquote(s) for s in str(answer).split(",") if s.strip()]
        return answer


# %% [markdown]
# ## Tests


# %%
def test_unquote():
    assert _unquote('  "hi" ') == "hi"
    assert _unquote("'a'") == "a"
    assert _unquote("plain") == "plain"


# %%
if test():
    test_unquote()
    print("agents.rlm tests passed")

# %% [markdown]
# ## End
