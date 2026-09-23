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
# # CodeAct agent
#
# smolagents `CodeAgent`. A long `task.vars["document"]` is exposed as a
# `get_document()` tool; the deepresearchqa path uses a web-search sub-agent.

# %%
from juplit import test

# %%
from pathlib import Path

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.agents._helpers import _conn, _openai_model, _smolagent_tools
from dolores.unified.benchmarks import Task
import prompts


# %%
@register_agent
class CodeactAgent(Agent):
    name = "codeact"

    def _answer(self, task: Task, log_dir: Path) -> str:
        from smolagents import CodeAgent, tool, LogLevel
        from dolores.unified.benchmarks.deepresearch_agent import (
            litellm_model, create_web_search_agent_tool,
        )

        cfg = self.cfg
        base, key = _conn(cfg)
        sm_model = _openai_model(cfg)

        bname = cfg.benchmark.benchmark_name()
        smolagent_tools = _smolagent_tools(task)

        # Inject document as a tool for benchmarks that provide a long document.
        if task.vars.get("document"):
            doc_text = task.vars["document"]

            @tool
            def get_document() -> str:
                """Return the full document text for this task. The document may be very long."""
                return doc_text

            smolagent_tools.append(get_document)

        if bname == "phantomwiki":
            instructions = prompts.PHANTOM_WIKI_CODEACT_INSTRUCTIONS
            executor_kwargs = {}
        elif bname == "synthworlds":
            instructions = prompts.SYNTHWORLDS_CODEACT_INSTRUCTIONS
            executor_kwargs = {}
        elif bname == "deepresearchqa":
            instructions = None
            executor_kwargs = {"timeout_seconds": None}
            lm = litellm_model(
                model_id=f"hosted_vllm/{cfg.model}", api_base=base, api_key=key,
            )
            search_agent_fn = create_web_search_agent_tool(lm, run_logs_dir=str(log_dir))
            smolagent_tools = [tool(search_agent_fn)]
        else:
            instructions = None
            executor_kwargs = {}

        agent = CodeAgent(
            tools=smolagent_tools,
            model=sm_model,
            max_steps=cfg.max_steps,
            instructions=self._instructions(instructions),
            verbosity_level=LogLevel.ERROR,
            executor_kwargs=executor_kwargs,
        )
        return str(agent.run(task.question)).strip()


# %%
if test():
    # Live smoke (needs a served model):
    #   uv run python -m dolores.unified run --config configs/debug_helloworld.yaml --task-id fib_7
    pass

# %% [markdown]
# ## End
