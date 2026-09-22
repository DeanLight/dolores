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
# # Deep Research agent
#
# smolagents manager `CodeAgent` orchestrating a web-search (or custom-tool)
# sub-agent.

# %%
from juplit import test

# %%
from pathlib import Path

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.benchmarks import Task
from prompts import (
    SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS,
    SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS,
    PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS,
    PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS,
)


# %%
def _make_manager_agent(model, run_logs_dir: str, *, max_steps: int, tools: list | None = None,
                        subagent_instr: str = None, orch_instructions: str = None,
                        manager_model=None):
    """Build a manager CodeAgent with either a web-search or custom-tool subagent."""
    from smolagents import CodeAgent, LogLevel
    from dolores.unified.benchmarks.deepresearch_agent import (
        _create_web_search_agent, _create_search_agent, visualizer, TextInspectorTool,
    )

    text_limit = 100000
    if tools is None:
        search_agent = _create_web_search_agent(model, run_logs_dir=run_logs_dir,
                                                instructions=subagent_instr)
        manager_tools = [visualizer, TextInspectorTool(model, text_limit)]
    else:
        search_agent = _create_search_agent(model, tools=tools, run_logs_dir=run_logs_dir,
                                            instructions=subagent_instr)
        manager_tools = []

    return CodeAgent(
        model=manager_model if manager_model is not None else model,
        tools=manager_tools,
        max_steps=max_steps,
        verbosity_level=LogLevel.ERROR,
        instructions=orch_instructions,
        executor_kwargs={"timeout_seconds": None},
        additional_authorized_imports=["*"],
        planning_interval=4,
        managed_agents=[search_agent],
    )


# %%
@register_agent
class DeepresearchAgent(Agent):
    name = "deepresearch"

    def _answer(self, task: Task, log_dir: Path) -> str:
        from dolores.unified.benchmarks.deepresearch_agent import litellm_model

        cfg = self.cfg
        base, key = cfg.inference.client_kwargs().get("base_url"), cfg.inference.client_kwargs().get("api_key")

        model = litellm_model(model_id=f"hosted_vllm/{cfg.model}", api_base=base, api_key=key)
        manager_model = None
        if cfg.no_thinking:
            manager_model = litellm_model(
                model_id=f"hosted_vllm/{cfg.model}", api_base=base, api_key=key,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )

        bname = cfg.benchmark.benchmark_name()
        if bname == "phantomwiki":
            tools = list(task.tools.values())
            subagent_instr = PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS
            orch_instructions = PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS
        elif bname == "synthworlds":
            tools = list(task.tools.values())
            subagent_instr = SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS
            orch_instructions = SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS
        else:
            tools = None
            subagent_instr = None
            orch_instructions = None

        # Two prompts here, and the substantive one is the sub-agent's: it is the
        # ReAct translation. A config can override either.
        agent = _make_manager_agent(
            model, run_logs_dir=str(log_dir), max_steps=cfg.max_steps, tools=tools,
            subagent_instr=cfg.get("subagent_instructions") or subagent_instr,
            orch_instructions=self._instructions(orch_instructions),
            manager_model=manager_model,
        )
        return str(agent.run(task.question)).strip()


# %%
if test():
    # Live smoke (needs a served model):
    #   uv run python -m dolores.unified.cli run --config configs/debug_deepsearchqa.yaml --task-id <id>
    pass

# %% [markdown]
# ## End
