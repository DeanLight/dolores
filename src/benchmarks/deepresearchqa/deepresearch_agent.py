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
# # Deep Research Agent
#
# [Deep Research](https://huggingface.co/blog/open-deep-research) from smolagents.
# The repo is [here](https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research).

# %%
from juplit import test

# %%
import copy
import tempfile

from .open_deep_research.scripts.text_inspector_tool import TextInspectorTool
from .open_deep_research.scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from .open_deep_research.scripts.visual_qa import visualizer

from smolagents import (
    CodeAgent,
    GoogleSearchTool,
    LiteLLMModel,
    ToolCallingAgent,
    LogLevel,
    tool,
)

import json
import uuid
from pathlib import Path

_custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
)

# Defaults for ``SimpleTextBrowser``. ``downloads_folder`` is intentionally
# omitted — it gets populated per-call inside ``create_web_search_agent`` with
# a fresh ``tempfile.mkdtemp()`` so parallel agents never collide.
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "request_kwargs": {
        "headers": {"User-Agent": _user_agent},
        "timeout": 300,
    },
}

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
class _RunLoggerMixin:
    """Mixin: after each ``run()``, write the run's logs to disk.

    Each run produces one JSON file under ``run_logs_dir`` named
    ``{counter}_sub_{uuid6}.json``, where ``counter`` is a per-instance
    1-indexed sequence and ``uuid6`` is a short random suffix (collision
    guard if two instances ever share a folder). The payload shape is::

        {"answer": <agent.run result>, **create_logs(self)}

    Captures *after* ``super().run()`` returns — memory/monitor are still
    populated at that point; the next ``run(reset=True)`` is what wipes them.
    """
    def __init__(self, *args, run_logs_dir: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._run_logs_dir = Path(run_logs_dir)
        self._run_logs_dir.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)

        self._run_counter += 1
        payload = {"answer": str(result), **create_logs(self)}
        fname = f"{self._run_counter}_sub_{uuid.uuid4().hex[:6]}.json"
        (self._run_logs_dir / fname).write_text(
            json.dumps(payload, indent=2, default=str)
        )
        return result


class LoggingToolCallingAgent(_RunLoggerMixin, ToolCallingAgent):
    pass

# %%
def litellm_model(**kwargs):
    """Build the ``LiteLLMModel`` shared by the manager and search subagent.

    All LiteLLM kwargs (``model_id``, ``api_base``, ``api_key``, ...) go
    through ``**kwargs`` — useful for pointing at a local vLLM server.

    Example (vLLM):

        model = litellm_model(
            model_id="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY",
        )
    """
    return LiteLLMModel(
        custom_role_conversions=_custom_role_conversions,
        max_completion_tokens=8192,
        **kwargs,
    )

# %%
def _create_web_search_agent(model, run_logs_dir: str, instructions: str = None):
    """Web-browsing search subagent. Per-call ``tempfile.mkdtemp()`` for the
    browser's binary cache avoids TOCTOU collisions across parallel agents."""

    browser_config = copy.deepcopy(BROWSER_CONFIG)
    browser_config["downloads_folder"] = tempfile.mkdtemp(prefix="downloads_")
    browser = SimpleTextBrowser(**browser_config)

    text_limit = 100000
    WEB_TOOLS = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
        TextInspectorTool(model, text_limit),
    ]
    search_agent = LoggingToolCallingAgent(
        model=model,
        tools=WEB_TOOLS,
        max_steps=20,
        verbosity_level=LogLevel.ERROR,
        instructions=instructions,
        planning_interval=4,
        name="search_agent",
        description="""A team member that will search the internet to answer your question.
    Ask him for all your questions that require browsing the web.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two webpages.
    Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.
    """,
        provide_run_summary=True,
        run_logs_dir=run_logs_dir,
    )
    search_agent.prompt_templates["managed_agent"]["task"] += """You can navigate to .txt online files.
    If a non-html page is in another format, especially .pdf or a Youtube video, use tool 'inspect_file_as_text' to inspect it.
    Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    return search_agent


def create_web_search_agent_tool(model, run_logs_dir: str, instructions: str = None):

    agent = _create_web_search_agent(model, run_logs_dir=run_logs_dir, instructions=instructions)

    def search_agent(query: str) -> str:
        """A team member that will search the internet to answer your question. Ask him for all your questions that require browsing the web. Provide him as much context as possible, in particular if you need to search on a specific timeframe! And don't hesitate to provide him with a complex search task, like finding a difference between two webpages. Your request must be a real sentence, not a google search! Like "Find me this information (...)" rather than a few keywords.

        Args:
            query: The search query to be answered by the web search agent.
        Returns:
            The string answer returned by the web search agent after searching the web.
        """
        return agent.run(query)

    return search_agent

# %%
def _create_search_agent(
    model,
    tools: list,
    run_logs_dir: str,
    instructions: str = None
):
    """Like ``create_web_search_agent`` but with caller-supplied ``tools``
    (plain Python functions — wrapped here with ``@tool``)."""
    search_agent = LoggingToolCallingAgent(
        model=model,
        tools=[tool(t) for t in tools],
        max_steps=20,
        verbosity_level=LogLevel.ERROR,
        instructions=instructions,
        planning_interval=4,
        name="search_agent",
        description=(
            "A team member that will search for information to answer your question. "
            "Ask him for any question that requires digging through the available sources. "
            "Provide him as much context as possible. "
            "Your request must be a real sentence, not a keyword query — like "
            '"Find me this information (...)" rather than a few keywords.'
        ),
        provide_run_summary=True,
        run_logs_dir=run_logs_dir,
    )
    search_agent.prompt_templates["managed_agent"]["task"] += """If after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""
    return search_agent


def create_search_agent_tool(model, tools: list, run_logs_dir: str, instructions: str = None):

    agent = _create_search_agent(model, tools=tools, run_logs_dir=run_logs_dir, instructions=instructions)

    def search_agent(query: str) -> str:
        """A team member that will search for information using the available tools to answer your question. Provide as much context as possible. Your request must be a real sentence, not a keyword query — like "Find me this information (...)" rather than a few keywords.

        Args:
            query: The search query to be answered by the search agent.
        Returns:
            The string answer returned by the search agent.
        """
        return agent.run(query)

    return search_agent
