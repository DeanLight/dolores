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
# # DeepReasoner — SynthWorlds CLI
#
# Run one SynthWorlds task by task_id and log the agent's raw answer.
#
# Note: OPENAI_API_KEY is required for the dense retriever (text-embedding-3-small).
#
# ```
# python -m dolores.synthworld_cli \
#     -c configs/agents/synthworld.yaml \
#     --set task_id=<id> version=synthworld_0.0.1 run_id=my_run
# ```

# %%
from juplit import test

# %%
import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated

import structlog
from cyclopts import App, Parameter
from openai import AsyncOpenAI
from structlog.contextvars import bound_contextvars

from deep_reasoner.cli_utils import build_client, load_and_validate, make_agent
from deep_reasoner.core import checkLogs, configure_structlog_fixture
from deep_reasoner.deepreasoner import Func, PlanExec, agent_context
from deep_reasoner.cli_base import MainConfig, make_base_tools, run_cli
from deep_reasoner.logging_utils import (
    agent_log_processor,
    agent_run_log_dir,
    format_config_models_for_help,
    redact_config_for_log,
    write_initial_qa,
    write_run_metadata,
)

from benchmarks import synthworlds

logger = structlog.get_logger(__name__)

# %%
class SynthworldConfig(MainConfig):
    """Config for a single SynthWorlds agent run.

    ``benchmark`` is always ``"synthworld"`` — no variable parameters.
    OPENAI_API_KEY must be set for the dense retriever (text-embedding-3-small).
    """
    task_id: str
    benchmark: str = "synthworld"

# %%
def make_tools(cfg: SynthworldConfig, client: AsyncOpenAI, openai_api_key: str) -> dict:
    return {
        **make_base_tools(cfg, client),
        'search': Func(synthworlds.create_retriever_tool(openai_api_key)),
    }

# %%
async def _main_async(cfg: SynthworldConfig) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "SynthWorlds retriever requires OPENAI_API_KEY in your environment "
            "(used for text-embedding-3-small)."
        )

    question = synthworlds.get_task(cfg.task_id)
    cfg.task = question
    try:
        expected_answer = synthworlds.get_answer(cfg.task_id)
    except Exception:
        expected_answer = None

    cfg_for_log = redact_config_for_log(cfg.model_dump(mode="json"))
    run_dir = agent_run_log_dir(cfg.log_dir, cfg.task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, config=cfg_for_log)
    write_initial_qa(run_dir, task_id=cfg.task_id, question=question, expected_answer=expected_answer)
    logger.info("Writing run logs", run_dir=run_dir.resolve())

    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = build_client(cfg.client)
    try:
        agent = make_agent(cfg, make_tools(cfg, client, openai_api_key), client)

        with bound_contextvars(benchmark=cfg.benchmark, task_id=cfg.task_id, version=cfg.version):
            agent_context.reset()
            with checkLogs(namespace="deep_reasoner", level=logging.INFO):
                result = await PlanExec(agent).call_async(cfg.task, namespaces=cfg.root_namespaces)
                f1 = em = score_error = None
                try:
                    f1, em = synthworlds.score(cfg.task_id, str(result))
                except Exception as exc:
                    score_error = str(exc)
                logger.info(
                    "agent.result",
                    question=cfg.task, answer=result, task_id=cfg.task_id,
                    expected_answer=expected_answer, f1=f1, em=em, score_error=score_error,
                )
        print(f"Answer:\n{result}")
    finally:
        await client.close()


def main(cfg: SynthworldConfig) -> None:
    asyncio.run(_main_async(cfg))

# %%
app = App(
    help="Run a DeepReasoner agent on a single SynthWorlds task.",
    help_epilogue=format_config_models_for_help((SynthworldConfig,)),
    help_format="markdown",
)


@app.default
def cli(
    main_config: Annotated[Path, Parameter(help="Main YAML config file (supports _compose).")],
    set_: Annotated[
        list[str],
        Parameter(
            name="--set", consume_multiple=True, negative=(),
            help="`--set task_id=<id> version=synthworld_0.0.1 run_id=my_run`",
        ),
    ] = [],
) -> None:
    """Load configs, apply overrides, validate, and run one SynthWorlds task."""
    run_cli(main, SynthworldConfig, main_config, set_)


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()
