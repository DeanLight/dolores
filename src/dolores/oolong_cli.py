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
# # DeepReasoner — Oolong CLI
#
# Run one Oolong task by task_id and log the agent's raw answer.
#
# ```
# python -m dolores.oolong_cli \
#     -c model-configs/generic.yaml -c configs/agents/oolong.yaml \
#     --set limit=500 seed=42 task_id=abc123 version=0.0.1 run_id=exp1
# ```

# %%
from juplit import test

# %%
import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated

import openai
import structlog
from cyclopts import App, Parameter
from openai import AsyncOpenAI
from pydantic import model_validator
from structlog.contextvars import bound_contextvars

from deep_reasoner.cli_utils import build_client, load_and_validate, make_agent
from deep_reasoner.core import checkLogs, configure_structlog_fixture
from deep_reasoner.deepreasoner import PlanExec, Var, agent_context
from deep_reasoner.cli_base import MainConfig, make_base_tools, run_cli
from deep_reasoner.logging_utils import (
    agent_log_processor,
    agent_run_log_dir,
    format_config_models_for_help,
    redact_config_for_log,
    write_initial_qa,
    write_run_metadata,
)

from benchmarks import oolong

logger = structlog.get_logger(__name__)

# %%
class OolongConfig(MainConfig):
    """Config for a single Oolong agent run.

    ``task`` is populated at runtime — do not set it in YAML.
    ``benchmark`` is auto-derived as ``oolong_{limit}_{seed}``.
    """
    task_id: str
    limit: int = 500
    seed: int = 42
    parse_api_key_env: str = "OPENAI_API_KEY"

    @model_validator(mode='after')
    def _set_benchmark(self) -> 'OolongConfig':
        self.benchmark = f"oolong_{self.limit}_{self.seed}"
        return self

# %%
async def _check_openai_connectivity(api_key: str) -> None:
    client = openai.AsyncOpenAI(api_key=api_key)
    try:
        await client.models.list()
    except openai.AuthenticationError as exc:
        raise ValueError(f"OpenAI API key is invalid or unauthorized: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Cannot reach OpenAI API (needed for oolong scoring): {exc}") from exc
    finally:
        await client.close()


async def _main_async(cfg: OolongConfig) -> None:
    parse_api_key = os.getenv(cfg.parse_api_key_env) or os.getenv("OPENAI_API_KEY")
    if not parse_api_key:
        raise ValueError(
            f"Oolong scoring requires an OpenAI API key. "
            f"Set {cfg.parse_api_key_env} (or OPENAI_API_KEY) in your environment."
        )
    await _check_openai_connectivity(parse_api_key)

    document, question = oolong.get_task(cfg.task_id)
    cfg.task = question
    try:
        expected_answer = oolong.get_answer(cfg.task_id)
    except Exception:
        expected_answer = None

    cfg_for_log = redact_config_for_log(cfg.model_dump(mode="json"))
    run_dir = agent_run_log_dir(cfg.log_dir, cfg.task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, config=cfg_for_log)
    write_initial_qa(run_dir, task_id=cfg.task_id, question=question, expected_answer=expected_answer)
    logger.info("Writing run logs", run_dir=run_dir.resolve())

    client = build_client(cfg.client)
    try:
        agent = make_agent(cfg, make_base_tools(cfg, client), client)

        with bound_contextvars(benchmark=cfg.benchmark, task_id=cfg.task_id, version=cfg.version):
            agent_context.reset()
            with checkLogs(namespace="deep_reasoner", level=logging.INFO):
                result = await PlanExec(agent).call_async(
                    cfg.task, namespaces=cfg.root_namespaces,
                    document=Var(document, "DnD game text"),
                )
                parsed_output = f1 = em = score_error = None
                try:
                    parsed_output = oolong.parse(cfg.task_id, result, api_key=parse_api_key)
                    score_val = oolong.score(cfg.task_id, parsed_output)
                    f1 = score_val
                    em = float(score_val == 1.0)
                except Exception as exc:
                    score_error = str(exc)
                logger.info(
                    "agent.result",
                    question=cfg.task, answer=result, task_id=cfg.task_id,
                    parsed_output=parsed_output, expected_answer=expected_answer,
                    f1=f1, em=em, score_error=score_error,
                )
        print(f"Answer:\n{result}")
    finally:
        await client.close()


def main(cfg: OolongConfig) -> None:
    asyncio.run(_main_async(cfg))

# %%
app = App(
    help="Run a DeepReasoner agent on a single Oolong task.",
    help_epilogue=format_config_models_for_help((OolongConfig,)),
    help_format="markdown",
)


@app.default
def cli(
    main_config: Annotated[Path, Parameter(help="Main YAML config file (supports _compose).")],
    set_: Annotated[
        list[str],
        Parameter(
            name="--set", consume_multiple=True, negative=(),
            help="`--set limit=500 seed=42 task_id=abc123 version=0.0.1 run_id=exp1`",
        ),
    ] = [],
) -> None:
    """Load main config, apply overrides, validate, and run one Oolong task."""
    run_cli(main, OolongConfig, main_config, set_)


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()
