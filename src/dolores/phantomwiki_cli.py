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
# # DeepReasoner — PhantomWiki CLI
#
# Run one PhantomWiki task by task_id and log the agent's raw answer.
#
# ```
# python -m dolores.phantomwiki_cli \
#     -c model-configs/generic.yaml -c configs/agents/phantomwiki.yaml \
#     --set size=50 seed=1 task_id=abc123 version=0.0.3 run_id=exp1
# ```

# %%
from juplit import test

# %%
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any

import structlog
from cyclopts import App, Parameter
from openai import AsyncOpenAI
from pydantic import model_validator
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

from benchmarks import phantomwiki

logger = structlog.get_logger(__name__)

# %%
def _parse_phantom_output(raw_output: Any) -> list[str]:
    """Best-effort parse of model output into list[str] for phantomwiki.score."""
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

# %%
class PhantomWikiConfig(MainConfig):
    """Config for a single PhantomWiki agent run.

    ``task`` is populated at runtime — do not set it in YAML.
    ``benchmark`` is auto-derived as ``phantomwiki_{size}_{seed}``.
    """
    task_id: str
    size: int = 50
    seed: int = 1

    @model_validator(mode='after')
    def _set_benchmark(self) -> 'PhantomWikiConfig':
        self.benchmark = f"phantomwiki_{self.size}_{self.seed}"
        return self

# %%
def make_tools(cfg: PhantomWikiConfig, client: AsyncOpenAI, retrieve_article_fn: Any, search_fn: Any) -> dict:
    return {
        **make_base_tools(cfg, client),
        'retrieve_article': Func(retrieve_article_fn),
        'search': Func(search_fn),
    }

# %%
async def _main_async(cfg: PhantomWikiConfig) -> None:
    question, retrieve_article_fn, search_fn = phantomwiki.get_task(cfg.size, cfg.seed, cfg.task_id)
    cfg.task = question
    try:
        expected_answer = phantomwiki.get_answer(cfg.task_id, cfg.size, cfg.seed)
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
        agent = make_agent(cfg, make_tools(cfg, client, retrieve_article_fn, search_fn), client)

        with bound_contextvars(benchmark=cfg.benchmark, task_id=cfg.task_id, version=cfg.version):
            agent_context.reset()
            with checkLogs(namespace="deep_reasoner", level=logging.INFO):
                result = await PlanExec(agent).call_async(cfg.task, namespaces=cfg.root_namespaces)
                parsed_output = _parse_phantom_output(result)
                f1 = em = score_error = None
                try:
                    f1, em = phantomwiki.score(cfg.size, cfg.seed, cfg.task_id, parsed_output)
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


def main(cfg: PhantomWikiConfig) -> None:
    asyncio.run(_main_async(cfg))

# %%
app = App(
    help="Run a DeepReasoner agent on a single PhantomWiki task.",
    help_epilogue=format_config_models_for_help((PhantomWikiConfig,)),
    help_format="markdown",
)


@app.default
def cli(
    main_config: Annotated[Path, Parameter(help="Main YAML config file (supports _compose).")],
    set_: Annotated[
        list[str],
        Parameter(
            name="--set", consume_multiple=True, negative=(),
            help="`--set size=50 seed=1 task_id=abc123 version=0.0.3 run_id=exp1`",
        ),
    ] = [],
) -> None:
    """Load main config, apply overrides, validate, and run one PhantomWiki task."""
    run_cli(main, PhantomWikiConfig, main_config, set_)


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()
