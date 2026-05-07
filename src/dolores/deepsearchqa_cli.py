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
# # DeepReasoner — DeepResearchQA CLI
#
# Run one DeepResearchQA task by task_id and log the agent's raw answer.
#
# Note: SERPER_API_KEY required for web search; OPENAI_API_KEY required for rescore judge.
#
# ```
# python -m dolores.deepsearchqa_cli \
#     -c configs/agents/deepsearchqa.yaml \
#     --set task_id=<id> version=deepsearchqa_0.0.1 run_id=my_run
# ```

# %%
from juplit import test

# %%
import asyncio
import json
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
from deep_reasoner.deepreasoner import Func, agent_context
from deep_reasoner.cli_base import MainConfig, make_base_tools, run_cli
from deep_reasoner.eval.view_sub_json import convert_task_dir
from deep_reasoner.logging_utils import (
    agent_log_processor,
    agent_run_log_dir,
    format_config_models_for_help,
    redact_config_for_log,
    write_initial_qa,
    write_run_metadata,
)

from benchmarks import deepresearchqa
from benchmarks.deepresearchqa.deepresearch_agent import (
    create_web_search_agent_tool,
    litellm_model,
)

logger = structlog.get_logger(__name__)

# %%
class DeepSearchQAConfig(MainConfig):
    """Config for a single DeepResearchQA agent run.

    ``benchmark`` is always ``"deepsearchqa"`` — no variable parameters.
    SERPER_API_KEY must be set for the web search agent.
    ``search_model_id`` must be in LiteLLM format, e.g. ``hosted_vllm/Qwen/Qwen3-32B``.
    """
    task_id: str
    benchmark: str = "deepsearchqa"
    search_model_id: str | None = None
    search_api_base: str | None = None
    search_api_key_env: str = "OPENAI_API_KEY"

# %%
def make_tools(cfg: DeepSearchQAConfig, client: AsyncOpenAI, run_dir: Path) -> dict:
    search_model_id = cfg.search_model_id or cfg.model
    if not search_model_id:
        raise ValueError(
            "Missing model. Set `search_model_id` (LiteLLM format, e.g. hosted_vllm/Qwen/Qwen3-32B) "
            "or `model` in YAML, or --set search_model_id=..."
        )
    search_api_key = os.getenv(cfg.search_api_key_env) or os.getenv("OPENAI_API_KEY") or ""
    lm = litellm_model(
        model_id=search_model_id,
        api_base=cfg.search_api_base,
        api_key=search_api_key,
        timeout=cfg.client.read_timeout,
    )
    search_agent = create_web_search_agent_tool(lm, run_logs_dir=str(run_dir))
    return {
        **make_base_tools(cfg, client),
        'search': Func(
            search_agent,
            description=(
                "Web search: search(query) -> str. "
                "Performs a Serper web search and returns a summary of the top results. "
                "Use to retrieve factual information from the web. "
                "Call multiple times with different queries to follow multi-hop chains."
            ),
        ),
    }

# %%
async def _main_async(cfg: DeepSearchQAConfig) -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError(
            "DeepResearchQA web search agent requires SERPER_API_KEY in your environment."
        )

    question = deepresearchqa.get_task(cfg.task_id)
    cfg.task = question
    try:
        expected_answer = deepresearchqa.get_answer(cfg.task_id)
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
        agent = make_agent(cfg, make_tools(cfg, client, run_dir), client)

        with bound_contextvars(benchmark=cfg.benchmark, task_id=cfg.task_id, version=cfg.version):
            agent_context.reset()
            with checkLogs(namespace="deep_reasoner", level=logging.INFO):
                result = await agent._run_async(cfg.task, vars={})
                f1 = em = score_error = score_reasoning = None
                try:
                    _, score_val, score_reasoning = deepresearchqa.score_judge(cfg.task_id, str(result))
                    f1 = float(score_val)
                    em = float(score_val)
                except Exception as exc:
                    score_error = str(exc)
                logger.info(
                    "agent.result",
                    question=cfg.task, answer=result, task_id=cfg.task_id,
                    expected_answer=expected_answer, f1=f1, em=em,
                    score_error=score_error, score_reasoning=score_reasoning,
                )
        print(f"Answer:\n{result}")

        written = convert_task_dir(run_dir)
        if written:
            logger.info("sub_json.converted", count=len(written), run_dir=str(run_dir))
    finally:
        await client.close()


def main(cfg: DeepSearchQAConfig) -> None:
    asyncio.run(_main_async(cfg))

# %%
app = App(
    help="Run a DeepReasoner agent on a single DeepResearchQA task.",
    help_epilogue=format_config_models_for_help((DeepSearchQAConfig,)),
    help_format="markdown",
)


@app.default
def cli(
    main_config: Annotated[Path, Parameter(help="Main YAML config file (supports _compose).")],
    set_: Annotated[
        list[str],
        Parameter(
            name="--set", consume_multiple=True, negative=(),
            help="`--set task_id=<id> version=deepsearchqa_0.0.1 run_id=my_run`",
        ),
    ] = [],
) -> None:
    """Load configs, apply overrides, validate, and run one DeepResearchQA task."""
    run_cli(main, DeepSearchQAConfig, main_config, set_)


async def _rescore_async(
    run_dir: Path,
    *,
    force: bool = False,
    concurrency: int = 50,
    dry_run: bool = False,
) -> None:
    qa_paths = sorted(run_dir.glob("*/qa.json"))
    if not qa_paths:
        print(f"No qa.json files found under {run_dir}")
        return

    to_score: list[tuple[Path, str, str]] = []
    skipped = 0
    for qa_path in qa_paths:
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        answer = qa.get("answer")
        if answer is None:
            skipped += 1
            continue
        if not force and qa.get("f1") is not None:
            skipped += 1
            continue
        to_score.append((qa_path, qa["task_id"], str(answer)))

    print(f"Found {len(qa_paths)} qa.json files: {len(to_score)} to rescore, {skipped} skipped")
    if dry_run or not to_score:
        return

    pairs = [(task_id, answer) for _, task_id, answer in to_score]
    results = await asyncio.to_thread(
        deepresearchqa.score_judge_batch, pairs, max_workers=concurrency
    )

    errors = 0
    for (qa_path, _, _), (_, score_val, reasoning) in zip(to_score, results):
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        try:
            qa["f1"] = float(score_val)
            qa["em"] = float(score_val)
            qa["score_reasoning"] = reasoning
            qa["score_error"] = None
        except Exception as exc:
            qa["score_error"] = str(exc)
            errors += 1
        qa_path.write_text(
            json.dumps(qa, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    print(f"Rescored {len(to_score) - errors}/{len(to_score)} tasks ({errors} errors)")


@app.command
def rescore(
    run_dir: Path,
    *,
    force: Annotated[bool, Parameter(help="Rescore even tasks that already have f1 scores.")] = False,
    concurrency: Annotated[int, Parameter(help="Max concurrent judge requests.")] = 50,
    dry_run: Annotated[bool, Parameter(help="Print what would be rescored without writing.")] = False,
) -> None:
    """Rescore qa.json files in a run directory using the LLM-as-judge scorer."""
    asyncio.run(_rescore_async(run_dir, force=force, concurrency=concurrency, dry_run=dry_run))


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()
