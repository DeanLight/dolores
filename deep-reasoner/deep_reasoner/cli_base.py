"""Template CLI entrypoint — copy this file and customise for your eval environment.

Quickstart
----------
1. Copy to your project, e.g. ``my_eval/cli.py``.
2. Register a script in ``pyproject.toml``::

       [project.scripts]
       my-agent = "my_eval.cli:main_cli"

3. Extend ``MainConfig`` with any tool-specific fields.
4. Fill in ``make_tools`` and ``make_agent``.
5. Run::

       deep-reasoner CONFIG.yaml "Your task here"
       deep-reasoner CONFIG.yaml "Your task" --set max_iter=10 --var label=demo --var-read doc=./file.txt
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import structlog
from cyclopts import App, Parameter
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from structlog.contextvars import bound_contextvars

from deep_reasoner.cli_utils import (
    ClientConfig,
    build_client,
    load_main_and_validate,
    make_agent,
)
from deep_reasoner.core import checkLogs, configure_structlog_fixture
from deep_reasoner.deepreasoner import Func, PlanExec, Var, agent_context
from deep_reasoner.llm import make_llm
from deep_reasoner.logging_utils import (
    agent_log_processor,
    agent_run_log_dir,
    format_config_models_for_help,
    redact_config_for_log,
    write_run_metadata,
)

logger = structlog.get_logger(__name__)


def _yaml_spec_to_var(spec: Any) -> Var:
    if isinstance(spec, dict) and "value" in spec:
        return Var(spec["value"], str(spec.get("description", "")))
    return Var(spec, "")


def load_initial_var_files(main_config: Path, rel_paths: list[str]) -> dict[str, Var]:
    """Load ``Var`` entries from YAML files (paths relative to *main_config*'s directory)."""
    if not rel_paths:
        return {}
    import yaml

    base = main_config.resolve().parent
    merged: dict[str, Var] = {}
    for rel in rel_paths:
        path = Path(rel)
        full = path if path.is_absolute() else (base / path)
        data = yaml.safe_load(full.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"initial_var_files entry {rel!r}: expected a mapping at top level")
        for name, spec in data.items():
            merged[str(name)] = _yaml_spec_to_var(spec)
    return merged


def merge_cli_var_assignments(literals: list[str], from_files: list[str]) -> dict[str, Var]:
    """Parse ``--var name=value`` and ``--var-read name=path`` flags."""
    out: dict[str, Var] = {}
    for raw in literals:
        key, sep, rest = raw.partition("=")
        if not sep:
            raise ValueError(f"--var expects name=value, got {raw!r}")
        out[key.strip()] = Var(rest, "")
    for raw in from_files:
        key, sep, rest = raw.partition("=")
        if not sep:
            raise ValueError(f"--var-read expects name=path, got {raw!r}")
        path = Path(rest.strip()).expanduser()
        out[key.strip()] = Var(path.read_text(encoding="utf-8"), f"file:{path.name}")
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class MainConfig(BaseModel):
    """Full agent config populated from layered YAML files.

    Add tool-specific fields here (model names, URLs, etc.); keep secrets in env vars.
    All fields can be set from YAML or overridden via ``--set key=value ...``.

    Example YAML::

        model: "meta-llama/llama-3.3-70b-instruct"
        system_prompt: "You are a helpful agent."
        initial_var_files: ["vars/demo_doc.yaml"]   # optional; paths relative to this file
    """
    model_config = ConfigDict(extra="allow")

    # LLM
    model: str | None = None
    planner_model: str | None = None
    llm_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Agent
    system_prompt: str
    max_iter: int = 30
    models: list[Any] = Field(default_factory=list)
    prompt_template_variables: dict[str, Any] = Field(default_factory=dict)

    # Logging / eval bookkeeping
    benchmark: str = "default"
    version: str = "0"
    task_id: str = "cli"
    log_dir: str = "logs"

    # HTTP client
    client: ClientConfig = Field(default_factory=ClientConfig)
    load_dotenv: bool = True

    # Namespace filter for the root plan_exec call — list of name patterns (regex).
    # Passed as namespaces= to PlanExec; None (default) means no filter, root sees all.
    root_namespaces: list[str] | None = None

    # YAML files (paths relative to this config file) whose top-level keys become REPL Vars.
    initial_var_files: list[str] = Field(default_factory=list)

    # ── Add your own fields below ──────────────────────────────────────────
    helper_model: str = ""


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def make_base_tools(cfg: MainConfig, client: AsyncOpenAI) -> dict[str, Func]:
    """Return the three tools every benchmark CLI injects: Var, Func, llm."""
    return {
        'Var': Func(
            Var,
            description="A wrapper for a variable with a value and a description. "
            "Used to pass variables to nested plan_exec calls.",
        ),
        'Func': Func(
            Func,
            description="A wrapper for a function with a value and a description. "
            "Used to pass functions to nested plan_exec calls.",
        ),
        'llm': Func(
            make_llm(client=client, model=cfg.model),
            description="Single LLM call: llm(prompt) -> str. "
            "Concurrent batch: llm.batch([prompt1, prompt2, ...]) -> list[str]. "
            "Use for language tasks on data you already hold "
            "(summarise, extract, classify, rewrite). "
            "Embed the data directly in the prompt string.",
        ),
    }


def make_tools(cfg: MainConfig, client: AsyncOpenAI) -> dict[str, Func]:
    """Build the tools injected into the agent REPL (template — extend per benchmark)."""
    return make_base_tools(cfg, client)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main_async(cfg: MainConfig, task: str, initial_vars: dict[str, Var]) -> None:
    if not (task or "").strip():
        raise ValueError("Missing task (second CLI argument).")

    cfg_for_log = redact_config_for_log(cfg.model_dump(mode="json"))
    run_dir = agent_run_log_dir(cfg.log_dir, cfg.task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(run_dir, config=cfg_for_log)
    logger.info("Writing run logs", run_dir=run_dir.resolve())

    client = build_client(cfg.client)
    try:
        tools = make_tools(cfg, client)
        agent = make_agent(cfg, tools, client)

        with bound_contextvars(
            benchmark=cfg.benchmark, task_id=cfg.task_id, version=cfg.version,
        ):
            agent_context.reset()
            with checkLogs(namespace="deep_reasoner", level=logging.INFO):
                result = PlanExec(agent)(task, namespaces=cfg.root_namespaces, **initial_vars)
                logger.info("agent.result", question=task, answer=result)

        print(f"Answer:\n{result}")
    finally:
        await client.close()


def main(cfg: MainConfig, task: str, initial_vars: dict[str, Var]) -> None:
    asyncio.run(_main_async(cfg, task, initial_vars))


# ---------------------------------------------------------------------------
# Shared CLI runner
# ---------------------------------------------------------------------------

def run_cli(
    main_fn,
    ConfigClass,
    main_config: Path,
    set_args,
    *,
    task: str,
    var_literals: list[str],
    var_reads: list[str],
) -> None:
    """Load main config, merge initial REPL vars, validate, run."""
    try:
        cfg = load_main_and_validate(ConfigClass, main_config, set_args=list(set_args))
        if cfg.load_dotenv:
            load_dotenv()
        configure_structlog_fixture(
            console=True,
            extra_processors=[agent_log_processor(cfg.log_dir)],
            default_level=logging.INFO,
        )
        file_vars = load_initial_var_files(main_config, cfg.initial_var_files)
        cli_vars = merge_cli_var_assignments(var_literals, var_reads)
        initial_vars = {**file_vars, **cli_vars}
        main_fn(cfg, task, initial_vars)
    except SystemExit:
        raise
    except Exception:
        logger.exception("cli.error")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = App(
    help="Run a DeepReasoner agent from one main YAML config.",
    help_epilogue=format_config_models_for_help((ClientConfig, MainConfig)),
    help_format="markdown",
)


@app.default
def cli(
    main_config: Annotated[
        Path,
        Parameter(help="Main YAML config file (supports _compose)."),
    ],
    task: Annotated[
        str,
        Parameter(help="Task or question for the agent (second positional argument)."),
    ],
    set_: Annotated[
        list[str],
        Parameter(
            name="--set",
            consume_multiple=True,
            negative=(),
            help=(
                "One or more key=value overrides after a single flag, e.g. "
                "`--set max_iter=5 client.base_url=http://...`"
            ),
        ),
    ] = [],
    var_: Annotated[
        list[str],
        Parameter(
            name="--var",
            consume_multiple=True,
            negative=(),
            help='Inject a REPL Var: --var name=value (repeatable).',
        ),
    ] = [],
    var_read: Annotated[
        list[str],
        Parameter(
            name="--var-read",
            consume_multiple=True,
            negative=(),
            help='Inject a REPL Var from file text: --var-read name=/path/to/file (repeatable).',
        ),
    ] = [],
) -> None:
    """Load main config, apply overrides, validate, and run the agent."""
    run_cli(main, MainConfig, main_config, set_, task=task, var_literals=var_, var_reads=var_read)


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()
