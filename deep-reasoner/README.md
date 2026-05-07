# deep-reasoner

Core library for **DeepReasoner**: an iterative CodeAct-style agent with a Python REPL loop, `plan_exec` sub-agents, Jinja2-templated system prompts, and OpenAI-compatible chat backends.

Benchmark CLIs and paper configs may live in a separate repo; this package is the runtime those workflows depend on.

## Requirements

- Python **3.12**
- An OpenAI-compatible HTTP API (hosted or local vLLM)

## Install

From a checkout (editable):

```bash
git clone https://github.com/SafeDesign-ai/symbolic-decomposer.git
cd symbolic-decomposer
uv sync
# or: uv pip install -e .
```

Install a specific branch/revision from Git without cloning:

```bash
uv add "deep-reasoner @ git+https://github.com/SafeDesign-ai/symbolic-decomposer.git@branch-or-sha"
```

Traditional pip:

```bash
pip install "deep-reasoner @ git+https://github.com/SafeDesign-ai/symbolic-decomposer.git"
```

Optional local GPU stack: `uv sync --extra vllm` (heavy). Dev tools: `uv sync --group dev`.

## CLI

The `deep-reasoner` command takes **config first**, **task second**, then optional flags.

```bash
# Remote API — set OPENAI_API_KEY (and OPENAI_BASE_URL if not using the default for your provider)
uv run deep-reasoner configs/examples/remote_api_question.yaml \
  "What is the boiling point of water at 3000 metres above sea level?"

uv run deep-reasoner configs/examples/remote_api_question.yaml \
  "Your question" --set max_iter=10 --var label=demo --var-read ctx=./notes.txt

# Local OpenAI-compatible server (e.g. vLLM on port 8555)
uv run deep-reasoner configs/examples/local_vllm_question.yaml "Your question"
```

Configs under `configs/examples/` compose shared agent settings from `_debug_agent_body.yaml`. Optional `initial_var_files` in YAML load REPL `Var` entries from paths **relative to that config file**; `--var` / `--var-read` add more from the shell. See `configs/README.md`.

## Use in Python

Build an async OpenAI-compatible client, load prompt and options from a YAML file (for example under `configs/examples/`), wire tools, then call the agent:

```python
import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from openai import AsyncOpenAI

from deep_reasoner.cli_utils import make_agent
from deep_reasoner.deepreasoner import Func, Var, agent_context
from deep_reasoner.llm import make_llm

async def run():
    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    cfg_path = Path("configs/examples/remote_api_question.yaml")
    main = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    body = yaml.safe_load((cfg_path.parent / "_debug_agent_body.yaml").read_text(encoding="utf-8"))
    model_id = main["model"]

    llm_caller = make_llm(client=client, model=model_id, stop=body.get("llm_kwargs", {}).get("stop"))
    tools = {
        "Var": Func(Var, description="Var(value, description)."),
        "Func": Func(Func, description="Func(fn, description)."),
        "llm": Func(llm_caller, description="LLM call or batch."),
    }
    cfg = SimpleNamespace(
        model=model_id,
        planner_model=None,
        llm_kwargs=body.get("llm_kwargs", {}),
        system_prompt=body["system_prompt"],
        models=body.get("models", []),
        max_iter=30,
        prompt_template_variables={},
    )
    agent = make_agent(cfg, tools, client)
    agent_context.reset()
    answer = agent("Your task here.", vars={})
    print(answer)
    await client.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

`PlanExec(agent)` wraps the same loop for sub-tasks; see `deep_reasoner.deepreasoner`.

## Tests

```bash
uv run pytest deep_reasoner/
```

Optional live API checks in `deepreasoner.py` run only with:

```bash
DEEP_REASONER_LIVE_TESTS=1 uv run pytest deep_reasoner/
```

## Project layout (core)

```
deep_reasoner/
  deepreasoner.py   # DeepReasoner, PlanExec, Var, Func
  llm.py            # make_llm, AsyncCaller, disk cache helpers
  cli_utils.py      # YAML + Dynaconf + ClientConfig + make_agent
  cli_base.py       # reference CLI (`deep-reasoner` console_script)
  config/           # shipped example agent YAML (e.g. debug_agent.yaml)
  core.py           # package paths, structlog, live-test client helpers
  code_exec.py      # REPL / execution context
  logging_utils.py  # run log dirs, yaml/json log processors
```

## License

See repository metadata.
