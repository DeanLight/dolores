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
# # obs — observability: context, logging, LLM-call patches
#
# Five things live here:
#
# 1. **Run ID and log path helpers** — `make_run_id`, `generate_run_id`,
#    `run_log_dir`, `agent_run_log_dir`.
#
# 2. **Config and run metadata helpers** — `redact_config_for_log`,
#    `git_info`, `write_run_metadata`, `write_initial_qa`,
#    `format_config_models_for_help`.
#
# 3. **`agent_context`** — a `conflit.Context` with tracked vars (`node_id`,
#    `node_name`, `depth`, `ancestry`, `agent`, `run_id`, `log_dir`).
#    Use `push_agent(...)` to enter a new context frame.
#
# 4. **Structlog processors** — `_obs_log_processor` (routes `llm.call`
#    events to `calls.jsonl`), `agent_log_processor` (routes legacy per-task
#    events to per-task directories), `install_run_logger()`.
#
# 5. **`install_patches()`** — monkey-patch `litellm` and `openai` so every
#    completion call emits a structlog `llm.call` event with normalised token
#    counts.  Safe to call multiple times (idempotent).
#
# Log layout per run:
# ```
# logs/<benchmark>/<agent>/<model>[-nothink]/<run_id>/
#     calls.jsonl   — one JSON object per LLM call
#     result.json   — final answer + metadata
#     thread.yaml   — full conversation thread (written by agents in PR C+)
# ```

# %%
from juplit import test

# %%
import functools
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import textwrap
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import coolname
import structlog
from conflit import Context, TrackedVar
from deep_reasoner.core import configure_structlog_fixture

from config import Paths
from dolores.unified.token_matched import ContextWindowExceeded, active_budget

logger = structlog.get_logger(__name__)

# %% [markdown]
# ## Run ID and log path helpers

# %%
def make_run_id() -> str:
    """Return a sortable, unique run identifier: ``<utc-ts>_<8-hex>``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{secrets.token_hex(4)}"


def generate_run_id() -> str:
    """Return a human-readable run ID using coolname (e.g. ``"swift-orange-bear"``)."""
    return coolname.generate_slug(3)


def run_log_dir(
    benchmark: str,
    agent: str,
    model: str,
    run_id: str,
    *,
    no_thinking: bool = False,
    base: Path | None = None,
) -> Path:
    """Return (but do not create) the run-specific log directory.

    Layout: ``<base>/<benchmark>/<agent>/<model>[-nothink]/<run_id>/``
    """
    if base is None:
        base = Paths.LOGS_DIR
    model_dir = f"{model}-nothink" if no_thinking else model
    return base / benchmark / agent / model_dir / run_id


def _sanitize_segment(value: object, *, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def agent_run_log_dir(log_base_dir: str | Path, task_id: object) -> Path:
    """Return per-task log directory: ``{log_base_dir}/{task_id}/``."""
    return Path(log_base_dir) / _sanitize_segment(task_id, default="task")


# %% [markdown]
# ## Config and run metadata helpers

# %%
def format_config_models_for_help(classes: Iterable[type]) -> str:
    """Return a markdown block listing Pydantic model source for CLI --help text."""
    import inspect
    parts: list[str] = [
        "",
        "### Config class definitions",
        "",
        "Pydantic models for YAML and `--set` (field names and nesting must match).",
        "",
    ]
    for cls in classes:
        src = inspect.getsource(cls).rstrip()
        parts.append(f"```python\n{src}\n```")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def redact_config_for_log(
    cfg_dict: dict[str, Any],
    *,
    redact_keys: Iterable[str] = ("api_key",),
) -> dict[str, Any]:
    """Recursively replace sensitive keys with ``"<redacted>"``."""
    redact = {str(k) for k in redact_keys}

    def rec(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: ("<redacted>" if k in redact else rec(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [rec(v) for v in x]
        return x

    return rec(cfg_dict)


def git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Best-effort git metadata for reproducibility."""
    try:
        root = (
            Path(repo_root)
            if repo_root is not None
            else Path(__file__).resolve().parent.parent
        )
        head = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=str(root)
        ) != 0 or subprocess.call(
            ["git", "diff", "--cached", "--quiet"], cwd=str(root)
        ) != 0
        return {"hash": head, "dirty": bool(dirty)}
    except Exception:
        return {"hash": None, "dirty": None}


def write_run_metadata(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    argv: list[str] | None = None,
    repo_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``metadata.json`` into *run_dir* and return the path."""
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv) if argv is None else argv,
        "git": git_info(repo_root=repo_root),
        "config": config,
    }
    if extra:
        payload.update(extra)
    path = run_dir / "metadata.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def write_initial_qa(
    run_dir: str | Path,
    *,
    task_id: str,
    question: str,
    expected_answer: Any,
) -> Path:
    """Write a ``qa.json`` stub before the agent runs (``answer=null`` initially)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "qa.json"
    path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "question": question,
                "expected_answer": expected_answer,
                "answer": None,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


# %% [markdown]
# ## agent_context

# %%
def _fresh_node_name(parent: str, inputs: dict) -> str:
    agent = inputs.get("agent") or parent or "node"
    node_id = inputs.get("node_id", 0)
    return f"{agent}-{node_id}"


agent_context: Context = Context(
    TrackedVar("node_id", default=0, shared=True, derive=lambda p, _: p + 1),
    TrackedVar("node_name", default="root", derive=_fresh_node_name),
    TrackedVar("depth", default=0, derive=lambda p, _: p + 1),
    TrackedVar("ancestry", default=(), derive=lambda p, inputs: p + (inputs["node_id"],)),
    TrackedVar("agent", default="", derive=lambda p, inputs: inputs.get("agent", p)),
    TrackedVar("run_id", default="", derive=lambda p, inputs: inputs.get("run_id", p)),
    TrackedVar("log_dir", default="", derive=lambda p, inputs: inputs.get("log_dir", p)),
    structlog_keys=["node_id", "node_name", "depth", "agent", "run_id"],
)


@contextmanager
def push_agent(
    agent: str,
    run_id: str | None = None,
    log_dir: str | Path | None = None,
):
    """Context manager that enters a new agent_context frame.

    At the top level, pass all three.  For nested sub-agents, only ``agent``
    is required — ``run_id`` and ``log_dir`` are inherited from the parent.
    """
    kwargs: dict[str, Any] = {"agent": agent}
    if run_id is not None:
        kwargs["run_id"] = run_id
    if log_dir is not None:
        kwargs["log_dir"] = str(log_dir)
    with agent_context.bind(**kwargs) as ctx:
        yield ctx


# %% [markdown]
# ## Token usage normalisation

# %%
def _to_dict(obj) -> dict:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def _normalize_usage(response) -> dict[str, int]:
    """Extract normalised token counts from an OpenAI/litellm response object."""
    usage = _to_dict(getattr(response, "usage", None))

    prompt_tokens: int = usage.get("prompt_tokens") or 0
    completion_tokens: int = usage.get("completion_tokens") or 0
    total_tokens: int = usage.get("total_tokens") or (prompt_tokens + completion_tokens)

    # cached_tokens: OpenAI prompt_tokens_details.cached_tokens
    #                OR Anthropic cache_read_input_tokens
    ptd = _to_dict(usage.get("prompt_tokens_details"))
    cached_tokens: int = (ptd.get("cached_tokens") or 0) or (usage.get("cache_read_input_tokens") or 0)

    # reasoning_tokens: OpenAI completion_tokens_details.reasoning_tokens
    ctd = _to_dict(usage.get("completion_tokens_details"))
    reasoning_tokens: int = ctd.get("reasoning_tokens") or 0

    # cache_creation_tokens: Anthropic cache_creation_input_tokens
    cache_creation_tokens: int = usage.get("cache_creation_input_tokens") or 0

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_creation_tokens": cache_creation_tokens,
    }


# %% [markdown]
# ## Structlog processors

# %%
_logger_installed = False
_logger_lock = threading.Lock()


def _obs_log_processor(log_logger, method, event_dict: dict) -> dict:
    """Structlog processor: route llm.call events to calls.jsonl in the current run dir."""
    event = event_dict.get("event", "")
    log_dir = agent_context.log_dir.get()
    if not log_dir or not event.startswith("llm."):
        return event_dict

    run_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if event == "llm.call":
        record = {k: v for k, v in event_dict.items() if k != "event"}
        with open(run_dir / "calls.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    return event_dict


def agent_log_processor(base_dir: str | Path = "logs"):
    """Return a structlog processor that persists per-task events to disk.

    Handles ``llm.*`` → ``llm_calls.jsonl``, ``agent.result`` → ``qa.json``,
    ``agent.config`` → ``config.json``, and agent message threads → ``*.yaml``.
    """
    base_dir = Path(base_dir)

    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def _append_jsonl(path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    def _messages_to_yaml(messages: list[dict], ancestry: object) -> str:
        lines = [f"ancestry: {repr(ancestry)}", "messages:"]
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = _sanitize_segment(msg.get("role"), default="unknown")
            content = str(msg.get("content", ""))
            lines.append(f"  - {role}: |")
            msg_lines = content.splitlines() or [""]
            for line in msg_lines:
                wrapped = textwrap.wrap(line, width=100) or [""]
                for wline in wrapped:
                    lines.append(f"      {wline}")
        return "\n".join(lines) + "\n"

    def _event_run_dir(event_dict: dict[str, Any]) -> Path | None:
        task_id = event_dict.get("task_id")
        if not task_id:
            return None
        run_dir = agent_run_log_dir(base_dir, task_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def processor(log_logger, method_name, event_dict):
        run_dir = _event_run_dir(event_dict)
        if run_dir is None:
            return event_dict
        event_name = str(event_dict.get("event") or "")

        if event_name.startswith("llm"):
            _append_jsonl(run_dir / "llm_calls.jsonl", event_dict)

        if event_name == "agent.result":
            _write_json(
                run_dir / "qa.json",
                {
                    "task_id": event_dict.get("task_id"),
                    "question": event_dict.get("question"),
                    "answer": event_dict.get("answer"),
                    "parsed_output": event_dict.get("parsed_output"),
                    "expected_answer": event_dict.get("expected_answer"),
                    "f1": event_dict.get("f1"),
                    "em": event_dict.get("em"),
                    "score_error": event_dict.get("score_error"),
                    "score_reasoning": event_dict.get("score_reasoning"),
                },
            )
            return event_dict

        if event_name == "agent.config":
            payload = event_dict.get("config")
            if payload is not None:
                _write_json(run_dir / "config.json", payload)
            return event_dict

        if not event_name.startswith("agent"):
            return event_dict

        messages = event_dict.get("messages")
        if not isinstance(messages, list):
            return event_dict

        node_id = _sanitize_segment(event_dict.get("node_id"), default="root")
        depth = _sanitize_segment(event_dict.get("depth"), default="0")
        node_name = _sanitize_segment(event_dict.get("node_name"), default="node")
        ancestry = event_dict.get("ancestry", ())
        (run_dir / f"n_{node_id}_d_{depth}_{node_name}.yaml").write_text(
            _messages_to_yaml(messages, ancestry),
            encoding="utf-8",
        )
        return event_dict

    return processor


def install_run_logger() -> None:
    """Configure structlog (once) to persist LLM call events under the current run dir.

    Idempotent — safe to call multiple times.
    """
    global _logger_installed
    with _logger_lock:
        if _logger_installed:
            return
        configure_structlog_fixture(
            console=False,
            extra_processors=[_obs_log_processor],
            default_level=logging.INFO,
        )
        _logger_installed = True


# %% [markdown]
# ## LLM-call patches

# %%
_patches_installed = False
_patches_lock = threading.Lock()


def _before_completion(kwargs: dict) -> dict:
    """Let the active token budget (if any) set sampling params on the way in."""
    budget = active_budget()
    return budget.before_call(kwargs) if budget else kwargs


def _is_context_window_error(exc: Exception) -> bool:
    """True when ``exc`` is a backend context-length overflow.

    litellm maps these to ``ContextWindowExceededError``; the raw vLLM/openai 400
    carries "maximum context length" in its message, so match both without
    importing litellm here.
    """
    if type(exc).__name__ == "ContextWindowExceededError":
        return True
    msg = str(exc).lower()
    return "context length" in msg or "maximum context" in msg or "reduce the length" in msg


def _raise_if_context_window(exc: Exception) -> None:
    """Translate a context-window overflow into ``ContextWindowExceeded``.

    Raised as a ``BaseException`` so agent frameworks' ``except Exception`` cannot
    swallow it and retry the same oversized prompt; the caller re-raises ``exc``
    unchanged when it is not a context overflow.
    """
    if _is_context_window_error(exc):
        raise ContextWindowExceeded(str(exc)) from exc


def _emit_llm_event(kwargs: dict, response) -> None:
    usage = _normalize_usage(response)
    model = getattr(response, "model", None) or kwargs.get("model", "")

    reasoning_content = None
    try:
        msg = response.choices[0].message
        reasoning_content = getattr(msg, "reasoning_content", None)
    except Exception:
        pass

    logger.info(
        "llm.call",
        model=model,
        reasoning_content=reasoning_content,
        **usage,
    )

    # Charge the attempt's budget last, so the call is always logged before a
    # BudgetExhausted unwinds the run.
    budget = active_budget()
    if budget:
        budget.after_call(response, usage)


def _install_litellm_patches() -> None:
    import litellm

    orig_completion = litellm.completion
    orig_acompletion = litellm.acompletion

    @functools.wraps(orig_completion)
    def _completion(*args, **kwargs):
        kwargs = _before_completion(kwargs)
        try:
            response = orig_completion(*args, **kwargs)
        except Exception as exc:
            _raise_if_context_window(exc)
            raise
        _emit_llm_event(kwargs, response)
        return response

    @functools.wraps(orig_acompletion)
    async def _acompletion(*args, **kwargs):
        kwargs = _before_completion(kwargs)
        try:
            response = await orig_acompletion(*args, **kwargs)
        except Exception as exc:
            _raise_if_context_window(exc)
            raise
        _emit_llm_event(kwargs, response)
        return response

    litellm.completion = _completion
    litellm.acompletion = _acompletion


def _install_openai_patches() -> None:
    from openai.resources.chat.completions import AsyncCompletions, Completions

    orig_sync = Completions.create
    orig_async = AsyncCompletions.create

    @functools.wraps(orig_sync)
    def _sync_create(self, *args, **kwargs):
        kwargs = _before_completion(kwargs)
        try:
            response = orig_sync(self, *args, **kwargs)
        except Exception as exc:
            _raise_if_context_window(exc)
            raise
        _emit_llm_event(kwargs, response)
        return response

    @functools.wraps(orig_async)
    async def _async_create(self, *args, **kwargs):
        kwargs = _before_completion(kwargs)
        try:
            response = await orig_async(self, *args, **kwargs)
        except Exception as exc:
            _raise_if_context_window(exc)
            raise
        _emit_llm_event(kwargs, response)
        return response

    Completions.create = _sync_create
    AsyncCompletions.create = _async_create


def install_patches() -> None:
    """Monkey-patch litellm and openai to emit structlog ``llm.call`` events.

    Idempotent — safe to call multiple times.
    """
    global _patches_installed
    with _patches_lock:
        if _patches_installed:
            return
        _install_litellm_patches()
        _install_openai_patches()
        _patches_installed = True


# %% [markdown]
# ## Tests

# %%
def test_agent_context_smoke():
    start_id = agent_context.node_id.get()

    with push_agent("my-agent", run_id="r1", log_dir="/tmp/run1") as ctx:
        assert agent_context.agent.get() == "my-agent"
        assert agent_context.run_id.get() == "r1"
        assert agent_context.log_dir.get() == "/tmp/run1"
        assert agent_context.depth.get() == 1
        assert agent_context.node_id.get() == start_id + 1
        outer_id = agent_context.node_id.get()

        with push_agent("sub-agent") as inner:
            assert agent_context.agent.get() == "sub-agent"
            assert agent_context.run_id.get() == "r1"
            assert agent_context.log_dir.get() == "/tmp/run1"
            assert agent_context.depth.get() == 2
            assert agent_context.node_id.get() == outer_id + 1
            assert outer_id in agent_context.ancestry.get()

    assert agent_context.agent.get() == ""
    assert agent_context.depth.get() == 0


def test_make_run_id():
    rid = make_run_id()
    ts, hex_part = rid.split("_")
    assert len(ts) == 15  # YYYYmmddTHHMMSS
    assert len(hex_part) == 8
    assert make_run_id() != make_run_id()


def test_generate_run_id():
    rid = generate_run_id()
    assert isinstance(rid, str)
    assert len(rid) > 0
    assert generate_run_id() != generate_run_id()


def test_sanitize_segment():
    assert _sanitize_segment(None) == "unknown"
    assert _sanitize_segment("") == "unknown"
    assert _sanitize_segment("hello world") == "hello_world"
    assert _sanitize_segment("a/b:c") == "a_b_c"


def test_agent_run_log_dir():
    p = agent_run_log_dir("logs", "task-42")
    assert p == Path("logs/task-42")


def test_redact_config_for_log():
    cfg = {"model": "gpt-4", "client": {"api_key": "secret", "base_url": "http://x"}}
    redacted = redact_config_for_log(cfg)
    assert redacted["client"]["api_key"] == "<redacted>"
    assert redacted["client"]["base_url"] == "http://x"
    assert redacted["model"] == "gpt-4"


def test_write_initial_qa():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = write_initial_qa(Path(tmp), task_id="t1", question="Q?", expected_answer="A")
        qa = json.loads(path.read_text())
        assert qa["task_id"] == "t1"
        assert qa["answer"] is None
        assert qa["expected_answer"] == "A"


def test_write_run_metadata():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = write_run_metadata(Path(tmp), config={"model": "gpt-4"}, argv=["test"])
        meta = json.loads(path.read_text())
        assert meta["config"]["model"] == "gpt-4"
        assert meta["argv"] == ["test"]
        assert "timestamp_utc" in meta


def test_patches_idempotent():
    global _patches_installed
    was = _patches_installed
    _patches_installed = False
    try:
        import litellm
        orig = litellm.completion
        install_patches()
        after_first = litellm.completion
        install_patches()
        after_second = litellm.completion
        assert after_first is after_second
    finally:
        _patches_installed = was


def test_normalized_usage_extraction():
    from types import SimpleNamespace

    response = SimpleNamespace(
        model="gpt-4o",
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
            cache_creation_input_tokens=0,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(reasoning_content="<think>...</think>"))],
    )
    usage = _normalize_usage(response)
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 150
    assert usage["cached_tokens"] == 20
    assert usage["reasoning_tokens"] == 10
    assert usage["cache_creation_tokens"] == 0

    anthropic_response = SimpleNamespace(
        model="claude-opus-4-7",
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=280,
            prompt_tokens_details=None,
            completion_tokens_details=None,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=5,
        ),
        choices=[SimpleNamespace(message=SimpleNamespace(reasoning_content=None))],
    )
    usage2 = _normalize_usage(anthropic_response)
    assert usage2["cached_tokens"] == 30
    assert usage2["cache_creation_tokens"] == 5
    assert usage2["reasoning_tokens"] == 0
