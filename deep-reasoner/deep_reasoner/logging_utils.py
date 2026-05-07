# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Logging Utilities
#
# Structured logging with run metadata capture and log directory organization.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import coolname

from juplit import test

# %% [markdown]
# ## Run ID generation

# %%
def generate_run_id() -> str:
    """Create a human-readable run ID using coolname (e.g. "swift-orange-bear")."""
    return coolname.generate_slug(3)

# %%
if test():
    run_id = generate_run_id()
    assert isinstance(run_id, str)
    assert len(run_id) > 0

# %% [markdown]
# ## Path utilities

# %%
def _sanitize_segment(value: object, *, default: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = default
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def agent_run_log_dir(
    log_base_dir: str | Path,
    task_id: object,
) -> Path:
    """Per-task directory for logs.

    Layout::

        {log_base_dir}/{task_id}/
    """
    t = _sanitize_segment(task_id, default="task")
    return Path(log_base_dir) / t

# %%
if test():
    assert _sanitize_segment(None) == "unknown"
    assert _sanitize_segment("") == "unknown"
    assert _sanitize_segment("hello world") == "hello_world"
    assert _sanitize_segment("a/b:c") == "a_b_c"

    p = agent_run_log_dir("logs", "task-42")
    assert p == Path("logs/task-42")

# %% [markdown]
# ## Config helpers

# %%
def format_config_models_for_help(classes: Iterable[type]) -> str:
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



# %%
if test():
    from pydantic import BaseModel, RootModel
    s = format_config_models_for_help((BaseModel, RootModel))
    assert "### Config class definitions" in s
    assert "```python" in s
    assert "BaseModel" in s
    print(s)


# %%

def redact_config_for_log(
    cfg_dict: dict[str, Any],
    *,
    redact_keys: Iterable[str] = ("api_key",),
) -> dict[str, Any]:
    redact = {str(k) for k in redact_keys}

    def rec(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: ("<redacted>" if k in redact else rec(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [rec(v) for v in x]
        return x

    return rec(cfg_dict)


# %%
if test():
    cfg = {"model": "gpt-4", "client": {"api_key": "secret", "base_url": "http://x"}}
    redacted = redact_config_for_log(cfg)
    assert redacted["client"]["api_key"] == "<redacted>"
    assert redacted["client"]["base_url"] == "http://x"
    assert redacted["model"] == "gpt-4"

# %% [markdown]
# ## Git info

# %%
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

# %% [markdown]
# ## Metadata writers

# %%
def write_run_metadata(
    run_dir: str | Path,
    *,
    config: dict[str, Any],
    argv: list[str] | None = None,
    repo_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
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
    """Write a qa.json stub before the agent runs (answer=null initially)."""
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

# %%
if test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        qa_path = write_initial_qa(tmp_path, task_id="t1", question="Q?", expected_answer="A")
        qa = json.loads(qa_path.read_text())
        assert qa["task_id"] == "t1"
        assert qa["answer"] is None
        assert qa["expected_answer"] == "A"

        meta_path = write_run_metadata(tmp_path, config={"model": "gpt-4"}, argv=["test"])
        meta = json.loads(meta_path.read_text())
        assert meta["config"]["model"] == "gpt-4"
        assert meta["argv"] == ["test"]
        assert "timestamp_utc" in meta

# %% [markdown]
# ## Structlog processor

# %%
def agent_log_processor(base_dir: str | Path = "logs"):
    """Return a structlog processor that persists events to disk."""
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

    def processor(logger, method_name, event_dict):
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


# %%
if test():
    import shutil
    import tempfile

    log_base = Path(tempfile.mkdtemp())
    try:
        proc = agent_log_processor(log_base)
        event = {
            "event": "not.agent",
            "benchmark": "b",
            "version": "v",
            "task_id": "t",
        }
        out = proc(None, None, event)
        assert isinstance(out, dict)
        run_dir = agent_run_log_dir(log_base, "t")
        assert run_dir.is_dir()
    finally:
        shutil.rmtree(log_base, ignore_errors=True)


# %%
# ! poe sync
