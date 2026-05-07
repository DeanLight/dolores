"""Convert sub-agent JSON logs (from deepsearchqa) to readable YAML."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import App, Parameter

app = App(help="Convert a sub-agent JSON log to a readable YAML trace.")


def _parse_tool_call(content: str) -> str:
    """Render 'Calling tools: [...]' content as tool_name(key=val, ...) lines."""
    lines = content.strip().split("\n", 1)
    if len(lines) < 2:
        return content
    try:
        calls = ast.literal_eval(lines[1])
        parts = []
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "?")
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            else:
                arg_str = repr(args)
            parts.append(f"{name}({arg_str})")
        return "\n".join(parts)
    except Exception:
        return content


def _parse_observation(content: str) -> str:
    """Strip leading 'Observation:\\n' prefix."""
    if content.startswith("Observation:\n"):
        return content[len("Observation:\n"):]
    return content


def convert_sub_json(data: dict) -> dict:
    """Convert parsed sub-agent JSON dict to a YAML-friendly dict."""
    steps = data.get("steps", [])
    if not steps:
        return {"answer": data.get("answer"), "total_usage": data.get("total_usage"), "messages": []}

    # The last step's messages contain the full execution conversation.
    # Steps 1+ accumulate history; step 0 is the planning-only call.
    exec_steps = [s for s in steps if any(m["role"] == "system" for m in s["messages"])]
    full_messages = exec_steps[-1]["messages"] if exec_steps else steps[-1]["messages"]

    out_messages = []
    seen: set[tuple[str, str]] = set()

    for msg in full_messages:
        role = msg["role"]
        content = (msg.get("content") or "").strip()

        if role == "system":
            continue
        if not content:
            continue

        dedup_key = (role, content[:120])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        if role == "tool-call":
            out_messages.append({"tool_call": _parse_tool_call(content)})
        elif role == "tool-response":
            out_messages.append({"observation": _parse_observation(content)})
        else:
            out_messages.append({role: content})

    return {
        "answer": data.get("answer"),
        "total_usage": data.get("total_usage"),
        "messages": out_messages,
    }


def _clean(s: str) -> str:
    """Strip trailing whitespace per line so PyYAML can use literal block style."""
    return "\n".join(line.rstrip() for line in s.split("\n"))


def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Use literal block style for multiline strings."""
    cleaned = _clean(data)
    if "\n" in cleaned:
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned)


class _LiteralDumper(yaml.Dumper):
    pass


_LiteralDumper.add_representer(str, _str_representer)


def to_yaml(converted: dict) -> str:
    return yaml.dump(converted, Dumper=_LiteralDumper, allow_unicode=True, sort_keys=False, width=100)


def convert_task_dir(task_dir: Path, overwrite: bool = False) -> list[Path]:
    """Convert all *_sub_*.json files in a task log directory to YAML.

    Returns the list of YAML paths written.
    """
    written: list[Path] = []
    for json_path in sorted(task_dir.glob("*_sub_*.json")):
        out_path = json_path.with_suffix(".yaml")
        if out_path.exists() and not overwrite:
            continue
        data = json.loads(json_path.read_text())
        out_path.write_text(to_yaml(convert_sub_json(data)), encoding="utf-8")
        written.append(out_path)
    return written


def convert_log_dir(log_dir: Path, overwrite: bool = False) -> list[Path]:
    """Recursively convert all *_sub_*.json files under a log directory.

    Walks any depth, so it works for a task dir, a run dir, a log base dir, or
    an experiment log dir.
    """
    written: list[Path] = []
    for json_path in sorted(log_dir.rglob("*_sub_*.json")):
        out_path = json_path.with_suffix(".yaml")
        if out_path.exists() and not overwrite:
            continue
        data = json.loads(json_path.read_text())
        out_path.write_text(to_yaml(convert_sub_json(data)), encoding="utf-8")
        written.append(out_path)
    return written


@app.default
def cli(
    path: Annotated[
        Path,
        Parameter(name="path", help="Path to a *_sub_*.json file."),
    ],
    output: Annotated[
        Path | None,
        Parameter(name=["-o", "--output"], help="Write YAML here (default: <path>.yaml next to source)."),
    ] = None,
    stdout: Annotated[
        bool,
        Parameter(name=["--stdout"], help="Print YAML to stdout instead of writing a file."),
    ] = False,
) -> None:
    """Convert a single sub-agent JSON log to readable YAML."""
    data = json.loads(path.read_text())
    converted = convert_sub_json(data)
    yaml_text = to_yaml(converted)

    if stdout:
        sys.stdout.write(yaml_text)
        return

    out_path = output or path.with_suffix(".yaml")
    out_path.write_text(yaml_text, encoding="utf-8")
    print(f"Written: {out_path}")


@app.command
def dir(
    log_dir: Annotated[
        Path,
        Parameter(
            name="log_dir",
            help=(
                "Directory to search for *_sub_*.json files. "
                "Accepts a task log dir, run dir, log base dir, or any ancestor — "
                "the tool walks recursively."
            ),
        ),
    ],
    overwrite: Annotated[
        bool,
        Parameter(name=["--overwrite"], help="Re-convert even if a .yaml already exists."),
    ] = False,
) -> None:
    """Recursively convert all *_sub_*.json files under a log directory to YAML."""
    written = convert_log_dir(log_dir, overwrite=overwrite)
    if written:
        for p in written:
            print(f"Written: {p}")
        print(f"\n{len(written)} file(s) converted.")
    else:
        print("No new sub JSON files found (use --overwrite to re-convert existing).")


def main_cli() -> None:
    app()


def main_cli_dir() -> None:
    app["dir"]()
