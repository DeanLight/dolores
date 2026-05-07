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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from juplit import test

# %% [markdown]
# # Eval Analysis
#
# Aggregates and analyzes results from experiment log directories.

# %%

import asyncio
import glob as _glob
import itertools
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from juplit import test

# %% [markdown]
# ## Parsing helpers

# %%
def _extract_agent_configs_from_argv(argv: list[str]) -> list[str]:
    """Extract repeated -c/--config values from logged argv."""
    configs: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in {"-c", "--config"} and i + 1 < len(argv):
            configs.append(argv[i + 1])
            i += 2
            continue
        i += 1
    return configs


def _parse_entities_from_answer(raw_output: Any) -> list[str]:
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
    out = []
    for part in re.split(r"[\n,]+", text):
        cleaned = re.sub(r"^\s*[-*•\d\.\)\(]+\s*", "", part).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _cfg_from_run_meta(run_meta: dict[str, Any]) -> dict[str, Any]:
    """Normalize run_metadata.json into a flat cfg dict."""
    exp = run_meta.get("experiment_config", {})
    extra = exp.get("benchmark_extra_params", {})
    cfg: dict[str, Any] = {}
    for field in ("model", "version", "benchmark", "max_iter"):
        if field in exp:
            cfg[field] = exp[field]
    for field in ("size", "seed"):
        if field in extra:
            cfg[field] = extra[field]
    return cfg

# %%
if test():
    assert _extract_agent_configs_from_argv(["-c", "a.yaml", "--config", "b.yaml"]) == [
        "a.yaml", "b.yaml"
    ]
    assert _extract_agent_configs_from_argv([]) == []
    assert _extract_agent_configs_from_argv(["--other", "val"]) == []

    assert _parse_entities_from_answer(["a", "b"]) == ["a", "b"]
    assert _parse_entities_from_answer('["x", "y"]') == ["x", "y"]
    assert _parse_entities_from_answer("a, b\nc") == ["a", "b", "c"]
    assert _parse_entities_from_answer("") == []
    assert _parse_entities_from_answer([" ", ""]) == []

# %% [markdown]
# ## Scoring helpers

# %%
async def _score_from_qa_if_needed(
    cfg: dict[str, Any],
    task_id: str,
    qa: dict[str, Any],
    openai_api_key: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Backfill f1/em from qa answer when explicit scores are absent.

    Returns (f1, em, score_error, parsed_output). parsed_output is non-None
    only when it was freshly computed here (i.e. not already in qa).
    oolong.parse (the LLM call) runs in a thread and is gated by semaphore.
    """
    f1 = qa.get("f1")
    em = qa.get("em")
    score_error = qa.get("score_error")
    if f1 is not None and em is not None:
        return f1, em, score_error, None

    benchmark = str(cfg.get("benchmark") or "")
    answer = qa.get("answer")
    if answer is None:
        return None, None, score_error, None

    try:
        from benchmarks import hello_world, phantomwiki

        if benchmark.startswith("helloworld"):
            seed = int(cfg.get("seed", 0))
            f1, em = hello_world.score(task_id, answer, seed=seed)
            return f1, em, score_error, None
        if benchmark.startswith("phantomwiki"):
            size = int(cfg.get("size"))
            seed = int(cfg.get("seed"))
            parsed = qa.get("parsed_output") or _parse_entities_from_answer(answer)
            f1, em = phantomwiki.score(size, seed, task_id, parsed)
            return f1, em, score_error, None
        if benchmark.startswith("oolong"):
            from benchmarks import oolong
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Oolong scoring requires an OpenAI API key. "
                    "Set OPENAI_API_KEY in your environment (e.g. via .envrc) "
                    "or pass --openai-api-key."
                )
            had_parsed = qa.get("parsed_output") is not None
            if had_parsed:
                parsed = qa.get("parsed_output")
            else:
                _sem = semaphore or asyncio.Semaphore(1)
                async with _sem:
                    parsed = await asyncio.to_thread(
                        oolong.parse, task_id, str(answer), api_key=api_key
                    )
            score_val = oolong.score(task_id, parsed)
            f1 = float(score_val)
            em = float(score_val == 1.0)
            return f1, em, score_error, (None if had_parsed else parsed)
        if benchmark.startswith("synthworld"):
            from benchmarks import synthworlds
            f1, em = synthworlds.score(task_id, str(answer))
            return f1, em, score_error, None
        if benchmark.startswith("deepsearchqa"):
            from benchmarks import deepresearchqa
            _, score_val, _ = await asyncio.to_thread(
                deepresearchqa.score_judge, task_id, str(answer)
            )
            f1 = float(score_val)
            em = float(score_val)
            return f1, em, score_error, None
    except Exception as exc:
        score_error = str(exc)
    return None, None, score_error, None

# %% [markdown]
# ## Task result loading

# %%
async def load_task_result(
    task_dir: Path,
    run_meta: dict | None = None,
    openai_api_key: str | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> dict | None:
    """Load scores and config metadata from a single task directory."""
    qa_path = task_dir / "qa.json"
    scores_path = task_dir / "scores.json"
    meta_path = task_dir / "metadata.json"

    if not qa_path.exists() and not scores_path.exists() and not meta_path.exists() and not run_meta:
        return None

    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    cfg = metadata.get("config", {})
    argv = metadata.get("argv", [])

    if not cfg and run_meta:
        cfg = _cfg_from_run_meta(run_meta)

    task_id = qa.get("task_id") or scores.get("task_id") or task_dir.name
    run_id = task_dir.parent.name
    benchmark_version = task_dir.parent.parent.name

    f1 = qa.get("f1", scores.get("f1"))
    em = qa.get("em", scores.get("em"))
    score_error = qa.get("score_error")
    qa_updates: dict[str, Any] = {}

    if f1 is None or em is None:
        f1, em, score_error, new_parsed = await _score_from_qa_if_needed(
            cfg, task_id, qa, openai_api_key=openai_api_key, semaphore=semaphore
        )
        if f1 is not None:
            qa_updates.update({"f1": f1, "em": em, "score_error": score_error})
        if new_parsed is not None:
            qa_updates["parsed_output"] = new_parsed

    # Backfill expected_answer for oolong runs that predate the fix (cheap dataset lookup, no LLM).
    benchmark_str = str(cfg.get("benchmark") or "")
    if benchmark_str.startswith("oolong") and qa.get("expected_answer") is None:
        try:
            from benchmarks import oolong as _oolong
            qa_updates["expected_answer"] = _oolong.get_answer(task_id)
        except Exception:
            pass

    if qa_updates and qa_path.exists():
        qa.update(qa_updates)
        qa_path.write_text(
            json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    agent_configs = _extract_agent_configs_from_argv(argv)
    if not agent_configs and run_meta:
        agent_configs = run_meta.get("experiment_config", {}).get("agent_configs", [])
    agent_config_key = " | ".join(agent_configs) if agent_configs else "<unknown>"

    agent_id = cfg.get("version")
    completed = f1 is not None and em is not None

    return {
        "completed": completed,
        "task_dir": str(task_dir),
        "agent_id": agent_id,
        "task_id": task_id,
        "run_id": run_id,
        "benchmark_version": benchmark_version,
        "f1": f1,
        "em": em,
        "parsed_output": qa.get("parsed_output", scores.get("parsed_output")),
        "score_error": score_error,
        "model": cfg.get("model"),
        "benchmark": cfg.get("benchmark"),
        "version": cfg.get("version"),
        "size": cfg.get("size"),
        "limit": cfg.get("limit"),
        "seed": cfg.get("seed"),
        "max_iter": cfg.get("max_iter"),
        "agent_configs": agent_configs,
        "agent_config_key": agent_config_key,
    }

# %% [markdown]
# ## Collection functions

# %%
async def collect_results(
    log_base_dir: str | Path,
    openai_api_key: str | None = None,
    max_concurrent_scoring: int = 10,
) -> pd.DataFrame:
    """Walk log_base_dir and collect finished task results into a DataFrame."""
    log_base_dir = Path(log_base_dir)
    semaphore = asyncio.Semaphore(max_concurrent_scoring)
    task_dirs = [d for d in sorted(log_base_dir.glob("*/*/*")) if d.is_dir()]

    async def _load(task_dir: Path) -> dict | None:
        record = await load_task_result(task_dir, openai_api_key=openai_api_key, semaphore=semaphore)
        if record is not None:
            record["log_base_dir"] = str(log_base_dir)
        return record

    results = await asyncio.gather(*[_load(d) for d in task_dirs])
    records = [r for r in results if r is not None]
    return pd.DataFrame(records) if records else pd.DataFrame()


async def collect_results_from_run_dirs(
    patterns: list[str],
    openai_api_key: str | None = None,
    max_concurrent_scoring: int = 10,
) -> pd.DataFrame:
    """Collect results from run directories matched by glob patterns."""
    run_dirs: set[Path] = set()
    for pattern in patterns:
        matched = _glob.glob(str(pattern), recursive=True)
        for m in matched:
            p = Path(m)
            if p.is_dir():
                run_dirs.add(p.resolve())
        if not matched:
            p = Path(pattern)
            if p.is_dir():
                run_dirs.add(p.resolve())

    semaphore = asyncio.Semaphore(max_concurrent_scoring)

    async def _load_run(run_dir: Path) -> list[dict]:
        run_meta: dict[str, Any] = {}
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                run_meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        _TASK_MARKERS = ("qa.json", "scores.json", "metadata.json")
        task_dirs = [
            d for d in sorted(run_dir.iterdir())
            if d.is_dir() and any((d / f).exists() for f in _TASK_MARKERS)
        ]
        results = await asyncio.gather(*[
            load_task_result(d, run_meta=run_meta, openai_api_key=openai_api_key, semaphore=semaphore)
            for d in task_dirs
        ])
        return [r for r in results if r is not None]

    run_results = await asyncio.gather(*[_load_run(d) for d in sorted(run_dirs)])
    records = [r for run in run_results for r in run]
    return pd.DataFrame(records) if records else pd.DataFrame()


async def collect_results_from_experiment_configs(
    config_paths: list[str | Path],
    openai_api_key: str | None = None,
    max_concurrent_scoring: int = 10,
) -> pd.DataFrame:
    """Load and merge results from multiple experiment YAML configs."""
    async def _load_cfg(cfg_path: Path) -> pd.DataFrame | None:
        cfg = yaml.safe_load(cfg_path.read_text())
        if not isinstance(cfg, dict):
            raise ValueError(f"Experiment config at {cfg_path} must be a YAML mapping")
        log_base_dir = cfg.get("log_dir", cfg.get("log_base_dir", "logs"))
        df = await collect_results(log_base_dir, openai_api_key=openai_api_key, max_concurrent_scoring=max_concurrent_scoring)
        if not df.empty:
            df["experiment_config"] = str(cfg_path)
            return df
        return None

    dfs = await asyncio.gather(*[_load_cfg(Path(p)) for p in config_paths])
    non_empty = [df for df in dfs if df is not None]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def discover_runs(log_base_dir: str | Path) -> pd.DataFrame:
    """Summarize all runs found under log_base_dir."""
    log_base_dir = Path(log_base_dir)
    rows: list[dict] = []
    for run_dir in sorted(log_base_dir.glob("*/*")):
        if not run_dir.is_dir():
            continue
        benchmark_version = run_dir.parent.name
        run_id = run_dir.name
        task_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        n_tasks = len(task_dirs)
        n_finished = sum(
            1 for d in task_dirs
            if (d / "scores.json").exists() or (d / "qa.json").exists()
        )

        meta: dict[str, Any] = {}
        meta_path = run_dir / "run_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        row: dict[str, Any] = {
            "benchmark_version": benchmark_version,
            "run_id": run_id,
            "n_tasks": n_tasks,
            "n_finished": n_finished,
        }
        for k, v in meta.items():
            if k not in row and not isinstance(v, (dict, list)):
                row[k] = v
        cfg = meta.get("experiment_config", {})
        if isinstance(cfg, dict):
            for field in ("model", "version", "benchmark", "agent_configs"):
                if field in cfg and field not in row:
                    row[field] = cfg[field]
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# %% [markdown]
# ## Analysis functions

# %%
def summarize(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate mean F1 / EM grouped by the given columns."""
    if df.empty:
        return df
    group_cols = by if by is not None else ["benchmark", "run_id", "model", "version"]
    group_cols = [c for c in group_cols if c in df.columns]

    completed = df[df["f1"].notna() & df["em"].notna()]
    agg = (
        completed.groupby(group_cols)[["f1", "em"]]
        .agg(mean_f1=("f1", "mean"), mean_em=("em", "mean"), n_completed=("f1", "count"))
        .round(3)
        .reset_index()
    )
    totals = df.groupby(group_cols).size().reset_index(name="n_total")
    result = agg.merge(totals, on=group_cols, how="outer")
    result["n_completed"] = result["n_completed"].fillna(0).astype(int)
    result["n_incomplete"] = result["n_total"] - result["n_completed"]

    df_filled = df.copy()
    df_filled["f1"] = df_filled["f1"].fillna(0.0)
    df_filled["em"] = df_filled["em"].fillna(0.0)
    agg_all = (
        df_filled.groupby(group_cols)[["f1", "em"]]
        .agg(mean_f1_all=("f1", "mean"), mean_em_all=("em", "mean"))
        .round(3)
        .reset_index()
    )
    result = result.merge(agg_all, on=group_cols, how="left")
    return result


def benchmark_config_table(df: pd.DataFrame) -> pd.DataFrame:
    """Comparison table by benchmark and agent id across experiments."""
    if df.empty:
        return df
    group_cols = [
        c for c in ["benchmark", "run_id", "agent_id", "model", "version", "experiment_config"]
        if c in df.columns
    ]
    completed = df[df["f1"].notna() & df["em"].notna()]
    out = (
        completed.groupby(group_cols, dropna=False)
        .agg(mean_f1=("f1", "mean"), mean_em=("em", "mean"), n_completed=("f1", "count"))
        .reset_index()
    )
    totals = df.groupby(group_cols, dropna=False).size().reset_index(name="n_total")
    out = out.merge(totals, on=group_cols, how="outer")
    out["n_completed"] = out["n_completed"].fillna(0).astype(int)
    out["n_incomplete"] = out["n_total"] - out["n_completed"]
    out["benchmark_rank"] = (
        out.groupby("benchmark")["mean_f1"]
        .rank(method="dense", ascending=False, na_option="bottom")
        .astype(int)
    )
    return out.sort_values(
        ["benchmark", "benchmark_rank", "mean_em"], ascending=[True, True, False]
    ).round(3)


def build_outperformance_report(df: pd.DataFrame) -> dict[str, Any]:
    """Pairwise per-task outperformance report for YAML export."""
    report: dict[str, Any] = {"benchmarks": {}}
    if df.empty:
        return report

    needed_cols = {"benchmark", "task_id", "agent_id", "run_id", "f1", "em"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {sorted(needed_cols)}")

    grouped = df.groupby(["benchmark", "task_id"], dropna=False)
    for (benchmark, task_id), g in grouped:
        rows = g[["agent_id", "run_id", "agent_config_key", "f1", "em"]].dropna(
            subset=["agent_id", "run_id", "f1", "em"]
        ).to_dict("records")
        comparisons = []
        for left, right in itertools.permutations(rows, 2):
            lf1, lem = float(left["f1"]), float(left["em"])
            rf1, rem = float(right["f1"]), float(right["em"])
            if (lf1 > rf1) or (lf1 == rf1 and lem > rem):
                comparisons.append({
                    "winner": {"agent_id": left["agent_id"], "run_id": left["run_id"]},
                    "loser": {"agent_id": right["agent_id"], "run_id": right["run_id"]},
                    "winner_scores": {"f1": lf1, "em": lem},
                    "loser_scores": {"f1": rf1, "em": rem},
                })
        if not comparisons:
            continue
        bench_entry = report["benchmarks"].setdefault(str(benchmark), {"tasks": {}})
        bench_entry["tasks"][str(task_id)] = {"comparisons": comparisons}
    return report


def write_outperformance_yaml(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write per-datapoint outperformance report to YAML."""
    output_path = Path(output_path)
    report = build_outperformance_report(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(report, sort_keys=False))
    return output_path


def resolve_analysis_output_paths(
    analysis_base_dir: str | Path,
    analysis_run_name: str | None = None,
) -> dict[str, Path]:
    """Build default output paths under a run-scoped analysis directory."""
    analysis_base_dir = Path(analysis_base_dir)
    run_name = analysis_run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = analysis_base_dir / run_name
    return {
        "run_dir": run_dir,
        "full_results": run_dir / "full_results.csv",
        "benchmark_config_table": run_dir / "benchmark_config_table.csv",
        "outperformance_yaml": run_dir / "outperformance.yaml",
    }


def list_problem_dirs(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return task directories for failed and incomplete tasks."""
    if "task_dir" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'task_dir' column (use collect_results* functions)"
        )
    incomplete = sorted(df.loc[~df["completed"], "task_dir"].tolist())
    failed = sorted(df.loc[df["completed"] & (df["em"] == 0), "task_dir"].tolist())
    return {"incomplete": incomplete, "failed": failed}


# %% [markdown]
# ### CLi

# %%
from cyclopts import App, Parameter
from typing import Annotated, Any


app = App(help="Aggregate eval results across experiment configs into comparison tables.")


@app.default
def cli(
    run_dir: Annotated[
        list[str],
        Parameter(
            name=["-r", "--run-dir"],
            consume_multiple=True,
            negative=(),
            help=(
                "Glob patterns matching run directories (each containing run_metadata.json). "
                "Supports ** for recursive matching, e.g. 'logs/my_exp/**/run-*'. "
                "May be repeated for multiple patterns."
            ),
        ),
    ] = [],
    log_dir: Annotated[
        Path | None,
        Parameter(name=["-l", "--log-dir"], help="Single log base directory to scan."),
    ] = None,
    experiment_config: Annotated[
        list[Path],
        Parameter(
            name=["-c", "--experiment-config"],
            consume_multiple=True,
            negative=(),
            help="One or more experiment YAML files to load and aggregate.",
        ),
    ] = [],
    analysis_base_dir: Annotated[
        Path,
        Parameter(name="--analysis-base-dir", help="Base directory for analysis artifacts."),
    ] = Path("logs/analysis"),
    analysis_run_name: Annotated[
        str | None,
        Parameter(name="--analysis-run-name", help="Run-name subdirectory under analysis-base-dir."),
    ] = None,
    output: Annotated[
        Path | None,
        Parameter(name=["-o", "--output"], help="Optional CSV path to save full results (overrides default)."),
    ] = None,
    table_output: Annotated[
        Path | None,
        Parameter(name="--table-output", help="Optional CSV path to save benchmark/config table (overrides default)."),
    ] = None,
    outperformance_yaml: Annotated[
        Path | None,
        Parameter(name="--outperformance-yaml", help="Optional YAML path for outperformance report (overrides default)."),
    ] = None,
    by: Annotated[
        list[str],
        Parameter(
            name="--by",
            consume_multiple=True,
            negative=(),
            help="Columns to group by (default: benchmark run_id model version).",
        ),
    ] = [],
    verbose: Annotated[
        bool,
        Parameter(name=["-v", "--verbose"], help="Show all failed/incomplete directories (default: first 5 only)."),
    ] = False,
    openai_api_key: Annotated[
        str | None,
        Parameter(
            name="--openai-api-key",
            help=(
                "OpenAI API key for Oolong scoring. "
                "Falls back to the OPENAI_API_KEY environment variable (e.g. set via .envrc)."
            ),
        ),
    ] = None,
) -> None:
    """Print summaries and write analysis artifacts under a run directory."""
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import datasets as _hf_datasets
        _hf_datasets.logging.set_verbosity_error()
        _hf_datasets.disable_progress_bar()
    except ImportError:
        pass

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")

    if run_dir:
        df = asyncio.run(collect_results_from_run_dirs(run_dir, openai_api_key=api_key))
    elif experiment_config:
        df = asyncio.run(collect_results_from_experiment_configs(experiment_config, openai_api_key=api_key))
    elif log_dir is not None:
        df = asyncio.run(collect_results(log_dir, openai_api_key=api_key))
    else:
        raise ValueError("Pass --run-dir, --log-dir, or at least one --experiment-config")

    if df.empty:
        print("No finished results found.")
        return

    summary = summarize(df, by=by or None)
    print("\n=== Aggregate Summary ===")
    print(summary.to_string(index=False))

    comp = benchmark_config_table(df)
    print("\n=== Benchmark x Agent Comparison ===")
    print(comp.to_string(index=False))
    n_total = len(df)
    n_completed = int(df["f1"].notna().sum())
    n_succeeded = int((df["em"] == 1).sum())
    n_failed = n_completed - n_succeeded
    print(f"\n{n_total} total tasks ({n_completed} completed, {n_total - n_completed} incomplete, {n_succeeded} succeeded, {n_failed} failed)")

    problem_dirs = list_problem_dirs(df)
    _PREVIEW = 5
    for label, paths in [("Incomplete", problem_dirs["incomplete"]), ("Failed (em=0)", problem_dirs["failed"])]:
        if not paths:
            continue
        shown = paths if verbose else paths[:_PREVIEW]
        truncated = len(paths) - len(shown)
        print(f"\n=== {label} Task Directories ===")
        for p in shown:
            print(p)
        if truncated:
            print(f"  ... and {truncated} more (run with -v to see all)")

    # Stable ordering and schema for full-results artifact.
    sort_cols = [c for c in ["run_id", "task_id", "agent_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    ordered_cols = ["agent_id"] + [c for c in df.columns if c != "agent_id"]
    df = df[ordered_cols]

    paths = resolve_analysis_output_paths(analysis_base_dir, analysis_run_name)
    final_output = output or paths["full_results"]
    final_table_output = table_output or paths["benchmark_config_table"]
    final_outperformance_yaml = outperformance_yaml or paths["outperformance_yaml"]

    Path(final_output).parent.mkdir(parents=True, exist_ok=True)
    Path(final_table_output).parent.mkdir(parents=True, exist_ok=True)
    Path(final_outperformance_yaml).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(final_output, index=False)
    comp.to_csv(final_table_output, index=False)
    out_yaml_path = write_outperformance_yaml(df, final_outperformance_yaml)

    print("\n=== Analysis Artifacts ===")
    print(f"analysis_run_dir: {paths['run_dir']}")
    print(f"full_results_csv: {final_output}")
    print(f"benchmark_config_table_csv: {final_table_output}")
    print(f"outperformance_yaml: {out_yaml_path}")


@app.command
def tables(
    pattern: Annotated[
        str,
        Parameter(help="Regex matched against log-dir names under --root (e.g. 'qwen3_32b_.*')."),
    ],
    root: Annotated[
        Path,
        Parameter(name=["--root"], help="Directory to search for log_dirs (default: logs)."),
    ] = Path("logs"),
    openai_api_key: Annotated[
        str | None,
        Parameter(
            name="--openai-api-key",
            help="OpenAI API key for Oolong scoring. Falls back to OPENAI_API_KEY env var.",
        ),
    ] = None,
) -> None:
    """Print a compact results table for each log_dir whose name matches the regex."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    rx = re.compile(pattern)
    matched = sorted(d for d in root.iterdir() if d.is_dir() and rx.search(d.name))

    if not matched:
        print(f"No directories under '{root}' matched pattern '{pattern}'.")
        return

    for log_dir in matched:
        run_dirs = sorted(str(d) for d in log_dir.iterdir() if d.is_dir())
        if not run_dirs:
            print(f"\n=== {log_dir.name} — no run directories ===")
            continue

        df = asyncio.run(collect_results_from_run_dirs(run_dirs, openai_api_key=api_key))
        n_total = len(df)
        n_completed = int(df["f1"].notna().sum()) if not df.empty else 0
        print(f"\n=== {log_dir.name}  ({n_completed}/{n_total} completed) ===")

        if df.empty:
            print("  no finished results")
            continue

        summary = summarize(df)
        print(summary.to_string(index=False))


@app.command
def discover(
    log_dir: Annotated[
        Path,
        Parameter(name=["-l", "--log-dir"], help="Log base directory to scan for runs."),
    ],
) -> None:
    """List all runs found in a log directory with task completion counts."""
    df = discover_runs(log_dir)
    if df.empty:
        print(f"No runs found under {log_dir}")
        return
    print(df.to_string(index=False))


def main_cli() -> None:
    app()


# if __name__ == "__main__":
#     main_cli()

# %% [markdown]
# ## Tests

# %%
if test():
    import tempfile, json

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Simulate two agents on the same benchmark/task set across two distinct run_ids
        for run_id, cfg_path, task_id, f1, em, model in [
            ("run1", "configs/agents/a.yaml", "task_a", 0.8, 1.0, "model_a"),
            ("run1", "configs/agents/a.yaml", "task_b", 0.6, 0.0, "model_a"),
            ("run2", "configs/agents/b.yaml", "task_a", 0.7, 0.0, "model_b"),
            ("run2", "configs/agents/b.yaml", "task_b", 0.9, 1.0, "model_b"),
        ]:
            task_dir = base / "phantomwiki_50_1_0.0.3" / run_id / task_id
            task_dir.mkdir(parents=True)
            (task_dir / "qa.json").write_text(
                json.dumps({"f1": f1, "em": em, "task_id": task_id, "parsed_output": ["x"], "answer": "x"})
            )
            (task_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "argv": ["symbd-phantomwiki", "-c", cfg_path, "--set", f"task_id={task_id}"],
                        "config": {
                            "model": model,
                            "benchmark": "phantomwiki_50_1",
                            "version": "0.0.3" if "a.yaml" in cfg_path else "0.0.3_tiny",
                            "size": 50,
                            "seed": 1,
                        },
                    }
                )
            )

        df = asyncio.run(collect_results(base))
        assert len(df) == 4
        assert sorted(df["agent_config_key"].unique().tolist()) == ["configs/agents/a.yaml", "configs/agents/b.yaml"]

        # summarize now groups by run_id too — two runs → two rows
        summary = summarize(df)
        assert len(summary) == 2
        assert set(summary.columns) >= {"benchmark", "run_id", "model", "version", "mean_f1_all", "mean_em_all"}
        # all tasks complete in this fixture, so _all columns equal the completed-only columns
        for _, row in summary.iterrows():
            assert row["mean_f1_all"] == row["mean_f1"]
            assert row["mean_em_all"] == row["mean_em"]

        comp = benchmark_config_table(df)
        # run_id is now in grouping — still two distinct (run_id, agent_id) pairs → 2 rows
        assert len(comp) == 2
        assert sorted(comp["benchmark_rank"].unique().tolist()) == [1, 2]
        assert sorted(comp["agent_id"].tolist()) == ["0.0.3", "0.0.3_tiny"]

        report = build_outperformance_report(df)
        assert "phantomwiki_50_1" in report["benchmarks"]
        assert "task_a" in report["benchmarks"]["phantomwiki_50_1"]["tasks"]
        first_cmp = report["benchmarks"]["phantomwiki_50_1"]["tasks"]["task_a"]["comparisons"][0]
        assert "agent_id" in first_cmp["winner"]
        assert "run_id" in first_cmp["winner"]

        out_yaml = base / "outperformance.yaml"
        write_outperformance_yaml(df, out_yaml)
        assert out_yaml.exists()
        loaded = yaml.safe_load(out_yaml.read_text())
        assert "benchmarks" in loaded

        # experiment-config based aggregation
        exp_cfg = base / "exp1.yaml"
        exp_cfg.write_text(yaml.safe_dump({"log_base_dir": str(base)}))
        df_from_cfg = asyncio.run(collect_results_from_experiment_configs([exp_cfg]))
        assert len(df_from_cfg) == 4
        assert df_from_cfg["experiment_config"].iloc[0] == str(exp_cfg)

        # default analysis output layout under base/run
        p = resolve_analysis_output_paths(base / "analysis", "myrun")
        assert p["run_dir"] == base / "analysis" / "myrun"
        assert p["full_results"] == base / "analysis" / "myrun" / "full_results.csv"
        assert p["benchmark_config_table"] == base / "analysis" / "myrun" / "benchmark_config_table.csv"
        assert p["outperformance_yaml"] == base / "analysis" / "myrun" / "outperformance.yaml"

# %%
if test():
    import tempfile

    # empty log dir → empty DataFrame
    with tempfile.TemporaryDirectory() as tmp:
        df = asyncio.run(collect_results(tmp))
        assert df.empty

    # discover_runs: surfaces run dirs with/without run_metadata.json
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Create two run dirs — one with run_metadata.json, one without
        run_dir_a = base / "phantomwiki_50_1_0.0.3" / "run_a"
        run_dir_b = base / "phantomwiki_50_1_0.0.3" / "run_b"
        for d in (run_dir_a, run_dir_b):
            d.mkdir(parents=True)
            # simulate a finished task
            task_dir = d / "task_x"
            task_dir.mkdir()
            (task_dir / "qa.json").write_text(json.dumps({"f1": 0.5, "em": 0.0, "task_id": "task_x"}))

        (run_dir_a / "run_metadata.json").write_text(
            json.dumps({"run_id": "run_a", "experiment_config": {"version": "0.0.3", "model": "gpt-4"}})
        )

        disc = discover_runs(base)
        assert len(disc) == 2
        assert "run_id" in disc.columns
        assert "n_finished" in disc.columns
        # run_a has metadata, run_b doesn't — both rows present
        assert sorted(disc["run_id"].tolist()) == ["run_a", "run_b"]
        # n_finished should be 1 for both (one task dir with qa.json)
        assert disc[disc["run_id"] == "run_a"]["n_finished"].iloc[0] == 1
        # model lifted from metadata for run_a
        assert disc[disc["run_id"] == "run_a"]["model"].iloc[0] == "gpt-4"

    # empty dir
    with tempfile.TemporaryDirectory() as tmp:
        disc = discover_runs(tmp)
        assert disc.empty

# %%
if test():
    import tempfile

    # collect_results_from_run_dirs: reads config from run_metadata.json
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        run_dir = base / "phantomwiki_50_1_phantom_0.0.1" / "my-run"
        task_dir = run_dir / "task_x"
        task_dir.mkdir(parents=True)
        (task_dir / "qa.json").write_text(
            json.dumps({"f1": 0.8, "em": 1.0, "task_id": "task_x", "parsed_output": ["x"], "answer": "x"})
        )
        # No per-task metadata.json — config comes entirely from run_metadata.json
        (run_dir / "run_metadata.json").write_text(json.dumps({
            "run_id": "my-run",
            "benchmark_version": "phantomwiki_50_1_phantom_0.0.1",
            "experiment_config": {
                "benchmark": "phantomwiki",
                "benchmark_extra_params": {"size": 50, "seed": 1},
                "version": "phantom_0.0.1",
                "model": "Qwen/Qwen3-32B",
                "agent_configs": ["configs/agents/phantomwiki.yaml"],
            },
        }))

        # Match via explicit path
        df = asyncio.run(collect_results_from_run_dirs([str(run_dir)]))
        assert len(df) == 1
        assert df["model"].iloc[0] == "Qwen/Qwen3-32B"
        assert df["version"].iloc[0] == "phantom_0.0.1"
        assert df["benchmark"].iloc[0] == "phantomwiki"
        assert df["size"].iloc[0] == 50
        assert df["agent_config_key"].iloc[0] == "configs/agents/phantomwiki.yaml"

        # Match via glob pattern
        df2 = asyncio.run(collect_results_from_run_dirs([str(base / "*" / "*")]))
        assert len(df2) == 1

        # Non-matching pattern → empty
        df3 = asyncio.run(collect_results_from_run_dirs([str(base / "nonexistent" / "*")]))
        assert df3.empty 

# %%
# ! poe sync
