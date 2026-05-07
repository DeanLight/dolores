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
# # Eval Experiment Runner
#
# Orchestrates parallel execution of benchmark tasks across agents.

# %%

import json
import logging
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import structlog
from cyclopts import App, Parameter
from pydantic import BaseModel, ConfigDict, Field

from deep_reasoner.cli_utils import load_settings
from deep_reasoner.core import checkLogs, configure_structlog_fixture
from juplit import test

from vllm_utils import (
    VllmConfig,
    find_free_port,
    start_vllm,
    stop_vllm,
    wait_for_vllm,
)

logger = structlog.get_logger(__name__)

# %% [markdown]
# ## Config models

# %%
class ServerConfig(BaseModel):
    """Pointer to an existing OpenAI-compatible inference server."""
    base_url: str
    api_key_env: str = "OPENAI_API_KEY"


class AgentRunConfig(BaseModel):
    """One agent configuration: the YAML files to layer and eval bookkeeping fields."""
    configs: list[str]
    version: str = "0"
    extra_params: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    """One benchmark slice to evaluate."""
    name: str
    key_limit: int | list[str] | None = None
    key_limit_seed: int = 0
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level experiment config loaded from a main config YAML."""
    model_config = ConfigDict(extra="allow")

    log_dir: str = "logs"
    batch_size: int = 32
    task_timeout: int = 600
    vllm_wait_timeout: int = 600
    vllm: VllmConfig | None = None
    server: ServerConfig | None = None
    agent: AgentRunConfig
    benchmark: BenchmarkConfig
    resume: bool = False

# %%
if test():
    # Pydantic validation
    cfg = ExperimentConfig(
        agent=AgentRunConfig(configs=["a.yaml"]),
        benchmark=BenchmarkConfig(name="phantomwiki", extra_params={"size": 50, "seed": 1}),
    )
    assert cfg.batch_size == 32
    assert cfg.agent.version == "0"
    assert cfg.benchmark.name == "phantomwiki"

    vllm = VllmConfig(model="Qwen/Qwen3-32B", port=8000)
    assert vllm.gpu_memory_utilization == 0.9

# %% [markdown]
# ## Task enumeration helpers

# %%
def _resolve_benchmark_name(name: str) -> str:
    """Normalize benchmark naming variants to a canonical name."""
    return name.strip().lower().replace("_", "")


def _apply_key_limit(
    all_ids: list[str], key_limit: int | list[str] | None, key_limit_seed: int
) -> list[str]:
    """Apply optional key limiting by count or explicit task id list."""
    if key_limit is None:
        return all_ids
    if isinstance(key_limit, int):
        if key_limit <= 0:
            return []
        if key_limit >= len(all_ids):
            return all_ids
        sampler = random.Random(key_limit_seed)
        return sampler.sample(all_ids, key_limit)
    if isinstance(key_limit, list):
        return list(key_limit)
    raise ValueError("Invalid key_limit value. Expected None, int, or list[str].")

# %%
if test():
    assert _resolve_benchmark_name("PhantomWiki") == "phantomwiki"
    assert _resolve_benchmark_name("hello_world") == "helloworld"

    ids = [str(i) for i in range(10)]
    assert _apply_key_limit(ids, None, 0) == ids
    assert len(_apply_key_limit(ids, 3, 0)) == 3
    assert _apply_key_limit(ids, 0, 0) == []
    assert _apply_key_limit(ids, 100, 0) == ids
    assert _apply_key_limit(ids, ["a", "b"], 0) == ["a", "b"]

# %%
def list_all_task_ids(bench: BenchmarkConfig) -> list[str]:
    """Resolve all task ids for a benchmark slice (before key_limit)."""
    from benchmarks import hello_world, oolong, phantomwiki  # external package

    canonical_name = _resolve_benchmark_name(bench.name)

    if canonical_name == "phantomwiki":
        size = bench.extra_params.get("size")
        seed = bench.extra_params.get("seed")
        if size is None or seed is None:
            raise ValueError(
                "phantomwiki benchmark requires extra_params.size and extra_params.seed"
            )
        return phantomwiki.list_test_ids(size, seed)

    if canonical_name == "helloworld":
        seed = bench.extra_params.get("seed")
        if seed is None:
            raise ValueError("helloworld benchmark requires extra_params.seed")
        return hello_world.list_test_ids(seed)

    if canonical_name == "oolong":
        limit = bench.extra_params.get("limit", 500)
        seed = bench.extra_params.get("seed", 42)
        return oolong.list_test_ids(limit=limit, seed=seed)

    if canonical_name == "synthworld":
        from benchmarks import synthworlds
        return synthworlds.list_test_ids()

    if canonical_name == "deepsearchqa":
        from benchmarks import deepresearchqa
        return deepresearchqa.list_test_ids(limit=None, seed=42)

    raise ValueError(f"Unknown benchmark: {bench.name!r}. Add support in list_all_task_ids().")


def get_task_ids(bench: BenchmarkConfig) -> list[str]:
    """Resolve the key-limited task list for a benchmark config slice."""
    return _apply_key_limit(list_all_task_ids(bench), bench.key_limit, bench.key_limit_seed)


def task_is_finished(log_base_dir: str | Path, task: dict) -> bool:
    """Return True if the task already has a completed qa.json (answer is not null)."""
    task_dir = Path(log_base_dir) / task["task_id"]
    qa_path = task_dir / "qa.json"
    if not qa_path.exists():
        return False
    try:
        return json.loads(qa_path.read_text()).get("answer") is not None
    except Exception:
        return False


def enumerate_tasks(exp_cfg: ExperimentConfig) -> list[dict]:
    """Return all (bench, task_id) combinations to execute."""
    return [
        {"bench": exp_cfg.benchmark, "task_id": task_id}
        for task_id in get_task_ids(exp_cfg.benchmark)
    ]


def plan_task_execution(exp_cfg: ExperimentConfig) -> tuple[list[dict], dict[str, int]]:
    """Plan tasks and compute task-state counts used by main and dry-run."""
    all_task_ids = list_all_task_ids(exp_cfg.benchmark)
    selected_task_ids = get_task_ids(exp_cfg.benchmark)
    selected_set = set(selected_task_ids)

    n_finished = 0
    n_incomplete = 0
    n_not_started = 0
    tasks_to_run: list[dict] = []

    for task_id in selected_task_ids:
        task = {"bench": exp_cfg.benchmark, "task_id": task_id}
        task_dir = Path(exp_cfg.log_dir) / task_id
        if task_is_finished(exp_cfg.log_dir, task):
            n_finished += 1
            if not exp_cfg.resume:
                tasks_to_run.append(task)
        elif task_dir.exists():
            n_incomplete += 1
            tasks_to_run.append(task)
        else:
            n_not_started += 1
            tasks_to_run.append(task)

    n_not_started_excluded_by_key_limit = 0
    for task_id in all_task_ids:
        if task_id in selected_set:
            continue
        task_dir = Path(exp_cfg.log_dir) / task_id
        if not task_dir.exists():
            n_not_started_excluded_by_key_limit += 1

    stats = {
        "n_selected": len(selected_task_ids),
        "n_total_unlimited": len(all_task_ids),
        "n_found_already_run": n_finished,
        "n_found_incomplete_rerun": n_incomplete,
        "n_found_not_started_run": n_not_started,
        "n_not_started_excluded_by_key_limit": n_not_started_excluded_by_key_limit,
        "n_actual_run": len(tasks_to_run),
    }
    return tasks_to_run, stats


def log_task_list(tasks: list[dict], path: Path) -> None:
    """Persist the full task list to disk before any execution starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "task_id": t["task_id"],
            "bench": t["bench"].model_dump(),
        }
        for t in tasks
    ]
    path.write_text(json.dumps(serializable, indent=2))


def save_run_metadata(
    log_base_dir: str | Path,
    exp_cfg: ExperimentConfig,
    n_tasks: int | None = None,
) -> Path:
    """Write run_metadata.json into {log_base_dir}/."""
    run_dir = Path(log_base_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_tasks": n_tasks,
        "experiment_config": {
            "benchmark": exp_cfg.benchmark.name,
            "agent_configs": exp_cfg.agent.configs,
            "version": exp_cfg.agent.version,
            "model": (exp_cfg.vllm.model if exp_cfg.vllm else None),
            "server_base_url": (exp_cfg.server.base_url if exp_cfg.server else None),
            "batch_size": exp_cfg.batch_size,
            "task_timeout": exp_cfg.task_timeout,
        },
    }
    out = run_dir / "run_metadata.json"
    out.write_text(json.dumps(meta, indent=2))
    return out

# %% [markdown]
# ## Subprocess helpers

# %%
def _benchmark_cli_cmd(bench_name: str) -> list[str]:
    """Return the python -m invocation for the given benchmark CLI."""
    canonical_name = _resolve_benchmark_name(bench_name)
    _base = "dolores"
    mapping = {
        "phantomwiki": ["-m", f"{_base}.phantomwiki_cli"],
        "helloworld": ["-m", f"{_base}.helloworld_cli"],
        "oolong":     ["-m", f"{_base}.oolong_cli"],
        "synthworld":  ["-m", f"{_base}.synthworld_cli"],
        "deepsearchqa": ["-m", f"{_base}.deepsearchqa_cli"],
    }
    try:
        return mapping[canonical_name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark for CLI dispatch: {bench_name!r}.") from exc


def build_cli_command(task: dict, exp_cfg: ExperimentConfig) -> list[str]:
    """Build the benchmark CLI subprocess command for one task."""
    agent = exp_cfg.agent
    bench = task["bench"]

    cmd = [sys.executable] + _benchmark_cli_cmd(bench.name)
    if len(agent.configs) != 1:
        raise ValueError(
            f"agent.configs must contain exactly one main config path, got {len(agent.configs)}"
        )
    cmd.append(agent.configs[0])

    set_args = [
        f"task_id={task['task_id']}",
        f"version={agent.version}",
        f"log_dir={exp_cfg.log_dir}",
    ]
    for key, value in bench.extra_params.items():
        set_args.append(f"{key}={value}")
    for key, value in agent.extra_params.items():
        set_args.append(f"{key}={value}")

    if exp_cfg.vllm:
        # Use the dynamic port assigned at runtime, not any hardcoded server.base_url.
        api_key_env = exp_cfg.server.api_key_env if exp_cfg.server else "VLLM_API_KEY"
        set_args += [
            f"client.base_url=http://localhost:{exp_cfg.vllm.port}/v1",
            f"client.api_key_env={api_key_env}",
        ]
    elif exp_cfg.server:
        set_args += [
            f"client.base_url={exp_cfg.server.base_url}",
            f"client.api_key_env={exp_cfg.server.api_key_env}",
        ]

    if exp_cfg.vllm:
        set_args.append(f"model={exp_cfg.vllm.model}")
        set_args.append(f"search_api_base=http://localhost:{exp_cfg.vllm.port}/v1")

    cmd.extend(["--set"] + set_args)
    return cmd


def run_with_crash_handling(cmd: list[str], timeout: int = 600) -> dict:
    """Run a CLI command in a subprocess and return a status dict."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stderr_tail = (result.stderr or "")[-2000:]
        stdout_tail = (result.stdout or "")[-2000:]
        if result.returncode == 2:
            return {
                "status": "failed",
                "cmd": cmd,
                "returncode": result.returncode,
                "stderr": stderr_tail,
                "stdout": stdout_tail,
            }
        if result.returncode != 0:
            return {
                "status": "failed",
                "cmd": cmd,
                "returncode": result.returncode,
                "stderr": stderr_tail,
                "stdout": stdout_tail,
            }
        return {"status": "ok", "cmd": cmd}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "cmd": cmd}
    except Exception as exc:
        return {"status": "error", "cmd": cmd, "error": str(exc)}



# %%
if test():
    import sys

    agent = AgentRunConfig(configs=["model-configs/debug.yaml"], version="0.0.3")
    bench = BenchmarkConfig(
        name="phantomwiki",
        key_limit=None,
        extra_params={"size": 50, "seed": 1},
    )
    exp = ExperimentConfig(
        agent=agent,
        benchmark=bench,
        server=ServerConfig(base_url="https://api.example.com/v1", api_key_env="MY_KEY"),
    )
    task = {"bench": bench, "task_id": "abc123"}
    cmd = build_cli_command(task, exp)

    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "dolores.phantomwiki_cli"]
    assert cmd[3] == "model-configs/debug.yaml"
    assert "--set" in cmd
    set_idx = cmd.index("--set")
    set_part = cmd[set_idx + 1 :]
    assert any(s.startswith("task_id=") for s in set_part)
    assert any(s == "size=50" for s in set_part)
    assert any(s == "seed=1" for s in set_part)
    assert any(s.startswith("client.base_url=") for s in set_part)

    bench_hw = BenchmarkConfig(
        name="hello_world",
        key_limit=1,
        extra_params={"seed": 0},
    )
    exp_hw = ExperimentConfig(agent=agent, benchmark=bench_hw)
    task_hw = {"bench": bench_hw, "task_id": "fib_3"}
    cmd_hw = build_cli_command(task_hw, exp_hw)
    assert cmd_hw[0] == sys.executable
    assert cmd_hw[1:3] == ["-m", "dolores.helloworld_cli"]
    assert any(s == "seed=0" for s in cmd_hw)

    py = sys.executable
    r = run_with_crash_handling([py, "-c", "print('ok')"], timeout=10)
    assert r["status"] == "ok"

    r = run_with_crash_handling([py, "-c", "raise SystemExit(1)"], timeout=10)
    assert r["status"] == "failed"

    r = run_with_crash_handling([py, "-c", "raise SystemExit(2)"], timeout=10)
    assert r["status"] == "failed"

    r = run_with_crash_handling([py, "-c", "import time; time.sleep(5)"], timeout=1)
    assert r["status"] == "timeout"

# %% [markdown]
# ## Main orchestration

# %%
def main(exp_cfg: ExperimentConfig) -> None:
    tasks, plan_stats = plan_task_execution(exp_cfg)

    logger.info(
        "eval_experiment.plan",
        resume=exp_cfg.resume,
        n_selected=plan_stats["n_selected"],
        n_total_unlimited=plan_stats["n_total_unlimited"],
        n_found_already_run=plan_stats["n_found_already_run"],
        n_found_incomplete_rerun=plan_stats["n_found_incomplete_rerun"],
        n_found_not_started_run=plan_stats["n_found_not_started_run"],
        n_not_started_excluded_by_key_limit=plan_stats["n_not_started_excluded_by_key_limit"],
        n_actual_run=plan_stats["n_actual_run"],
    )

    # Clean up incomplete task directories before rerunning those tasks.
    for t in tasks:
        task_dir = Path(exp_cfg.log_dir) / t["task_id"]
        if task_dir.exists() and not task_is_finished(exp_cfg.log_dir, t):
            shutil.rmtree(task_dir)

    n_tasks = len(tasks)
    for i, task in enumerate(tasks, 1):
        task["task_num"] = i

    task_list_path = Path(exp_cfg.log_dir) / "task_list.json"
    log_task_list(tasks, task_list_path)

    save_run_metadata(exp_cfg.log_dir, exp_cfg, n_tasks=plan_stats["n_selected"])

    logger.info(
        "eval_experiment.scheduled",
        n_tasks=n_tasks,
        log_dir=exp_cfg.log_dir,
        batch_size=exp_cfg.batch_size,
    )

    experiment_t0 = time.monotonic()
    vllm_proc = None
    if exp_cfg.vllm:
        exp_cfg.vllm.port = find_free_port()
        vllm_proc = start_vllm(exp_cfg.vllm)
        wait_for_vllm(exp_cfg.vllm.port, timeout=exp_cfg.vllm_wait_timeout)

    def _run_one(task: dict) -> dict:
        bench = task["bench"]
        logger.info(
            "eval_experiment.task_start",
            task_id=task["task_id"],
            task_num=task["task_num"],
            n_tasks=n_tasks,
            benchmark=bench.name,
        )
        t0 = time.monotonic()
        try:
            cmd = build_cli_command(task, exp_cfg)
            result = run_with_crash_handling(cmd, exp_cfg.task_timeout)
        except Exception as exc:
            result = {"status": "error", "cmd": None, "error": f"task setup failed: {exc}"}
        duration_s = round(time.monotonic() - t0, 2)
        kwargs = dict(
            task_id=task["task_id"],
            task_num=task["task_num"],
            n_tasks=n_tasks,
            benchmark=bench.name,
            status=result["status"],
            duration_s=duration_s,
        )
        if result["status"] in {"failed", "error", "timeout"}:
            if result.get("stderr"):
                kwargs["stderr"] = result["stderr"]
            if result.get("stdout"):
                kwargs["stdout"] = result["stdout"]
            if result.get("error"):
                kwargs["error"] = result["error"]
            logger.error("eval_experiment.task_done", **kwargs)
        else:
            logger.info("eval_experiment.task_done", **kwargs)
        return result

    results = []
    unexpected_error = None
    try:
        with ThreadPoolExecutor(max_workers=exp_cfg.batch_size) as pool:
            futures = {pool.submit(_run_one, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"status": "error", "cmd": None, "error": str(exc)})
    except Exception as exc:
        unexpected_error = str(exc)
        logger.exception("eval_experiment.run_error")
    finally:
        if vllm_proc is not None:
            stop_vllm(vllm_proc)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_incomplete = sum(1 for r in results if r["status"] in {"failed", "error", "timeout"})
    status = "failed" if (unexpected_error or n_incomplete) else "ok"
    log_fn = logger.error if status == "failed" else logger.info
    log_fn("eval_experiment.complete", status=status, n_done=n_ok, n_incomplete=n_incomplete,
           duration_s=round(time.monotonic() - experiment_t0, 2))

# %%
if test():
    import tempfile

    agent = AgentRunConfig(configs=["model-configs/debug.yaml"], version="0.0.3")
    bench = BenchmarkConfig(
        name="phantomwiki",
        key_limit=None,
        extra_params={"size": 50, "seed": 1},
    )
    exp = ExperimentConfig(
        agent=agent,
        benchmark=bench,
        server=ServerConfig(base_url="https://api.example.com/v1", api_key_env="MY_KEY"),
    )
    task = {"bench": bench, "task_id": "abc123"}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "task_list.json"
        log_task_list([task], path)
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["task_id"] == "abc123"
        assert "run_id" not in data[0]

# %%
if test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        exp = ExperimentConfig(
            agent=AgentRunConfig(configs=["cfg/agent.yaml"], version="0.1"),
            benchmark=BenchmarkConfig(name="phantomwiki", extra_params={"size": 50, "seed": 1}),
            server=ServerConfig(base_url="https://api.example.com/v1", api_key_env="MY_KEY"),
        )
        out = save_run_metadata(tmp, exp)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "run_id" not in data
        assert data["experiment_config"]["version"] == "0.1"
        assert data["experiment_config"]["agent_configs"] == ["cfg/agent.yaml"]

# %%
if test():
    import sys
    import tempfile

    class _FakeLogger:
        def __init__(self):
            self.events = []

        def info(self, event, **kwargs):
            self.events.append(("info", event, kwargs))

        def error(self, event, **kwargs):
            self.events.append(("error", event, kwargs))

        def exception(self, event, **kwargs):
            self.events.append(("exception", event, kwargs))

    orig_logger = logger
    orig_build_cli_command = build_cli_command
    orig_run_with_crash_handling = run_with_crash_handling
    orig_plan_task_execution = plan_task_execution

    try:
        fake_logger = _FakeLogger()
        logger = fake_logger

        def _fake_plan_task_execution(exp_cfg):
            bench = exp_cfg.benchmark
            return (
                [{"bench": bench, "task_id": "fib_3"}],
                {
                    "n_selected": 1,
                    "n_total_unlimited": 1,
                    "n_found_already_run": 0,
                    "n_found_incomplete_rerun": 0,
                    "n_found_not_started_run": 1,
                    "n_not_started_excluded_by_key_limit": 0,
                    "n_actual_run": 1,
                },
            )

        def _fake_build_cli_command(task, exp_cfg):
            return [sys.executable, "-c", "print('x')"]

        def _fake_run_with_crash_handling(cmd, timeout=600):
            return {
                "status": "failed",
                "cmd": cmd,
                "returncode": 1,
                "stderr": "boom",
            }

        plan_task_execution = _fake_plan_task_execution
        build_cli_command = _fake_build_cli_command
        run_with_crash_handling = _fake_run_with_crash_handling

        with tempfile.TemporaryDirectory() as tmp:
            exp_fail = ExperimentConfig(
                batch_size=1,
                task_timeout=3,
                agent=AgentRunConfig(configs=["model-configs/debug.yaml"]),
                benchmark=BenchmarkConfig(name="hello_world", key_limit=1, extra_params={"seed": 0}),
            )
            main(exp_fail)

        complete_events = [
            (level, kwargs)
            for level, event, kwargs in fake_logger.events
            if event == "eval_experiment.complete"
        ]
        assert len(complete_events) == 1
        level, complete = complete_events[0]
        assert level == "error"
        assert complete["status"] == "failed"
        assert complete["n_done"] == 0
        assert complete["n_incomplete"] == 1
    finally:
        logger = orig_logger
        build_cli_command = orig_build_cli_command
        run_with_crash_handling = orig_run_with_crash_handling
        plan_task_execution = orig_plan_task_execution

# %% [markdown]
# ## limit_run

# %%
def limit_run(
    source_run_dir: Path | str,
    key_limit: int | list[str] | None,
    key_limit_seed: int = 0,
    new_run_id: str | None = None,
    output_log_base_dir: str | Path | None = None,
) -> str:
    """Create a new run by subsetting task results from an existing run directory."""
    source_run_dir = Path(source_run_dir)
    bench_version_dir = source_run_dir.parent

    all_task_ids = sorted(
        p.name for p in source_run_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    selected_ids = _apply_key_limit(all_task_ids, key_limit, key_limit_seed)

    if new_run_id is None:
        from deep_reasoner.logging_utils import generate_run_id
        new_run_id = generate_run_id()

    if output_log_base_dir is not None:
        out_bench_version_dir = Path(output_log_base_dir) / bench_version_dir.name
    else:
        out_bench_version_dir = bench_version_dir

    new_run_dir = out_bench_version_dir / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    source_meta_path = source_run_dir / "run_metadata.json"
    meta: dict = {}
    if source_meta_path.exists():
        meta = json.loads(source_meta_path.read_text())
    meta.update({
        "run_id": new_run_id,
        "benchmark_version": bench_version_dir.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source_run_id": source_run_dir.name,
        "source_run_dir": str(source_run_dir.resolve()),
        "key_limit": key_limit,
        "key_limit_seed": key_limit_seed,
        "n_tasks_selected": len(selected_ids),
        "n_tasks_source": len(all_task_ids),
    })
    (new_run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    for task_id in selected_ids:
        src = (source_run_dir / task_id).resolve()
        dst = new_run_dir / task_id
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    logger.info(
        "eval_experiment.limit_run_done",
        new_run_id=new_run_id,
        n_selected=len(selected_ids),
        n_total=len(all_task_ids),
    )
    return new_run_id


# %% [markdown]
# ## CLI
#
# ```
# symbd-eval-experiment -c experiments/pw_50.yaml
# ```

# %%
def _lower_keys(obj):
    if isinstance(obj, dict):
        return {k.lower(): _lower_keys(v) for k, v in obj.items()}
    return obj


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base in-place. Dicts are merged; all other types replace."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


_MAIN_CONFIG_SKIP_KEYS = {"_compose", "description"}


def load_main_config(main_path: Path) -> "Dynaconf":
    """Load a main config YAML, resolve _compose layers, apply inline overrides.

    Order: _compose files are loaded first (later overrides earlier), then
    any inline keys in the main config (excluding _compose/description) are
    merged on top via settings.update — so the main config always wins.

    Compose files are deep-merged so nested dicts (e.g. vllm) accumulate keys
    across files rather than the last file winning the whole dict.
    """
    import yaml

    from dynaconf import Dynaconf

    with open(main_path) as f:
        main_cfg = yaml.safe_load(f) or {}

    compose_paths = [Path(p) for p in main_cfg.get("_compose", [])]
    overrides = {k: v for k, v in main_cfg.items() if k not in _MAIN_CONFIG_SKIP_KEYS}

    merged: dict = {}
    for path in compose_paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    _deep_merge(merged, overrides)

    settings = Dynaconf(environments=False, load_dotenv=True)
    if merged:
        settings.update(merged)
    return settings


app = App(help="Run a batch of eval tasks from a main config YAML.")


@app.default
def cli(
    main_config: Annotated[
        Path,
        Parameter(help="Main experiment config YAML (with _compose list)."),
    ],
    resume: Annotated[
        bool,
        Parameter(name="--resume", negative="--no-resume", help="Skip tasks that already have a finished qa.json (answer is not null)."),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(
            name="--dry-run",
            negative="--no-dry-run",
            help="Pretty-print merged config and planned task-state counts, then exit.",
        ),
    ] = False,
) -> None:
    """Load a main experiment config and run all tasks."""
    from deep_reasoner.core import pretty_print_config

    configure_structlog_fixture(console=True, default_level=logging.INFO)
    try:
        settings = load_main_config(main_config)
        exp_cfg = ExperimentConfig.model_validate(_lower_keys(settings.as_dict()))
        if resume:
            exp_cfg.resume = True
        if dry_run:
            pretty_print_config(_lower_keys(settings.as_dict()), title=str(main_config))
            _, plan_stats = plan_task_execution(exp_cfg)
            logger.info(
                "eval_experiment.dry_run_plan",
                resume=exp_cfg.resume,
                n_selected=plan_stats["n_selected"],
                n_total_unlimited=plan_stats["n_total_unlimited"],
                n_found_already_run=plan_stats["n_found_already_run"],
                n_found_incomplete_rerun=plan_stats["n_found_incomplete_rerun"],
                n_found_not_started_run=plan_stats["n_found_not_started_run"],
                n_not_started_excluded_by_key_limit=plan_stats["n_not_started_excluded_by_key_limit"],
                n_actual_run=plan_stats["n_actual_run"],
            )
            return
        logger.info("eval_experiment.config_loaded", main_config=str(main_config.resolve()))
        main(exp_cfg)
    except SystemExit:
        raise
    except Exception:
        logger.exception("eval_experiment.cli_error")
        raise SystemExit(1)


@app.command
def preview(
    *configs: Annotated[
        Path,
        Parameter(help="Main config YAML files to preview."),
    ],
) -> None:
    """Pretty-print the first config fully, then show diffs for subsequent ones."""
    from deep_reasoner.core import diff_configs, pretty_print_config, pretty_print_config_diff

    if not configs:
        print("No configs provided.")
        return

    loaded = []
    for path in configs:
        settings = load_main_config(path)
        loaded.append((path, _lower_keys(settings.as_dict())))

    base_path, base_cfg = loaded[0]
    pretty_print_config(base_cfg, title=str(base_path))

    for path, cfg in loaded[1:]:
        diff = diff_configs(base_cfg, cfg)
        pretty_print_config_diff(diff, title=f"{path} (diff from {base_path.name})")


def _update_task_state(states: dict[str, str], tid: str, task_dir: Path) -> None:
    if states.get(tid) == "done":
        return
    qa_path = task_dir / "qa.json"
    try:
        if qa_path.exists() and json.loads(qa_path.read_text()).get("answer") is not None:
            states[tid] = "done"
        else:
            states.setdefault(tid, "incomplete")
    except Exception:
        states.setdefault(tid, "incomplete")


def _scan_task_states(log_dir: Path, task_ids: list[str]) -> dict[str, str]:
    """Scan log_dir and return {task_id: 'done'|'incomplete'} for seen tasks.

    Handles both flat layout (<log_dir>/<task_id>/) and run-level layout
    (<log_dir>/<run_id>/<task_id>/) by checking whether each immediate subdir
    name is a known task_id.
    """
    states: dict[str, str] = {}
    if not log_dir.is_dir():
        return states
    task_set = set(task_ids)
    for subdir in log_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name in task_set:
            # Flat layout: log_dir/<task_id>/
            _update_task_state(states, subdir.name, subdir)
        else:
            # Run-level layout: log_dir/<run_id>/<task_id>/
            for task_dir in subdir.iterdir():
                if task_dir.is_dir() and task_dir.name in task_set:
                    _update_task_state(states, task_dir.name, task_dir)
    return states


@app.command
def plan(
    *main_configs: Annotated[
        Path,
        Parameter(help="Main experiment config YAML files."),
    ],
) -> None:
    """Show task-state counts for one or more experiment configs as a table."""
    import logging as _logging

    for _noisy in ("datasets", "huggingface_hub", "filelock"):
        _logging.getLogger(_noisy).setLevel(_logging.ERROR)
    configure_structlog_fixture(console=False, default_level=_logging.ERROR)

    _HEADERS = ["config", "selected", "total", "done", "incomplete", "not_started", "excluded", "to_run"]

    rows: list[list] = []
    for cfg_path in main_configs:
        try:
            settings = load_main_config(cfg_path)
            exp_cfg = ExperimentConfig.model_validate(_lower_keys(settings.as_dict()))

            all_task_ids = list_all_task_ids(exp_cfg.benchmark)
            selected_task_ids = get_task_ids(exp_cfg.benchmark)
            selected_set = set(selected_task_ids)

            states = _scan_task_states(Path(exp_cfg.log_dir), selected_task_ids)
            n_done = sum(1 for s in states.values() if s == "done")
            n_incomplete = sum(1 for s in states.values() if s == "incomplete")
            n_not_started = len(selected_task_ids) - len(states)
            n_excluded = sum(1 for tid in all_task_ids if tid not in selected_set)
            n_to_run = n_incomplete + n_not_started if exp_cfg.resume else len(selected_task_ids)

            rows.append([
                cfg_path.name,
                len(selected_task_ids), len(all_task_ids),
                n_done, n_incomplete, n_not_started, n_excluded, n_to_run,
            ])
        except Exception as exc:
            rows.append([cfg_path.name, f"ERROR: {exc}"] + [""] * (len(_HEADERS) - 2))

    col_widths = [max(len(str(r[i])) for r in [_HEADERS] + rows) for i in range(len(_HEADERS))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*_HEADERS))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        app()


if __name__ == "__main__":
    main_cli()

# %% [markdown]
# ## Limit Run — subset an existing run
#
# `symbd-limit-run` creates a new run by subsetting the task results of an existing run directory using the same `key_limit`/`key_limit_seed` sampling logic as the experiment runner. Selected task directories are symlinked (no data copied), making comparisons with limited runs fair.
#
# ```bash
# symbd-limit-run -s logs/my_exp/phantomwiki_50_1_0.0.3/blue-pigeon -k 20 --key-limit-seed 42
# ```

# %%
def limit_run(
    source_run_dir: Path | str,
    key_limit: int | list[str] | None,
    key_limit_seed: int = 0,
    new_run_id: str | None = None,
    output_log_base_dir: str | Path | None = None,
) -> str:
    """Create a new run by subsetting task results from an existing run directory.

    Discovers all task_id subdirectories in ``source_run_dir``, applies the same
    ``key_limit``/``key_limit_seed`` sampling logic used during experiment runs,
    creates a new run directory, and symlinks the selected task directories into it.
    No task data is copied — only the directory structure and ``run_metadata.json``
    are written.

    Returns the new run_id.
    """
    source_run_dir = Path(source_run_dir)
    bench_version_dir = source_run_dir.parent  # {log_base_dir}/{bench_version}

    # Discover task_ids: every non-hidden subdirectory is a task
    all_task_ids = sorted(
        p.name for p in source_run_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    selected_ids = _apply_key_limit(all_task_ids, key_limit, key_limit_seed)

    if new_run_id is None:
        from deep_reasoner.logging_utils import generate_run_id
        new_run_id = generate_run_id()

    if output_log_base_dir is not None:
        out_bench_version_dir = Path(output_log_base_dir) / bench_version_dir.name
    else:
        out_bench_version_dir = bench_version_dir

    new_run_dir = out_bench_version_dir / new_run_id
    new_run_dir.mkdir(parents=True, exist_ok=True)

    # Build run_metadata.json — extend source metadata so analysis tools can read it
    source_meta_path = source_run_dir / "run_metadata.json"
    meta: dict = {}
    if source_meta_path.exists():
        meta = json.loads(source_meta_path.read_text())
    meta.update({
        "run_id": new_run_id,
        "benchmark_version": bench_version_dir.name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source_run_id": source_run_dir.name,
        "source_run_dir": str(source_run_dir.resolve()),
        "key_limit": key_limit,
        "key_limit_seed": key_limit_seed,
        "n_tasks_selected": len(selected_ids),
        "n_tasks_source": len(all_task_ids),
    })
    (new_run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # Symlink each selected task directory into the new run
    for task_id in selected_ids:
        src = (source_run_dir / task_id).resolve()
        dst = new_run_dir / task_id
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    logger.info(
        "eval_experiment.limit_run_done",
        new_run_id=new_run_id,
        new_run_dir=str(new_run_dir),
        source_run_id=source_run_dir.name,
        n_selected=len(selected_ids),
        n_total=len(all_task_ids),
    )
    return new_run_id


#| eval: false
limit_run_app = App(help="Create a limited subset run from an existing run's log directory.")


@limit_run_app.default
def limit_run_cli(
    source_run_dir: Annotated[
        Path,
        Parameter(name=["-s", "--source-run-dir"], help="Path to the source run directory (the {bench_version}/{run_id} dir)."),
    ],
    key_limit: Annotated[
        int,
        Parameter(name=["-k", "--key-limit"], help="Number of tasks to randomly sample from the source run."),
    ],
    key_limit_seed: Annotated[
        int,
        Parameter(name=["--key-limit-seed"], help="Random seed for task sampling (default 0)."),
    ] = 0,
    run_id: Annotated[
        str | None,
        Parameter(name=["--run-id"], help="New run ID (auto-generated if omitted)."),
    ] = None,
    output_log_base_dir: Annotated[
        str | None,
        Parameter(name=["--output-log-base-dir"], help="Output log base dir (defaults to same parent as source run)."),
    ] = None,
) -> None:
    """Subset an existing run by sampling tasks with key_limit/key_limit_seed logic."""
    configure_structlog_fixture(console=True, default_level=logging.INFO)
    limit_run(
        source_run_dir=source_run_dir,
        key_limit=key_limit,
        key_limit_seed=key_limit_seed,
        new_run_id=run_id,
        output_log_base_dir=output_log_base_dir,
    )


def limit_run_main_cli() -> None:
    with checkLogs(namespace="__main__", level=logging.INFO):
        limit_run_app()


# %%
if test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        source_run = tmp / "bench_v1" / "old-run"
        task_ids = [f"task_{i:03d}" for i in range(10)]
        for tid in task_ids:
            d = source_run / tid
            d.mkdir(parents=True)
            (d / "qa.json").write_text(json.dumps({"task_id": tid}))
        (source_run / "run_metadata.json").write_text(
            json.dumps({"run_id": "old-run", "experiment_config": {"benchmark": "phantomwiki"}})
        )

        new_id = limit_run(source_run, key_limit=4, key_limit_seed=7, new_run_id="new-run")
        assert new_id == "new-run"

        new_run = tmp / "bench_v1" / "new-run"
        symlinked = sorted(p.name for p in new_run.iterdir() if p.is_symlink())
        assert len(symlinked) == 4

        for name in symlinked:
            assert (new_run / name).resolve() == (source_run / name).resolve()

        meta = json.loads((new_run / "run_metadata.json").read_text())
        assert meta["run_id"] == "new-run"
        assert meta["source_run_id"] == "old-run"
        assert meta["key_limit"] == 4
        assert meta["key_limit_seed"] == 7
        assert meta["n_tasks_selected"] == 4
        assert meta["n_tasks_source"] == 10
        assert meta["experiment_config"]["benchmark"] == "phantomwiki"

# %%
# ! poe sync
