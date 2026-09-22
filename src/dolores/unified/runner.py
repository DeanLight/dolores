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
# # runner — parent/child fan-out
#
# `dispatch` enumerates a benchmark's tasks, skips ones already done, brings the
# inference backend up once (fast-fail), and fans out one child subprocess per
# task. Each child re-runs `python -m dolores.unified.cli run --config <yaml> --task-id <id>` so
# the YAML config is the single source of truth across parent and children.

# %%
from juplit import test

# %%
import json
import random
import sys
from pathlib import Path

from config import Paths
from core import count_attempts, run_subprocess_pool

# %%
_SHUFFLE_BENCHMARKS = {"phantomwiki", "synthworlds"}


# %%
def _base_log_dir(agent_name: str, cfg) -> Path:
    """``<base>/<benchmark>/<agent>/<model>[-nothink]/`` — matches run_log_dir's parent.

    ``base`` is the config's ``log_dir`` when set (so resume counts attempts under the
    same tree the runs write to), else the default ``logs/``.
    """
    model_dir = f"{cfg.model}-nothink" if cfg.no_thinking else cfg.model
    base = Path(cfg.get("log_dir")) if cfg.get("log_dir") else Paths.LOGS_DIR
    return base / cfg.benchmark.benchmark_name() / agent_name / model_dir


def plan_work(agent_name: str, cfg, *, task_ids=None, shuffle=None):
    """Return ``(work, all_ids)`` — the ``(task_id, attempt)`` pairs still owed.

    With the default ``samples = 1`` this is one pair per not-yet-done task, i.e.
    the pre-sampling behaviour. With ``samples = k`` a task is only finished once
    it has k completed attempts, which is what lets a re-run draw sample k+1.
    """
    bench = cfg.benchmark
    all_ids = list(task_ids) if task_ids is not None else bench.list_task_ids()
    do_shuffle = shuffle if shuffle is not None else (bench.benchmark_name() in _SHUFFLE_BENCHMARKS)
    if do_shuffle:
        random.shuffle(all_ids)
    samples = cfg.get("samples", 1)
    done = count_attempts(_base_log_dir(agent_name, cfg))
    work = [(t, i) for t in all_ids for i in range(done[t], samples)]
    return work, all_ids


def dispatch(agent_name: str, cfg, *, config_path, task_ids=None, max_workers=8,
             shuffle=None, child_args=None) -> None:
    """Parent: bring the backend up, then fan out one child per pending task."""
    work, all_ids = plan_work(agent_name, cfg, task_ids=task_ids, shuffle=shuffle)
    samples = cfg.get("samples", 1)
    total = len(all_ids) * samples
    print(f"Running {len(work)} / {total} attempts -- (Total {total - len(work)} done)")

    # Record the planned totals at the log-tree root so watch_experiments (and any
    # other reader) can show done/total without parsing the buffered child stdout.
    base = _base_log_dir(agent_name, cfg)
    base.mkdir(parents=True, exist_ok=True)
    (base / "run_metadata.json").write_text(json.dumps(
        {"n_tasks": total, "n_unique_tasks": len(all_ids), "samples": samples}
    ))

    abs_config = str(Path(config_path).resolve())

    def cmd_builder(item: tuple[str, int]) -> list[str]:
        tid, attempt = item
        return [
            sys.executable, "-m", "dolores.unified.cli", "run",
            "--config", abs_config, "--task-id", str(tid), "--attempt", str(attempt),
            *(child_args or []),
        ]

    cfg.inference.wait_until_ready()
    try:
        run_subprocess_pool(cmd_builder, work, max_workers)
    finally:
        stop = getattr(cfg.inference, "stop", None)
        if callable(stop):
            stop()


# %% [markdown]
# ## Tests


# %%
def test_plan_work_skips_done():
    import json
    import tempfile
    from unittest.mock import patch

    from dolores.unified.agents.base import AgentCfg
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    bench = HelloWorld()
    cfg = AgentCfg(model="stub", inference=OpenAIBackend(api_base="http://x", api_key="k"),
                   benchmark=bench)
    with tempfile.TemporaryDirectory() as tmp, patch.object(Paths, "LOGS_DIR", Path(tmp)):
        base = _base_log_dir("cot", cfg)
        done = base / "somerun"
        done.mkdir(parents=True)
        (done / "qa.json").write_text(json.dumps({"task_id": "fib_7", "answer": "13"}))

        work, all_ids = plan_work("cot", cfg)
        assert set(all_ids) == {"fib_7", "shmib_5", "fib_3"}
        assert set(work) == {("shmib_5", 0), ("fib_3", 0)}

        # samples=3: the done task still owes attempts 1 and 2.
        cfg_sampled = AgentCfg(model="stub", inference=cfg.inference, benchmark=bench,
                               samples=3, temperature=0.8)
        work3, _ = plan_work("cot", cfg_sampled)
        assert set(work3) == {
            ("fib_7", 1), ("fib_7", 2),
            ("shmib_5", 0), ("shmib_5", 1), ("shmib_5", 2),
            ("fib_3", 0), ("fib_3", 1), ("fib_3", 2),
        }, work3


# %%
if test():
    test_plan_work_skips_done()
    print("runner tests passed")

# %% [markdown]
# ## End
