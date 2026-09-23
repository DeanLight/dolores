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
# # core
#
# Shared utilities for baseline runners.

# %%
from juplit import test

# %%
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

# %%
def find_tested_ids(log_dir: Path) -> set[str]:
    """Return task IDs that already have a completed qa.json under log_dir.

    Reads log_dir/*/qa.json. A run is complete when answer is not None.
    Skips corrupt or incomplete files silently.
    """
    tested: set[str] = set()
    for qa_path in Path(log_dir).glob("*/qa.json"):
        try:
            data = json.loads(qa_path.read_text())
            if data.get("answer") is not None:
                tid = data.get("task_id")
                if tid:
                    tested.add(tid)
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return tested


# %%
def count_attempts(log_dir: Path) -> Counter:
    """Return ``task_id -> number of completed attempts`` under log_dir.

    Same scan as ``find_tested_ids``, but counting: Arm B needs to know how many
    samples a task already has, not merely whether it has one.
    """
    counts: Counter = Counter()
    for qa_path in Path(log_dir).glob("*/qa.json"):
        try:
            data = json.loads(qa_path.read_text())
            if data.get("answer") is not None:
                tid = data.get("task_id")
                if tid:
                    counts[tid] += 1
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return counts


# %%
def run_subprocess_pool(
    cmd_builder: Callable[[str], list[str]],
    work: list[str],
    max_workers: int,
    timeout: float | None = None,
) -> None:
    """Fan out task IDs as subprocess workers; retry each indefinitely on failure.

    ``timeout`` (seconds) kills a worker that runs too long. A timed-out task is
    recorded and not retried: it would most likely time out again.
    """

    def _launch(tid: str) -> None:
        while True:
            try:
                r = subprocess.run(
                    cmd_builder(tid),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"⏱ {tid} timed out after {timeout}s — not retried")
                return
            if r.returncode == 0:
                return
            tail = (r.stderr or "").strip().splitlines()[-1:]
            msg = tail[0] if tail else f"exit {r.returncode}"
            print(f"↻ {tid} failed ({msg}) — retrying...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_launch, tid): tid for tid in work}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                future.result()
                print(f"✓ {tid}")
            except Exception as exc:
                print(f"✗ {tid} — {exc}")


# %%
def test_find_tested_ids():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # Complete run — answer set
        r1 = base / "run1"
        r1.mkdir()
        (r1 / "qa.json").write_text(json.dumps({"task_id": "t1", "answer": "hello"}))

        # Complete run — answer set
        r2 = base / "run2"
        r2.mkdir()
        (r2 / "qa.json").write_text(json.dumps({"task_id": "t2", "answer": "world"}))

        # Incomplete — answer is None
        r3 = base / "run3"
        r3.mkdir()
        (r3 / "qa.json").write_text(json.dumps({"task_id": "t3", "answer": None}))

        # Corrupt JSON — should be skipped
        r4 = base / "run4"
        r4.mkdir()
        (r4 / "qa.json").write_text("not json{{{")

        result = find_tested_ids(base)
        assert result == {"t1", "t2"}, f"expected {{t1, t2}}, got {result}"


def test_count_attempts():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        for i, (tid, answer) in enumerate(
            [("t1", "a"), ("t1", "b"), ("t2", "c"), ("t3", None)]
        ):
            run = base / f"run{i}"
            run.mkdir()
            (run / "qa.json").write_text(json.dumps({"task_id": tid, "answer": answer}))
        corrupt = base / "run_bad"
        corrupt.mkdir()
        (corrupt / "qa.json").write_text("not json{{{")

        counts = count_attempts(base)
        assert counts["t1"] == 2, counts
        assert counts["t2"] == 1, counts
        assert counts["t3"] == 0, counts   # answer is None -> incomplete
        assert counts["never_seen"] == 0


def test_run_subprocess_pool_timeout_is_not_retried():
    import sys

    calls = []

    def cmd_builder(tid):
        calls.append(tid)
        return [sys.executable, "-c", "import time; time.sleep(5)"]

    run_subprocess_pool(cmd_builder, ["slow"], max_workers=1, timeout=0.5)
    assert calls == ["slow"], "a timed-out task must not be relaunched"


# %%
if test():
    test_find_tested_ids()
    test_count_attempts()
    test_run_subprocess_pool_timeout_is_not_retried()
    print("core tests passed")
