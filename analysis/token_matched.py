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
# # token_matched — reporting the token-fair comparison
#
# Reads the saved attempts and reports both arms next to what they actually
# spent:
#
# * **Arm A** — accuracy at a `2 × k_b` budget, plus how often the budget bound
#   (`budget_report`).
# * **Arm B** — `majority@k` (fair) and `best@k` (oracle ceiling) with the vote
#   share behind each winner (`aggregate`), and the accuracy-vs-k curve re-scored
#   from the same attempts (`accuracy_vs_k`).
#
# The voting rules live on the benchmark, not here — deciding "same answer"
# belongs to whoever owns the scoring. This module only loads, groups and
# delegates.

# %%
from juplit import test

# %%
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from dolores.unified.benchmarks import Benchmark, get_benchmark, list_benchmarks
from config import Paths

# %% [markdown]
# ## Loading attempts


# %%
def _tokens_from_calls(run_dir: Path) -> int:
    """Sum ``total_tokens`` over a run's ``calls.jsonl`` (0 when it has none).

    Fallback for attempts written before ``tokens_spent`` was recorded in qa.json.
    """
    calls = run_dir / "calls.jsonl"
    if not calls.exists():
        return 0
    total = 0
    for line in calls.read_text().splitlines():
        try:
            total += json.loads(line).get("total_tokens") or 0
        except json.JSONDecodeError:
            continue
    return total


def _identity_from_path(run_dir: Path) -> dict:
    """``benchmark``/``agent``/``model`` from a run directory.

    The layout is ``logs/<benchmark>/<agent>/<model>/<run_id>/``, but a model name
    contains a slash (`Qwen/Qwen3-32B`), so the model is *two* directories deep and
    counting backwards from the run id mislabels every field. Count forward from
    the logs root instead, and treat everything between the agent and the run id as
    the model.

    A config's ``log_dir`` can nest the tree deeper (``logs/token_matched/samplek/
    <benchmark>/...``), so anchor on the last path part that names a registered
    benchmark when there is one, and on the logs root otherwise.
    """
    parts = run_dir.parts
    root = Paths.LOGS_DIR.name
    names = set(list_benchmarks())
    bench_idx = [i for i, p in enumerate(parts[:-3]) if p in names]
    if bench_idx:
        rel = parts[bench_idx[-1]:]
    elif root in parts:
        rel = parts[len(parts) - 1 - parts[::-1].index(root) + 1:]
    else:
        rel = parts
    if len(rel) < 4:
        raise ValueError(f"cannot read benchmark/agent/model from {run_dir}")
    return {
        "benchmark": rel[0],
        "agent": rel[1],
        "model": "/".join(rel[2:-1]),
    }


def load_attempts(log_glob: str) -> pd.DataFrame:
    """One row per completed attempt found under *log_glob*.

    Expects the standard layout ``logs/<benchmark>/<agent>/<model>/<run_id>/qa.json``.
    """
    rows = []
    # glob.glob, not Path().glob: the latter refuses absolute patterns.
    for qa_path in sorted(Path(p) for p in glob.glob(f"{log_glob.rstrip('/')}/*/qa.json")):
        try:
            data = json.loads(qa_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("answer") is None:
            continue
        run_dir = qa_path.parent
        rows.append({
            **_identity_from_path(run_dir),
            "task_id": data.get("task_id"),
            "attempt": data.get("attempt", 0),
            "answer": data.get("answer"),
            "gold": data.get("expected_answer"),
            "score": _scalar(data.get("scores")),
            "tokens_spent": data.get("tokens_spent") or _tokens_from_calls(run_dir),
            "stop_reason": data.get("stop_reason", "finished"),
            "seed": data.get("seed"),
            "run_dir": str(run_dir),
        })
    return pd.DataFrame(rows)


def _scalar(score) -> float:
    """Benchmarks return either a float or an ``(f1, em)`` pair — take the first."""
    if isinstance(score, (tuple, list)):
        return float(score[0]) if score else 0.0
    if score is None:
        return 0.0
    return float(score)


# %% [markdown]
# ## Arm B — majority@k and best@k, delegated to the benchmark


# %%
def aggregate(df: pd.DataFrame, bench: Benchmark, k: int | None = None) -> pd.DataFrame:
    """One row per task: the fair majority result and the oracle ceiling.

    *k* truncates to the first k attempts, which is what makes the accuracy-vs-k
    curve free — the attempts are already on disk.
    """
    rows = []
    for task_id, group in df.groupby("task_id", sort=True):
        group = group.sort_values("attempt")
        if k is not None:
            group = group.head(k)
        answers = list(group["answer"])
        if not answers:
            continue
        majority, vote_share, majority_score = bench.majority_at_k(task_id, answers)
        # best_at_k is the oracle rule, so the gold answer is handed to it explicitly.
        _, oracle_score = bench.best_at_k(task_id, answers, gold=group["gold"].iloc[0])
        rows.append({
            "task_id": task_id,
            "n_attempts": len(answers),
            "majority_answer": majority,
            "vote_share": vote_share,
            "majority_score": _scalar(majority_score),
            "oracle_score": _scalar(oracle_score),
            "tokens_spent": int(group["tokens_spent"].sum()),
        })
    return pd.DataFrame(rows)


def accuracy_vs_k(df: pd.DataFrame, bench: Benchmark) -> pd.DataFrame:
    """``majority@j`` and ``best@j`` for j = 1..k, re-scored from the same attempts."""
    max_k = int(df["attempt"].max()) + 1 if len(df) else 0
    rows = []
    for j in range(1, max_k + 1):
        at_j = aggregate(df, bench, k=j)
        if at_j.empty:
            continue
        rows.append({
            "k": j,
            "majority_at_k": at_j["majority_score"].mean(),
            "best_at_k": at_j["oracle_score"].mean(),
            "mean_vote_share": at_j["vote_share"].mean(),
            "tokens_spent": int(at_j["tokens_spent"].sum()),
        })
    return pd.DataFrame(rows)


# %% [markdown]
# ## Arm A — accuracy next to spend, and how often the budget bound


# %%
def budget_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, agent, model): accuracy, tokens spent, and the bind rate."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["benchmark", "agent", "model"], sort=True)
    return pd.DataFrame({
        "n_attempts": grouped.size(),
        "mean_score": grouped["score"].mean(),
        "total_tokens": grouped["tokens_spent"].sum(),
        "mean_tokens": grouped["tokens_spent"].mean(),
        "out_of_budget_share": grouped["stop_reason"].apply(
            lambda s: (s == "out_of_budget").mean()
        ),
    }).reset_index()


def token_summary(log_glob: str) -> pd.DataFrame:
    """Mean/total tokens per task per cell — how a `2 × k_b` budget is read off.

    Also the input to the paper's headline ratio: take Deep Reasoner's mean over a
    baseline's mean per benchmark, then average those per-benchmark ratios.
    """
    df = load_attempts(log_glob)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["benchmark", "agent", "model"], sort=True)["tokens_spent"]
    return pd.DataFrame({
        "n_attempts": grouped.size(),
        "mean_tokens": grouped.mean(),
        "total_tokens": grouped.sum(),
    }).reset_index()


# %% [markdown]
# ## CLI


# %%
def main() -> None:
    parser = argparse.ArgumentParser(prog="token-matched-analysis")
    parser.add_argument("-r", "--runs", required=True,
                        help="Glob of run dirs, e.g. 'logs/oolong/codeact/Qwen-Qwen3-32B'.")
    parser.add_argument("-b", "--benchmark", default=None,
                        help="Benchmark name for the voting rules. Default: infer from the path.")
    parser.add_argument("-o", "--out", default="logs/analysis/token_matched",
                        help="Directory for the CSV tables.")
    parser.add_argument("--set", dest="bench_kwargs", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Benchmark constructor kwarg, repeatable. Needed when the "
                             "run used a non-default variant the path does not record, "
                             "e.g. PhantomWiki size 500: --set size=500.")
    args = parser.parse_args()

    df = load_attempts(args.runs)
    if df.empty:
        raise SystemExit(f"no completed attempts under {args.runs!r}")

    def _typed(value: str):
        # ints stay ints (PhantomWiki.size/seed are validated as ints); leave the rest as-is.
        try:
            return int(value)
        except ValueError:
            return value

    kwargs = {}
    for item in args.bench_kwargs:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        kwargs[key.strip()] = _typed(value.strip())

    name = args.benchmark or df["benchmark"].iloc[0]
    bench = get_benchmark(name, **kwargs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = {
        "attempts": df,
        "aggregate": aggregate(df, bench),
        "accuracy_vs_k": accuracy_vs_k(df, bench),
        "budget_report": budget_report(df),
    }
    for label, table in tables.items():
        table.to_csv(out / f"{label}.csv", index=False)
        print(f"\n=== {label} ===")
        print(table.to_string(index=False))
    print(f"\nWrote {len(tables)} tables to {out}")


if __name__ == "__main__":
    main()


# %% [markdown]
# ## Tests


# %%
class _StubBenchmark(Benchmark):
    """Scores 1.0 for "13"; clusters on the plain string."""

    def list_task_ids(self):
        return ["t1"]

    def get_task(self, task_id):
        raise NotImplementedError

    def score(self, task_id, answer, **kwargs):
        return 1.0 if str(answer).strip() == "13" else 0.0


def _attempts_frame(answers: list[str], tokens: int = 10) -> pd.DataFrame:
    return pd.DataFrame([
        {"benchmark": "hello_world", "agent": "cot", "model": "stub", "task_id": "t1",
         "attempt": i, "answer": a, "gold": 13, "score": 1.0 if a == "13" else 0.0,
         "tokens_spent": tokens, "stop_reason": "finished", "seed": i, "run_dir": f"r{i}"}
        for i, a in enumerate(answers)
    ])


def test_identity_from_nested_log_dir():
    nested = Path("logs/token_matched/samplek/phantomwiki/react/Qwen/Qwen3-32B/20260101T000000_ab")
    assert _identity_from_path(nested) == {
        "benchmark": "phantomwiki", "agent": "react", "model": "Qwen/Qwen3-32B"}
    flat = Path("logs/oolong/codeact/Qwen/Qwen3-32B/20260101T000000_ab")
    assert _identity_from_path(flat) == {
        "benchmark": "oolong", "agent": "codeact", "model": "Qwen/Qwen3-32B"}


def test_aggregate_reports_majority_and_oracle():
    bench = _StubBenchmark()
    # The majority is wrong ("12"); the oracle still finds the right answer.
    df = _attempts_frame(["12", "12", "13"])
    row = aggregate(df, bench).iloc[0]
    assert row["majority_answer"] == "12"
    assert row["vote_share"] == 2 / 3
    assert row["majority_score"] == 0.0
    assert row["oracle_score"] == 1.0
    assert row["tokens_spent"] == 30
    assert row["n_attempts"] == 3


def test_aggregate_truncates_to_k():
    bench = _StubBenchmark()
    df = _attempts_frame(["13", "12", "12"])
    assert aggregate(df, bench, k=1).iloc[0]["majority_answer"] == "13"
    assert aggregate(df, bench, k=3).iloc[0]["majority_answer"] == "12"
    # k beyond what exists is a truncation, not an error.
    assert aggregate(df, bench, k=99).iloc[0]["n_attempts"] == 3


def test_accuracy_vs_k_anchored_at_first_attempt():
    bench = _StubBenchmark()
    df = _attempts_frame(["13", "12", "12"])
    curve = accuracy_vs_k(df, bench)
    assert list(curve["k"]) == [1, 2, 3]
    # j=1 is exactly the single-sample number we have today.
    assert curve.iloc[0]["majority_at_k"] == 1.0
    assert curve.iloc[0]["best_at_k"] == 1.0
    # The oracle never drops as k grows.
    assert list(curve["best_at_k"]) == sorted(curve["best_at_k"])


def test_budget_report_counts_binding_runs():
    df = _attempts_frame(["13", "12"])
    df.loc[0, "stop_reason"] = "out_of_budget"
    row = budget_report(df).iloc[0]
    assert row["n_attempts"] == 2
    assert row["out_of_budget_share"] == 0.5
    assert row["total_tokens"] == 20
    assert row["mean_score"] == 0.5


def test_load_attempts_reads_layout_and_skips_incomplete():
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        # A real model name has a slash in it, so the run dir sits one level
        # deeper than the naive <benchmark>/<agent>/<model>/<run_id> reading.
        run = Path(tmp) / "logs" / "hello_world" / "cot" / "Qwen" / "Qwen3-32B" / "run1"
        run.mkdir(parents=True)
        (run / "qa.json").write_text(json.dumps({
            "task_id": "t1", "answer": "13", "expected_answer": 13, "scores": [1.0, 1.0],
            "attempt": 2, "tokens_spent": 42, "stop_reason": "out_of_budget", "seed": 7,
        }))
        # An unfinished attempt is not a row.
        blank = Path(tmp) / "logs" / "hello_world" / "cot" / "Qwen" / "Qwen3-32B" / "run2"
        blank.mkdir(parents=True)
        (blank / "qa.json").write_text(json.dumps({"task_id": "t1", "answer": None}))
        # tokens fall back to calls.jsonl when qa.json predates tokens_spent.
        older = Path(tmp) / "logs" / "hello_world" / "cot" / "Qwen" / "Qwen3-32B" / "run3"
        older.mkdir(parents=True)
        (older / "qa.json").write_text(json.dumps({"task_id": "t1", "answer": "12"}))
        (older / "calls.jsonl").write_text(
            json.dumps({"total_tokens": 5}) + "\n" + json.dumps({"total_tokens": 6}) + "\n"
        )

        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            df = load_attempts("logs/hello_world/cot/Qwen/Qwen3-32B")
        finally:
            os.chdir(cwd)

    assert len(df) == 2
    first = df[df["attempt"] == 2].iloc[0]
    assert first["benchmark"] == "hello_world"
    assert first["agent"] == "cot"
    assert first["model"] == "Qwen/Qwen3-32B"   # not "Qwen3-32B", and agent is not "Qwen"
    assert first["score"] == 1.0
    assert first["tokens_spent"] == 42
    assert first["stop_reason"] == "out_of_budget"
    assert df[df["answer"] == "12"].iloc[0]["tokens_spent"] == 11


# %%
if test():
    test_identity_from_nested_log_dir()
    test_aggregate_reports_majority_and_oracle()
    test_aggregate_truncates_to_k()
    test_accuracy_vs_k_anchored_at_first_attempt()
    test_budget_report_counts_binding_runs()
    test_load_attempts_reads_layout_and_skips_incomplete()
    print("analysis.token_matched tests passed")

# %% [markdown]
# ## End
