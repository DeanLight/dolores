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
# # Runs
#
# Load the results of unified runs (`python -m dolores.unified run`) for the paper
# tables. Every run config writes its attempts to one tree,
# `<log_dir>/<benchmark>/<agent>/<model>[-nothink]/<run_id>/qa.json`, so results are
# found through the configs that produced them: the config says which method,
# model and benchmark variant (e.g. PhantomWiki size) a directory holds, which the
# directory path alone cannot (sizes and ablations share benchmark/agent/model).

# %%
from juplit import test

# %%
import json
from pathlib import Path


from config import Paths
from dolores.unified.cli import load_config

# %%
METHOD_LABELS = {
    "react": "ReAct",
    "codeact": "CodeAct",
    "deepresearch": "Deep Research",
    "rlm": "RLM",
    "deep_reasoner": "Deep Reasoner (ours)",
}

# The three models the paper reports, in table order.
PAPER_MODELS = ["qwen3_8b", "qwen3_32b", "llama3_70b"]


# %%
def run_tree(cfg: dict) -> Path:
    """The directory a config's runs land in — mirrors ``runner._base_log_dir``."""
    model_dir = cfg["model"] + ("-nothink" if cfg.get("no_thinking") else "")
    base = Path(cfg["log_dir"]) if cfg.get("log_dir") else Paths.LOGS_DIR
    if not base.is_absolute():
        base = Paths.ROOT / base
    return base / cfg["benchmark"]["name"] / cfg["agent"] / model_dir


# Deep Reasoner configs left out of the main table: SynthWorlds has two planner
# versions (synthworlds.yaml = agent v01, synthworlds_v2.yaml = agent v02); the
# table reports one of them.
DR_EXCLUDE = ("synthworlds_v2",)


def paper_configs(benchmark: str, models=PAPER_MODELS, variant: str = "",
                  exclude=DR_EXCLUDE) -> list[Path]:
    """The configs behind the paper's main table for one benchmark.

    Baselines come from ``configs/baselines/<model>/``, Deep Reasoner from
    ``configs/deep_reasoner/<model><variant>/`` (``variant`` e.g. ``_decomp``),
    minus the Deep Reasoner configs named in ``exclude``.
    """
    root = Paths.ROOT / "configs"
    paths: list[Path] = []
    for model in models:
        paths += sorted((root / "baselines" / model).glob(f"{benchmark}_*.yaml"))
        paths += sorted(p for p in (root / "deep_reasoner" / f"{model}{variant}").glob(f"{benchmark}*.yaml")
                        if p.stem not in exclude)
    return paths


def load_results(config_path: Path | str) -> list[dict]:
    """One dict per task: the latest completed attempt-0 run of this config.

    Keys: the qa.json fields plus ``method``, ``agent``, ``model`` (short name, with
    ``-nothink`` when set), ``benchmark`` (the config's benchmark block), ``config``
    and ``qa_path``.
    """
    cfg = load_config(config_path)
    model_str = cfg["model"].split("/")[-1] + ("-nothink" if cfg.get("no_thinking") else "")
    latest: dict[str, tuple[str, Path, dict]] = {}
    for qa_path in sorted(run_tree(cfg).glob("*/qa.json")):
        try:
            data = json.loads(qa_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("answer") is None or data.get("attempt", 0) != 0:
            continue
        run_id = qa_path.parent.name
        tid = data.get("task_id")
        if tid not in latest or run_id > latest[tid][0]:
            latest[tid] = (run_id, qa_path, data)
    return [
        {
            **data,
            "method": METHOD_LABELS.get(cfg["agent"], cfg["agent"]),
            "agent": cfg["agent"],
            "model": model_str,
            "benchmark": cfg["benchmark"],
            "config": str(config_path),
            "qa_path": str(qa_path),
        }
        for _, qa_path, data in latest.values()
    ]


def save_field(row: dict, **fields) -> None:
    """Write extra fields (e.g. a cached judge score) back into the row's qa.json."""
    path = Path(row["qa_path"])
    data = json.loads(path.read_text())
    data.update(fields)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))


# %% [markdown]
# ## Tests


# %%
def test_load_results_takes_latest_attempt0_per_task():
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg_path = tmp / "c.yaml"
        cfg_path.write_text(
            "agent: react\nmodel: Qwen/Qwen3-32B\n"
            f"log_dir: {tmp / 'logs'}\n"
            "inference: {kind: openai, api_base: 'http://x', api_key: k}\n"
            "benchmark: {name: phantomwiki, size: 500, seed: 1}\n"
        )
        tree = tmp / "logs" / "phantomwiki" / "react" / "Qwen" / "Qwen3-32B"

        def write(run_id, **qa):
            (tree / run_id).mkdir(parents=True)
            (tree / run_id / "qa.json").write_text(json.dumps(qa))

        write("20260101T000000_a", task_id="t1", answer="old", attempt=0)
        write("20260102T000000_b", task_id="t1", answer="new", attempt=0)
        write("20260103T000000_c", task_id="t1", answer="sample", attempt=1)
        write("20260104T000000_d", task_id="t2", answer=None, attempt=0)

        with patch.object(Paths, "ROOT", tmp):
            rows = load_results(cfg_path)
        assert [(r["task_id"], r["answer"]) for r in rows] == [("t1", "new")]
        assert rows[0]["method"] == "ReAct" and rows[0]["model"] == "Qwen3-32B"
        assert rows[0]["benchmark"] == {"name": "phantomwiki", "size": 500, "seed": 1}

        save_field(rows[0], judge_score=1.0)
        assert json.loads(Path(rows[0]["qa_path"]).read_text())["judge_score"] == 1.0


def test_paper_configs_cover_every_method():
    agents = {load_config(p)["agent"] for p in paper_configs("phantomwiki")}
    assert agents == set(METHOD_LABELS)
    dr_synth = [p for p in paper_configs("synthworlds") if p.parent.parent.name == "deep_reasoner"]
    assert len(dr_synth) == len(PAPER_MODELS), "one SynthWorlds Deep Reasoner config per model"
    assert paper_configs("oolong", variant="_decomp")


# %%
if test():
    test_load_results_takes_latest_attempt0_per_task()
    test_paper_configs_cover_every_method()
    print("analysis.runs tests passed")

# %% [markdown]
# ## End
