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
# # CLI — `python -m dolores.unified run`
#
# One entry point for every agent. A YAML config names the `agent`, `model`,
# `benchmark`, and `inference` backend; reserved keys are split out and anything
# else flows into the `AgentCfg` as agent-specific extras.
#
# ```
# python -m dolores.unified run --config configs/debug_helloworld.yaml [--task-id fib_7] [--max-workers N]
# ```
#
# With `--task-id` it runs one task (child mode); without, it fans out via the
# runner (parent mode). `analyze` is deferred to PR E.

# %%
from juplit import test

# %%
import argparse
from pathlib import Path

import yaml

from dolores.unified.agents.base import AgentCfg, get_agent_cls
from dolores.unified.benchmarks import get_benchmark
from dolores.unified.inference import make_backend
from dolores.unified.runner import dispatch

# %%
# Top-level YAML keys consumed by the framework/runner; everything else is an
# agent-specific extra fed to AgentCfg.
RESERVED_KEYS = {"agent", "model", "benchmark", "inference",
                 "max_steps", "max_workers", "no_thinking", "shuffle", "task_timeout"}
# Token-matched runs add token_budget / samples / temperature / instructions
# (plus subagent_instructions, which only deepresearch reads).
# Those are NOT reserved: they ride through as AgentCfg extras, which is how the
# agent reads them (cfg.get("token_budget") and friends).


# %%
def load_config(path: str | Path) -> dict:
    """Load a run config, resolving ``_compose`` includes.

    ``_compose`` lists other config files to layer under this one, in order; keys
    in this file win. It keeps the 48 per-model/benchmark/baseline configs from
    repeating a model block, a benchmark block and a full prompt each.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    includes = cfg.pop("_compose", [])
    merged: dict = {}
    for include in includes:
        # Composed paths are repo-relative, like configs/main's _compose lists.
        merged.update(load_config(include))
    merged.update(cfg)
    return merged


def parse_config(cfg_dict: dict, *, token_budget: int | None = None,
                 samples: int | None = None,
                 api_base: str | None = None, api_key: str | None = None):
    """Split a config dict into ``(agent_name, AgentCfg, runner_opts)``.

    Reserved keys build the backend/benchmark/typed-fields; leftover keys become
    AgentCfg extras (e.g. ``agent_config``, ``max_depth``).

    ``token_budget`` / ``samples`` override the config's values when given, so a
    smoke test can run a real cell at a fraction of its cost without editing (and
    risking a commit of) the generated configs. ``api_base`` / ``api_key`` override
    the config's inference endpoint, so the runner can point the client at whatever
    free port it started vllm on rather than the config's placeholder.
    """
    d = dict(cfg_dict)
    d.pop("_compose", None)
    if token_budget is not None:
        d["token_budget"] = token_budget
    if samples is not None:
        d["samples"] = samples
    agent_name = d.pop("agent")
    inference_spec = dict(d.pop("inference"))
    if api_base is not None:
        inference_spec["api_base"] = api_base
    if api_key is not None:
        inference_spec["api_key"] = api_key
    inference = make_backend(**inference_spec)
    bench_spec = dict(d.pop("benchmark"))
    benchmark = get_benchmark(bench_spec.pop("name"), **bench_spec)
    model = d.pop("model")
    max_steps = d.pop("max_steps", 75)
    no_thinking = d.pop("no_thinking", False)
    runner_opts = {
        "max_workers": d.pop("max_workers", 8),
        "shuffle": d.pop("shuffle", None),
        # Seconds before a task's child process is killed and recorded as timed out
        # (not retried). None = no limit.
        "task_timeout": d.pop("task_timeout", None),
    }
    cfg = AgentCfg(
        model=model, inference=inference, benchmark=benchmark,
        max_steps=max_steps, no_thinking=no_thinking, **d,
    )
    budget = cfg.get("token_budget")
    if budget is not None and budget <= 0:
        # 0 is the placeholder the shipped try-hard configs carry until someone
        # fills in the real figure, so refusing it here is the gate, not an edge case.
        raise ValueError(
            f"token_budget must be positive, got {budget} — fill it in from the "
            "mean tokens per task reported by analysis.token_matched"
        )
    samples = cfg.get("samples")
    if samples is not None and samples < 1:
        # Same gate for the cells whose k_b is not known yet.
        raise ValueError(
            f"samples must be at least 1, got {samples} — this cell's k_b is not "
            "known yet, so its sample count cannot be set"
        )
    return agent_name, cfg, runner_opts


# %%
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m dolores.unified")
    sub = parser.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run", help="Run an agent over a benchmark from a YAML config.")
    runp.add_argument("--config", required=True, help="Path to the run config YAML.")
    runp.add_argument("--task-id", dest="task_id", default=None,
                      help="Run exactly one task (child mode). Omit to fan out (parent mode).")
    runp.add_argument("--attempt", type=int, default=0,
                      help="Which sampled attempt of the task to run (child mode).")
    runp.add_argument("--max-workers", dest="max_workers", type=int, default=None,
                      help="Override max_workers from the config (parent mode).")
    runp.add_argument("--limit", type=int, default=None,
                      help="Run only the first N tasks (parent mode) — for sanity slices.")
    runp.add_argument("--token-budget", dest="token_budget", type=int, default=None,
                      help="Override the config's token_budget — e.g. a small value "
                           "to smoke-test that the budget actually binds.")
    runp.add_argument("--samples", type=int, default=None,
                      help="Override the config's samples — e.g. 2 to smoke-test "
                           "sampling without paying for the real attempt count.")
    runp.add_argument("--api-base", dest="api_base", default=None,
                      help="Override the config's inference api_base — the runner "
                           "passes the free port it started vllm on.")
    runp.add_argument("--api-key", dest="api_key", default=None,
                      help="Override the config's inference api_key.")
    args = parser.parse_args(argv)

    if args.command == "run":
        agent_name, cfg, runner_opts = parse_config(
            load_config(args.config), token_budget=args.token_budget, samples=args.samples,
            api_base=args.api_base, api_key=args.api_key)

        if args.task_id is not None:
            # Child mode — run one task.
            agent = get_agent_cls(agent_name)(cfg)
            agent.run(cfg.benchmark.get_task(args.task_id), attempt=args.attempt)
            return

        # Parent mode — fan out.
        max_workers = args.max_workers if args.max_workers is not None else runner_opts["max_workers"]
        # Take the slice before plan_work shuffles, so --limit 5 is the same 5 tasks
        # every time and a sanity run stays comparable across baselines.
        task_ids = cfg.benchmark.list_task_ids()[:args.limit] if args.limit else None
        # The children re-read the same YAML, so an override only the parent knows
        # about would be silently dropped on the way down.
        child_args = []
        if args.token_budget is not None:
            child_args += ["--token-budget", str(args.token_budget)]
        if args.samples is not None:
            child_args += ["--samples", str(args.samples)]
        if args.api_base is not None:
            child_args += ["--api-base", args.api_base]
        if args.api_key is not None:
            child_args += ["--api-key", args.api_key]
        dispatch(agent_name, cfg, config_path=args.config, task_ids=task_ids,
                 max_workers=max_workers, shuffle=runner_opts["shuffle"],
                 task_timeout=runner_opts["task_timeout"],
                 child_args=child_args)


# %%
if __name__ == "__main__":
    main()


# %% [markdown]
# ## Tests


# %%
def test_parse_config_splits_reserved_and_extras():
    from dolores.unified.benchmarks import HelloWorld
    from dolores.unified.inference import OpenAIBackend

    agent_name, cfg, runner_opts = parse_config({
        "agent": "deep_reasoner",
        "model": "Qwen/Qwen3-32B",
        "inference": {"kind": "openai", "api_base": "http://localhost:8555/v1", "api_key": "EMPTY"},
        "benchmark": {"name": "hello_world"},
        "max_steps": 12,
        "no_thinking": True,
        "max_workers": 4,
        "agent_config": "configs/agents/debug.yaml",  # extra
        "max_depth": 6,                                  # extra
    })
    assert agent_name == "deep_reasoner"
    assert cfg.model == "Qwen/Qwen3-32B"
    assert isinstance(cfg.inference, OpenAIBackend)
    assert isinstance(cfg.benchmark, HelloWorld)
    assert cfg.max_steps == 12
    assert cfg.no_thinking is True
    assert cfg.get("agent_config") == "configs/agents/debug.yaml"
    assert cfg.get("max_depth") == 6
    assert runner_opts == {"max_workers": 4, "shuffle": None, "task_timeout": None}


def test_parse_config_token_matched_keys():
    _, cfg, _ = parse_config({
        "agent": "codeact",
        "model": "Qwen/Qwen3-32B",
        "inference": {"kind": "openai", "api_base": "http://x", "api_key": "k"},
        "benchmark": {"name": "hello_world"},
        "token_budget": 1_000_000,
        "samples": 205,
        "temperature": 0.8,
        "instructions": "be thorough",
    })
    assert cfg.get("token_budget") == 1_000_000
    assert cfg.get("samples") == 205
    assert cfg.get("temperature") == 0.8
    assert cfg.get("instructions") == "be thorough"

    try:
        parse_config({
            "agent": "codeact", "model": "m",
            "inference": {"kind": "openai", "api_base": "http://x", "api_key": "k"},
            "benchmark": {"name": "hello_world"},
            "token_budget": 0,
        })
        assert False, "should have raised"
    except ValueError:
        pass


def _prompt_constant(name: str) -> str:
    """The text of a prompts.py constant, normalised the way the configs store it.

    Reading the source rather than importing keeps this honest: it compares the
    config against the file a reviewer reads, not a value some import rebound.

    The configs strip leading/trailing blank lines and trailing spaces so YAML can
    hold the prompt as a readable `|` block instead of one escaped double-quoted
    line — whitespace only, no words change. This applies the same normalisation
    so the comparison stays exact on everything that matters.
    """
    import re

    src = Path("src/prompts.py").read_text()
    match = re.search(rf'^{name} = """', src, re.M)
    assert match, f"no such prompt constant: {name}"
    start = match.end()
    text = src[start:src.index('"""', start)]
    return "\n".join(line.rstrip() for line in text.strip("\n").split("\n"))


# ReAct and Deep Research cannot run Oolong — its long-context documents exceed
# their context window — so those two cells are excluded everywhere, matching the
# combo list in scripts/sbatch_baselines.sh.
IMPOSSIBLE_CELLS = {("oolong", "react"), ("oolong", "deepresearch")}
BENCHMARK_NAMES = ["phantomwiki", "synthworlds", "oolong", "deepresearchqa"]
BASELINE_NAMES = ["react", "codeact", "rlm", "deepresearch"]


def _runnable_cells() -> list[tuple[str, str]]:
    return [(bench, baseline)
            for bench in BENCHMARK_NAMES for baseline in BASELINE_NAMES
            if (bench, baseline) not in IMPOSSIBLE_CELLS]


def test_cli_overrides_budget_and_samples():
    """--token-budget / --samples override the config, and reach the children."""
    import os
    from unittest.mock import patch

    raw = load_config("configs/token_matched/synthworlds_codeact_samplek.yaml")
    assert raw["samples"] == 3 and "token_budget" not in raw

    with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
        _, cfg, _ = parse_config(raw, samples=2, token_budget=20_000)
    assert cfg.get("samples") == 2
    assert cfg.get("token_budget") == 20_000

    # Omitted overrides leave the config alone.
    with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
        _, untouched, _ = parse_config(load_config(
            "configs/token_matched/synthworlds_codeact_samplek.yaml"))
    assert untouched.get("samples") == 3

    # The gates still apply to an override, not just to a config value.
    for bad in ({"token_budget": 0}, {"samples": 0}):
        try:
            parse_config(load_config(
                "configs/token_matched/synthworlds_codeact_samplek.yaml"), **bad)
            assert False, f"should have refused {bad}"
        except ValueError:
            pass


def test_dispatch_forwards_overrides_to_children():
    """A parent-only override would be silently dropped: children re-read the YAML."""
    import tempfile
    from unittest.mock import patch

    from dolores.unified import runner
    from dolores.unified.agents.base import AgentCfg
    from dolores.unified.benchmarks import HelloWorld
    from config import Paths
    from dolores.unified.inference import OpenAIBackend

    bench = HelloWorld()
    cfg = AgentCfg(model="stub", benchmark=bench, samples=2, temperature=0.8,
                   inference=OpenAIBackend(api_base="http://x", api_key="k"))
    seen = []

    def _capture(cmd_builder, work, max_workers, **_):
        seen.extend(cmd_builder(item) for item in work)

    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(Paths, "LOGS_DIR", Path(tmp)), \
            patch.object(runner, "run_subprocess_pool", _capture), \
            patch.object(OpenAIBackend, "wait_until_ready", lambda self: None):
        runner.dispatch("cot", cfg, config_path="configs/debug_helloworld.yaml",
                        child_args=["--token-budget", "20000", "--samples", "2"])

    assert seen, "no child commands were built"
    for cmd in seen:
        assert "--token-budget" in cmd and "20000" in cmd
        assert "--samples" in cmd and "2" in cmd


def test_token_matched_grid_is_complete():
    """Every runnable cell, both arms: 14 cells x {try_hard, samplek} = 28 configs."""
    cells = _runnable_cells()
    assert len(cells) == 14
    expected = {
        f"{bench}_{baseline}_{arm}.yaml"
        for bench, baseline in cells for arm in ("try_hard", "samplek")
    }
    found = {p.name for p in Path("configs/token_matched").glob("*.yaml")}
    assert found == expected, f"missing {expected - found}, unexpected {found - expected}"


def test_token_matched_configs_are_valid():
    """Every token-matched config parses and sets exactly one arm."""
    import os
    from unittest.mock import patch

    configs = sorted(Path("configs/token_matched").glob("*.yaml"))
    assert configs, "no token-matched configs found"

    for path in configs:
        raw = yaml.safe_load(path.read_text())

        try_hard = "token_budget" in raw
        samplek = "samples" in raw
        assert try_hard == path.stem.endswith("_try_hard"), f"{path}: name and arm disagree"
        assert samplek == path.stem.endswith("_samplek"), f"{path}: name and arm disagree"
        assert try_hard != samplek, (
            f"{path}: set exactly one arm — budgeting and sampling the same run "
            f"would spend k x 2 x k_b, which matches nothing"
        )
        if samplek:
            assert raw.get("temperature"), f"{path}: samples needs a temperature"

        with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
            _, cfg, _ = parse_config(raw)
        assert cfg.benchmark is not None


def test_token_matched_numbers_match_the_ratio_csv():
    """Every budget and sample count is derived from the committed measurements.

    `analysis/dr_vs_baseline_token_ratios.csv` is the source; nothing here is
    transcribed by hand, so a config cannot drift from the numbers it claims.
    """
    import csv
    import math

    csv_benchmark = {
        "phantomwiki_500_1": "phantomwiki",
        "synthworlds": "synthworlds",
        "oolong-real": "oolong",
        "deepresearchqa": "deepresearchqa",
    }
    measured = {}
    with open("analysis/dr_vs_baseline_token_ratios.csv") as handle:
        for row in csv.DictReader(handle):
            cell = (csv_benchmark[row["benchmark"]], row["baseline_method"])
            if row["status"] != "ok":
                # The CSV's own record that the pair never ran, matching the cells
                # we exclude for context width.
                assert cell in IMPOSSIBLE_CELLS, f"{cell} is {row['status']} but still in the grid"
                continue
            measured[cell] = (
                float(row["dr_over_baseline_x_increase"]),
                round(2 * float(row["dr_mean_total_tok_per_task"])),
            )

    assert set(measured) == set(_runnable_cells()), "the CSV and the grid disagree"

    for (bench, baseline), (k, budget) in measured.items():
        try_hard = yaml.safe_load(
            Path(f"configs/token_matched/{bench}_{baseline}_try_hard.yaml").read_text())
        samplek = yaml.safe_load(
            Path(f"configs/token_matched/{bench}_{baseline}_samplek.yaml").read_text())

        assert try_hard["token_budget"] == budget, (
            f"{bench}x{baseline}: budget is {try_hard['token_budget']}, "
            f"but 2x Deep Reasoner's spend is {budget}"
        )
        assert samplek["samples"] == math.ceil(k), (
            f"{bench}x{baseline}: {samplek['samples']} samples, expected ceil({k})"
        )
        # Rounding up is the fair direction: it never spends less than Deep Reasoner.
        assert samplek["samples"] >= k


def test_no_token_matched_placeholders_remain():
    """Every cell has real numbers now, so nothing should still be gated."""
    for path in Path("configs/token_matched").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text())
        assert raw.get("token_budget", 1) > 0, f"{path} still has a placeholder budget"
        assert raw.get("samples", 1) >= 1, f"{path} still has a placeholder sample count"


def test_inlined_prompts_match_prompts_py():
    """The inlined prompt text is the real thing, not a paraphrase of it.

    Each config's `instructions` must start with the exact constant the agent
    would otherwise have used, so a token-matched run differs from a normal run
    only in the token matching (and, for Arm A, the appended don't-give-up text).
    """
    cases = {
        "token_matched/phantomwiki_react_try_hard.yaml": "PHANTOM_WIKI_REACT_INSTRUCTIONS",
        "token_matched/phantomwiki_react_samplek.yaml": "PHANTOM_WIKI_REACT_INSTRUCTIONS",
        "token_matched/phantomwiki_deepresearch_samplek.yaml": "PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS",
        "token_matched/synthworlds_codeact_try_hard.yaml": "SYNTHWORLDS_CODEACT_INSTRUCTIONS",
        "token_matched/synthworlds_codeact_samplek.yaml": "SYNTHWORLDS_CODEACT_INSTRUCTIONS",
        "token_matched/synthworlds_rlm_samplek.yaml": "SYNTHWORLDS_RLM_INSTRUCTIONS",
        "base/prompt_phantomwiki_react.yaml": "PHANTOM_WIKI_REACT_INSTRUCTIONS",
        "base/prompt_phantomwiki_codeact.yaml": "PHANTOM_WIKI_CODEACT_INSTRUCTIONS",
        "base/prompt_phantomwiki_rlm.yaml": "PHANTOM_WIKI_RLM_INSTRUCTIONS",
        "base/prompt_phantomwiki_deepresearch.yaml": "PHANTOM_WIKI_DEEPRESEARCH_INSTRUCTIONS",
        "base/prompt_synthworlds_react.yaml": "SYNTHWORLDS_REACT_INSTRUCTIONS",
        "base/prompt_synthworlds_codeact.yaml": "SYNTHWORLDS_CODEACT_INSTRUCTIONS",
        "base/prompt_synthworlds_rlm.yaml": "SYNTHWORLDS_RLM_INSTRUCTIONS",
        "base/prompt_synthworlds_deepresearch.yaml": "SYNTHWORLDS_DEEPRESEARCH_INSTRUCTIONS",
    }
    for rel, constant in cases.items():
        text = (Path("configs") / rel).read_text()
        assert "instructions: |" in text, (
            f"{rel}: the prompt must be a readable YAML literal block, not an "
            f"escaped one-liner"
        )
        raw = yaml.safe_load(text)
        inlined = raw["instructions"]
        expected = _prompt_constant(constant)
        assert inlined.startswith(expected), f"{rel} does not match {constant} verbatim"
        tail = inlined[len(expected):].strip()
        assert not tail or tail.startswith("Do not give up."), (
            f"{rel} appends something other than the Arm A don't-give-up text"
        )


def test_prompt_provenance_matches_the_table():
    """The author's provenance table, asserted (see the table in prompts.py).

    Two claims per cell: whether it has extra instructions at all, and — where the
    table says how the prompt was built — how many worked examples it carries.
    """
    import re

    # cell -> constant, or None where the table says the framework default is used.
    expected = {
        ("synthworlds", "react"): "SYNTHWORLDS_REACT_INSTRUCTIONS",
        ("phantomwiki", "react"): "PHANTOM_WIKI_REACT_INSTRUCTIONS",
        ("deepresearchqa", "react"): None,
        ("synthworlds", "codeact"): "SYNTHWORLDS_CODEACT_INSTRUCTIONS",
        ("phantomwiki", "codeact"): "PHANTOM_WIKI_CODEACT_INSTRUCTIONS",
        ("deepresearchqa", "codeact"): None,
        ("oolong", "codeact"): None,
        ("synthworlds", "rlm"): "SYNTHWORLDS_RLM_INSTRUCTIONS",
        ("phantomwiki", "rlm"): "PHANTOM_WIKI_RLM_INSTRUCTIONS",
        ("deepresearchqa", "rlm"): None,
        ("oolong", "rlm"): None,
        # Deep Research's substantive prompt is the sub-agent's ReAct translation.
        ("synthworlds", "deepresearch"): "SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS",
        ("phantomwiki", "deepresearch"): "PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS",
        ("deepresearchqa", "deepresearch"): None,
    }
    assert set(expected) == set(_runnable_cells()), "provenance table and grid disagree"

    src = Path("src/prompts.py").read_text()
    for (bench, baseline), constant in expected.items():
        if constant is None:
            continue
        assert re.search(rf'^{constant} = """', src, re.M), (
            f"{baseline} on {bench} should use {constant}, which does not exist"
        )

    def worked_examples(constant: str) -> int:
        return len(re.findall(r"^Task:\s*", _prompt_constant(constant), re.M))

    # "handcrafted on 6 reasoning types", and CodeAct is "translated from ReAct (3/6)".
    assert worked_examples("SYNTHWORLDS_REACT_INSTRUCTIONS") == 6
    assert worked_examples("SYNTHWORLDS_CODEACT_INSTRUCTIONS") == 3
    # Deep Research translates all six.
    assert worked_examples("SYNTHWORLDS_DEEPRESEARCH_REACT_INSTRUCTIONS") == 6
    # PhantomWiki's translations keep the same example set as its ReAct prompt.
    pw = worked_examples("PHANTOM_WIKI_REACT_INSTRUCTIONS")
    assert pw == worked_examples("PHANTOM_WIKI_CODEACT_INSTRUCTIONS") == \
        worked_examples("PHANTOM_WIKI_DEEPRESEARCH_REACT_INSTRUCTIONS"), (
            "the PhantomWiki translations no longer share ReAct's example set"
        )

    # "17 REPL examples for how to use tools", and SynthWorlds' tool examples.
    def repl_blocks(constant: str) -> int:
        return len(re.findall(r"```repl", _prompt_constant(constant)))

    assert repl_blocks("PHANTOM_WIKI_RLM_INSTRUCTIONS") == 17
    assert repl_blocks("SYNTHWORLDS_RLM_INSTRUCTIONS") > 0


def test_deepresearch_configs_carry_both_prompts():
    """Deep Research runs two prompts, so its configs must show both."""
    for cell in ("phantomwiki_deepresearch", "synthworlds_deepresearch"):
        for arm in ("try_hard", "samplek"):
            raw = yaml.safe_load(
                Path(f"configs/token_matched/{cell}_{arm}.yaml").read_text())
            assert raw.get("instructions"), f"{cell}_{arm}: no orchestrator prompt"
            assert raw.get("subagent_instructions"), (
                f"{cell}_{arm}: the sub-agent prompt is the ReAct translation — "
                f"without it the config does not show what the cell ran"
            )
            bench = cell.split("_")[0]
            constant = ("PHANTOM_WIKI" if bench == "phantomwiki" else "SYNTHWORLDS")
            expected = _prompt_constant(f"{constant}_DEEPRESEARCH_REACT_INSTRUCTIONS")
            assert raw["subagent_instructions"].startswith(expected)


def test_baseline_configs_compose_and_parse():
    """The non-token-matched configs mirror the old runs: no budget, no sampling."""
    import os
    from unittest.mock import patch

    configs = sorted(Path("configs/baselines").rglob("*.yaml"))
    expected = len(_runnable_cells()) * 3   # 14 cells x 3 models
    assert len(configs) == expected, f"expected {expected} baseline configs, found {len(configs)}"
    assert not [p for p in configs if p.stem in {"oolong_react", "oolong_deepresearch"}], (
        "Oolong x ReAct / Deep Research do not fit those agents' context window"
    )

    for path in configs:
        merged = load_config(path)
        assert "_compose" not in merged, f"{path}: _compose should be resolved away"
        assert merged["model"] and merged["benchmark"], f"{path}: composition lost a key"
        for key in ("token_budget", "samples", "temperature"):
            assert key not in merged, f"{path}: {key} would make this a token-matched run"

        with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
            agent_name, cfg, _ = parse_config(merged)
        assert agent_name in {"react", "codeact", "rlm", "deepresearch"}
        assert cfg.get("samples", 1) == 1


def test_deep_reasoner_configs_parse():
    """Every Deep Reasoner run config parses and names a planner YAML that exists."""
    import os
    from unittest.mock import patch

    configs = sorted(Path("configs/deep_reasoner").rglob("*.yaml"))
    assert configs, "no Deep Reasoner configs found"
    variants = {p.parent.name for p in configs}
    assert {"qwen3_32b", "qwen3_8b", "llama3_70b",
            "qwen3_32b_decomp", "qwen3_32b_nomodel"} <= variants

    for path in configs:
        merged = load_config(path)
        assert Path(merged["agent_config"]).is_file(), f"{path}: missing planner YAML"
        assert isinstance(merged["vllm_args"], list), f"{path}: vllm_args must be a list"
        assert merged["slurm"]["gpus"] >= merged["slurm"]["tensor_parallel"], path
        with patch.dict(os.environ, {"SERPER_API_KEY": "smoke-test-key"}):
            agent_name, cfg, runner_opts = parse_config(merged)
        assert agent_name == "deep_reasoner"
        assert cfg.max_steps == 30, f"{path}: the paper's planner ran max_iter=30"
        assert runner_opts["task_timeout"], f"{path}: missing the per-task timeout"
        if cfg.benchmark.benchmark_name() == "deepresearchqa":
            assert cfg.get("search_model_id"), f"{path}: DeepSearchQA needs a search model"


def test_parse_config_benchmark_kwargs():
    agent_name, cfg, _ = parse_config({
        "agent": "react",
        "model": "m",
        "inference": {"kind": "openai", "api_base": "http://x", "api_key": "k"},
        "benchmark": {"name": "phantomwiki", "size": 50, "seed": 1},
    })
    assert cfg.benchmark.benchmark_name() == "phantomwiki"
    assert cfg.max_steps == 75  # default


# %%
if test():
    test_parse_config_splits_reserved_and_extras()
    test_parse_config_token_matched_keys()
    test_cli_overrides_budget_and_samples()
    test_dispatch_forwards_overrides_to_children()
    test_token_matched_grid_is_complete()
    test_token_matched_configs_are_valid()
    test_token_matched_numbers_match_the_ratio_csv()
    test_no_token_matched_placeholders_remain()
    test_inlined_prompts_match_prompts_py()
    test_prompt_provenance_matches_the_table()
    test_deepresearch_configs_carry_both_prompts()
    test_baseline_configs_compose_and_parse()
    test_deep_reasoner_configs_parse()
    test_parse_config_benchmark_kwargs()
    print("cli tests passed")

# %% [markdown]
# ## End
