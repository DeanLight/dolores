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
# # CLI Utilities
#
# Layered YAML config loading via Dynaconf, CLI override parsing, and Pydantic validation.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from dynaconf import Dynaconf
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

from juplit import test

T = TypeVar("T", bound=BaseModel)

# %% [markdown]
# ## Loading settings

# %%
def load_settings(
    config_files: list[Path],
    overrides: dict[str, Any] | None = None,
) -> Dynaconf:
    """Stack YAML config files and apply in-process overrides.

    Parameters
    ----------
    config_files:
        Ordered list of YAML paths. Later files override earlier ones.
    overrides:
        Flat or nested dict merged on top of all YAML files.
        Typically built from CLI ``--set`` flags via :func:`parse_dotted_overrides`.
    """
    settings = Dynaconf(
        settings_files=[str(p) for p in config_files],
        environments=False,
        load_dotenv=True,
    )
    if overrides:
        settings.update(overrides, merge=True)
    return settings

# %% [markdown]
# ## Dotted key overrides

# %%
def parse_dotted_overrides(args: list[str]) -> dict[str, Any]:
    """Parse ``["key=value", "a.b=value"]`` into a nested dict.

    Dot-separated keys become nested dicts::

        parse_dotted_overrides(["client.base_url=http://localhost"])
        # -> {"client": {"base_url": "http://localhost"}}

    Values are YAML-parsed so booleans, ints, and floats round-trip correctly
    (e.g. ``enable_thinking=false`` → ``False``). Plain strings are unaffected.
    """
    result: dict[str, Any] = {}
    for arg in args:
        key, _, value = arg.partition("=")
        parts = key.strip().split(".")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = yaml.safe_load(value.strip())
    return result

# %%
if test():
    assert parse_dotted_overrides(["task_id=my-task"]) == {"task_id": "my-task"}
    assert parse_dotted_overrides(["client.base_url=http://localhost"]) == {
        "client": {"base_url": "http://localhost"}
    }
    assert parse_dotted_overrides(["max_iter=5", "client.base_url=http://localhost"]) == {
        "max_iter": 5,
        "client": {"base_url": "http://localhost"},
    }
    assert parse_dotted_overrides([]) == {}

# %% [markdown]
# ## Validation

# %%
def _lower_keys(obj: Any) -> Any:
    """Recursively lowercase all dict keys (Dynaconf uppercases them by default)."""
    if isinstance(obj, dict):
        return {k.lower(): _lower_keys(v) for k, v in obj.items()}
    return obj


def validate_config(settings: Dynaconf, config_cls: type[T]) -> T:
    """Validate a Dynaconf settings object into *config_cls*.

    Lowercases all keys before passing to ``model_validate`` so that Pydantic
    field names (lowercase) match Dynaconf's internally-uppercased keys.
    """
    return config_cls.model_validate(_lower_keys(settings.as_dict()))


def load_and_validate(
    config_cls: type[T],
    config_files: list[Path],
    overrides: dict[str, Any] | None = None,
    set_args: list[str] | None = None,
) -> T:
    """Load layered YAMLs, apply overrides, and validate into *config_cls*.

    Parameters
    ----------
    config_cls:
        Pydantic model class to validate into.
    config_files:
        Ordered list of YAML paths. Later files override earlier ones.
    overrides:
        Optional dict of overrides merged on top of all YAML files.
    set_args:
        Optional list of ``"key=value"`` strings (e.g. from a ``--set`` CLI flag).
        Dot-separated keys map to nested config fields.
        Merged on top of *overrides* when both are provided.
    """
    all_overrides: dict[str, Any] = {}
    if overrides:
        all_overrides.update(overrides)
    if set_args:
        all_overrides.update(parse_dotted_overrides(set_args))
    return validate_config(load_settings(config_files, all_overrides or None), config_cls)


_MAIN_CONFIG_SKIP_KEYS = {"_compose", "description"}


def load_main_and_validate(
    config_cls: type[T],
    main_config: Path,
    set_args: list[str] | None = None,
) -> T:
    """Load a single main YAML config (with optional _compose), then validate.

    Load order:
    1) files listed in ``_compose`` (later entries override earlier).
       Relative paths resolve against the main config file's directory.
    2) inline keys in the main config (excluding metadata keys),
    3) CLI ``--set`` dotted overrides.
    """
    with open(main_config, encoding="utf-8") as f:
        main_cfg = yaml.safe_load(f) or {}

    base = main_config.resolve().parent
    compose_files: list[Path] = []
    for p in main_cfg.get("_compose", []):
        pp = Path(p)
        compose_files.append(pp if pp.is_absolute() else (base / pp))
    inline_overrides = {
        k: v for k, v in main_cfg.items() if k not in _MAIN_CONFIG_SKIP_KEYS
    }
    set_overrides = parse_dotted_overrides(set_args or [])

    settings = load_settings(compose_files, inline_overrides or None)
    if set_overrides:
        settings.update(set_overrides, merge=True)
    return validate_config(settings, config_cls)

# %%
if test():
    import tempfile

    import yaml
    from pydantic import BaseModel as _BaseModel

    class _Inner(_BaseModel):
        url: str = "http://default"
        retries: int = 3

    class _TestCfg(_BaseModel):
        name: str
        value: int = 0
        inner: _Inner = _Inner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(
            {"name": "from-yaml", "value": 10, "inner": {"url": "http://yaml", "retries": 5}},
            f,
        )
        tmp = Path(f.name)

    # baseline from YAML
    cfg = load_and_validate(_TestCfg, [tmp])
    assert cfg.name == "from-yaml"
    assert cfg.value == 10
    assert cfg.inner.url == "http://yaml"
    assert cfg.inner.retries == 5

    # flat dict override
    assert load_and_validate(_TestCfg, [tmp], overrides={"value": 99}).value == 99

    # nested dict override — only the specified key changes
    cfg2 = load_and_validate(_TestCfg, [tmp], overrides={"inner": {"url": "http://override"}})
    assert cfg2.inner.url == "http://override"

    # dotted set_args override a top-level field
    assert load_and_validate(_TestCfg, [tmp], set_args=["value=7"]).value == 7

    # dotted set_args override a nested field
    cfg3 = load_and_validate(_TestCfg, [tmp], set_args=["inner.url=http://set-arg"])
    assert cfg3.inner.url == "http://set-arg"
    assert cfg3.inner.retries == 5  # other nested fields unaffected

    # multiple dotted set_args at once
    cfg4 = load_and_validate(_TestCfg, [tmp], set_args=["inner.url=http://multi", "inner.retries=9"])
    assert cfg4.inner.url == "http://multi"
    assert cfg4.inner.retries == 9

    # set_args win over overrides
    cfg5 = load_and_validate(_TestCfg, [tmp], overrides={"value": 1}, set_args=["value=2"])
    assert cfg5.value == 2

    # _compose entries are resolved relative to the main config's directory
    import shutil

    _td = Path(tempfile.mkdtemp())
    try:
        (_td / "frag.yaml").write_text("extra: 1\nname: composed\n", encoding="utf-8")
        (_td / "main.yaml").write_text("_compose: [frag.yaml]\nvalue: 42\n", encoding="utf-8")

        class _Composed(_BaseModel):
            extra: int = 0
            name: str = ""
            value: int = 0

        _c = load_main_and_validate(_Composed, _td / "main.yaml")
        assert _c.extra == 1 and _c.name == "composed" and _c.value == 42
    finally:
        shutil.rmtree(_td, ignore_errors=True)

    tmp.unlink()

# %% [markdown]
# ## Shared client and agent helpers

# %%
class ClientConfig(BaseModel):
    """HTTP client settings for the OpenAI-compatible backend."""
    model_config = ConfigDict(extra="allow")

    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_retries: int = 5
    connect_timeout: float = 5.0
    read_timeout: float = 600.0
    write_timeout: float = 10.0
    max_connections: int = 500
    max_keepalive_connections: int = 100


def build_client(cfg: ClientConfig) -> AsyncOpenAI:
    """Build an AsyncOpenAI client from a ClientConfig."""
    api_key = os.getenv(cfg.api_key_env) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        base_url = cfg.base_url or ""
        if base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1"):
            api_key = ""
        else:
            raise ValueError(
                f"Missing API key. Set `{cfg.api_key_env}` (preferred) or `OPENAI_API_KEY` in your environment."
            )
    return AsyncOpenAI(
        base_url=cfg.base_url,
        api_key=api_key,
        max_retries=cfg.max_retries,
        timeout=httpx.Timeout(
            connect=cfg.connect_timeout,
            read=cfg.read_timeout,
            write=cfg.write_timeout,
            pool=None,
        ),
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=cfg.max_connections,
                max_keepalive_connections=cfg.max_keepalive_connections,
            )
        ),
    )



def make_agent(cfg: Any, tools: dict, client: AsyncOpenAI) -> Any:
    """Build a DeepReasoner from config, tools, and shared client.

    cfg must have: planner_model, model, llm_kwargs, prompt_template_variables,
    models, system_prompt, max_iter.
    """
    from deep_reasoner.deepreasoner import DeepReasoner
    from deep_reasoner.llm import make_llm

    planner_model = cfg.planner_model or cfg.model or os.getenv("MODEL")
    if not planner_model:
        raise ValueError("Missing model. Set `model` in YAML, MODEL env var, or --set model=...")

    planner_llm = make_llm(client=client, model=planner_model, **cfg.llm_kwargs)

    prompt_vars = dict(cfg.prompt_template_variables)
    prompt_vars.setdefault("models", cfg.models)

    return DeepReasoner(
        planner_llm=planner_llm,
        system_prompt=cfg.system_prompt,
        init_vars=tools,
        max_iter=cfg.max_iter,
        **prompt_vars,
    )

# %%
# ! poe sync
