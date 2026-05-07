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
# # Settings & config
#
# Paths and other configurations for the benchmarks

# %%
from juplit import test

# %%
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# %%
class Paths:
    # src/config.py → repo root is one level up from ``src/``.
    ROOT     = Path(__file__).resolve().parent.parent
    ENV_FILE = ROOT / ".env"
    LOGS_DIR = ROOT / "logs"
    ANALYSIS_DIR = ROOT / "analysis"

# %%
class RunContext(BaseModel):
    """Metadata for a single evaluation run. Serialisable to JSON."""
    model_config = {"extra": "allow"}

    model:     str                  # e.g. "Qwen/Qwen3-8B", "gpt-5-mini"
    benchmark: str                  # e.g. "phantomwiki", "oolong"
    method:    str                  # e.g. "react", "codeact", "cot"
    no_thinking: bool = False       # if True, model dir gets a "-nothink" suffix

    _log_stem: str | None = None   # cached stem, set on first access

    @staticmethod
    def get_log_dir(benchmark: str, method: str, model: str, no_thinking: bool = False) -> Path:
        """Return (and create) logs/<benchmark>/<method>/<model>[-nothink]/."""
        model_safe = model.replace("/", "-")
        if no_thinking:
            model_safe = f"{model_safe}-nothink"
        p = Paths.LOGS_DIR / benchmark / method / model_safe
        p.mkdir(parents=True, exist_ok=True)
        return p

    def log_dir(self) -> Path:
        """Return (and create) logs/<benchmark>/<method>/<model>[-nothink]/."""
        return self.get_log_dir(self.benchmark, self.method, self.model, self.no_thinking)

    @property
    def stem(self) -> str:
        """Shared ``<timestamp>_<uuid>`` stem (stable after first access)."""
        if self._log_stem is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:8]
            self._log_stem = f"{ts}_{uid}"
        return self._log_stem

    @property
    def identity(self) -> dict[str, str]:
        """All identity fields as a plain dict, stamped on every log line."""
        return {**self.model_dump(), "stem": self.stem}

    def save_answer(self, answer: str) -> Path:
        """Write answer + identity as JSON inside the ``<stem>/`` folder.

        Returns the path to the written file.
        """
        stem_dir = self.log_dir() / self.stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        p = stem_dir / "answer.json"
        data = {**self.identity, "answer": answer}
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def save(self, data: dict) -> Path:
        """Write identity + data as JSON to a log file. Returns the path."""
        p = self.log_dir() / (self.stem + ".json")
        merged = {**self.identity, **data}
        p.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

# %%
if test():
    ctx = RunContext(
        model="Qwen/Qwen3-8B",
        benchmark="phantomwiki",
        method="react",
        size=50,
        seed=1,
        max_steps=50,
    )

    print(ctx.model_dump_json(indent=2))
    print()
    print("log_dir:", ctx.log_dir())
    print(f"stem: {ctx.stem}")

# %%
if test():
    ctx.identity

# %%
if test():
    ctx.stem

# %%
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Paths.ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""

# %%
load_dotenv(Paths.ENV_FILE, override=True)
settings = Settings()

# %%
if test():
    Paths.ANALYSIS_DIR
