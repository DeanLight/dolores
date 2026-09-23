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
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# %%
class Paths:
    # src/config.py → repo root is one level up from ``src/``.
    ROOT     = Path(__file__).resolve().parent.parent
    ENV_FILE = ROOT / ".env"
    LOGS_DIR = ROOT / "logs"
    ANALYSIS_DIR = ROOT / "analysis"

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
