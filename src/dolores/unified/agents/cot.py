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
# # CoT agent
#
# Chain-of-thought: single OpenAI-compatible call, no tools. The answer is the
# last non-empty line of the response.

# %%
from juplit import test

# %%
from pathlib import Path

from dolores.unified.agents.base import Agent, register_agent
from dolores.unified.agents._helpers import _conn, _no_thinking_extra_body
from dolores.unified.benchmarks import Task

# %%
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Reason step by step, then give a concise final answer "
    "on the last line of your response. Do not include any extra explanation after the answer."
)


# %%
@register_agent
class CotAgent(Agent):
    name = "cot"

    def _answer(self, task: Task, log_dir: Path) -> str:
        import openai

        base, key = _conn(self.cfg)
        client = openai.OpenAI(base_url=base, api_key=key)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": task.question},
        ]
        response = client.chat.completions.create(
            model=self.cfg.model, messages=messages, **_no_thinking_extra_body(self.cfg)
        )
        raw = response.choices[0].message.content or ""
        lines = [l for l in raw.splitlines() if l.strip()]
        return lines[-1].strip() if lines else ""


# %% [markdown]
# ## Tests


# %%
def test_cot_smoke_scripted_client():
    """HelloWorld fib_7 with a fake OpenAI client returning '13' (no network)."""
    import json
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch

    import openai

    from dolores.unified.agents.base import AgentCfg
    from dolores.unified.benchmarks import HelloWorld
    from config import Paths
    from dolores.unified.inference import OpenAIBackend

    class _FakeClient:
        def __init__(self, *a, **k):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            msg = SimpleNamespace(content="Let me think...\n13")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    bench = HelloWorld()
    cfg = AgentCfg(model="stub", inference=OpenAIBackend(api_base="http://x/v1", api_key="k"),
                   benchmark=bench)
    with tempfile.TemporaryDirectory() as tmp, \
            patch.object(Paths, "LOGS_DIR", Path(tmp)), \
            patch.object(openai, "OpenAI", _FakeClient):
        result = CotAgent(cfg).run(bench.get_task("fib_7"))
        assert result.answer == "13"
        assert result.scores == (1.0, 1.0)
        qa = json.loads(next(Path(tmp).rglob("qa.json")).read_text())
        assert qa["answer"] == "13"
        assert qa["scores"] == [1.0, 1.0]


# %%
if test():
    test_cot_smoke_scripted_client()
    print("agents.cot tests passed")

# %% [markdown]
# ## End
