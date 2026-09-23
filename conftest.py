"""Pytest setup for the repository.

The vendored ``deep_reasoner`` package is written as notebooks whose ``if test():``
cells include live demos that call an LLM endpoint (``OPENAI_BASE_URL`` / ``MODEL``).
juplit's ``test()`` is true in *any* module once pytest is imported, so importing
``deep_reasoner`` from this repo's tests would run those demos. Import it once here
with pytest hidden, so the cached modules never execute them; this repo's own
tests stay offline.
"""
import importlib
import sys

_DEEP_REASONER_MODULES = (
    "deep_reasoner.core",
    "deep_reasoner.llm",
    "deep_reasoner.code_exec",
    "deep_reasoner.logging_utils",
    "deep_reasoner.deepreasoner",
    "deep_reasoner.cli_utils",
    "deep_reasoner.cli_base",
)

_pytest = sys.modules.pop("pytest", None)
try:
    for _name in _DEEP_REASONER_MODULES:
        importlib.import_module(_name)
finally:
    if _pytest is not None:
        sys.modules["pytest"] = _pytest
