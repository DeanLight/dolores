Authoritative benchmark and experiment YAMLs for paper runs live in **deepreasoner-baselines** (`configs/` there).

This repo keeps only **minimal examples** under `configs/examples/`:

- `remote_api_question.yaml` / `local_vllm_question.yaml` — runnable with the `deep-reasoner` CLI. Pass the task as the **second positional** argument (not in YAML).
- `_debug_agent_body.yaml` — shared `llm_kwargs`, `system_prompt`, and mental-model pack (same content as the legacy `configs/agents/debug.yaml` on `main`). The `*_question.yaml` files list it under `_compose`.
- Optional **`initial_var_files`** in the main YAML: list of paths (relative to that file) to small YAML mappings; each top-level key becomes a REPL `Var` before the run (overridable with `--var` / `--var-read` on the CLI).
- Paths in `_compose` are resolved **relative to the directory of the main YAML file** (so fragments can live next to the entry config).
- For tests and copy-paste without the examples tree, the wheel ships **`deep_reasoner/config/debug_agent.yaml`** (importable path helpers live in `deep_reasoner.core`).
