#!/usr/bin/env bash
# Quick reachability + auth check for the external services the benchmarks use.
# Run before submitting jobs:  bash scripts/preflight.sh
# (VLLM_API_KEY is for the local vLLM server and can't be checked here.)
set -u
pass=0; fail=0
_ok(){ printf '  OK    %s\n' "$1"; pass=$((pass+1)); }
_no(){ printf '  FAIL  %s\n' "$1"; fail=$((fail+1)); }

echo "[preflight] checking external services..."

# OpenAI — Oolong/DeepSearchQA judges + SynthWorlds embeddings retriever
if [[ -z "${OPENAI_API_KEY:-}" ]]; then _no "OpenAI (judges): OPENAI_API_KEY not set"
elif curl -sf -o /dev/null -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
then _ok "OpenAI (judges + embeddings) reachable"
else _no "OpenAI: key rejected or unreachable"; fi

# HuggingFace — token + hub (dataset + model downloads)
if [[ -z "${HF_TOKEN:-}" ]]; then _no "HuggingFace: HF_TOKEN not set"
elif curl -sf -o /dev/null -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2
then _ok "HuggingFace hub reachable (token valid)"
else _no "HuggingFace: token rejected or unreachable"; fi

# Serper — web search for DeepSearchQA / deepresearch
if [[ -z "${SERPER_API_KEY:-}" ]]; then _no "Serper (search): SERPER_API_KEY not set"
elif curl -sf -o /dev/null -X POST https://google.serper.dev/search \
       -H "X-API-KEY: $SERPER_API_KEY" -H "Content-Type: application/json" -d '{"q":"ping"}'
then _ok "Serper (web search) reachable"
else _no "Serper: key rejected or unreachable"; fi

echo "[preflight] ${pass} ok, ${fail} failed"
[[ $fail -eq 0 ]] || { echo "[preflight] fix the failures above before submitting jobs" >&2; exit 1; }
echo "[preflight] all services reachable"
