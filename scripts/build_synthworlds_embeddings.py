#!/usr/bin/env python
"""Build SynthWorlds SM retrieval embeddings (sm.npy + sm_docs.json).

The SynthWorlds benchmark's retriever tool (``retrieve_top_5``; Deep Reasoner's
``search``) needs a precomputed embedding matrix over the SM document corpus. The
two blobs are too large for git, so build them once before any SynthWorlds run;
without them ``_load_sm_embeddings`` raises FileNotFoundError and every retrieval
call fails.

Recipe: corpus = HF ``kenqgu/SynthWorlds`` config ``qa-sm-docs`` split ``test``
(6290 docs); embed each ``doc`` with OpenAI ``text-embedding-3-small`` in corpus
order so ``embs[i]`` ↔ ``docs[i]``. Writes into
``src/dolores/unified/benchmarks/data/``; the blobs are gitignored.

Needs OPENAI_API_KEY and the qa-sm-docs dataset (cached or online).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from openai import OpenAI

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "dolores" / "unified" / "benchmarks" / "data"
MODEL = "text-embedding-3-small"
BATCH = 1000


def main() -> None:
    ds = load_dataset("kenqgu/SynthWorlds", "qa-sm-docs", split="test")
    docs = [row["doc"] for row in ds]
    print(f"[build] {len(docs)} docs from qa-sm-docs test", flush=True)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embs: list[list[float]] = []
    for start in range(0, len(docs), BATCH):
        chunk = docs[start : start + BATCH]
        resp = client.embeddings.create(input=chunk, model=MODEL)
        embs.extend(d.embedding for d in resp.data)
        print(f"[build] embedded {start + len(chunk)}/{len(docs)}", flush=True)

    arr = np.asarray(embs, dtype=np.float32)
    assert arr.shape[0] == len(docs), (arr.shape, len(docs))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "sm.npy", arr)
    with open(DATA_DIR / "sm_docs.json", "w") as f:
        json.dump(docs, f)
    print(f"[build] wrote {DATA_DIR/'sm.npy'} {arr.shape} and sm_docs.json", flush=True)


if __name__ == "__main__":
    main()
