"""CLI: embed data/kb/chunks/chunks.jsonl and write data/kb/embeddings/.

Usage (from repo root, with .venv active):
    python -m retrieval.embed_kb

Costs real (tiny) money — one OpenAI API call. Safe to re-run: if the chunk_ids on
disk already match chunks.jsonl, it skips the API call instead of re-embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

MODEL = "text-embedding-3-small"
PRICE_PER_1M_TOKENS_USD = 0.02  # OpenAI's published rate for this model

REPO_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = REPO_ROOT / "data" / "kb" / "chunks" / "chunks.jsonl"
EMB_DIR = REPO_ROOT / "data" / "kb" / "embeddings"
EMB_PATH = EMB_DIR / "embeddings.npy"
IDS_PATH = EMB_DIR / "chunk_ids.json"


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# # if the chunk_ids on disk already match chunks.jsonl, we can skip the API call instead of re-embedding
def already_embedded(chunk_ids: list[str]) -> bool:
    if not (EMB_PATH.exists() and IDS_PATH.exists()):
        return False
    on_disk = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    return on_disk == chunk_ids


def main() -> None:
    load_dotenv()

    chunks = load_chunks()
    chunk_ids = [c["chunk_id"] for c in chunks]

    if already_embedded(chunk_ids):
        print(f"{EMB_PATH.relative_to(REPO_ROOT)} already matches chunks.jsonl — skipping (no API call).")
        return

    client = OpenAI()
    response = client.embeddings.create(model=MODEL, input=[c["text"] for c in chunks])

    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, vectors)
    IDS_PATH.write_text(json.dumps(chunk_ids), encoding="utf-8")

    tokens = response.usage.total_tokens
    cost = tokens / 1_000_000 * PRICE_PER_1M_TOKENS_USD
    print(f"Embedded {len(chunks)} chunks -> {vectors.shape} ({MODEL})")
    print(f"Tokens billed: {tokens} (~${cost:.5f})")
    print(f"Wrote {EMB_PATH.relative_to(REPO_ROOT)} and {IDS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
