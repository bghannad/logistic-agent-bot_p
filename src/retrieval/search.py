"""Hybrid (keyword + embedding) search over the embedded KB.

Two independent rankings of the same 63 chunks, merged with Reciprocal Rank Fusion
(RRF) — a standard way to combine rankings from different scoring systems (BM25
scores and cosine similarities aren't on comparable scales, so merging by rank
rather than raw score avoids having to hand-tune a weight between them).
"""

from __future__ import annotations

import json
import re

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from retrieval.embed_kb import CHUNKS_PATH, EMB_PATH, IDS_PATH, MODEL

RRF_K = 60  # standard constant from the RRF paper; not tuned for this project


def _load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# turning a sentence into a list of individual words so that a keyword-search algorithm,
# like BM25 can work with it.
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

# Returns a list of 63 chunk_ids, ordered by keyword overlap with the query (best first). 
# The BM25 algorithm is a standard way to score keyword overlap.
def _rank_by_score(chunk_ids: list[str], scores: np.ndarray) -> list[str]:
    order = np.argsort(scores)[::-1]
    return [chunk_ids[i] for i in order]

# Takes both ranked lists and gives each chunk_id a combined score based on where it ranked in each list (using the RRF formula). 
# Returns a single list of chunk_ids, best first.
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


class HybridIndex:
    """Loads the KB once; call .search() as many times as you want."""

    def __init__(self) -> None:
        load_dotenv()

        self.chunks = _load_chunks()
        self.chunk_ids = [c["chunk_id"] for c in self.chunks]
        self.by_id = {c["chunk_id"]: c for c in self.chunks}

        # Safety check: make sure the embeddings on disk match the current chunks, then load those precomputed vectors
        ids_on_disk = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        if ids_on_disk != self.chunk_ids:
            raise RuntimeError(
                "data/kb/embeddings/ is out of sync with chunks.jsonl — "
                "re-run `python -m retrieval.embed_kb`."
            )
        self.vectors = np.load(EMB_PATH)

        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks]) # makes b,m23 stats for KB
        self.client = OpenAI()

    # Tokenize the query, score it against every chunk using the precomputed BM25 stats, sort chunk_ids best-first.
    def _bm25_ranking(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        return _rank_by_score(self.chunk_ids, scores)

    # Embed the query via OpenAI (this is the one API call per search). 
    # Compare that query vector to all 63 chunk vectors via cosine similarity. Sort chunk_ids best-first.
    def _vector_ranking(self, query: str) -> list[str]:
        query_vec = np.array(
            self.client.embeddings.create(model=MODEL, input=[query]).data[0].embedding,
            dtype=np.float32,
        )
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        sims = (self.vectors @ query_vec) / norms
        return _rank_by_score(self.chunk_ids, sims)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        fused = reciprocal_rank_fusion([self._bm25_ranking(query), self._vector_ranking(query)])
        return [self.by_id[cid] for cid in fused[:top_k]] #Takes the top 10 chunk_ids from the fused ranking 
            #and converts each id back into the real chunk (text + metadata) using the lookup dict
