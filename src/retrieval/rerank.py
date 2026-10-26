"""Cross-encoder reranking of hybrid-search candidates.

A cross-encoder scores each (query, chunk) pair jointly (unlike embeddings, which
score them independently and compare vectors) — slower, so it only runs on the
small candidate set hybrid search already narrowed things down to, not the whole KB.
Runs locally, no API cost.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None

# A pretrained cross-encoder model, downloaded once and cached in _model,
def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model

# takes all the 15 pairs (query & each top 15 matches) and use cross-encoder to rerank them, 
# returning the top 5
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = _get_model().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
