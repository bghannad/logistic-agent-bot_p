"""retrieve(query) = hybrid search (wide net) -> rerank (narrow to the real answer).

CLI demo (from repo root, with .venv active):
    python -m retrieval.retrieve "my package is stuck in customs, what do I do"
"""

from __future__ import annotations

import sys

from retrieval.rerank import rerank
from retrieval.search import HybridIndex

CANDIDATE_K = 15  # how many hybrid-search results the reranker gets to choose from
FINAL_K = 5  # how many results retrieve() returns


def retrieve(index: HybridIndex, query: str, candidate_k: int = CANDIDATE_K, final_k: int = FINAL_K) -> list[dict]:
    candidates = index.search(query, top_k=candidate_k)
    return rerank(query, candidates, top_k=final_k)


def main() -> None:
    query = " ".join(sys.argv[1:]) or "my package is stuck in customs, what do I do"
    print(f"Query: {query!r}\n")

    index = HybridIndex()
    for result in retrieve(index, query):
        print(f"[{result['chunk_id']}] section: {result['section_heading']}")
        print(result["text"][:250].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
