"""Retrieval eval: for each hand-labeled question, does retrieve() surface the
right chunk in its final top-5 (hybrid search + rerank, same pipeline the
compliance specialist actually uses)?

Three metrics per case, all computed against the same top-5 result:
- Hit@5: did at least one expected chunk appear anywhere in the top 5? (binary)
- Precision@5: of the 5 chunks returned, what fraction were actually expected?
- Recall@5: of the chunks that were expected, what fraction did we actually get?
    Only matters for the handful of cases with 2 expected chunk_ids (a topic that
    got split across chunks) — recall can be < 1 there even on a "hit."

Usage (from repo root, with .venv active):
    python -m eval.retrieval_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from retrieval.retrieve import retrieve
from retrieval.search import HybridIndex

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main() -> None:
    cases = load_cases()
    index = HybridIndex()

    hits = 0
    precisions = []
    recalls = []

    for case in cases:
        results = retrieve(index, case["query"])
        result_ids = [r["chunk_id"] for r in results]
        expected = set(case["expected_chunk_ids"])

        relevant_retrieved = len(set(result_ids) & expected)
        precision = relevant_retrieved / len(result_ids) if result_ids else 0.0
        recall = relevant_retrieved / len(expected) if expected else 0.0
        precisions.append(precision)
        recalls.append(recall)

        rank = next((i + 1 for i, cid in enumerate(result_ids) if cid in expected), None)
        status = f"HIT (rank {rank})" if rank else "MISS"
        hits += rank is not None

        print(f"[{case['id']}] {status}  precision@5={precision:.2f}  recall@5={recall:.2f}")
        print(f"  query: {case['query']}")
        print(f"  expected: {sorted(expected)}")
        print(f"  got:      {result_ids}")
        print("----------------------------------------------------------------")

    n = len(cases)
    print(f"\nHit@5:       {hits}/{n} ({hits / n:.0%})")
    print(f"Precision@5: {sum(precisions) / n:.2f} (avg across cases)")
    print(f"Recall@5:    {sum(recalls) / n:.2f} (avg across cases)")


if __name__ == "__main__":
    main()
