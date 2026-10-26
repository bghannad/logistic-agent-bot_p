import json
from pathlib import Path

from ingestion.chunking import chunk_section, count_tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO_ROOT / "data" / "kb" / "chunks" / "chunks.jsonl"
RETRIEVAL_CASES_PATH = REPO_ROOT / "data" / "eval" / "retrieval_cases.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_chunk_section_never_exceeds_max_tokens():
    # A single long paragraph each, well over the 450-token budget in total —
    # tests the capping logic itself, not the real chunks.jsonl (whose stored
    # token_count includes the context header on top of this budget, so a few
    # real chunks legitimately read slightly over 450 — see conversation notes).
    paragraphs = [("word " * 200).strip() for _ in range(5)]  # ~200 tokens each
    sub_chunks = chunk_section(paragraphs, max_tokens=450, overlap_paragraphs=1)

    assert len(sub_chunks) > 1  # confirms it actually had to split
    for body in sub_chunks:
        assert count_tokens(body) <= 450


def test_real_chunks_file_has_63_unique_ids():
    chunks = _load_jsonl(CHUNKS_PATH)
    ids = [c["chunk_id"] for c in chunks]

    assert len(chunks) == 63
    assert len(set(ids)) == len(ids)


def test_retrieval_eval_chunk_ids_exist_in_real_chunks():
    chunk_ids = {c["chunk_id"] for c in _load_jsonl(CHUNKS_PATH)}
    cases = _load_jsonl(RETRIEVAL_CASES_PATH)

    for case in cases:
        for expected_id in case["expected_chunk_ids"]:
            assert expected_id in chunk_ids, f"{expected_id} (from {case['id']}) not in chunks.jsonl"
