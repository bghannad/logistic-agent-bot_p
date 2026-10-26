import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = REPO_ROOT / "data" / "kb" / "chunks" / "chunks.jsonl"
CHUNK_IDS_PATH = REPO_ROOT / "data" / "kb" / "embeddings" / "chunk_ids.json"


def test_chunk_ids_json_matches_chunks_jsonl_order():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        chunk_ids_from_chunks = [json.loads(line)["chunk_id"] for line in f]

    chunk_ids_from_embeddings = json.loads(CHUNK_IDS_PATH.read_text(encoding="utf-8"))

    assert chunk_ids_from_embeddings == chunk_ids_from_chunks
