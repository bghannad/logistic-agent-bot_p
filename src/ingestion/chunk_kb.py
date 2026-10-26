"""is the runner — a short script that loops over all 6 raw KB files, applies that
CLI: chunk every raw KB doc in data/kb/raw/ and write data/kb/chunks/chunks.jsonl.

Usage (from repo root, with .venv active):
    python -m ingestion.chunk_kb
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ingestion.chunking import chunk_document
from ingestion.parse_raw import parse_raw_file

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "kb" / "raw"
OUT_PATH = REPO_ROOT / "data" / "kb" / "chunks" / "chunks.jsonl"

# The two long, headerless docs get heading-aware splitting; the four short,
# single-topic docs are chunked whole (see docs/ROADMAP.md Sec. 8.4 discussion).
STRUCTURED_DOCS = {
    "cbp_internet_purchases.txt",
    "cbp_prohibited_restricted_items.txt",
}


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for path in sorted(RAW_DIR.glob("*.txt")):
        doc = parse_raw_file(path)
        chunks = chunk_document(doc, structured=path.name in STRUCTURED_DOCS)
        all_chunks.extend(chunks)

        token_counts = [c.token_count for c in chunks]
        headings = [c.section_heading for c in chunks if c.section_heading]
        print(f"{path.name}: {len(chunks)} chunks, tokens min/avg/max = "
              f"{min(token_counts)}/{sum(token_counts)//len(token_counts)}/{max(token_counts)}")
        if headings:
            print(f"  sections: {headings}")

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks total to {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
