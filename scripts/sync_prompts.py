"""Sync prompts/*.txt to Langfuse as versioned prompts under the "production" label.

Only creates a new version when the local text actually differs from what's
currently live under "production" — running this with no changes is a no-op.

Usage (from repo root, with venv active):
    python scripts/sync_prompts.py            # sync for real
    python scripts/sync_prompts.py --dry-run   # show what would happen, no writes
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    from langfuse import get_client

    client = get_client()

    for path in sorted(PROMPTS_DIR.glob("*.txt")):
        name = path.stem
        local_text = path.read_text(encoding="utf-8")

        try:
            current = client.get_prompt(name, label="production", type="text")
            exists = True
            unchanged = current.prompt == local_text
        except Exception:
            exists = False
            unchanged = False

        if unchanged:
            print(f"[{name}] unchanged — skipping")
            continue

        action = "update" if exists else "create"
        if dry_run:
            print(f"[{name}] would {action} a new 'production' version (--dry-run, no write)")
            continue

        client.create_prompt(name=name, type="text", prompt=local_text, labels=["production"])
        print(f"[{name}] created new 'production' version ({action})")


if __name__ == "__main__":
    main()
