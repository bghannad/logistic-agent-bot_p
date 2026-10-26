"""Test-suite-wide safety nets.

- A dummy OPENAI_API_KEY, set before pytest collects any test module: several
  modules (agents/graph.py, agents/specialists.py) construct ChatOpenAI(...) at
  import time, which needs *some* key present to construct (never calls out at
  construction) — this runs at conftest import, before collection, so it's in
  place even on a fresh clone with no .env at all.
- An autouse fixture that strips any real Langfuse keys for every test, so the
  suite never talks to Langfuse even though a real .env with real keys sits
  right next to this repo locally.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")


@pytest.fixture(autouse=True)
def _no_real_langfuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
