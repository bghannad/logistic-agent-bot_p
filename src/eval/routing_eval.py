"""Routing eval: for each hand-labeled customer message, does the supervisor call
the right specialist tool(s) — zero, one, or more than one?

This checks the routing decision only, independent of answer quality (that's what
retrieval_eval.py is for). Single-turn (no checkpointer), same as chat.py.

Usage (from repo root, with .venv active):
    python -m eval.routing_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agents.graph import build_graph

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = REPO_ROOT / "data" / "eval" / "routing_cases.jsonl"


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def called_specialists(messages: list) -> set[str]:
    called = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            called.update(call["name"] for call in msg.tool_calls)
    return called


def main() -> None:
    cases = load_cases()
    graph = build_graph()

    exact_matches = 0
    for case in cases:
        result = graph.invoke({"messages": [HumanMessage(content=case["message"])]})
        actual = called_specialists(result["messages"])
        expected = set(case["expected_specialists"])

        missing = expected - actual
        extra = actual - expected
        is_match = not missing and not extra
        exact_matches += is_match

        print(f"[{case['id']}] {'MATCH' if is_match else 'MISMATCH'}")
        print(f"  message:  {case['message']}")
        print(f"  expected: {sorted(expected) or '(none)'}")
        print(f"  actual:   {sorted(actual) or '(none)'}")
        if missing:
            print(f"  missing:  {sorted(missing)}")
        if extra:
            print(f"  extra:    {sorted(extra)}")

    print(f"\nExact match: {exact_matches}/{len(cases)} ({exact_matches / len(cases):.0%})")


if __name__ == "__main__":
    main()
