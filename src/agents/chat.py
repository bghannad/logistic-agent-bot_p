"""CLI: send one message through the supervisor graph and print the routing + reply.

Usage (from repo root, with .venv active):
    python -m agents.chat "my package has been stuck in customs for 5 days"

Single-turn only — see graph.py docstring.
"""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage

from agents.graph import build_graph
from agents.tracing import flush, get_callbacks


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m agents.chat "your message here"')
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    print(f"Customer: {message}\n")

    graph = build_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"callbacks": get_callbacks()},
    )

    for msg in result["messages"]:
        role = msg.__class__.__name__
        if role == "AIMessage" and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"[supervisor -> {call['name']}] args={call['args']}")
        elif role == "ToolMessage":
            print(f"[{msg.name} result] {str(msg.content)[:200]}")
        elif role == "AIMessage" and msg.content:
            print(f"\nSupervisor reply:\n{msg.content}")

    flush()


if __name__ == "__main__":
    main()
