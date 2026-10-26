"""CLI: a real multi-turn chat with the supervisor graph.

Uses LangGraph's checkpointer so the graph remembers the conversation across turns
(keyed by one thread_id for the whole session) — this is what lets the supervisor
ask a clarifying question and pick the conversation back up once you answer it.

In-memory only: history is lost when you quit. A persistent checkpointer (e.g.
SqliteSaver) is a drop-in swap later if the demo needs to survive restarts — not
needed yet.

Usage (from repo root, with .venv active):
    python -m agents.converse
Type 'exit' to quit.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agents.graph import build_graph
from agents.tracing import flush, get_callbacks, session_metadata


def main() -> None:
    graph = build_graph(checkpointer=InMemorySaver())
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_callbacks(),
        "metadata": session_metadata(thread_id),
    }

    print("Chat started (type 'exit' to quit).\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
        print(f"\nBot: {result['messages'][-1].content}\n")

    flush()


if __name__ == "__main__":
    main()
