"""Optional Langfuse tracing for the supervisor graph.

Strictly additive: if LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set in .env,
get_callbacks() returns an empty list and every call site behaves exactly as before
with no tracing. The callback handler itself reads its keys/host from the same env
vars automatically (no explicit client init needed).

Usage, at any graph.invoke() call site:
    from agents.tracing import flush, get_callbacks
    graph.invoke({"messages": [...]}, config={"callbacks": get_callbacks(), ...})
    flush()  # see flush()'s docstring for why this call matters

"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def get_callbacks() -> list:
    if not tracing_enabled():
        return []

    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def session_metadata(session_id: str) -> dict:
    """Pass this as config["metadata"] so multi-turn conversations with the same
    session_id group together in Langfuse's Sessions view instead of showing as
    unrelated one-off traces. "langfuse_session_id" is the exact key Langfuse's
    LangChain integration looks for in trace metadata (confirmed against current
    Langfuse docs, not from memory)."""
    if not tracing_enabled():
        return {}
    return {"langfuse_session_id": session_id}


def flush() -> None:
    """Force any queued traces to send now.

    The Langfuse SDK batches traces on a background thread rather than sending
    each one immediately (so tracing doesn't add latency to every call) — fine for
    a long-running server, but a short CLI script can exit before that thread gets
    to send its queued batch, silently losing the trace. Call this right before
    a CLI entry point exits.
    """
    if not tracing_enabled():
        return

    from langfuse import get_client

    get_client().flush()
