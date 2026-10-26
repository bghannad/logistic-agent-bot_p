"""Prompt loading: versioned in git (prompts/*.txt), optionally synced to and
served from Langfuse's prompt management (see scripts/sync_prompts.py).

With Langfuse keys set: fetches the named prompt by label (default "production")
on every call — Langfuse caches internally (~60s default TTL), so moving the
"production" label to a different version changes the live prompt with no
redeploy. The local .txt file is passed as `fallback`, used only if the fetch
fails. When a real (non-fallback) prompt is used, its metadata is attached to the
returned ChatPromptTemplate so Langfuse links traces to the exact prompt version
(confirmed against Langfuse's docs: https://langfuse.com/docs/prompt-management/features/link-to-traces).

Without Langfuse keys: reads the local .txt file directly. No `langfuse` import
happens on this path at all.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.tracing import tracing_enabled

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _read_local_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def _mustache_to_fstring(text: str) -> str:
    """Convert Langfuse's {{var}} syntax to Python/LangChain's {var}, escaping
    every other literal brace so it isn't mistaken for a format variable."""
    placeholders: dict[str, str] = {}

    def protect(match: re.Match) -> str:
        key = f"\x00{len(placeholders)}\x00"
        placeholders[key] = "{" + match.group(1) + "}"
        return key

    protected = re.sub(r"\{\{\s*(\w+)\s*\}\}", protect, text)
    escaped = protected.replace("{", "{{").replace("}", "}}")
    for key, value in placeholders.items():
        escaped = escaped.replace(key, value)
    return escaped


def get_prompt_template(name: str) -> ChatPromptTemplate:
    if not tracing_enabled():
        system_text = _mustache_to_fstring(_read_local_prompt(name))
        return ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder("messages"),
        ])

    from langfuse import get_client

    prompt = get_client().get_prompt(
        name,
        label=os.environ.get("PROMPT_LABEL", "production"),
        type="text",
        fallback=_read_local_prompt(name),
    )

    template = ChatPromptTemplate.from_messages([
        ("system", prompt.get_langchain_prompt()),
        MessagesPlaceholder("messages"),
    ])
    if not prompt.is_fallback:
        template.metadata = {"langfuse_prompt": prompt}
    return template
