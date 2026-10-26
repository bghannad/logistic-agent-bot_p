"""The three specialist agents, each exposed to the supervisor as one callable tool.

Each specialist is its own small LLM call (optionally with its own tool, for
tracking). The supervisor never sees check_shipment_status or retrieve() directly —
it only sees these three tools, and decides which (if any) to call.
Whys:
1. One small LLM call per specialist: each gets a short, focused system prompt for one job. 
That’s less confusing for the model than one giant prompt covering everything.
2. Different inputs per specialist: tracking gets a tool, compliance gets retrieved chunks 
pasted into its prompt, and escalation gets only a judgment call.
3. Wrapped as @tool: that’s how a LangGraph supervisor “calls” another agent. 
It uses the same mechanism as calling a plain function, and the docstring tells the supervisor when to pick it.

"""

from __future__ import annotations

import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState

from agents.prompts import get_prompt_template
from agents.tools import check_shipment_status
from retrieval.retrieve import retrieve
from retrieval.search import HybridIndex

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

_hybrid_index: HybridIndex | None = None


def _get_index() -> HybridIndex:
    global _hybrid_index
    if _hybrid_index is None:
        _hybrid_index = HybridIndex()
    return _hybrid_index


def tracking_agent_run(user_message: str) -> str:
    llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools([check_shipment_status])
    chain = get_prompt_template("tracking") | llm

    messages = [HumanMessage(content=user_message)]
    response = chain.invoke({"messages": messages})

    if response.tool_calls:
        messages.append(response)
        for call in response.tool_calls:
            result = check_shipment_status.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        response = chain.invoke({"messages": messages})

    return response.content


def compliance_agent_run(user_message: str) -> str:
    chunks = retrieve(_get_index(), user_message)
    context = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)

    llm = ChatOpenAI(model=MODEL, temperature=0)
    chain = get_prompt_template("compliance") | llm
    response = chain.invoke({"messages": [HumanMessage(content=user_message)], "context": context})
    return response.content


def escalation_agent_run(conversation_summary: str) -> str:
    llm = ChatOpenAI(model=MODEL, temperature=0)
    chain = get_prompt_template("escalation") | llm
    response = chain.invoke({"messages": [HumanMessage(content=conversation_summary)]})
    return response.content


def _latest_human_message(state: dict) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


@tool
def call_tracking_agent(state: Annotated[dict, InjectedState]) -> str:
    """Call when the customer's message is about real-time shipment status: where a
    package is, delays, a customs hold, or a delivery attempt."""
    return tracking_agent_run(_latest_human_message(state))


@tool
def call_compliance_agent(state: Annotated[dict, InjectedState]) -> str:
    """Call when the question needs customs/import policy or FTC consumer-protection
    knowledge: duty or value thresholds, prohibited or restricted items, or refund
    rules for an order that never arrived."""
    return compliance_agent_run(_latest_human_message(state))


@tool
def call_escalation_agent(conversation_summary: str) -> str:
    """Call when the situation might need human escalation: suspected fraud or scam,
    a lost package the other specialists can't resolve, or a customer who seems to
    need a person. Pass a brief summary of the situation so far."""
    return escalation_agent_run(conversation_summary)
