"""The supervisor graph: one LLM with the three specialists bound as tools.

Standard LangGraph tool-calling loop (ToolNode + tools_condition), just applied to
specialist agents instead of plain functions:

    supervisor --(tool calls present)--> specialists --> supervisor --(none)--> END

The model itself decides whether to call zero, one, or more than one specialist —
this is the "genuine model-driven tool use" piece from docs/ROADMAP.md Sec. 1,
applied to routing, not just to the tracking tool.

build_graph() takes an optional checkpointer (see agents/converse.py for the
multi-turn chatbot that uses one); with none, it's single-turn, as used by chat.py.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.specialists import call_compliance_agent, call_escalation_agent, call_tracking_agent

load_dotenv()
MODEL = os.environ.get("AGENT_MODEL", "gpt-4o-mini")

SUPERVISOR_SYSTEM_PROMPT = SystemMessage(content=(
    "You are the supervisor for a delivery company's support chatbot. You never "
    "answer from your own knowledge — you route to specialists and combine their "
    "findings into one clear reply. You have three specialist tools: "
    "call_tracking_agent, call_compliance_agent, call_escalation_agent. Call "
    "whichever ones are genuinely needed — zero, one, or more than one — based on "
    "the customer's message. If you don't have enough information yet (e.g. a "
    "tracking question with no tracking number given), ask a clarifying question "
    "instead of guessing or calling a tool anyway. Once you have what you need, "
    "write one coherent final reply to the customer — don't just paste the "
    "specialists' findings back verbatim. If a specialist reports that it doesn't "
    "have an answer (e.g. the knowledge base doesn't cover something), relay that "
    "honestly — do NOT fill the gap with your own outside knowledge, even if it "
    "feels like common sense or you're confident it's correct."
))

_tools = [call_tracking_agent, call_compliance_agent, call_escalation_agent]
_supervisor_llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(_tools)


def supervisor_node(state: MessagesState) -> dict:
    response = _supervisor_llm.invoke([SUPERVISOR_SYSTEM_PROMPT, *state["messages"]])
    return {"messages": [response]}


def build_graph(checkpointer=None):
    graph = StateGraph(MessagesState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("specialists", ToolNode(_tools))
    graph.set_entry_point("supervisor")
    graph.add_conditional_edges("supervisor", tools_condition, {"tools": "specialists", END: END})
    graph.add_edge("specialists", "supervisor")
    return graph.compile(checkpointer=checkpointer)
