"""Local web UI for the supervisor chatbot.

Wraps the same LangGraph supervisor + checkpointer from agents/converse.py behind
a tiny HTTP API, so the chatbot runs in a browser instead of a terminal. This is
also the thing Docker/Azure will containerize later (Sec. 8 step 8) — same app,
just packaged differently. In-memory checkpointer, same as converse.py: history is
lost on server restart.

Usage (from repo root, with .venv active):
    PYTHONPATH=src uvicorn web.main:app --reload --app-dir src
Then open http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agents.graph import build_graph
from agents.tracing import flush, get_callbacks, session_metadata

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_graph = build_graph(checkpointer=InMemorySaver())


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    config = {
        "configurable": {"thread_id": req.session_id},
        "callbacks": get_callbacks(),
        "metadata": session_metadata(req.session_id),
    }
    result = _graph.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    flush()
    return ChatResponse(reply=result["messages"][-1].content)
