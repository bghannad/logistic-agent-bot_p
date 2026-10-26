"""The /chat endpoint, with the real graph swapped for a fake — no OpenAI call,
no LangGraph execution, no network. get_callbacks/flush are also stubbed out as
a second layer of protection against any accidental real call."""

from fastapi.testclient import TestClient

from web import main


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGraph:
    def __init__(self) -> None:
        self.last_config = None

    def invoke(self, _input, config=None):
        self.last_config = config
        return {"messages": [FakeMessage("fixed reply")]}


def test_chat_endpoint_returns_reply_and_passes_session_id_as_thread_id(monkeypatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr(main, "_graph", fake_graph)
    monkeypatch.setattr(main, "get_callbacks", lambda: [])
    monkeypatch.setattr(main, "flush", lambda: None)

    client = TestClient(main.app)
    response = client.post("/chat", json={"session_id": "abc-123", "message": "hello"})

    assert response.status_code == 200
    assert response.json() == {"reply": "fixed reply"}
    assert fake_graph.last_config["configurable"]["thread_id"] == "abc-123"
