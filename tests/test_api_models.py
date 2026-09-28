"""Регрессия Ф4-2: OpenAI-совместимый /v1/models для клиентов (Open WebUI)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from veles.api.app import create_app


def test_models_open_without_token() -> None:
    app = create_app(pipeline=None, api_token="")
    client = TestClient(app)
    r = client.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert ids == ["veles-knowledge", "veles-assistant"]
    assert all(m["object"] == "model" for m in data["data"])


def test_models_respects_bearer_token() -> None:
    app = create_app(pipeline=None, api_token="secret-token")
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "veles-knowledge"


def _empty_retrieval_pipeline(llm):
    """Пайплайн, где retrieval не находит ничего (честный abstain-путь)."""
    from veles.indexing.indexer import IndexBundle
    from test_api import FakeEmbedder

    class EmptyVector:
        def search(self, q, top_k=10):
            return []

    class EmptyBm25:
        def search(self, q, top_k=10):
            return []

    bundle = IndexBundle(vector=EmptyVector(), bm25=EmptyBm25(), node_meta={
        "n1": {"text": "Норма управляемости руководителя - от 5 до 9 пунктов.",
               "access_groups": ["personal"], "title": "Пересказ",
               "document_id": "lectures/x.txt", "heading_path": ["Менеджмент"],
               "provenance": {"adapter": "txt-md:0.1", "source_id": "x.txt", "lines": [1, 1]}},
    })
    from veles.api.app import VelesPipeline
    return VelesPipeline(bundle, FakeEmbedder(), llm=llm)


def test_veles_assistant_fallback_on_abstain() -> None:
    """veles-assistant: вне корпуса -> общий ответ GLM с пометкой, без цитат."""
    from fastapi.testclient import TestClient

    class StubLLM:
        enabled = True

        def answer_general(self, question: str) -> str:
            return f"Общий ответ на {question}"

    app = create_app(pipeline=_empty_retrieval_pipeline(StubLLM()))
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "veles-assistant",
        "messages": [{"role": "user", "content": "как приготовить борщ"}],
    })
    d = r.json()
    assert d["model"] == "veles-assistant"
    content = d["choices"][0]["message"]["content"]
    assert "Общий ответ" in content
    assert "В базе знаний ответа" in content
    assert d["veles"]["decision"] == "abstain"


def test_veles_knowledge_stays_pure_kb_on_abstain() -> None:
    """veles-knowledge: abstain БЕЗ fallback - база остаётся строгой."""
    from fastapi.testclient import TestClient

    class NoLLM:
        enabled = False

    app = create_app(pipeline=_empty_retrieval_pipeline(NoLLM()))
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "veles-knowledge",
        "messages": [{"role": "user", "content": "как приготовить борщ"}],
    })
    d = r.json()
    content = d["choices"][0]["message"]["content"]
    assert "в базе знаний нет" in content
    assert "общий ответ" not in content
