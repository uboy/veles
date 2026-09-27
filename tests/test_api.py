"""Тесты Ф4: фасад (канонический + OpenAI-совместимый), codex-сигнал, деградация LLM.

Без реальных моделей: фейковый эмбеддер + ручной IndexBundle (как в test_slice).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.api.llm import LLMConfig, OpenAICompatibleProvider
from veles.indexing.indexer import IndexBundle
from veles.policy.answer import decide
from veles.retrieval.retriever import Citation, Evidence


def make_bundle():
    class FakeVector:
        def search(self, q, top_k=10):
            return [("n1", 0.95), ("n2", 0.3)]

    class FakeBm25:
        def search(self, q, top_k=10):
            return [("n1", 5.0), ("n2", 0.4)]

    meta = {
        "n1": {
            "text": "Норма управляемости руководителя - от 5 до 9 пунктов.",
            "access_groups": ["personal"],
            "title": "Пересказ Общий менеджмент",
            "document_id": "lectures/Общий менеджмент/лекция2/пересказ.txt",
            "heading_path": ["Общий менеджмент"],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "пересказ.txt", "lines": [82, 82]},
        },
        "n2": {
            "text": "Коучинг работает с целями и мотивацией.",
            "access_groups": ["personal"],
            "title": "Коучинг",
            "document_id": "lectures/Коучинг/лекция1/пересказ.txt",
            "heading_path": [],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "коучинг.txt", "lines": None},
        },
    }
    return IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta)


class FakeEmbedder:
    dim = 4

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


def make_pipeline(llm: OpenAICompatibleProvider | None = None) -> VelesPipeline:
    return VelesPipeline(make_bundle(), FakeEmbedder(), llm)


def test_knowledge_query_answer_with_citations():
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "какая норма управляемости", "top_k": 3})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "answer"
    assert "5 до 9" in d["answer"]
    assert d["citations"][0]["title"].startswith("Пересказ")
    assert d["citations"][0]["lines"] == [82, 82]
    assert d["request_id"]


def test_knowledge_query_abstain_on_unknown_codex():
    """Ф3-урок: вопрос ссылается на кодекс, которого нет в корпусе -> abstain."""
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "какая ставка НДС согласно НК РФ"})
    d = r.json()
    assert d["decision"] == "abstain"


def test_chat_completions_shape_and_content():
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "whatever",
        "messages": [
            {"role": "system", "content": "ты помощник"},
            {"role": "user", "content": "какая норма управляемости"},
        ],
    })
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["model"] == "whatever"
    assert "Источники:" in d["choices"][0]["message"]["content"]
    assert d["veles"]["decision"] == "answer"


def test_llm_degradation_to_extractive():
    """LLM настроен, но недоступен -> фасад отвечает извлечённым фрагментом (200)."""
    bad_llm = OpenAICompatibleProvider(
        LLMConfig(base_url="http://127.0.0.1:1/v1", model="x", timeout_s=1)
    )
    app = create_app(pipeline=make_pipeline(llm=bad_llm))
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "какая норма управляемости"})
    d = r.json()
    assert r.status_code == 200 and d["decision"] == "answer"
    assert d["diagnostics"]["llm_used"] is False
    assert d["diagnostics"]["llm_error"]


def test_llm_synthesis_path_with_stub():
    """Доступный LLM -> синтезированный ответ, llm_used=True."""

    class StubLLM(OpenAICompatibleProvider):
        def __init__(self):  # noqa: D107 - конфиг не нужен заглушке
            super().__init__(LLMConfig(base_url="stub", model="stub"))

        def synthesize(self, question, evidence_texts):
            return "Синтезированный ответ [1]."

    app = create_app(pipeline=make_pipeline(StubLLM()))
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "какая норма управляемости"})
    d = r.json()
    assert d["diagnostics"]["llm_used"] is True
    assert d["answer"].startswith("Синтезированный")


def test_auth_token_required():
    app = create_app(pipeline=make_pipeline(), api_token="secret")
    client = TestClient(app)
    assert client.post("/v1/knowledge/query", json={"query": "q"}).status_code == 401
    ok = client.post(
        "/v1/knowledge/query",
        json={"query": "q"},
        headers={"Authorization": "Bearer secret"},
    )
    assert ok.status_code == 200


def test_decide_codex_miss_in_policy():
    ev = Evidence("n1", "t", 0.03, 1, 1, [Citation("t", "d", [], "s", None, "n1")])
    d = decide([ev], codex_refs=["нк рф"], doc_titles={"Гражданский кодекс.pdf"})
    assert d.decision == "abstain"
    d2 = decide([ev], codex_refs=["нк рф"], doc_titles={"Налоговый кодекс Российской Федерации.pdf"})
    assert d2.decision == "answer"
