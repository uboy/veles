"""Регрессия /v1/knowledge/stats: оглавление базы для /stats /sources /toc моста."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_api import make_pipeline
from veles.api.app import create_app


def test_stats_shape_and_totals() -> None:
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.get("/v1/knowledge/stats")
    assert r.status_code == 200
    d = r.json()
    assert d["totals"] == {"documents": 2, "nodes": 2}
    assert d["embed_model"]
    titles = {doc["title"] for doc in d["documents"]}
    assert titles == {"Пересказ Общий менеджмент", "Коучинг"}
    for doc in d["documents"]:
        assert isinstance(doc["sections"], list)
        assert doc["nodes"] >= 1
    managment = next(doc for doc in d["documents"] if "менеджмент" in doc["title"].lower())
    assert managment["sections"] == ["Общий менеджмент"]


def test_stats_respects_bearer_token() -> None:
    app = create_app(pipeline=make_pipeline(), api_token="secret-token")
    client = TestClient(app)
    assert client.get("/v1/knowledge/stats").status_code == 401
    r = client.get("/v1/knowledge/stats", headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200
