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
    assert [m["id"] for m in data["data"]] == ["veles-knowledge"]
    assert data["data"][0]["object"] == "model"


def test_models_respects_bearer_token() -> None:
    app = create_app(pipeline=None, api_token="secret-token")
    client = TestClient(app)
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "veles-knowledge"
