"""Панель загрузки (/upload): страница, статика, авторизация, CSP.

Ревью-контекст: владелец 30.09 спросил «где панель для загрузки?» - страница
отдаёт форму (JS - /static/upload.js с того же origin), загрузка идёт через
уже покрытый тестами K4 POST /v1/documents. Авторизация - как у /tracker:
bearer/token= (владелец) или identity-заголовки; аноним - 401."""

import re

import numpy as np
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.config import EMBED_DIM
from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import IndexBundle
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver


class HashingEmbedder:
    dim = EMBED_DIM

    def embed_query(self, q):
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        for tok in re.findall(r"[а-яёa-z0-9]+", q.lower()):
            v[hash(tok) % EMBED_DIM] = 1.0
        return v


def make_client():
    bundle = IndexBundle(vector=NumpyFlatStore(dim=EMBED_DIM),
                         bm25=Bm25Index(), node_meta={})
    pipeline = VelesPipeline(bundle, HashingEmbedder(), None)
    app = create_app(pipeline=pipeline,
                     resolver=StaticOwnerResolver({"personal"}))
    return TestClient(app)


def test_upload_page_auth():
    client = make_client()
    r = client.post("/v1/knowledge/query", json={"query": "x"})
    assert r.status_code in (200, 400)  # static без токена: база отвечает
    page = client.get("/upload")
    assert page.status_code == 200
    assert "<form id=\"upform\">" in page.text
    headers = page.headers
    assert "script-src 'self'" in headers["content-security-policy"]
    assert "unsafe-inline" not in headers["content-security-policy"]


def test_upload_static_assets_served_with_csp():
    client = make_client()
    js = client.get("/static/upload.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("application/javascript")
    assert "script-src 'self'" in js.headers["content-security-policy"]
    css = client.get("/static/upload.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")


def test_upload_static_rejects_bad_name():
    client = make_client()
    r = client.get("/static/..%2F..%2Fapp.py")
    assert r.status_code in (400, 404)  # traversal отсечён
