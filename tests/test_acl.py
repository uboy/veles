"""Регрессия ACL-шва (Ф5 groundwork): pre-retrieval trimming по группам.

Критерий AC5: запрос без прав -> 0 фрагментов закрытого документа в контексте.
Индетичность приходит от IdentityResolver (инвариант 5), не от клиента.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_api import FakeEmbedder
from veles.api.app import VelesPipeline, create_app
from veles.indexing.indexer import IndexBundle
from veles.policy.identity import StaticOwnerResolver
from veles.retrieval.retriever import Citation, Evidence


class FakeVector:
    def search(self, q, top_k=10):
        return [("pub1", 0.90), ("sec1", 0.85)]  # закрытый узел даже ВЫШЕ

    def save(self, dir_path):
        raise NotImplementedError


class FakeBm25:
    def search(self, q, top_k=10):
        return [("pub1", 5.0), ("sec1", 4.5)]

    def save(self, dir_path):
        raise NotImplementedError


def make_acl_bundle() -> IndexBundle:
    base = {
        "text": "Публичная норма управляемости от 5 до 9.",
        "access_groups": ["personal"],
        "title": "Публичный документ",
        "document_id": "lectures/pub.txt",
        "heading_path": ["Публичный"],
        "provenance": {"adapter": "txt-md:0.1", "source_id": "pub.txt", "lines": [1, 1]},
    }
    secret = dict(base, text="Зарплатная ведомость: конфиденциальные суммы.",
                  access_groups=["finance-restricted"], title="Закрытый документ",
                  document_id="finance/secret.txt", heading_path=["Зарплаты"],
                  provenance={"adapter": "txt-md:0.1", "source_id": "secret.txt", "lines": [2, 2]})
    meta = {"pub1": base, "sec1": secret}
    return IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta)


def make_evidence(nid: str) -> Evidence:
    return Evidence(
        node_version_id=nid, text="x", rrf=0.03,
        citations=[Citation(title="t", document_id="d", heading_path=[],
                            source_id="s", lines=None, node_version_id=nid)],
    )


def test_leakage_zero_without_rights() -> None:
    """AC5: без прав закрытый узел не попадает в evidence, даже релевантнее."""
    p = VelesPipeline(make_acl_bundle(), FakeEmbedder(), llm=None)
    result = p.answer("зарплатная ведомость", top_k=5, use_llm=False,
                      allowed_groups={"personal"})
    texts = [ev["text"] for ev in result["evidence"]]
    assert texts, "публичный узел должен остаться в выдаче"
    assert "Зарплатная ведомость: конфиденциальные суммы." not in texts


def test_restricted_visible_with_rights() -> None:
    p = VelesPipeline(make_acl_bundle(), FakeEmbedder(), llm=None)
    result = p.answer("зарплатная ведомость", top_k=5, use_llm=False,
                      allowed_groups={"personal", "finance-restricted"})
    texts = [ev["text"] for ev in result["evidence"]]
    assert "Зарплатная ведомость: конфиденциальные суммы." in texts


def test_resolver_wired_via_http() -> None:
    app = create_app(pipeline=VelesPipeline(make_acl_bundle(), FakeEmbedder(), llm=None),
                     resolver=StaticOwnerResolver(groups={"personal"}))
    client = TestClient(app)
    d = client.post("/v1/knowledge/query", json={"query": "зарплатная ведомость", "use_llm": False}).json()
    assert all(ev["text"] != "Зарплатная ведомость: конфиденциальные суммы." for ev in d["evidence"])
    app2 = create_app(pipeline=VelesPipeline(make_acl_bundle(), FakeEmbedder(), llm=None),
                      resolver=StaticOwnerResolver(groups={"personal", "finance-restricted"}))
    d2 = TestClient(app2).post("/v1/knowledge/query",
                               json={"query": "зарплатная ведомость", "use_llm": False}).json()
    assert any(ev["text"] == "Зарплатная ведомость: конфиденциальные суммы." for ev in d2["evidence"])


def test_static_resolver_from_env(monkeypatch) -> None:
    monkeypatch.setenv("VELES_ALLOWED_GROUPS", "personal, finance-restricted")
    assert StaticOwnerResolver.from_env().allowed_groups() == {"personal", "finance-restricted"}
    monkeypatch.delenv("VELES_ALLOWED_GROUPS")
    assert "personal" in StaticOwnerResolver.from_env().allowed_groups()
