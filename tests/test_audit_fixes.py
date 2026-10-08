"""Тесты фиксов аудита 08.10: ссылки без владельческого токена в identity-режиме,
таймаут разбора в отдельном процессе, отказ chunked-загрузки без размера,
дозаполнение выдачи для узких групп."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from veles.api.app import attach_citation_links, create_app, render_chat_content
from veles.api.llm import OpenAICompatibleProvider
from veles.ingest.queue import IsolatedTimeout, _run_isolated
from veles.indexing.indexer import IndexBundle
from veles.retrieval.retriever import hybrid_search


# --- Фикс 1: владельческий токен не утекает в ссылки вне static-режима --------


def _result() -> dict:
    return {
        "decision": "answer",
        "answer": "Ответ.",
        "citations": [{"n": 1, "title": "Док", "source_id": "d.md",
                        "document_id": "docs/d.md", "node_version_id": "n1",
                        "asset": "pic-001.png", "lines": [1, 1], "lost_force": False}],
        "diagnostics": {"confidence": 0.5, "reason": "", "conflicts": []},
    }


def test_no_mint_no_token_links_in_citations():
    out = attach_citation_links(_result(), token="owner-token",
                                mint_token_links=False)
    assert "source_url" not in out["citations"][0]
    assert "asset_url" not in out["citations"][0]
    assert "owner-token" not in repr(out)


def test_static_mode_keeps_token_links():
    out = attach_citation_links(_result(), token="owner-token",
                                mint_token_links=True)
    assert "token=owner-token" in out["citations"][0]["source_url"]
    assert "token=owner-token" in out["citations"][0]["asset_url"]


def test_render_no_mint_no_token_in_text():
    text = render_chat_content(_result(), token="owner-token", mint_token_links=False)
    assert "owner-token" not in text
    assert "Источники:" in text  # источники остаются списком названий


# --- Фикс 2: разбор в отдельном процессе с таймаутом --------------------------


def _sleepy(seconds: float) -> str:
    time.sleep(seconds)
    return "done"


def _boom() -> str:
    raise ValueError("разбор сломался")


def _ok() -> str:
    return "parsed"


def test_isolated_returns_result():
    assert _run_isolated(_ok, (), 10) == "parsed"


def test_isolated_propagates_exception():
    with pytest.raises(ValueError, match="разбор сломался"):
        _run_isolated(_boom, (), 10)


def test_isolated_timeout_kills_hang():
    t0 = time.monotonic()
    with pytest.raises(IsolatedTimeout):
        _run_isolated(_sleepy, (30.0,), 1.0)
    assert time.monotonic() - t0 < 15


# --- Фикс 3: chunked без Content-Length отклоняется до чтения тела ------------


class FakeVector:
    def search(self, q, top_k=10):
        return [("n1", 0.9)]


class FakeBm25:
    def search(self, q, top_k=10):
        return [("n1", 5.0)]


def _pipeline():
    from veles.api.app import VelesPipeline

    meta = {
        "n1": {
            "text": "Норма управляемости - от 5 до 9 пунктов.",
            "access_groups": ["personal"], "title": "Док",
            "document_id": "docs/d.md", "heading_path": [],
            "provenance": {"adapter": "t", "source_id": "d.md", "lines": None},
        }
    }

    class FakeEmbedder:
        dim = 4

        def embed_query(self, text):
            import numpy as np

            return np.zeros(4, dtype=np.float32)

    return VelesPipeline(
        IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta),
        FakeEmbedder(), None,
    )


def test_upload_without_content_length_rejected_411(monkeypatch, tmp_path):
    monkeypatch.setenv("VELES_DATA", str(tmp_path))
    monkeypatch.setenv("VELES_UPLOAD_MAX_MB", "10")
    app = create_app(pipeline=_pipeline())
    client = TestClient(app)
    # httpx с итератором тела не знает размер -> уходит без Content-Length
    r = client.post(
        "/v1/documents",
        content=iter([b"x" * 1024]),
        headers={"Authorization": "Bearer t",
                 "Content-Type": "multipart/form-data; boundary=x"},
    )
    assert r.status_code == 411, r.text
    assert "Content-Length" in r.json()["detail"]


# --- Фикс 4: узкая группа получает полные top_k (глубина каналов добирается) --


def _narrow_group_bundle():
    """40 узлов: 0-29 в чужой группе, 30-39 в personal; каналы отдают по порядку
    индекса - раньше узкая группа оставалась с пустой выдачей на глубине top_k*5."""

    class DeepVector:
        def search(self, q, top_k=10):
            return [(f"n{i}", 1.0 / (i + 1)) for i in range(min(top_k, 40))]

    class DeepBm25:
        def search(self, q, top_k=10):
            return [(f"n{i}", 1.0 / (i + 1)) for i in range(min(top_k, 40))]

    meta = {}
    for i in range(40):
        meta[f"n{i}"] = {
            "text": f"текст узла {i} про отпуск и приказы",
            "access_groups": ["other"] if i < 30 else ["personal"],
            "title": "Кадровый регламент", "document_id": "docs/hr.md",
            "heading_path": [],
            "provenance": {"adapter": "t", "source_id": "hr.md", "lines": None},
        }
    return IndexBundle(vector=DeepVector(), bm25=DeepBm25(), node_meta=meta)


def test_narrow_group_gets_full_topk():
    import numpy as np

    class FakeEmbedder:
        dim = 4

        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    evidence = hybrid_search(
        _narrow_group_bundle(), np.zeros(4, dtype=np.float32),
        "отпуск", top_k=5, allowed_groups={"personal"},
    )
    ids = [e.node_version_id for e in evidence]
    assert len(evidence) == 5  # раньше было 0 - неавторизованные съели глубину
    assert all(e.citations[0].document_id == "docs/hr.md" for e in evidence)
    assert "n30" in ids  # первый personal-узел добрался из глубины каналов


# --- Фикс 5 (аудит 08.10, ранжирование): тема вопроса в названии документа ----


def _titled_bundle():
    """10 узлов 'Setup Guide' наверху каналов, 5 узлов 'Sync Build' - глубже;
    по словам содержания обе темы совпадают с вопросом, различие - название."""

    class OrderedVector:
        def search(self, q, top_k=10):
            ids = [f"s{i}" for i in range(10)] + [f"b{i}" for i in range(5)]
            return [(i, 1.0 / (n + 1)) for n, i in enumerate(ids[:top_k])]

    meta = {}
    for i in range(10):
        meta[f"s{i}"] = {
            "text": "repo sync and build instructions for the platform",
            "access_groups": ["personal"], "title": "Setup Guide",
            "document_id": "docs/setup.md", "heading_path": [],
            "provenance": {"adapter": "t", "source_id": "setup.md", "lines": None},
        }
    for i in range(5):
        meta[f"b{i}"] = {
            "text": "repo sync and build instructions for the platform",
            "access_groups": ["personal"], "title": "Sync Build",
            "document_id": "docs/sync-build.md", "heading_path": [],
            "provenance": {"adapter": "t", "source_id": "sync-build.md", "lines": None},
        }
    return IndexBundle(vector=OrderedVector(), bm25=OrderedVector(), node_meta=meta)


def test_title_boost_lifts_matching_document():
    import numpy as np

    class FakeEmbedder:
        dim = 4

        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    ev = hybrid_search(_titled_bundle(), np.zeros(4, dtype=np.float32),
                       "how to build and sync openharmony", top_k=5,
                       title_boost=1.0)
    assert any(e.citations[0].document_id == "docs/sync-build.md" for e in ev), \
        [e.citations[0].document_id for e in ev]


def test_title_boost_zero_keeps_order():
    import numpy as np

    class FakeEmbedder:
        dim = 4

        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    ev = hybrid_search(_titled_bundle(), np.zeros(4, dtype=np.float32),
                       "how to build and sync openharmony", top_k=3,
                       title_boost=0.0)
    assert all(e.citations[0].document_id == "docs/setup.md" for e in ev)
