"""Регрессия ссылок на источники: /v1/source (HTML-сборка, оригинал, авторизация)
и кликабельные цитаты/сноски в render_chat_content."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.test_api import make_pipeline
from veles.api import app as app_module
from veles.api.app import create_app, render_chat_content


def _make_canon(tmp_path):
    canon = tmp_path / "data" / "canonical-test"
    doc_dir = canon / "lectures_x.txt"
    doc_dir.mkdir(parents=True)
    (doc_dir / "manifest.json").write_text(
        json.dumps({"title": "Тестовый документ", "document_id": "lectures/x.txt",
                    "revision_id": "r1", "parser_profile": "txt-md:0.1",
                    "access_groups": ["personal"], "sha256": "a", "status": "current",
                    "ingested_at": "2026-09-28T10:00:00+00:00", "ingested_by": "denis@corp",
                    "derived_from": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    nodes = [
        {"logical_node_id": "lectures/x.txt#root", "node_version_id": "root0",
         "node_type": "root", "parent_id": None, "heading_path": [], "text": "x",
         "provenance": {"adapter": "txt-md:0.1", "source_id": "x.txt", "lines": None},
         "access_groups": ["personal"]},
        {"logical_node_id": "lectures/x.txt#p1", "node_version_id": "nv1",
         "node_type": "paragraph", "parent_id": "lectures/x.txt#root",
         "heading_path": ["Раздел 1"], "text": "Норма управляемости от 5 до 9.",
         "provenance": {"adapter": "txt-md:0.1", "source_id": "x.txt", "lines": [7, 9]},
         "access_groups": ["personal"]},
    ]
    (doc_dir / "nodes.jsonl").write_text(
        "\n".join(json.dumps(n, ensure_ascii=False) for n in nodes), encoding="utf-8"
    )


def test_source_html_view_auth_anchors(tmp_path, monkeypatch):
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    app = create_app(pipeline=None, api_token="t")
    client = TestClient(app)
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t", "node": "nv1"})
    assert r.status_code == 200
    assert "Норма управляемости" in r.text
    assert "id='n-nv1'" in r.text
    assert "Тестовый документ" in r.text
    assert client.get("/v1/source/lectures/x.txt").status_code == 401
    assert client.get("/v1/source/lectures/x.txt", params={"token": "bad"}).status_code == 401
    assert client.get("/v1/source/lectures/none.txt", params={"token": "t"}).status_code == 404


def test_source_serves_original_file(tmp_path, monkeypatch):
    _make_canon(tmp_path)
    srcroot = tmp_path / "src"
    srcroot.mkdir()
    (srcroot / "x.txt").write_text("ОРИГИНАЛ-ФАЙЛ", encoding="utf-8")
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(app_module, "VELES_SOURCE_ROOTS", [srcroot])
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert "ОРИГИНАЛ-ФАЙЛ" in r.text
    assert "text/plain" in r.headers["content-type"]


def test_citations_and_footnotes_are_links():
    result = {
        "decision": "answer",
        "diagnostics": {"reason": "ok"},
        "answer": "Норма 5-9 [1], коучинг [2].",
        "citations": [
            {"n": 1, "title": "Пересказ", "source_id": "x.txt", "lines": [7, 9],
             "document_id": "lectures/x.txt", "node_version_id": "nv1"},
            {"n": 2, "title": "Коучинг", "source_id": "k.txt", "lines": None,
             "document_id": "lectures/k.txt", "node_version_id": "nv2"},
        ],
    }
    out = render_chat_content(result, token="tk")
    assert "](http://127.0.0.1:8400/v1/source/lectures/x.txt?token=tk&node=nv1)" in out
    assert "](http://127.0.0.1:8400/v1/source/lectures/k.txt?token=tk&node=nv2)" in out
    assert out.count("](http://") == 4  # 2 сноски + 2 строки источников


def test_source_shows_ingest_metadata(tmp_path, monkeypatch):
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert "2026-09-28T10:00:00" in r.text and "denis@corp" in r.text
