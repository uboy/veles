"""Регрессия ссылок на источники: /v1/source (HTML-сборка, оригинал, авторизация)
и кликабельные цитаты/сноски в render_chat_content."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tests.test_api import make_pipeline
from veles.api import app as app_module
from veles.api.app import (
    create_app,
    link_sig,
    render_chat_content,
    signed_source_url,
    source_url,
)


def _make_canon(tmp_path, title="Тестовый документ", heading="Раздел 1",
                text="Норма управляемости от 5 до 9."):
    canon = tmp_path / "data" / "canonical-test"
    doc_dir = canon / "lectures_x.txt"
    doc_dir.mkdir(parents=True)
    (doc_dir / "manifest.json").write_text(
        json.dumps({"title": title, "document_id": "lectures/x.txt",
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
         "heading_path": [heading], "text": text,
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
    assert "](http://127.0.0.1:8400/v1/source/lectures/x.txt?token=tk&node=nv1#n-nv1)" in out
    assert "](http://127.0.0.1:8400/v1/source/lectures/k.txt?token=tk&node=nv2#n-nv2)" in out
    assert out.count("](http://") == 4  # 2 сноски + 2 строки источников
    # ревью владельца 29.09 (демо сервер): текст сноски - название документа,
    # а не голый номер (иначе непонятно, куда ведёт ссылка)
    assert "[«Пересказ»](http://127.0.0.1:8400" in out
    assert "[«Коучинг»](http://127.0.0.1:8400" in out


def test_signed_citation_link_keeps_sig_and_adds_fragment():
    """Подписанная ссылка (identity-режимы): sig в query не меняется, фрагмент
    добавляется после sig - сервер фрагмент не видит, подпись валидна."""
    secret, doc, node, exp = "s", "lectures/x.txt", "nv1", 1234567890
    url = signed_source_url(secret, doc, node, exp)
    assert url.endswith(f"?node=nv1&exp={exp}&sig={link_sig(secret, doc, node, exp)}#n-nv1")


def test_citation_fragment_matches_render_anchor(tmp_path, monkeypatch):
    """K8: фрагмент #n-<node_version_id> из собранной цитаты совпадает с якорем
    id='n-<...>' в HTML-рендере /v1/source - браузер скроллит сам (корень без
    якоря: в индекс и рендер он не идёт)."""
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    page = client.get("/v1/source/lectures/x.txt", params={"token": "t"}).text
    fragment = source_url("lectures/x.txt", "nv1", "t").rsplit("#", 1)[1]
    assert f"id='{fragment}'" in page


def test_source_shows_ingest_metadata(tmp_path, monkeypatch):
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert "2026-09-28T10:00:00" in r.text and "denis@corp" in r.text


# --- ревью K6 P2-3: reflect-XSS на /v1/source ---


def test_source_html_escapes_document_texts(tmp_path, monkeypatch):
    """Негатив: title/heading_path/текст из данных документа экранируются -
    сырой payload не попадает в HTML (иначе падает по своему assert'у)."""
    _make_canon(tmp_path, title="Тест <script>alert('t')</script>",
                heading="Раздел <b>1</b>",
                text="Текст <img src=x onerror='alert(2)'>")
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert "&lt;script&gt;" in r.text
    assert "<script>alert('t')</script>" not in r.text
    assert "Раздел &lt;b&gt;1&lt;/b&gt;" in r.text
    assert "<b>1</b>" not in r.text
    assert "<img src=x" not in r.text


def test_source_rejects_malformed_node(tmp_path, monkeypatch):
    """node валидируется строго ([\\w.-]+): JS-инъекция через query -> 400;
    валидный node по-прежнему работает (фрагмент ссылок цитат ведёт на анкер)."""
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt",
                   params={"token": "t", "node": "';fetch('//attacker')//"})
    assert r.status_code == 400
    ok = client.get("/v1/source/lectures/x.txt", params={"token": "t", "node": "nv1"})
    assert ok.status_code == 200
    assert "id='n-nv1'" in ok.text


def test_source_security_headers(tmp_path, monkeypatch):
    """Страница источника несёт тот же набор заголовков, что /tracker (общая
    функция, K8); CSP строгий БЕЗ unsafe-inline - инлайн-скрипта скролла нет,
    к анкеру ведёт #fragment в самих ссылках."""
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.get("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    csp = r.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp
    assert "unsafe-inline" not in csp
    assert "script-src" not in csp
    assert "frame-ancestors 'none'" in csp
    assert "<script" not in r.text  # инлайн-скрипт скролла убран совсем


def test_source_head_supported(tmp_path, monkeypatch):
    """K8: HEAD на /v1/source - 200 с теми же заголовками (раньше маршрут был
    только GET и curl -I получал 405)."""
    _make_canon(tmp_path)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    client = TestClient(create_app(pipeline=None, api_token="t"))
    r = client.head("/v1/source/lectures/x.txt", params={"token": "t"})
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'none'" in r.headers["Content-Security-Policy"]
    # несуществующий документ под HEAD - тот же 404, что под GET
    assert client.head("/v1/source/lectures/none.txt",
                       params={"token": "t"}).status_code == 404
