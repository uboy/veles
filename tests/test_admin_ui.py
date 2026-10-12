"""Админ-панель (/admin): страница-оболочка, статика, CSP, группы в stats.

Панель - shell без секретов: страница и её статика (/static/admin.js|css)
отдаются БЕЗ токена даже при настроенном VELES_API_TOKEN (браузер грузит
подресурсы без заголовков; токен живёт в localStorage и ходит только в
Authorization JSON-запросов). Единственная правка данных API: каждый документ
/v1/knowledge/stats несёт groups - объединение access_groups живых видимых
узлов (кормит вкладки «Документы» и «Группы»); новых JSON-эндпоинтов нет.
Негатив: при настроенном токене все API-вкладки панели (stats/groups/jobs/
models) без bearer - 401, панель показывает просьбу ввести токен.
Админка v2: страница несёт массовую панель «Документов» (чекбоксы, смена
группы, массовое удаление), скрипт - кнопки задач и бейджи; новые эндпоинты
(POST {id}/groups, jobs cancel/retry/delete) - только под bearer владельца.
Индекс - ручная мета узлов; эмбеддер - детерминированный hashing (канон:
тесты без загрузки моделей), как в test_upload_ui.
"""

from __future__ import annotations

import re
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.audit import reset_audit
from veles.config import EMBED_DIM
from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import IndexBundle
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver

TOKEN = "admin-secret"


class HashingEmbedder:
    dim = EMBED_DIM

    def embed_query(self, q):
        v = np.zeros(EMBED_DIM, dtype=np.float32)
        for tok in re.findall(r"[а-яёa-z0-9]+", q.lower()):
            v[hash(tok) % EMBED_DIM] = 1.0
        return v


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    """Тесты не зависят от окружения машины и не пишут в репозиторий:
    token/identity-режим снят (кроме теста, где токен задаётся явно), файл
    групп и журнал джобов уезжают в tmp. Синглтон журнала аудита сбрасывается
    на каждый тест: он кэширует VELES_DATA при первом вызове - без сброса
    события ушли бы в tmp первого теста сессии."""
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    monkeypatch.delenv("VELES_LINK_SECRET", raising=False)
    monkeypatch.delenv("VELES_GROUPS_FILE", raising=False)
    for name in ("VELES_AUDIT", "VELES_AUDIT_ROTATE_MB", "VELES_AUDIT_KEEP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "veles-data"))
    reset_audit()
    yield tmp_path
    reset_audit()


def make_bundle() -> IndexBundle:
    """Мета четырёх узлов: живой документ в двух группах, superseded-узел,
    второй живой документ и tombstone - два последних в панель не попадают."""
    return IndexBundle(
        vector=NumpyFlatStore(dim=EMBED_DIM), bm25=Bm25Index(),
        node_meta={
            "n1": {"document_id": "uploads/reglament.md", "title": "Регламент закупок",
                   "access_groups": ["personal", "legal"], "status": "current",
                   "heading_path": ["Общие положения"]},
            "n2": {"document_id": "uploads/reglament.md", "title": "Регламент закупок",
                   "access_groups": ["personal"], "status": "superseded",
                   "heading_path": []},
            "n3": {"document_id": "uploads/otchet.md", "title": "Годовой отчёт",
                   "access_groups": ["personal"], "status": "current", "heading_path": []},
            "n4": {"document_id": "uploads/udalennyj.md", "title": "Удалённый документ",
                   "access_groups": ["secret-group"], "status": "deleted",
                   "heading_path": []},
        },
    )


def make_client(api_token: str | None = None) -> TestClient:
    app = create_app(
        pipeline=VelesPipeline(make_bundle(), HashingEmbedder(), llm=None),
        api_token=api_token,
        resolver=StaticOwnerResolver({"personal", "legal"}),
    )
    return TestClient(app)


# ------------------------------------------------------- страница и статика


def test_admin_page_served_without_token() -> None:
    """Страница - shell без секретов: 200 без bearer даже при настроенном
    токене; CSP со script-src 'self' и без unsafe-inline (инлайн-скриптов нет)."""
    client = make_client(api_token=TOKEN)
    r = client.get("/admin")
    assert r.status_code == 200
    assert 'src="/static/admin.js"' in r.text
    assert 'href="/static/admin.css"' in r.text
    csp = r.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp


def test_admin_static_assets_open_with_csp() -> None:
    """Статика панели отдаётся без токена (иначе shell не соберётся: браузер
    грузит js/css без заголовков); content-type и CSP - как у /upload."""
    client = make_client(api_token=TOKEN)
    js = client.get("/static/admin.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("application/javascript")
    assert "script-src 'self'" in js.headers["content-security-policy"]
    css = client.get("/static/admin.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")


def test_static_traversal_rejected() -> None:
    """traversal по /static отсечён (та же проверка имени, что у /upload)."""
    client = make_client()
    r = client.get("/static/..%2F..%2Fapp.py")
    assert r.status_code in (400, 404)


# ------------------------------------------------------------- groups в stats


def test_stats_documents_carry_groups() -> None:
    """groups документа - объединение access_groups живых узлов из того же
    снапшота; superseded и tombstone в список и в группы не попадают."""
    client = make_client()
    r = client.get("/v1/knowledge/stats")
    assert r.status_code == 200
    d = r.json()
    by_id = {doc["document_id"]: doc for doc in d["documents"]}
    assert set(by_id) == {"uploads/reglament.md", "uploads/otchet.md"}
    assert by_id["uploads/reglament.md"]["groups"] == ["legal", "personal"]
    assert by_id["uploads/otchet.md"]["groups"] == ["personal"]
    assert d["totals"] == {"documents": 2, "nodes": 2, "superseded_nodes": 1}
    for doc in d["documents"]:
        assert isinstance(doc["groups"], list)


# ---------------------------------------------------- авторизация API-вкладок


def test_api_tabs_require_bearer_when_token_set() -> None:
    """Негатив: при настроенном VELES_API_TOKEN все JSON-вкладки панели без
    bearer - 401 (панель просит токен), с bearer - 200 с ожидаемой структурой.
    Файл групп и журнал джобов уходят в tmp из clean_env (репозиторий не
    засоряется): groups_file/uploads_root читают окружение на вызов."""
    client = make_client(api_token=TOKEN)
    assert client.get("/admin").status_code == 200
    for path in ("/v1/knowledge/stats", "/v1/groups", "/v1/documents/jobs",
                 "/v1/models"):
        assert client.get(path).status_code == 401, path

    auth = {"Authorization": f"Bearer {TOKEN}"}
    stats = client.get("/v1/knowledge/stats", headers=auth)
    assert stats.status_code == 200
    body = stats.json()
    assert {"totals", "embed_model", "documents"} <= set(body)
    assert {"document_id", "title", "nodes", "sections", "groups"} \
        <= set(body["documents"][0])

    gr = client.get("/v1/groups", headers=auth)
    assert gr.status_code == 200
    items = gr.json()["groups"]
    assert items and all({"id", "title"} <= set(g) for g in items)
    assert not any(g["id"].startswith("user:") for g in items)

    jobs = client.get("/v1/documents/jobs", headers=auth)
    assert jobs.status_code == 200
    assert jobs.json() == {"jobs": []}

    models = client.get("/v1/models", headers=auth)
    assert models.status_code == 200
    assert [m["id"] for m in models.json()["data"]]


# --------------------------------------------------- админка v2: wiring и гейты


def test_admin_page_wires_bulk_and_job_actions() -> None:
    """Страница несёт массовую панель «Документов» (v2: чекбоксы, смена
    группы, массовое удаление), а скрипт - вызовы новых эндпоинтов и бейджи."""
    client = make_client()
    page = client.get("/admin")
    assert page.status_code == 200
    for marker in ('id="bulkpanel"', 'id="bulkgroup"', 'id="bulkregroup"',
                   'id="bulkdelete"', 'id="bulkprogress"'):
        assert marker in page.text, marker
    js = client.get("/static/admin.js").text
    for marker in ("selrow", "selall", "/groups", "/cancel",
                   "/retry", "jobact", "jobBadges", "уже в базе",
                   "дубль задачи"):
        assert marker in js, marker


def test_regroup_endpoint_guards() -> None:
    """POST /v1/documents/{id}/groups - только под bearer владельца (401 без
    него); валидация тела до поиска документа (400: пустой список, unknown
    группа); несуществующий документ с валидным телом - 404."""
    client = make_client(api_token=TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    url = "/v1/documents/uploads/reglament.md/groups"
    assert client.post(url, json={"groups": ["test-dept"]}).status_code == 401
    assert client.post(
        url, json={"groups": []}, headers=auth).status_code == 400
    assert client.post(
        url, json={"groups": ["нет-такой"]}, headers=auth).status_code == 400
    assert client.post(
        url, json={"groups": ["user:битый!!"]}, headers=auth).status_code == 400
    assert client.post(
        url, json={"groups": ["test-dept"]}, headers=auth).status_code == 404


def test_jobs_action_guards() -> None:
    """cancel/retry/delete задач: без bearer - 401, неизвестная задача - 404
    (права как у DELETE /v1/documents; список полей v2 в пустом ответе не
    ломает форму {jobs: []})."""
    client = make_client(api_token=TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.get("/v1/documents/jobs", headers=auth).json() == {"jobs": []}
    for method, path in (("POST", "/v1/documents/jobs/abc/cancel"),
                         ("POST", "/v1/documents/jobs/abc/retry"),
                         ("DELETE", "/v1/documents/jobs/abc")):
        assert client.request(method, path).status_code == 401, path
        assert client.request(method, path, headers=auth).status_code == 404, path


# --------------------------------------------------------- вкладка «Аудит»


def test_admin_page_wires_audit_tab() -> None:
    """Страница несёт вкладку «Аудит» (кнопка в нав + секция): фильтры
    действие/кто/подстрока/период, «Обновить», «CSV», «Показать ещё»;
    скрипт - запрос /v1/admin/audit, offset-пагинация и экспорт через blob."""
    client = make_client()
    page = client.get("/admin")
    assert page.status_code == 200
    for marker in ('data-tab="audit"', 'id="tab-audit"', 'id="auditaction"',
                   'id="auditactor"', 'id="auditq"', 'id="auditsince"',
                   'id="audituntil"', 'id="auditrefresh"', 'id="auditcsv"',
                   'id="auditbox"', 'id="auditmore"'):
        assert marker in page.text, marker
    assert "Показать ещё" in page.text
    js = client.get("/static/admin.js").text
    for marker in ("/v1/admin/audit", "auditQuery", "renderAudit", "auditCsv",
                   "URLSearchParams", "URL.createObjectURL",
                   "audit_access_denied", "auth_ldap_fail", "job_retry"):
        assert marker in js, marker


def test_audit_tab_endpoint_contract() -> None:
    """Запрос вкладки (точные параметры auditQuery: action, actor, q,
    since, until, limit=100, offset) обрабатывается бэкендом: поиск пишется
    событием search, фильтры отсекают, вторая страница «Показать ещё» пуста
    при том же total; без bearer (вкладка без токена) - 401, панель покажет
    просьбу ввести токен."""
    client = make_client(api_token=TOKEN)
    auth = {"Authorization": f"Bearer {TOKEN}"}
    assert client.post(
        "/v1/knowledge/query", json={"query": "порог закупки", "use_llm": False},
        headers=auth,
    ).status_code == 200

    page1 = client.get("/v1/admin/audit?action=search&limit=100&offset=0",
                       headers=auth)
    assert page1.status_code == 200
    body = page1.json()
    assert body["total"] == 1 and body["events"][0]["action"] == "search"

    assert client.get("/v1/admin/audit?q=" + quote("порог закупки"),
                      headers=auth).json()["total"] == 1
    assert client.get("/v1/admin/audit?actor=owner",
                      headers=auth).json()["total"] == 1
    assert client.get("/v1/admin/audit?action=upload",
                      headers=auth).json()["total"] == 0
    assert client.get("/v1/admin/audit?since=2000-01-01",
                      headers=auth).json()["total"] == 1
    # «Показать ещё»: offset за пределами - событий нет, total прежний
    page2 = client.get("/v1/admin/audit?action=search&limit=100&offset=100",
                       headers=auth)
    assert page2.json()["total"] == 1 and page2.json()["events"] == []
    # вкладка без токена: 401 (static-режим), не 403
    assert client.get("/v1/admin/audit?action=search").status_code == 401


def test_audit_csv_neutralizes_formulas() -> None:
    """Ревью раунд 1 (CWE-1236): экспорт CSV обязан нейтрализовать ведущие
    =,+,-,@ и таб/CR (префикс '), до удвоения кавычек - details несут текст
    поиска/вопроса любого пользователя, исполнение формулы - Excel/LibreOffice
    админа. Поведенчески JS pytest не исполняет - проверяется наличие
    нейтрализации в коде экспорта (синтаксис - node --check вне pytest)."""
    js = make_client().get("/static/admin.js").text
    assert "function csvSafe" in js
    # класс опасных первых символов присутствует дословно
    assert "/^[=+\\-@\\t\\r]/" in js
    # csvSafe применяется ДО удвоения кавычек в csvCell
    assert js.index("csvSafe(v).replace") > js.index("function csvSafe")
