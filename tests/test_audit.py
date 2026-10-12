"""Журнал аудита действий (план docs/plan/admin-audit): запись, ротация,
фильтры, never-throw, точки записи фасада, доступ только админа.

Юнит: событие {ts, actor, admin, action, target, details, ip} в
<VELES_DATA>/audit/audit.jsonl; actor-маппинг (uid/owner/anonymous); обрезка
details до 500; ротация по размеру с хранением KEEP последних; фильтры
actor/action/q/since/until/limit/offset; битые строки чтение пропускает;
ошибка записи (журнал-путь занят файлом, файл только-чтение) - warning в лог,
запрос не ломается; VELES_AUDIT=0 выключает запись.
TestClient: search/ask пишут текст запроса; upload/документы/job_* пишут свои
события (цепочка с failed-задачей через сломанный pipeline_fn очереди -
детерминированно, без гонок); не-админ на /v1/admin/audit - 403 и событие
отказа; успешные чтения аудита и health/models/stats не логируются; пароль и
токен в журнале отсутствуют (LDAP-фейк и bearer). Индекс - настоящий
build_index; эмбеддер - детерминированный hashing (канон: тесты без загрузки
моделей), как в test_delete.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.audit import AuditLog, actor_of, get_audit, reset_audit
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import IngestQueue
from veles.indexing.indexer import build_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import RequestIdentity, StaticOwnerResolver
from veles.policy.ldap_identity import LdapAuthError, LdapIdentity

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

DOC_TEXT = "# Регламент закупок\n\nПорог единственной закупки - 350 тысяч рублей.\n"
DOC_TITLE = "Регламент закупок"


class HashingEmbedder:
    """Тот же стаб, что в test_delete: sublinear tf по хеш-бакетам."""

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture(autouse=True)
def audit_env(tmp_path, monkeypatch):
    """Изоляция окружения: VELES_DATA в tmp, аудит-переменные сняты,
    синглтон журнала пересоздан (env читается при первом вызове)."""
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "veles-data"))
    monkeypatch.delenv("VELES_AUDIT", raising=False)
    monkeypatch.delenv("VELES_AUDIT_ROTATE_MB", raising=False)
    monkeypatch.delenv("VELES_AUDIT_KEEP", raising=False)
    monkeypatch.delenv("VELES_RATE_LIMIT_RPM", raising=False)
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_GROUPS", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_JWT_SECRET", raising=False)
    monkeypatch.delenv("VELES_ADMIN_EMAILS", raising=False)
    monkeypatch.delenv("VELES_LDAP_URL", raising=False)
    reset_audit()
    yield tmp_path / "veles-data"
    reset_audit()


@pytest.fixture()
def kb_env(tmp_path, monkeypatch):
    """tmp-окружение с настоящим (пустым) индексом: канон и INDEX_DIR в tmp -
    upload через TestClient доходит до done, документ попадает в канон
    (нужно regroup/delete), как kb_env в test_delete."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    index_dir = data / "index"
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    monkeypatch.setattr("veles.config.INDEX_DIR", index_dir)
    monkeypatch.delenv("VELES_QUOTA_DOCS", raising=False)
    monkeypatch.delenv("VELES_QUOTA_MB", raising=False)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    return {"data": data, "canonical": canonical, "index_dir": index_dir, "bundle": bundle}


def make_static_client(bundle) -> TestClient:
    app = create_app(
        pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver({"personal"}),
    )
    return TestClient(app)


def journal_text(data) -> str:
    f = data / "audit" / "audit.jsonl"
    return f.read_text(encoding="utf-8") if f.exists() else ""


def journal_events(data) -> list[dict]:
    text = journal_text(data)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def actions(data) -> list[str]:
    return [e["action"] for e in journal_events(data)]


# ------------------------------------------------------------------- юнит


def test_event_fields_and_truncation(tmp_path) -> None:
    """Событие несёт ровно 7 полей; ts - ISO-8601 UTC; actor без личности -
    owner; длинные details/target обрезаются до 500 (план: текст запроса)."""
    log = AuditLog(tmp_path)
    log.log("search", target="т" * 600, details="д" * 600, ip="10.0.0.9")
    lines = journal_text(tmp_path).splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert set(event) == {"ts", "actor", "admin", "action", "target", "details", "ip"}
    ts = datetime.fromisoformat(event["ts"])
    assert ts.tzinfo is not None and ts.utcoffset().total_seconds() == 0
    assert event["actor"] == "owner"
    assert event["admin"] is False
    assert event["action"] == "search"
    assert event["target"] == "т" * 500
    assert event["details"] == "д" * 500
    assert event["ip"] == "10.0.0.9"


def test_actor_of_mapping() -> None:
    """actor: user_id личности; пустой user_id - owner (bearer-владелец);
    личности нет (подписанная ссылка) - anonymous."""
    assert actor_of(None) == "anonymous"
    assert actor_of(RequestIdentity(user_id="", admin=True, groups={"all"})) == "owner"
    assert actor_of(
        RequestIdentity(user_id="ivanov@corp.ru", admin=False, groups=set())
    ) == "ivanov@corp.ru"


def test_disabled_by_env(monkeypatch, tmp_path) -> None:
    """VELES_AUDIT=0 выключает запись (чтение не ломается - файла просто нет);
    без переменной запись работает."""
    monkeypatch.setenv("VELES_AUDIT", "0")
    reset_audit()
    get_audit().log("search", details="не запишется")
    assert journal_text(tmp_path / "veles-data") == ""
    monkeypatch.delenv("VELES_AUDIT")
    reset_audit()
    get_audit().log("search", details="запишется")
    assert len(journal_events(tmp_path / "veles-data")) == 1


def test_rotation_keeps_last_files(tmp_path) -> None:
    """Ротация по размеру: переполненный файл -> audit.jsonl.1, старшие
    сдвигаются, хранится ровно KEEP ротированных, текущий - свежий."""
    log = AuditLog(tmp_path, rotate_mb=1, keep=2)

    def marker(tag: str) -> None:
        # старый файл под завязку -> следующий log() обязан ротировать
        audit_dir = tmp_path / "audit"
        if log.path.exists():
            with log.path.open("a", encoding="utf-8") as f:
                f.write("x" * 700 + "\n")
            need = 1024 * 1024 - log.path.stat().st_size + 1
            with log.path.open("a", encoding="utf-8") as f:
                f.write("x" * max(need, 0))
        else:
            audit_dir.mkdir()
        log.log("search", details=tag)

    marker("v1")
    marker("v2")
    marker("v3")
    rotated = sorted(p.name for p in (tmp_path / "audit").glob("audit.jsonl.*"))
    assert rotated == ["audit.jsonl.1", "audit.jsonl.2"], rotated
    # .1 - самый свежий ротированный (событие v2 + мусор набивки), .2 - старее (v1)
    first = json.loads(
        (tmp_path / "audit" / "audit.jsonl.1").read_text(encoding="utf-8").splitlines()[0]
    )
    second = json.loads(
        (tmp_path / "audit" / "audit.jsonl.2").read_text(encoding="utf-8").splitlines()[0]
    )
    assert first["details"] == "v2" and second["details"] == "v1"
    assert log.path.stat().st_size < 1024 * 1024


def test_read_filters(tmp_path) -> None:
    """Фильтры чтения: actor/action - точное совпадение; q - подстрока по
    target/details без регистра; since/until - границы включительно, наивное
    время считается UTC; limit/offset - страница с total до среза."""
    log = AuditLog(tmp_path)
    log.log("search", actor="a@corp.ru", details="как согласовать закупку")
    log.log("upload", actor="b@corp.ru", target="регламент-закупок.pdf",
            details="uploads/b/регламент-закупок.pdf")
    log.log("document_delete", actor="a@corp.ru", target="черновик.md", details="узлов: 3")

    assert log.read(actor="a@corp.ru")["total"] == 2
    assert log.read(action="upload")["total"] == 1
    assert log.read(q="закуп")["total"] == 2  # details первой и target второй
    assert log.read(q="ЧЕРНОВИК")["total"] == 1  # без регистра
    assert log.read(q="нет-такой-подстроки")["total"] == 0
    assert log.read(actor="a@corp.ru", action="document_delete")["total"] == 1

    page = log.read(limit=2, offset=0)
    assert page["total"] == 3 and len(page["events"]) == 2
    assert log.read(limit=2, offset=2)["total"] == 3
    assert len(log.read(limit=2, offset=2)["events"]) == 1
    assert log.read(offset=99)["events"] == []

    assert log.read(since="2099-01-01T00:00:00+00:00")["total"] == 0
    assert log.read(until="2099-01-01T00:00:00+00:00")["total"] == 3
    # наивная граница без пояса - UTC, а не TypeError
    assert log.read(since="2000-01-01")["total"] == 3
    with pytest.raises(ValueError):
        log.read(since="завтра утром")


def test_time_filter_boundaries(tmp_path) -> None:
    """Границы since/until включительны: событие на границе попадает в выборку."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True)
    rows = [
        {"ts": "2026-10-12T10:00:00+00:00", "actor": "a", "admin": False,
         "action": "search", "target": "", "details": "вопрос 1", "ip": ""},
        {"ts": "2026-10-12T11:00:00+00:00", "actor": "b", "admin": False,
         "action": "search", "target": "", "details": "вопрос 2", "ip": ""},
        {"ts": "2026-10-12T12:00:00+00:00", "actor": "c", "admin": False,
         "action": "search", "target": "", "details": "вопрос 3", "ip": ""},
    ]
    (audit_dir / "audit.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    log = AuditLog(tmp_path)
    assert log.read(since="2026-10-12T11:00:00+00:00")["total"] == 2
    assert log.read(until="2026-10-12T11:00:00+00:00")["total"] == 2
    both = log.read(since="2026-10-12T11:00:00+00:00", until="2026-10-12T11:00:00+00:00")
    assert both["total"] == 1 and both["events"][0]["actor"] == "b"


def test_read_skips_broken_lines(tmp_path) -> None:
    """Битые/недописанные строки JSONL чтение пропускает (never-throw):
    мусор и оборванный хвост не читаются как события, хорошие не теряются."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(parents=True)
    good = {"ts": "2026-10-12T10:00:00+00:00", "actor": "a", "admin": False,
            "action": "search", "target": "", "details": "ok", "ip": ""}
    (audit_dir / "audit.jsonl").write_text(
        json.dumps(good) + "\n"                       # хорошая
        + "{битый json\n"                              # мусор
        + json.dumps(good) + "\n"                      # хорошая
        + '{"ts": "2026-10-13T10:00:00+00' + "\n"      # недописанная (хвост)
        + "\n",                                        # пустая
        encoding="utf-8",
    )
    result = AuditLog(tmp_path).read()
    assert result["total"] == 2


def test_never_throw_audit_path_is_file(tmp_path, caplog) -> None:
    """Каталог журнала занят ФАЙЛОМ (mkdir невозможен): log() не бросает,
    пишет один warning - запрос не ломается."""
    (tmp_path / "audit").write_text("не каталог", encoding="utf-8")
    log = AuditLog(tmp_path)
    with caplog.at_level(logging.WARNING, logger="veles.audit"):
        log.log("search", details="первый")   # поломка -> warning
        log.log("search", details="второй")   # та же поломка - без повторного warning
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "не записано" in warnings[0].getMessage()


def test_never_throw_readonly_file(tmp_path, caplog) -> None:
    """Файл журнала только-чтение: append невозможен - log() не бросает,
    warning в логе; после возврата прав запись восстанавливается."""
    log = AuditLog(tmp_path)
    log.log("search", details="до поломки")
    assert len(journal_events(tmp_path)) == 1
    os.chmod(log.path, 0o444)
    try:
        with caplog.at_level(logging.WARNING, logger="veles.audit"):
            log.log("search", details="не влезет")
        assert any("не записано" in r.getMessage() for r in caplog.records)
        assert len(journal_events(tmp_path)) == 1  # прежнее событие на месте
    finally:
        os.chmod(log.path, 0o666)  # чтобы tmp_path прибрался
    log.log("search", details="после возврата прав")
    assert len(journal_events(tmp_path)) == 2


# ------------------------------------------------------- TestClient: фасад


def test_health_and_models_not_logged(kb_env) -> None:
    """НЕ логируются (план): healthz/readyz/models/stats/admin/статика -
    журнал после всех чтений пуст."""
    client = make_static_client(kb_env["bundle"])
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code in (200, 503)
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/knowledge/stats").status_code == 200
    assert client.get("/admin").status_code == 200
    assert client.get("/static/admin.js").status_code == 200
    assert journal_text(kb_env["data"]) == ""


def test_search_and_ask_logged(kb_env) -> None:
    """Поиск и вопрос пишут search/ask с текстом запроса (решение владельца),
    actor=owner (static), порядок хронологический."""
    client = make_static_client(kb_env["bundle"])
    q = "какой порог единственной закупки"
    assert client.post(
        "/v1/knowledge/query", json={"query": q, "use_llm": False}
    ).status_code == 200
    question = "сколько страниц в регламенте"
    assert client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": question}]},
    ).status_code == 200
    events = journal_events(kb_env["data"])
    assert [e["action"] for e in events] == ["search", "ask"]
    assert events[0]["details"] == q and events[0]["actor"] == "owner"
    assert events[1]["details"] == question and events[1]["actor"] == "owner"


def test_token_not_in_journal(kb_env, monkeypatch) -> None:
    """Токен владельца (bearer) нигде в журнале не появляется."""
    token = "owner-secret-token-abc123"
    monkeypatch.setenv("VELES_API_TOKEN", token)
    app = create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver({"personal"}),
    )
    client = TestClient(app)
    r = client.post(
        "/v1/knowledge/query", json={"query": "порог закупки", "use_llm": False},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    # и отказ по чужому токену тоже не пишет событий (401 - до точки записи)
    assert client.post(
        "/v1/knowledge/query", json={"query": "порог закупки", "use_llm": False},
        headers={"Authorization": "Bearer chuzhoj-token"},
    ).status_code == 401
    text = journal_text(kb_env["data"])
    assert token not in text and "chuzhoj-token" not in text
    assert len(journal_events(kb_env["data"])) == 1


def test_ldap_events_and_no_password(kb_env, monkeypatch) -> None:
    """Вход LDAP: auth_ldap_fail на неверный пароль (actor=попытанный логин)
    и auth_ldap_ok на успех; пароль (и верный, и неверный) и выпущенный jwt
    в журнале отсутствуют."""
    monkeypatch.setattr("veles.api.app.ldap_is_configured", lambda: True)
    monkeypatch.setenv("VELES_IDENTITY_JWT_SECRET", "jwt-secret-тест")
    good_password = "Верный-Пароль-иванова"

    def fake_authenticate(username: str, password: str) -> LdapIdentity:
        if password == good_password:
            return LdapIdentity(
                user_id=f"{username}@corp.ru", groups=frozenset({"all", "devs"})
            )
        raise LdapAuthError("неверные учётные данные")

    monkeypatch.setattr("veles.api.app.ldap_authenticate", fake_authenticate)
    client = TestClient(create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None),
    ))
    wrong = client.post(
        "/v1/auth/ldap", json={"username": "ivanov", "password": "Неверный-пароль"}
    )
    assert wrong.status_code == 401
    ok = client.post(
        "/v1/auth/ldap", json={"username": "petrov", "password": good_password}
    )
    assert ok.status_code == 200
    token = ok.json()["access_token"]

    events = journal_events(kb_env["data"])
    assert [e["action"] for e in events] == ["auth_ldap_fail", "auth_ldap_ok"]
    assert events[0]["actor"] == "ivanov"  # попытанный логин, не личность
    assert events[1]["actor"] == "petrov@corp.ru"
    assert "devs" in events[1]["details"]
    text = journal_text(kb_env["data"])
    assert good_password not in text
    assert "Неверный-пароль" not in text
    assert token not in text


# ------------------------------------------------ TestClient: документы/jobs


def wait_done(client: TestClient, job_id: str, timeout_s: float = 15.0) -> dict:
    queue = client.app.state.veles_ingest
    assert queue is not None and queue.wait_idle(timeout_s), "очередь не опустела"
    r = client.get(f"/v1/documents/jobs/{job_id}")
    assert r.status_code == 200
    return r.json()


def test_document_and_job_actions_logged(kb_env) -> None:
    """Цепочка значимых действий пишет свои события: upload (2), job_retry,
    job_cancel, document_regroup, document_delete, job_delete. Упавшая задача
    для retry получается детерминированно: очередь со сломанным pipeline_fn
    (воркер честно ставит failed, L0 остаётся), затем живая очередь."""
    client = make_static_client(kb_env["bundle"])

    r1 = client.post("/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE})
    assert r1.status_code == 202
    done = wait_done(client, r1.json()["job_id"])
    assert done["status"] == "done"
    doc_id = done["document_id"]

    # вторая загрузка обязана упасть: очередь с ломающимся pipeline_fn
    def _broken_pipeline():
        raise RuntimeError("пайплайн отключён тестом")

    client.app.state.veles_ingest = IngestQueue(pipeline_fn=_broken_pipeline)
    r2 = client.post(
        "/v1/documents", json={"text": "Командировки: авансовый отчёт", "title": "Командировки"}
    )
    assert r2.status_code == 202
    failed_job = r2.json()["job_id"]
    failed = wait_done(client, failed_job)
    assert failed["status"] == "failed"

    # живая очередь: retry собирает документ, cancel/delete идут по журналу
    client.app.state.veles_ingest = IngestQueue(
        pipeline_fn=client.app.state.veles["pipeline_fn"]
    )
    retry = client.post(f"/v1/documents/jobs/{failed_job}/retry")
    assert retry.status_code == 202, retry.text
    retried_job = retry.json()["job_id"]
    assert wait_done(client, retried_job)["status"] == "done"

    cancel = client.post(f"/v1/documents/jobs/{failed_job}/cancel")
    assert cancel.status_code == 200  # завершённая - no-op, событие всё равно пишется
    regroup = client.post(f"/v1/documents/{doc_id}/groups", json={"groups": ["test-dept"]})
    assert regroup.status_code == 200, regroup.text
    delete = client.delete(f"/v1/documents/{doc_id}")
    assert delete.status_code == 200, delete.text
    # негатив: несуществующий документ - 404 и БЕЗ события
    assert client.delete("/v1/documents/owner/нет-такого.md").status_code == 404
    job_delete = client.delete(f"/v1/documents/jobs/{retried_job}")
    assert job_delete.status_code == 200, job_delete.text

    events = journal_events(kb_env["data"])
    assert actions(kb_env["data"]) == [
        "upload", "upload", "job_retry", "job_cancel",
        "document_regroup", "document_delete", "job_delete",
    ]
    by_action: dict[str, list[dict]] = {}
    for e in events:
        by_action.setdefault(e["action"], []).append(e)
    assert all(e["actor"] == "owner" for e in events)
    uploads = by_action["upload"]
    assert {e["target"] for e in uploads} == {"Регламент_закупок.md", "Командировки.md"}
    assert uploads[0]["details"] == doc_id
    assert by_action["job_retry"][0]["target"] == failed_job
    assert retried_job in by_action["job_retry"][0]["details"]
    assert by_action["job_cancel"][0]["target"] == failed_job
    assert "failed" in by_action["job_cancel"][0]["details"]
    assert by_action["document_regroup"][0]["target"] == doc_id
    assert "test-dept" in by_action["document_regroup"][0]["details"]
    assert by_action["document_delete"][0]["target"] == doc_id
    assert "узлов" in by_action["document_delete"][0]["details"]
    assert by_action["job_delete"][0]["target"] == retried_job


def test_document_read_logged(kb_env, monkeypatch) -> None:
    """Ревью раунд 1: чтение полного текста /v1/source пишется (document_read)
    в ОБЕИХ ветках - оригинал из VELES_SOURCE_ROOTS и текстовая сборка канона;
    вне исключений плана (healthz/models/stats/static)."""
    client = make_static_client(kb_env["bundle"])
    r = client.post("/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE})
    assert r.status_code == 202
    done = wait_done(client, r.json()["job_id"])
    doc_id = done["document_id"]
    filename = doc_id.rsplit("/", 1)[-1]

    # сборка из канона (VELES_SOURCE_ROOTS пуст)
    assert client.get(f"/v1/source/{doc_id}").status_code == 200

    # оригинал: корень источников с файлом L0
    corpus = kb_env["data"].parent / "corpus"
    (corpus / "owner").mkdir(parents=True)
    (corpus / "owner" / filename).write_text(DOC_TEXT, encoding="utf-8")
    monkeypatch.setattr("veles.api.app.VELES_SOURCE_ROOTS", [corpus])
    assert client.get(f"/v1/source/{doc_id}").status_code == 200

    reads = [e for e in journal_events(kb_env["data"])
             if e["action"] == "document_read"]
    assert len(reads) == 2
    assert all(e["target"] == doc_id and e["actor"] == "owner" for e in reads)
    assert "канона" in reads[0]["details"]
    assert "оригинал" in reads[1]["details"]


def test_tracker_read_logged(kb_env, monkeypatch) -> None:
    """Ревью раунд 1: чтение файла трекера /tracker/file пишется
    (tracker_read) - свой файл (не-админ) и чужой (админ; details несёт чей
    каталог)."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")
    tracker = kb_env["data"] / "tracker"
    (tracker / "ivanov@corp.ru" / "task").mkdir(parents=True)
    (tracker / "ivanov@corp.ru" / "task" / "zametka.md").write_text(
        "- задача", encoding="utf-8")
    monkeypatch.setenv("TRACKER_ROOT", str(tracker))
    app = create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None)
    )
    ivanov = {"X-OpenWebUI-User-Email": "ivanov@corp.ru"}
    boss = {"X-OpenWebUI-User-Email": "boss@corp.ru"}
    client = TestClient(app)

    assert client.get("/tracker/file/task/zametka.md",
                      headers=ivanov).status_code == 200
    assert client.get("/tracker/file/task/zametka.md",
                      headers=boss).status_code == 200
    reads = [e for e in journal_events(kb_env["data"])
             if e["action"] == "tracker_read"]
    assert len(reads) == 2
    assert reads[0]["actor"] == "ivanov@corp.ru" and reads[0]["admin"] is False
    assert reads[0]["target"] == "task/zametka.md"
    assert reads[1]["actor"] == "boss@corp.ru" and reads[1]["admin"] is True
    assert "ivanov@corp.ru" in reads[1]["details"]


# ------------------------------------------------------ TestClient: доступ


def test_admin_reads_audit_non_admin_403(kb_env, monkeypatch) -> None:
    """GET /v1/admin/audit: админ читает (фильтры actor/action/since работают),
    не-админ - 403 и событие audit_access_denied; успешные чтения аудита сами
    событий не создают; мусорная дата - 400."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    app = create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None)
    )
    ivanov = {"X-OpenWebUI-User-Email": "ivanov@corp.ru"}
    boss = {"X-OpenWebUI-User-Email": "boss@corp.ru"}
    owner = {"Authorization": "Bearer owner-token"}
    client = TestClient(app)

    # pre-условие: одно событие поиска от пользователя
    assert client.post(
        "/v1/knowledge/query", json={"query": "порог закупки", "use_llm": False},
        headers=ivanov,
    ).status_code == 200

    denied = client.get("/v1/admin/audit", headers=ivanov)
    assert denied.status_code == 403
    denials = [e for e in journal_events(kb_env["data"])
               if e["action"] == "audit_access_denied"]
    assert len(denials) == 1
    assert denials[0]["actor"] == "ivanov@corp.ru" and denials[0]["admin"] is False

    ok = client.get("/v1/admin/audit", headers=boss)
    assert ok.status_code == 200
    body = ok.json()
    assert {"events", "total", "limit", "offset"} <= set(body)
    assert [e["action"] for e in body["events"]] == ["search", "audit_access_denied"]
    reads_before = len(journal_events(kb_env["data"]))

    filtered = client.get(
        "/v1/admin/audit?action=audit_access_denied&actor=ivanov%40corp.ru",
        headers=boss,
    )
    assert filtered.status_code == 200
    fbody = filtered.json()
    assert fbody["total"] == 1 and fbody["events"][0]["actor"] == "ivanov@corp.ru"
    assert client.get(
        "/v1/admin/audit?since=2000-01-01", headers=boss
    ).status_code == 200
    assert client.get(
        "/v1/admin/audit?since=завтра", headers=boss
    ).status_code == 400
    assert client.get(
        "/v1/admin/audit?limit=0", headers=boss
    ).status_code == 422  # Query-валидация ge=1
    # успешные чтения (3 запроса) событий не добавили
    assert len(journal_events(kb_env["data"])) == reads_before

    # bearer-владелец = админ
    assert client.get("/v1/admin/audit", headers=owner).status_code == 200


def test_static_owner_reads_audit(kb_env, monkeypatch) -> None:
    """Static-профиль: владелец-админ читает журнал; при настроенном токене
    без bearer - 401 (как остальные API-вкладки панели)."""
    client = make_static_client(kb_env["bundle"])
    client.post("/v1/knowledge/query", json={"query": "порог закупки", "use_llm": False})
    r = client.get("/v1/admin/audit?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1 and body["events"][0]["action"] == "search"

    monkeypatch.setenv("VELES_API_TOKEN", "owner-secret")
    app = create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver({"personal"}),
    )
    guarded = TestClient(app)
    assert guarded.get("/v1/admin/audit").status_code == 401
    assert guarded.get(
        "/v1/admin/audit", headers={"Authorization": "Bearer owner-secret"}
    ).status_code == 200
