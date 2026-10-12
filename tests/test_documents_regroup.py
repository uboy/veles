"""Админка v2 (docs/plan/admin-ui/v2-spec.md): смена групп документа и задачи.

§1 Смена групп: POST /v1/documents/{id}/groups - только админ (инвариант 5,
чужой клиент группы менять не может), переписывает access_groups в каноне
(манифест + узлы) и в метах живого индекса (overlay-патч), права применяются
немедленно: doc в группе A находит до, в группе B - после, чужая группа не
находит; рестарт (load_index с overlay) документ не возвращает в старую
группу. Сквозной сценарий владельца: 2 документа -> массовая смена группы ->
поиск подтверждает -> массовое удаление -> поиск пуст.

§2 Задачи: cancel (queued снимается, running помечается failed «отменена
владельцем»), retry (повторная постановка сохранённого L0), delete (убрать
завершённую из журнала); exists_in_base (документ уже в каноне) и
duplicate_of (другая активная задача с тем же файлом) в списке задач.
Сценарий: задача-фол -> перезапуск -> отмена -> удаление из истории.

Индекс - настоящий build_index; эмбеддер - детерминированный hashing (канон:
тесты без загрузки моделей), dim = EMBED_DIM, чтобы overlay докатывался через
load_index, - как в test_delete.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.canonical.store import _doc_dir, find_document_manifest, save_document
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import IngestQueue, uploads_root, user_dir
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver
from veles.retrieval.retriever import hybrid_search

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

DOC_TITLE = "Регламент закупок"
DOC_QUERY = "какой порог единственной закупки"
DOC_MARK = "350 тысяч"
DOC_V1 = """\
# Регламент закупок

Порог единственной закупки составляет 350 тысяч рублей.

Закупки свыше порога требуют обязательного тендера.
"""

DOC2_TITLE = "Командировки"
DOC2_QUERY = "как оформить авансовый отчёт"
DOC2_MARK = "авансовый отчёт"
DOC2_V1 = """\
# Командировки

После поездки сотрудник сдаёт авансовый отчёт в бухгалтерию в течение трёх дней.
"""

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""
SEED_ID = "05_vyshka_academic/lectures/коучинг.md"


class HashingEmbedder:
    """Тот же стаб, что в test_delete/test_documents_ingest: sublinear tf."""

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


@pytest.fixture()
def kb_env(tmp_path, monkeypatch):
    """tmp-окружение как в test_delete: канон, INDEX_DIR + мини-индекс с одним
    засеянным документом; поверх - конфиг групп с группой legal (petrov)."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    index_dir = data / "index"
    corpus = tmp_path / "corpus"
    seed = corpus / "05_vyshka_academic/lectures/коучинг.md"
    seed.parent.mkdir(parents=True)
    seed.write_text(SEED_DOC, encoding="utf-8", newline="\n")
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    monkeypatch.setattr("veles.config.INDEX_DIR", index_dir)
    monkeypatch.setenv("VELES_DATA", str(data))
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    # конфиг групп (v2): legal - только petrov; ivanov остаётся вне
    groups_conf = data / "groups.json"
    groups_conf.write_text(json.dumps({"groups": [
        {"id": "all", "title": "Видно всем", "members": []},
        {"id": "legal", "title": "Юристы", "members": ["petrov@corp.ru"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("VELES_GROUPS_FILE", str(groups_conf))
    return {"data": data, "canonical": canonical, "index_dir": index_dir,
            "bundle": bundle}


def make_identity_app(kb_env, monkeypatch):
    """Корпоративный профиль (как ACL-тест K4): иванов/петров - пользователи
    (петров в legal), boss - админ, owner - bearer-владелец."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    app = create_app(pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None))
    return {
        "ivanov": {"X-OpenWebUI-User-Email": "ivanov@corp.ru"},
        "petrov": {"X-OpenWebUI-User-Email": "petrov@corp.ru"},
        "boss": {"X-OpenWebUI-User-Email": "boss@corp.ru"},
        "owner": {"Authorization": "Bearer owner-token"},
        "app": app,
    }


def wait_done(client: TestClient, job_id: str, timeout_s: float = 15.0,
              headers: dict | None = None) -> dict:
    queue = client.app.state.veles_ingest
    assert queue is not None and queue.wait_idle(timeout_s), "очередь не опустела"
    r = client.get(f"/v1/documents/jobs/{job_id}", headers=headers or {})
    assert r.status_code == 200
    return r.json()


def upload_json(client: TestClient, text: str, title: str,
                headers: dict | None = None):
    return client.post("/v1/documents", json={"text": text, "title": title},
                       headers=headers or {})


def query_texts(client: TestClient, q: str, headers: dict | None = None) -> list[str]:
    d = client.post(
        "/v1/knowledge/query", json={"query": q, "use_llm": False},
        headers=headers or {},
    ).json()
    return [ev["text"] for ev in d["evidence"]]


def search_as(groups: set[str], q: str, bundle=None) -> list[str]:
    """Прямой поиск по бандлу от лица групп (рестарт: bundle - load_index)."""
    target = bundle if bundle is not None else None
    assert target is not None
    qvec = HashingEmbedder().embed_query(q)
    return [e.text for e in hybrid_search(
        target, qvec, q, top_k=8, allowed_groups=groups)]


def wait_status(client: TestClient, job_id: str, want: str,
                timeout_s: float = 15.0) -> dict:
    """Дождаться статуса джоба (cancel/retry асинхронны на стороне воркера)."""
    rec: dict = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/v1/documents/jobs/{job_id}")
        assert r.status_code == 200
        rec = r.json()
        if rec.get("status") == want:
            return rec
        time.sleep(0.02)
    raise AssertionError(f"задача {job_id} не достигла статуса {want!r}: {rec}")


# --------------------------------------------- смена групп: канон и живой поиск


def test_regroup_updates_canon_and_search(kb_env, monkeypatch) -> None:
    """Админ меняет группы: канон (манифест+узлы) и живой индекс переписаны,
    права применяются немедленно (legal находит, прежняя личная группа - нет),
    рестарт не откатывает; не-админ и аноним группы менять не могут."""
    users = make_identity_app(kb_env, monkeypatch)
    ivanov, petrov, boss = users["ivanov"], users["petrov"], users["boss"]
    client = TestClient(users["app"])

    r = upload_json(client, DOC_V1, DOC_TITLE, headers=ivanov)
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert wait_done(client, r.json()["job_id"], headers=ivanov)["status"] == "done"

    # до смены: личная группа иванова - он находит, петров нет
    assert any(DOC_MARK in t for t in query_texts(client, DOC_QUERY, headers=ivanov))
    assert not any(DOC_MARK in t for t in query_texts(client, DOC_QUERY, headers=petrov))

    regroup_url = f"/v1/documents/{doc_id}/groups"
    body = {"groups": ["legal"]}
    # инвариант 5: группы от клиента принимает только админ; аноним - 401
    assert client.post(regroup_url, json=body, headers=petrov).status_code == 403
    assert client.post(regroup_url, json=body, headers=ivanov).status_code == 403
    assert client.post(regroup_url, json=body).status_code == 401
    # валидация тела ДО поиска документа: пустой/неизвестный/личный-битый - 400
    assert client.post(regroup_url, json={"groups": []}, headers=boss).status_code == 400
    assert client.post(
        regroup_url, json={"groups": ["нет-такой"]}, headers=boss).status_code == 400
    assert client.post(
        regroup_url, json={"groups": ["user:не-email!!"]}, headers=boss).status_code == 400
    # несуществующий документ с валидным телом - 404
    assert client.post(
        "/v1/documents/petrov@corp.ru/нет-такого.md/groups",
        json=body, headers=boss).status_code == 404

    # админ меняет группы
    r = client.post(regroup_url, json=body, headers=boss)
    assert r.status_code == 200, r.text
    answer = r.json()
    assert answer["status"] == "regrouped"
    assert answer["document_id"] == doc_id
    assert answer["groups"] == ["legal"]
    assert answer["nodes"] >= 1
    assert answer["overlay_saved"] is True

    # канон переписан: манифест и каждый узел несут новый состав
    doc_dir = _doc_dir(doc_id, kb_env["canonical"])
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access_groups"] == ["legal"]
    node_groups = [
        json.loads(line)["access_groups"]
        for line in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert node_groups and all(g == ["legal"] for g in node_groups)
    # аудит-событие regroup
    events = [json.loads(l) for l in
              (kb_env["canonical"] / "ingest-audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "regroup"
    assert events[-1]["groups"] == ["legal"] and events[-1]["by"] == "boss@corp.ru"

    # права применены немедленно: legal находит, прежняя группа - нет
    assert any(DOC_MARK in t for t in query_texts(client, DOC_QUERY, headers=petrov))
    assert not any(DOC_MARK in t for t in query_texts(client, DOC_QUERY, headers=ivanov))
    # stats показывают новый состав групп документа
    stats = client.get("/v1/knowledge/stats", headers=boss).json()
    by_id = {d["document_id"]: d for d in stats["documents"]}
    assert by_id[doc_id]["groups"] == ["legal"]

    # рестарт: overlay-патч применён - legal находит, corp не возвращается
    fresh = load_index(kb_env["index_dir"])
    assert any(DOC_MARK in t for t in search_as({"legal"}, DOC_QUERY, fresh))
    assert not any(DOC_MARK in t for t in search_as({"corp"}, DOC_QUERY, fresh))


def test_bulk_regroup_then_delete_scenario(kb_env, monkeypatch) -> None:
    """Сквозной сценарий владельца (v2-spec, Проверки): загрузка 2 документов ->
    массовая смена группы -> поиск подтверждает -> массовое удаление -> поиск
    пуст; засеянный документ не задет."""
    users = make_identity_app(kb_env, monkeypatch)
    ivanov, petrov, boss = users["ivanov"], users["petrov"], users["boss"]
    client = TestClient(users["app"])

    docs: list[tuple[str, str, str, str]] = []  # (doc_id, query, mark, title)
    for text, title, q, mark in (
        (DOC_V1, DOC_TITLE, DOC_QUERY, DOC_MARK),
        (DOC2_V1, DOC2_TITLE, DOC2_QUERY, DOC2_MARK),
    ):
        r = upload_json(client, text, title, headers=ivanov)
        assert r.status_code == 202
        assert wait_done(client, r.json()["job_id"], headers=ivanov)["status"] == "done"
        docs.append((r.json()["document_id"], q, mark, title))

    # массовая смена группы: та же операция на каждый выбранный документ
    for doc_id, *_ in docs:
        r = client.post(f"/v1/documents/{doc_id}/groups",
                        json={"groups": ["legal"]}, headers=boss)
        assert r.status_code == 200, r.text

    # поиск подтверждает: петров (legal) видит оба, иванов не видит ни один
    for _doc_id, q, mark, _title in docs:
        assert any(mark in t for t in query_texts(client, q, headers=petrov)), q
        assert not any(mark in t for t in query_texts(client, q, headers=ivanov)), q

    # массовое удаление выбранных
    for doc_id, *_ in docs:
        assert client.delete(f"/v1/documents/{doc_id}", headers=boss).status_code == 200

    # поиск пуст: ни маркеров, ни документов в оглавлении (кроме засеянного)
    for _doc_id, q, mark, _title in docs:
        for headers in (petrov, boss):
            assert not any(mark in t for t in query_texts(client, q, headers=headers)), q
    stats = client.get("/v1/knowledge/stats", headers=boss).json()
    assert {d["document_id"] for d in stats["documents"]} == {SEED_ID}


# ------------------------------------------------ задачи: cancel/retry/delete


def _no_pipeline_app() -> TestClient:
    """Фасад без пайплайна: state["pipeline_fn"] подменён быстрым отказом -
    ни один маршрут не трогает загрузку настоящей модели эмбеддера."""
    app = create_app(pipeline=None, resolver=StaticOwnerResolver({"personal"}))
    app.state.veles["pipeline_fn"] = (
        lambda: (_ for _ in ()).throw(RuntimeError("индекса нет"))
    )
    return TestClient(app)


def _blocked_app() -> tuple[TestClient, threading.Event, threading.Event]:
    """Фасад с очередью, чей пайплайн управляется событиями: 1-й вызов висит на
    gate1, 3-й - на gate2, остальные падают сразу («индекс недоступен» - L0
    сохраняется, retry возможен). Пайплайн берётся ДО разбора, поэтому
    заблокированная задача стоит в running без дочернего процесса."""
    gate1, gate2 = threading.Event(), threading.Event()
    calls = {"n": 0}

    def pipeline():
        calls["n"] += 1
        if calls["n"] == 1:
            gate1.wait(timeout=20)
        elif calls["n"] == 3:
            gate2.wait(timeout=20)
        raise RuntimeError("пайплайн закрыт")

    queue = IngestQueue(pipeline_fn=pipeline)
    app = create_app(pipeline=None, resolver=StaticOwnerResolver({"personal"}))
    app.state.veles_ingest = queue  # API подхватит её вместо ленивого создания
    app.state.veles["pipeline_fn"] = (
        lambda: (_ for _ in ()).throw(RuntimeError("индекса нет"))
    )
    return TestClient(app), gate1, gate2


def test_job_fail_retry_cancel_delete(kb_env, monkeypatch) -> None:
    """Сценарий §2: задача-фол -> перезапуск -> отмена -> удаление из истории;
    duplicate_of (дубль активной задачи) и exists_in_base; активную удалить
    нельзя (409); отмена running снимает ожидание и ставит failed «отменена
    владельцем»; L0 у retry-способных провалов сохраняется."""
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    client, gate1, gate2 = _blocked_app()

    r = upload_json(client, DOC_V1, DOC_TITLE)
    assert r.status_code == 202
    job1 = r.json()["job_id"]
    wait_status(client, job1, "running")

    # дубль: та же задача (тот же файл) ставится второй, пока первая активна
    r2 = upload_json(client, DOC_V1, DOC_TITLE)
    assert r2.status_code == 202
    job1b = r2.json()["job_id"]
    jobs = {j["job_id"]: j for j in client.get("/v1/documents/jobs").json()["jobs"]}
    dups = {job1: jobs[job1]["duplicate_of"], job1b: jobs[job1b]["duplicate_of"]}
    # ровно одна из двух помечена дублем другой (created_at с точностью до
    # секунды - порядок пары определяется stable-сортировкой по job_id)
    assert (dups[job1] == job1b) != (dups[job1b] == job1)
    assert all(jobs[j]["exists_in_base"] is False for j in (job1, job1b))

    # активную (queued) удалить нельзя - сначала отмена
    assert client.delete(f"/v1/documents/jobs/{job1b}").status_code == 409
    rec = client.post(f"/v1/documents/jobs/{job1b}/cancel").json()
    assert rec["status"] == "failed" and "отменена" in rec["error"]

    # отмена running: воркер выйдет из заблокированного пайплайна и честно
    # пометит failed «отменена владельцем»; L0 сохранён (разбор не падал)
    assert client.post(f"/v1/documents/jobs/{job1}/cancel").status_code == 200
    gate1.set()
    rec = wait_status(client, job1, "failed")
    assert "отменена" in rec["error"]
    assert (uploads_root() / user_dir("") / "Регламент_закупок.md").is_file()

    # перезапуск упавшей: новая задача, старая остаётся в истории
    r = client.post(f"/v1/documents/jobs/{job1}/retry")
    assert r.status_code == 202, r.text
    job2 = r.json()["job_id"]
    assert job2 != job1 and r.json()["retried_from"] == job1
    assert wait_status(client, job2, "failed")  # пайплайн по-прежнему недоступен
    r = client.post(f"/v1/documents/jobs/{job2}/retry")
    assert r.status_code == 202
    job3 = r.json()["job_id"]
    wait_status(client, job3, "running")  # третий вызов пайплайна висит на gate2
    # перезапуск активной (running) - 409 «только упавшей»
    assert client.post(f"/v1/documents/jobs/{job3}/retry").status_code == 409

    # cancel по завершённой - запись без изменений (отменять нечего)
    before = client.get(f"/v1/documents/jobs/{job1}").json()
    assert client.post(f"/v1/documents/jobs/{job1}/cancel").json() == before

    # активную running удалить нельзя; отмена queued второй задачи - сразу failed
    assert client.delete(f"/v1/documents/jobs/{job3}").status_code == 409
    r = upload_json(client, DOC2_V1, DOC2_TITLE)
    job4 = r.json()["job_id"]
    rec = client.post(f"/v1/documents/jobs/{job4}/cancel").json()
    assert rec["status"] == "failed" and "отменена" in rec["error"]

    # удаление из истории: завершённая исчезает из журнала и из чтения
    gate2.set()
    wait_status(client, job3, "failed")
    assert client.delete(f"/v1/documents/jobs/{job3}").json()["status"] == "deleted"
    assert client.get(f"/v1/documents/jobs/{job3}").status_code == 404
    assert client.delete(f"/v1/documents/jobs/{job3}").status_code == 404
    assert job3 not in {j["job_id"] for j in
                        client.get("/v1/documents/jobs").json()["jobs"]}

    # неизвестная задача - 404 на всех трёх операциях
    for method, path in (("post", "/v1/documents/jobs/нет/cancel"),
                         ("post", "/v1/documents/jobs/нет/retry"),
                         ("delete", "/v1/documents/jobs/нет")):
        assert getattr(client, method)(path).status_code == 404, path


def test_jobs_exists_in_base_after_done(kb_env) -> None:
    """exists_in_base: после успешной задачи документ в каноне - бейдж «уже в
    базе» (до завершения поле False проверяется в blocked-сценарии выше, где
    воркер заведомо не успевает)."""
    app = create_app(
        pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver({"personal"}),
    )
    client = TestClient(app)
    r = upload_json(client, DOC_V1, DOC_TITLE)
    assert r.status_code == 202
    wait_done(client, r.json()["job_id"])
    jobs = client.get("/v1/documents/jobs").json()["jobs"]
    rec = next(j for j in jobs if j["job_id"] == r.json()["job_id"])
    assert rec["exists_in_base"] is True
    assert rec["duplicate_of"] == ""


def test_regroup_no_pipeline_reports_overlay(kb_env, monkeypatch) -> None:
    """Канон переписывается и без живого индекса (узлов патчить нет - nodes=0);
    патч-only overlay при этом пишется штатно, о чём ответ говорит честно."""
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    client = _no_pipeline_app()

    r = client.post(f"/v1/documents/{SEED_ID}/groups",
                    json={"groups": ["legal"]})
    assert r.status_code == 200, r.text
    answer = r.json()
    assert answer["nodes"] == 0  # в (пустом) индексе узлов документа нет
    assert answer["overlay_saved"] is True
    assert answer["groups"] == ["legal"]
    assert find_document_manifest(SEED_ID)["access_groups"] == ["legal"]
