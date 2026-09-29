"""K4: загрузка документов пользователем - POST /v1/documents, очередь, upsert.

Чек-лист брифа docs/plan/briefs/2026-09-29-k4-user-ingest.md: позитивный цикл
(загрузка -> джоб done -> поиск находит -> цитата ведёт на документ), ACL-негативы
(чужому не видно ни в поиске, ни в джобах; админ видит всё), квоты до очереди,
повторная загрузка -> r2 (старая ревизия только по revision-параметру), no-op по
sha256, рестарт-негатив очереди (не вечный running), форматные негативы.
Индекс - настоящий build_index; эмбеддер - детерминированный hashing (канон:
тесты без загрузки моделей), dim = EMBED_DIM, чтобы overlay докатывался через
load_index (валидация модели/размерности как в проде).
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.canonical.store import _doc_dir, save_document
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import (
    IngestQueue,
    QuotaExceeded,
    check_quota,
    slugify_title,
    uploads_root,
    user_dir,
    user_quota_lock,
)
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver
from veles.retrieval.retriever import hybrid_search

# dim = EMBED_DIM: load_index строит NumpyFlatStore с EMBED_DIM по умолчанию,
# overlay должен докладываться в тот же store без конфликтов размерности
HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

DOC_TITLE = "Регламент закупок"
DOC_SLUG = slugify_title(DOC_TITLE)  # Регламент_закупок (регистр сохраняется)
DOC_ID = f"{user_dir('')}/{DOC_SLUG}.md"
DOC_V1 = """\
# Регламент закупок

Порог единственной закупки составляет 350 тысяч рублей.

Закупки свыше порога требуют обязательного тендера.
"""
DOC_V2 = DOC_V1.replace("350 тысяч", "500 тысяч")
DOC_QUERY = "какой порог единственной закупки"
DOC_MARK_V1 = "350 тысяч"
DOC_MARK_V2 = "500 тысяч"

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""


class HashingEmbedder:
    """Тот же стаб, что в e2e/test_revisions: sublinear tf по хеш-бакетам."""

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
    """tmp-окружение: VELES_DATA (uploads/overlay), канон, INDEX_DIR + мини-индекс
    с одним засеянным документом, собранный настоящим build_index и сохранённый
    в INDEX_DIR (честная проверка overlay через load_index)."""
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
    monkeypatch.delenv("VELES_QUOTA_DOCS", raising=False)
    monkeypatch.delenv("VELES_QUOTA_MB", raising=False)
    return {"data": data, "canonical": canonical, "index_dir": index_dir, "bundle": bundle}


def make_static_client(bundle, groups=("personal",)) -> TestClient:
    """Личный профиль: владелец (user_id=''), группы из resolver."""
    app = create_app(
        pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver(set(groups)),
    )
    return TestClient(app)


def wait_done(client: TestClient, job_id: str, timeout_s: float = 15.0,
              headers: dict | None = None) -> dict:
    """Дождаться опустешения очереди и вернуть финальный статус джоба."""
    queue = client.app.state.veles_ingest
    assert queue is not None and queue.wait_idle(timeout_s), "очередь не опустела"
    r = client.get(f"/v1/documents/jobs/{job_id}", headers=headers or {})
    assert r.status_code == 200
    return r.json()


def upload_json(client: TestClient, text: str, title: str = DOC_TITLE):
    return client.post("/v1/documents", json={"text": text, "title": title})


def query_texts(client: TestClient, q: str = DOC_QUERY, headers: dict | None = None,
                **fields) -> list[str]:
    d = client.post(
        "/v1/knowledge/query", json={"query": q, "use_llm": False, **fields},
        headers=headers or {},
    ).json()
    return [ev["text"] for ev in d["evidence"]]


def upload_outcomes(canonical) -> list[str]:
    audit = (canonical / "ingest-audit.jsonl").read_text(encoding="utf-8")
    return [json.loads(l)["outcome"] for l in audit.splitlines() if '"upload"' in l]


# ------------------------------------------------------------- позитивный цикл


def test_upload_json_find_and_citation(kb_env) -> None:
    """Загрузка текста -> джоб done -> поиск находит -> цитата ведёт на документ."""
    client = make_static_client(kb_env["bundle"])
    r = upload_json(client, DOC_V1)
    assert r.status_code == 202
    body = r.json()
    assert body["job_id"] and body["status"] == "queued"
    assert body["document_id"] == DOC_ID

    job = wait_done(client, body["job_id"])
    assert job["status"] == "done", job
    assert job["action"] == "created" and job["revision_id"] == "r1"
    assert job["document_id"] == DOC_ID and job["title"] == DOC_TITLE

    d = client.post("/v1/knowledge/query",
                    json={"query": DOC_QUERY, "use_llm": False}).json()
    assert d["decision"] == "answer"
    assert DOC_MARK_V1 in d["answer"]
    cite = d["citations"][0]
    assert cite["document_id"] == DOC_ID
    assert cite["source_id"] == DOC_ID
    assert cite["title"] == DOC_TITLE  # title из запроса, не слаг имени файла
    assert cite["revision_id"] == "r1"
    # L0 лёг в uploads/<user>/, аудит - в ingest-audit.jsonl
    assert (uploads_root() / user_dir("") / f"{DOC_SLUG}.md").exists()
    events = [json.loads(l) for l in
              (kb_env["canonical"] / "ingest-audit.jsonl").read_text(encoding="utf-8").splitlines()]
    up = [e for e in events if e["event"] == "upload" and e["job_id"] == body["job_id"]]
    assert up and up[0]["outcome"] == "created" and up[0]["by"] == "owner"
    assert up[0]["size_bytes"] > 0 and up[0]["document_id"] == DOC_ID


def test_upload_multipart_txt(kb_env) -> None:
    client = make_static_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents",
        files={"file": ("заметка_совещание.txt", DOC_V1.encode("utf-8"), "text/plain")},
        data={"title": "Заметка совещание"},
    )
    assert r.status_code == 202
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done" and job["action"] == "created"
    assert job["title"] == "Заметка совещание"
    texts = query_texts(client, "порог закупки тендер")
    assert any(DOC_MARK_V1 in t for t in texts)


def test_stats_sees_uploaded(kb_env) -> None:
    """Регрессия: загруженный документ виден оглавлению фасада."""
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    stats = client.get("/v1/knowledge/stats").json()
    doc_ids = {d["document_id"] for d in stats["documents"]}
    assert DOC_ID in doc_ids


# --------------------------------------------------------------------- ACL K4


def test_acl_other_user_invisible_everywhere(kb_env, monkeypatch) -> None:
    """Чужой пользователь: документ невидим в поиске (в т.ч. revision=all),
    джоб 404, в списке джобов отсутствует; админ видит и джоб, и документ.
    Ревью K4 (P0): пути чтения под тем же ACL - source/asset чужому 404,
    stats чужих документов не содержит; владелец-загрузчик и bearer-владелец
    видят; подписанная ссылка на цитату работает без заголовков."""
    import hashlib as _hashlib
    import hmac as _hmac
    import time as _time

    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")  # читается на запросе
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    monkeypatch.setenv("VELES_LINK_SECRET", "sig-secret")
    app = create_app(pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None))
    ivanov = {"X-OpenWebUI-User-Email": "ivanov@corp.ru"}
    petrov = {"X-OpenWebUI-User-Email": "petrov@corp.ru"}
    boss = {"X-OpenWebUI-User-Email": "boss@corp.ru"}
    owner = {"Authorization": "Bearer owner-token"}
    ivanov_client = TestClient(app)
    petrov_client = TestClient(app)
    boss_client = TestClient(app)
    owner_client = TestClient(app)

    r = ivanov_client.post("/v1/documents", json={"text": DOC_V1, "title": DOC_TITLE},
                           headers=ivanov)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    doc_id = r.json()["document_id"]
    assert wait_done(ivanov_client, job_id, headers=ivanov)["status"] == "done"

    # документ несёт личную группу ivanov
    doc_dir = _doc_dir(doc_id, kb_env["canonical"])
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access_groups"] == ["user:ivanov@corp.ru"]

    # чужой в поиске: ни в дефолте, ни в revision=all; владелец и админ видят
    assert not any(DOC_MARK_V1 in t for t in query_texts(petrov_client, headers=petrov))
    assert not any(DOC_MARK_V1 in t for t in query_texts(petrov_client, headers=petrov, revision="all"))
    assert any(DOC_MARK_V1 in t for t in query_texts(ivanov_client, headers=ivanov))
    assert any(DOC_MARK_V1 in t for t in query_texts(boss_client, headers=boss))

    # джобы: чужой - 404 и пустой список; админ и владелец видят джоб
    assert petrov_client.get(f"/v1/documents/jobs/{job_id}", headers=petrov).status_code == 404
    assert job_id not in [j["job_id"] for j in
                          petrov_client.get("/v1/documents/jobs", headers=petrov).json()["jobs"]]
    assert boss_client.get(f"/v1/documents/jobs/{job_id}", headers=boss).status_code == 200
    assert ivanov_client.get(f"/v1/documents/jobs/{job_id}", headers=ivanov).status_code == 200

    # без личности - 401 ещё до очереди
    assert TestClient(app).post("/v1/documents", json={"text": "текст"}).status_code == 401

    # --- пути чтения (P0 ревью K4): source/asset/stats под тем же ACL ---
    # актив кладём в doc_dir руками (как save_document): парсер txt активов не создаёт
    (doc_dir / "assets").mkdir()
    (doc_dir / "assets" / "схема.png").write_bytes(b"\x89PNG-fake")

    # чужой: source и asset - 404, неотличимо от несуществующего документа
    assert petrov_client.get(f"/v1/source/{doc_id}", headers=petrov).status_code == 404
    assert petrov_client.get(f"/v1/asset/{doc_id}/схема.png", headers=petrov).status_code == 404
    # несуществующий документ для того же чужого - тоже 404 (неотличимость)
    assert petrov_client.get("/v1/source/owner/нет-такого.md", headers=petrov).status_code == 404

    # оглавление: чужой документ у петрова отсутствует (и в totals.nodes),
    # у загрузчика, админа и bearer-владельца - присутствует
    def stat_doc_ids(client, headers):
        d = client.get("/v1/knowledge/stats", headers=headers).json()
        return {doc["document_id"] for doc in d["documents"]}, d["totals"]["nodes"]

    petrov_docs, petrov_nodes = stat_doc_ids(petrov_client, petrov)
    assert doc_id not in petrov_docs
    assert petrov_nodes == 0  # засеянный документ несёт personal - петрову не виден тоже
    for client, headers in ((ivanov_client, ivanov), (boss_client, boss),
                            (owner_client, owner)):
        docs, nodes = stat_doc_ids(client, headers)
        assert doc_id in docs
        assert nodes >= 1  # узлы видимых документов в totals считаются
    # у админа видно больше, чем у чужого (его собственные + загрузки иванова)
    _, boss_nodes = stat_doc_ids(boss_client, boss)
    assert boss_nodes > petrov_nodes
    # владелец-загрузчик, админ и bearer-владелец читают source и asset
    for client, headers in ((ivanov_client, ivanov), (boss_client, boss),
                            (owner_client, owner)):
        assert client.get(f"/v1/source/{doc_id}", headers=headers).status_code == 200
        assert client.get(f"/v1/asset/{doc_id}/схема.png", headers=headers).status_code == 200

    # подписанная ссылка на цитату: работает без заголовков (sig скоупит
    # документ+узел, групповая проверка не применяется)
    nodes = [json.loads(l) for l in
             (doc_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    node_id = next(n["node_version_id"] for n in nodes if n.get("node_type") != "root")
    exp = int(_time.time()) + 600
    sig = _hmac.new(b"sig-secret", f"{doc_id}|{node_id}|{exp}".encode(),
                    _hashlib.sha256).hexdigest()
    r = TestClient(app).get(f"/v1/source/{doc_id}",
                            params={"node": node_id, "exp": str(exp), "sig": sig})
    assert r.status_code == 200
    # битая подпись - 401, как раньше
    assert TestClient(app).get(
        f"/v1/source/{doc_id}",
        params={"node": node_id, "exp": str(exp), "sig": "bad"}).status_code == 401


def test_access_groups_param_admin_only(kb_env, monkeypatch) -> None:
    """multipart access_groups: не-админ (identity) - 403; админ/владелец задаёт
    группы - документ виден только имеющей право группе."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    app = create_app(pipeline=VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None))
    user = {"X-OpenWebUI-User-Email": "petrov@corp.ru"}
    client = TestClient(app)
    r = client.post(
        "/v1/documents",
        files={"file": ("док.txt", "Текст про смету затрат".encode("utf-8"), "text/plain")},
        data={"access_groups": "finance-restricted"},
        headers=user,
    )
    assert r.status_code == 403
    assert "администратор" in r.json()["detail"]

    # static-владелец (админ) задаёт restricted: без права не видно, с правом видно
    owner = make_static_client(kb_env["bundle"], groups=("personal",))
    r2 = owner.post(
        "/v1/documents",
        files={"file": ("бюджет.txt", "Бюджет проекта: смета затрат 100".encode("utf-8"),
                        "text/plain")},
        data={"access_groups": "finance-restricted", "title": "Бюджет"},
    )
    assert r2.status_code == 202
    assert wait_done(owner, r2.json()["job_id"])["status"] == "done"
    assert not any("смета затрат" in t for t in query_texts(owner, "смета затрат бюджета"))
    entitled = make_static_client(kb_env["bundle"], groups=("personal", "finance-restricted"))
    assert any("смета затрат" in t for t in query_texts(entitled, "смета затрат бюджета"))


# ---------------------------------------------------------------------- квоты


def test_quota_docs_403_before_queue(kb_env, monkeypatch) -> None:
    monkeypatch.setenv("VELES_QUOTA_DOCS", "1")
    client = make_static_client(kb_env["bundle"])
    first = upload_json(client, DOC_V1)
    assert first.status_code == 202
    assert wait_done(client, first.json()["job_id"])["status"] == "done"

    second = upload_json(client, "Совсем другой текст про командировки", title="Командировки")
    assert second.status_code == 403
    assert "квота" in second.json()["detail"]
    # отказ ДО очереди: новых джобов нет
    jobs = client.get("/v1/documents/jobs").json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["document_id"] == DOC_ID
    assert not (uploads_root() / user_dir("") / "Командировки.md").exists()

    # 0 = лимит снят
    monkeypatch.setenv("VELES_QUOTA_DOCS", "0")
    third = upload_json(client, "Совсем другой текст про командировки", title="Командировки")
    assert third.status_code == 202
    assert wait_done(client, third.json()["job_id"])["status"] == "done"


def test_quota_size_413(kb_env, monkeypatch) -> None:
    monkeypatch.setenv("VELES_QUOTA_MB", "1")
    client = make_static_client(kb_env["bundle"])
    big = b"x" * (1536 * 1024)  # 1.5 МБ > 1 МБ
    r = client.post("/v1/documents", files={"file": ("большой.txt", big, "text/plain")})
    assert r.status_code == 413
    assert "МБ" in r.json()["detail"]
    assert not (uploads_root() / user_dir("") / "большой.txt").exists()


# ------------------------------------------- кап размера запроса (P1 ревью K4)


def test_upload_content_length_cap_413_before_read(kb_env, monkeypatch) -> None:
    """P1 ревью K4: Content-Length больше жёсткого капа VELES_UPLOAD_MAX_MB -
    413 ДО чтения тела (точная квота и парсер форм не вызываются), L0 и джобов
    нет; файл в пределах капа проходит."""
    monkeypatch.setenv("VELES_UPLOAD_MAX_MB", "1")
    client = make_static_client(kb_env["bundle"])
    big = b"x" * (1536 * 1024)  # 1.5 МБ > капа 1 МБ
    r = client.post("/v1/documents", files={"file": ("большой.txt", big, "text/plain")})
    assert r.status_code == 413
    assert "капа" in r.json()["detail"] and "VELES_UPLOAD_MAX_MB" in r.json()["detail"]
    assert not (uploads_root() / user_dir("") / "большой.txt").exists()
    assert client.get("/v1/documents/jobs").json()["jobs"] == []

    r2 = client.post("/v1/documents",
                     files={"file": ("небольшой.txt", "небольшой текст".encode("utf-8"),
                                     "text/plain")})
    assert r2.status_code == 202


def test_upload_content_length_fake_header_413(kb_env, monkeypatch) -> None:
    """P1: фейковый Content-Length больше капа при крошечном теле - 413 до
    чтения: заголовок проверяется раньше парсера формы."""
    monkeypatch.setenv("VELES_UPLOAD_MAX_MB", "1")
    client = make_static_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents",
        files={"file": ("хитрый.txt", b"tiny", "text/plain")},
        headers={"content-length": str(10 * 1024 * 1024)},
    )
    assert r.status_code == 413
    assert "капа" in r.json()["detail"]
    assert not (uploads_root() / user_dir("") / "хитрый.txt").exists()


def test_upload_content_length_quota_remainder_413(kb_env, monkeypatch) -> None:
    """P1: Content-Length больше свободного остатка квоты (с запасом на
    перезапись крупнейшего своего L0) - 413 до чтения; меньший размер доходит
    до точной проверки check_quota."""
    monkeypatch.setenv("VELES_QUOTA_MB", "1")
    client = make_static_client(kb_env["bundle"])
    first = client.post("/v1/documents",
                        files={"file": ("часть.txt", b"x" * (600 * 1024), "text/plain")})
    assert first.status_code == 202
    assert wait_done(client, first.json()["job_id"])["status"] == "done"

    # 1.2 МБ > остаток 0.4 МБ + запас на перезапись 0.6 МБ - ранний отказ
    second = client.post("/v1/documents",
                         files={"file": ("сверх.txt", b"x" * (1200 * 1024), "text/plain")})
    assert second.status_code == 413
    assert "остатка" in second.json()["detail"]
    assert not (uploads_root() / user_dir("") / "сверх.txt").exists()

    # 0.5 МБ: ранний гейт проходит (запас на перезапись), точная проверка
    # check_quota отсекает - прежняя семантика 413 сохранена
    third = client.post("/v1/documents",
                        files={"file": ("впритык.txt", b"x" * (500 * 1024), "text/plain")})
    assert third.status_code == 413
    assert "занято" in third.json()["detail"]


def test_quota_check_and_write_atomic_per_user(kb_env, monkeypatch) -> None:
    """P2 ревью K4 (TOCTOU): два одновременных потока выполняют check+write под
    per-user замком - квота не превышается, второй получает отказ по документам
    (без замка оба прошли бы проверку до записи)."""
    import threading as _threading

    monkeypatch.setenv("VELES_QUOTA_DOCS", "1")
    monkeypatch.delenv("VELES_QUOTA_MB", raising=False)
    barrier = _threading.Barrier(2)
    outcomes: list[str] = []

    def worker(i: int) -> None:
        barrier.wait(timeout=10)
        directory = user_dir("")
        try:
            with user_quota_lock(directory):
                l0 = uploads_root() / directory / f"f{i}.txt"
                check_quota(directory, 10, replacing=l0.exists())
                l0.parent.mkdir(parents=True, exist_ok=True)
                l0.write_bytes(b"x" * 10)
            outcomes.append("ok")
        except QuotaExceeded:
            outcomes.append("quota")

    threads = [_threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(outcomes) == ["ok", "quota"]
    assert len(list((uploads_root() / user_dir("")).glob("f*.txt"))) == 1


# ------------------------------------------------- ревизии, no-op, перезапись


def test_reupload_creates_r2_old_revision_only_via_param(kb_env) -> None:
    """Повторная загрузка того же document_id: r2, старая ревизия не в дефолтном
    поиске, доступна revision-параметром; node_version_id не меняются."""
    client = make_static_client(kb_env["bundle"])
    j1 = wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    assert (j1["action"], j1["revision_id"]) == ("created", "r1")
    bundle = client.app.state.veles["pipeline"].bundle
    keys_before = set(bundle.node_meta)
    r1_meta = {k: dict(m) for k, m in bundle.node_meta.items()}

    j2 = wait_done(client, upload_json(client, DOC_V2).json()["job_id"])
    assert (j2["action"], j2["revision_id"], j2["status"]) == ("updated", "r2", "done")

    # дефолт: только r2; старая - только по revision-параметру
    texts = query_texts(client)
    assert any(DOC_MARK_V2 in t for t in texts)
    assert all(DOC_MARK_V1 not in t for t in texts)
    texts_r1 = query_texts(client, revision="r1")
    assert any(DOC_MARK_V1 in t for t in texts_r1)
    assert all(DOC_MARK_V2 not in t for t in texts_r1)

    # node_version_id не менялись: все ключи на месте, у r1 статус сменился
    assert keys_before <= set(bundle.node_meta)
    assert set(bundle.node_meta) - keys_before, "узлы r2 добавлены"
    for nid, meta in r1_meta.items():
        if meta.get("document_id") == DOC_ID and meta.get("revision_id") == "r1":
            # nid (ключ меты) и есть node_version_id: остался тем же, изменился только статус
            assert nid in bundle.node_meta
            assert bundle.node_meta[nid]["status"] == "superseded"


def test_reupload_same_sha_is_noop(kb_env) -> None:
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    wait_done(client, upload_json(client, DOC_V2).json()["job_id"])
    bundle = client.app.state.veles["pipeline"].bundle
    meta_snapshot = json.dumps(
        {k: {kk: vv for kk, vv in m.items() if kk != "provenance"}
         for k, m in bundle.node_meta.items()}, ensure_ascii=False, sort_keys=True)

    j3 = wait_done(client, upload_json(client, DOC_V2).json()["job_id"])
    assert (j3["status"], j3["action"], j3["revision_id"]) == ("done", "unchanged", "r2")
    meta_now = json.dumps(
        {k: {kk: vv for kk, vv in m.items() if kk != "provenance"}
         for k, m in bundle.node_meta.items()}, ensure_ascii=False, sort_keys=True)
    assert meta_now == meta_snapshot, "no-op не должен менять индекс"
    assert upload_outcomes(kb_env["canonical"])[-1] == "unchanged"


def test_uploaded_doc_survives_index_reload(kb_env) -> None:
    """Overlay-долговечность: после загрузки свежий load_index (как рестарт
    фасада) находит документ без пересборки; патч ревизии тоже переносится."""
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    wait_done(client, upload_json(client, DOC_V2).json()["job_id"])

    fresh = load_index(kb_env["index_dir"])
    embedder = HashingEmbedder()
    qvec = embedder.embed_query(DOC_QUERY)
    texts = [e.text for e in hybrid_search(fresh, qvec, DOC_QUERY, top_k=8)]
    assert any(DOC_MARK_V2 in t for t in texts), "загрузка потерялась после reload"
    assert all(DOC_MARK_V1 not in t for t in texts), "патч superseded не доехал"
    statuses = {m["status"] for m in fresh.node_meta.values()
                if m.get("revision_id") == "r1" and m.get("document_id") == DOC_ID}
    assert statuses == {"superseded"}


def test_corrupt_overlay_degrades_to_main_index(kb_env) -> None:
    """Битый overlay не роняет load_index: основной индекс остаётся рабочим."""
    (kb_env["index_dir"] / "overlay.npz").write_bytes(b"not-an-npz")
    fresh = load_index(kb_env["index_dir"])
    assert fresh.node_meta, "основной индекс обязан работать"


# ------------------------------------------------------------ рестарт очереди


def test_queue_restart_marks_pending_job_failed(kb_env) -> None:
    """Рестарт-негатив: джоб чужого boot_id в queued/running при чтении статуса
    честно переводится в failed «сервис перезапущен», повторное чтение - тоже."""
    pipeline = VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None)
    q1 = IngestQueue(pipeline_fn=lambda: pipeline, boot_id="boot-1", start_worker=False)
    job = q1.submit(user_id="", filename=f"{DOC_SLUG}.md", title=DOC_TITLE,
                    document_id=DOC_ID, size_bytes=10, groups=["personal"])
    # джоб в журнале как queued с boot-1; память «умерла» вместе с очередью q1
    q2 = IngestQueue(pipeline_fn=lambda: pipeline, boot_id="boot-2", start_worker=False)
    rec = q2.lookup(job.job_id)
    assert rec["status"] == "failed"
    assert "перезапущен" in rec["error"]
    assert q2.lookup(job.job_id)["status"] == "failed"  # персистентно, не вечный running


def test_list_jobs_marks_stale_boot_failed(kb_env) -> None:
    """P2 ревью K4: список джобов применяет ту же boot-id-семантику, что lookup, -
    джоб чужого boot_id в queued показывается failed «сервис перезапущен»
    (не вечный running) и персистентно фиксируется в журнале; свой живой
    queued не трогается."""
    pipeline = VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None)
    q1 = IngestQueue(pipeline_fn=lambda: pipeline, boot_id="boot-1", start_worker=False)
    job = q1.submit(user_id="", filename=f"{DOC_SLUG}.md", title=DOC_TITLE,
                    document_id=DOC_ID, size_bytes=10, groups=["personal"])
    q2 = IngestQueue(pipeline_fn=lambda: pipeline, boot_id="boot-2", start_worker=False)
    rec = next(r for r in q2.list_jobs() if r["job_id"] == job.job_id)
    assert rec["status"] == "failed" and "перезапущен" in rec["error"]
    # отказ персистентен: третье поколение очереди читает failed из журнала
    q3 = IngestQueue(pipeline_fn=lambda: pipeline, boot_id="boot-3", start_worker=False)
    assert q3.lookup(job.job_id)["status"] == "failed"
    # живой queued своего boot_id остаётся queued
    live = q1.submit(user_id="", filename="другой.md", title="Другой",
                     document_id=f"{user_dir('')}/другой.md",
                     size_bytes=5, groups=["personal"])
    own = next(r for r in q1.list_jobs() if r["job_id"] == live.job_id)
    assert own["status"] == "queued"


def test_done_job_survives_facade_restart(kb_env) -> None:
    """Завершённый джоб после рестарта фасада остаётся done (чтение из журнала)."""
    client = make_static_client(kb_env["bundle"])
    job_id = upload_json(client, DOC_V1).json()["job_id"]
    assert wait_done(client, job_id)["status"] == "done"
    # «рестарт»: новое приложение, пустая память очереди, тот же VELES_DATA
    client2 = make_static_client(kb_env["bundle"])
    r = client2.get(f"/v1/documents/jobs/{job_id}")
    assert r.status_code == 200 and r.json()["status"] == "done"


# ---------------------------------------------------------- форматные негативы


def test_format_negatives(kb_env) -> None:
    client = make_static_client(kb_env["bundle"])
    # пустой файл - 400
    r = client.post("/v1/documents", files={"file": ("пусто.txt", b"", "text/plain")})
    assert r.status_code == 400 and "пустой" in r.json()["detail"]
    # пустой текст в JSON - 400
    r = upload_json(client, "   ")
    assert r.status_code == 400 and "пустой" in r.json()["detail"]
    # неподдерживаемый тип - 415 с перечнем форматов
    r = client.post("/v1/documents",
                    files={"file": ("программа.exe", b"MZ-fake", "application/x-msdownload")})
    assert r.status_code == 415
    assert "txt" in r.json()["detail"] and "pdf" in r.json()["detail"]
    # обход каталогов в имени - 400
    r = client.post("/v1/documents",
                    files={"file": ("../зловредный.txt", b"data", "text/plain")})
    assert r.status_code == 400
    # ни multipart, ни JSON - 415
    r = client.post("/v1/documents", content=b"raw", headers={"content-type": "text/plain"})
    assert r.status_code == 415
    # очередь пуста: ни один негатив не создал джоб
    assert client.get("/v1/documents/jobs").json()["jobs"] == []


def test_pdf_job_fails_clean_without_docling(kb_env, monkeypatch) -> None:
    """PDF идёт через docling-конвейер: без docling в окружении джоб честно
    failed с человеческой причиной (в стенд-venv docling есть - ветка не бьётся).
    P2 ревью K4: упавший разбор удаляет L0 - квота не съедается навсегда."""
    client = make_static_client(kb_env["bundle"])
    import builtins

    real_import = builtins.__import__

    def no_docling(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("docling недоступен (тест)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_docling)
    r = client.post("/v1/documents",
                    files={"file": ("документ.pdf", b"%PDF-1.4 fake", "application/pdf")})
    assert r.status_code == 202
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "failed" and "документ.pdf" in job["error"]
    assert upload_outcomes(kb_env["canonical"])[-1] == "failed"
    assert not (uploads_root() / user_dir("") / "документ.pdf").exists()


def test_discard_l0_warns_on_unlink_failure(kb_env, monkeypatch, caplog) -> None:
    """P2 ревью K4: L0 не удалился (права/гонка fs) - только WARN в лог,
    воркер не падает."""
    import logging as _logging
    from pathlib import Path as _Path

    pipeline = VelesPipeline(kb_env["bundle"], HashingEmbedder(), llm=None)
    queue = IngestQueue(pipeline_fn=lambda: pipeline, start_worker=False)
    job = queue.submit(user_id="", filename="недоступный.txt", title="Т",
                       document_id=f"{user_dir('')}/недоступный.txt",
                       size_bytes=1, groups=["personal"])

    def denied(self):
        raise OSError("denied")

    monkeypatch.setattr(_Path, "unlink", denied)
    with caplog.at_level(_logging.WARNING, logger="veles.ingest.queue"):
        queue._discard_l0(job)  # не бросает
    assert any("не удалён" in r.message for r in caplog.records)
