"""K10: удаление документов - DELETE /v1/documents/{id}, trash, tombstone, CLI.

Чек-лист брифа docs/plan/briefs/2026-09-29-k10-delete-docs.md: владелец удаляет
свой, чужой - 404 неотличимый, админ - любой; после DELETE поиск не находит
(в т.ч. revision=all/r1), source/asset 404, stats уменьшаются, L0 удалён,
trash на диске со всеми ревизиями, аудит-событие delete; рестарт фасада
(load_index с overlay-патчем) и полная пересборка документ не возвращают;
double-delete 404; квота освобождается; повторная загрузка того же текста
ищется сразу и после рестарта (regression сброса stale-патча overlay); CLI без
--yes показывает план, с --yes удаляет. Индекс - настоящий build_index;
эмбеддер - детерминированный hashing (канон: тесты без загрузки моделей),
dim = EMBED_DIM, чтобы overlay докатывался через load_index.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.canonical.store import _doc_dir, save_document, trash_root
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import uploads_root, user_dir
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver
from veles.retrieval.retriever import hybrid_search

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

DOC_TITLE = "Регламент закупок"
DOC_SLUG = "Регламент_закупок"
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
DOC_TEXTS = [DOC_MARK_V1, DOC_MARK_V2]

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""
SEED_ID = "05_vyshka_academic/lectures/коучинг.md"
SEED_QUERY = "коучинг цели мотивация"
SEED_MARK = "работает с целями"


class HashingEmbedder:
    """Тот же стаб, что в test_documents_ingest/e2e: sublinear tf по хеш-бакетам."""

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
    """tmp-окружение: VELES_DATA (uploads/trash), канон, INDEX_DIR + мини-индекс
    с одним засеянным документом (настоящий build_index, честный overlay через
    load_index) - как в test_documents_ingest."""
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
    app = create_app(
        pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None),
        resolver=StaticOwnerResolver(set(groups)),
    )
    return TestClient(app)


def make_identity_app(kb_env, monkeypatch):
    """Корпоративный профиль (как ACL-тест K4): иванов/петров - пользователи,
    boss - админ, owner - bearer-владелец."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")  # читается на запросе
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


def upload_json(client: TestClient, text: str, title: str = DOC_TITLE,
                headers: dict | None = None):
    return client.post("/v1/documents", json={"text": text, "title": title},
                       headers=headers or {})


def query_texts(client: TestClient, q: str, headers: dict | None = None,
                **fields) -> list[str]:
    d = client.post(
        "/v1/knowledge/query", json={"query": q, "use_llm": False, **fields},
        headers=headers or {},
    ).json()
    return [ev["text"] for ev in d["evidence"]]


def search_bundle(bundle, q: str) -> list[str]:
    qvec = HashingEmbedder().embed_query(q)
    return [e.text for e in hybrid_search(bundle, qvec, q, top_k=8)]


def doc_gone_from_search(client, headers=None, marks=DOC_TEXTS) -> bool:
    """Ни один маркер документа не ищется ни в дефолте, ни в revision=all,
    ни в revision=r1 (tombstone-фильтр безусловен)."""
    for rev in (None, "all", "r1"):
        fields = {"revision": rev} if rev else {}
        texts = query_texts(client, DOC_QUERY, headers=headers, **fields)
        assert not any(m in t for m in marks for t in texts), (rev, texts)
    return True


def seed_visible(client) -> bool:
    return any(SEED_MARK in t for t in query_texts(client, SEED_QUERY))


# ------------------------------------------------ владелец/чужой/админ + физика


def test_delete_owner_full_cycle(kb_env, monkeypatch) -> None:
    """Владелец удаляет свой: поиск закрывается СРАЗУ (default/all/r1), source и
    asset 404, stats уменьшились, L0 удалён, trash на диске с манифестом и
    узлами, аудит-событие delete; чужой и несуществующий - один 404, без
    личности - 401; double-delete 404; рестарт (load_index) и полная пересборка
    документ не возвращают, засеянный документ жив."""
    users = make_identity_app(kb_env, monkeypatch)
    ivanov, petrov, boss, owner = users["ivanov"], users["petrov"], users["boss"], users["owner"]
    client = TestClient(users["app"])

    r = upload_json(client, DOC_V1, headers=ivanov)
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert wait_done(client, r.json()["job_id"], headers=ivanov)["status"] == "done"

    # актив и source до удаления доступны владельцу
    doc_dir = _doc_dir(doc_id, kb_env["canonical"])
    (doc_dir / "assets").mkdir()
    (doc_dir / "assets" / "схема.png").write_bytes(b"\x89PNG-fake")
    assert client.get(f"/v1/source/{doc_id}", headers=ivanov).status_code == 200
    assert client.get(f"/v1/asset/{doc_id}/схема.png", headers=ivanov).status_code == 200

    def stat_nodes(headers) -> int:
        d = client.get("/v1/knowledge/stats", headers=headers).json()
        return d["totals"]["nodes"]

    nodes_before = stat_nodes(boss)
    assert nodes_before >= 1
    assert any(DOC_MARK_V1 in t for t in query_texts(client, DOC_QUERY, headers=ivanov))

    # чужой - 404; несуществующий - тот же 404 (неотличимость)
    foreign = client.delete(f"/v1/documents/{doc_id}", headers=petrov)
    assert foreign.status_code == 404
    missing = client.delete("/v1/documents/owner/нет-такого.md", headers=petrov)
    assert missing.status_code == 404
    assert foreign.json()["detail"] == missing.json()["detail"]
    # удаление не состоялось: документ на месте и ищется
    assert (doc_dir / "manifest.json").exists()
    assert any(DOC_MARK_V1 in t for t in query_texts(client, DOC_QUERY, headers=ivanov))
    # без личности - 401
    assert client.delete(f"/v1/documents/{doc_id}").status_code == 401

    # владелец удаляет свой
    r = client.delete(f"/v1/documents/{doc_id}", headers=ivanov)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "deleted" and body["document_id"] == doc_id
    assert body["nodes"] >= 1 and body["revisions"] == ["r1"]
    assert body["trash"] and re.fullmatch(r"\d{8}T\d{6}-.+", body["trash"][0])

    # поиск закрыт СРАЗУ: ни владельцу, ни админу, ни bearer-владельцу,
    # ни с revision=all/r1
    for headers in (ivanov, boss, owner):
        assert doc_gone_from_search(client, headers)
    # чтение закрыто: source и asset 404
    assert client.get(f"/v1/source/{doc_id}", headers=ivanov).status_code == 404
    assert client.get(f"/v1/asset/{doc_id}/схема.png", headers=ivanov).status_code == 404
    # stats уменьшились ровно на узлы документа, документа в оглавлении нет
    nodes_after = stat_nodes(boss)
    assert nodes_after == nodes_before - body["nodes"]
    assert doc_id not in {d["document_id"] for d in
                          client.get("/v1/knowledge/stats", headers=ivanov).json()["documents"]}
    # L0 удалён, канон исчез
    assert not (uploads_root() / doc_id).exists()
    assert not (doc_dir / "manifest.json").exists()

    # trash на диске: манифест, узлы, мета корзины
    entry = kb_env["data"] / "trash" / body["trash"][0]
    assert (entry / "manifest.json").exists() and (entry / "nodes.jsonl").exists()
    meta = json.loads((entry / "trash-meta.json").read_text(encoding="utf-8"))
    assert meta["document_id"] == doc_id and meta["deleted_by"] == "ivanov@corp.ru"
    assert doc_dir.name in entry.name  # имя каталога канона в имени корзины

    # аудит-событие delete
    events = [json.loads(l) for l in
              (kb_env["canonical"] / "ingest-audit.jsonl").read_text(encoding="utf-8").splitlines()]
    del_events = [e for e in events if e.get("event") == "delete"]
    assert del_events and del_events[-1]["document_id"] == doc_id
    assert del_events[-1]["by"] == "ivanov@corp.ru" and del_events[-1]["trash"] == body["trash"]

    # double-delete - 404
    assert client.delete(f"/v1/documents/{doc_id}", headers=ivanov).status_code == 404

    # рестарт фасада: load_index применяет overlay-патч - документ не всплывает
    fresh = load_index(kb_env["index_dir"])
    assert not any(DOC_MARK_V1 in t for t in search_bundle(fresh, DOC_QUERY))
    # полная пересборка: trash вне glob канона - документ не возвращается,
    # засеянный документ жив
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    bundle.save(kb_env["index_dir"], embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    rebuilt = load_index(kb_env["index_dir"])
    assert not any(DOC_MARK_V1 in t for t in search_bundle(rebuilt, DOC_QUERY))
    assert any(SEED_MARK in t for t in search_bundle(rebuilt, SEED_QUERY))


def test_delete_admin_and_bearer_can_delete_any(kb_env, monkeypatch) -> None:
    """Админ (заголовки) и bearer-владелец удаляют чужой документ; владелец без
    админа - нет (404)."""
    users = make_identity_app(kb_env, monkeypatch)
    ivanov, boss, owner = users["ivanov"], users["boss"], users["owner"]
    client = TestClient(users["app"])

    r = upload_json(client, DOC_V1, headers=ivanov)
    doc_id = r.json()["document_id"]
    wait_done(client, r.json()["job_id"], headers=ivanov)

    assert client.delete(f"/v1/documents/{doc_id}", headers=owner).status_code == 200
    assert doc_gone_from_search(client, ivanov)

    r2 = upload_json(client, "Отчёт по командировкам: авансовый отчёт", title="Командировки",
                     headers=ivanov)
    doc_id2 = r2.json()["document_id"]
    wait_done(client, r2.json()["job_id"], headers=ivanov)
    assert client.delete(f"/v1/documents/{doc_id2}", headers=boss).status_code == 200
    assert doc_gone_from_search(client, ivanov)


def test_delete_static_owner(kb_env) -> None:
    """Личный профиль (static): владелец-админ удаляет свою загрузку."""
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    assert any(DOC_MARK_V1 in t for t in query_texts(client, DOC_QUERY))

    r = client.delete(f"/v1/documents/{DOC_ID}")
    assert r.status_code == 200 and r.json()["nodes"] >= 1
    assert doc_gone_from_search(client)
    assert DOC_ID not in {d["document_id"] for d in
                          client.get("/v1/knowledge/stats").json()["documents"]}
    assert seed_visible(client)  # соседний документ не задет


# --------------------------------------------------------------- все ревизии


def test_delete_removes_all_revisions(kb_env) -> None:
    """DELETE снимает ДОКУМЕНТ: r2 текущая и r1 архивная уезжают в trash вместе,
    в поиске не остаются ни в каком revision-режиме (до и после рестарта)."""
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    j2 = wait_done(client, upload_json(client, DOC_V2).json()["job_id"])
    assert (j2["action"], j2["revision_id"]) == ("updated", "r2")

    r = client.delete(f"/v1/documents/{DOC_ID}")
    assert r.status_code == 200
    assert r.json()["revisions"] == ["r1", "r2"]

    entry = kb_env["data"] / "trash" / r.json()["trash"][0]
    # все ревизии в корзине: текущая r2 в корне, r1 - архивом
    manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["revision_id"] == "r2"
    assert (entry / "revision-r1" / "manifest.json").exists()
    assert (entry / "revision-r1" / "nodes.jsonl").exists()

    assert doc_gone_from_search(client, marks=DOC_TEXTS)
    fresh = load_index(kb_env["index_dir"])
    for rev in (None, "all", "r1", "r2"):
        qvec = HashingEmbedder().embed_query(DOC_QUERY)
        texts = [e.text for e in hybrid_search(
            fresh, qvec, DOC_QUERY, top_k=8, revision=rev)]
        assert not any(m in t for m in DOC_TEXTS for t in texts), rev
    # пересборка тоже не возвращает ни одну ревизию
    # mmap-карту матрицы освобождаем: Windows не даёт np.save поверх живой карты
    fresh.vector.close()
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    bundle.save(kb_env["index_dir"], embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    rebuilt = load_index(kb_env["index_dir"])
    assert not any(m in t for m in DOC_TEXTS for t in search_bundle(rebuilt, DOC_QUERY))


# -------------------------------------------------- tombstone засеянного (L2)


def test_delete_seeded_canon_document(kb_env) -> None:
    """Удаление корпусного документа из ОСНОВНОГО индекса (узлы не в overlay):
    исчезает сразу, рестарт и пересборка не возвращают, каталог канона пуст."""
    client = make_static_client(kb_env["bundle"])
    assert seed_visible(client)

    r = client.delete(f"/v1/documents/{SEED_ID}")
    assert r.status_code == 200 and r.json()["nodes"] >= 1
    assert not seed_visible(client)

    fresh = load_index(kb_env["index_dir"])
    assert not any(SEED_MARK in t for t in search_bundle(fresh, SEED_QUERY))

    # mmap-карту матрицы освобождаем: Windows не даёт np.save поверх живой карты
    fresh.vector.close()
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    bundle.save(kb_env["index_dir"], embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    assert not any(SEED_MARK in t for t in search_bundle(load_index(kb_env["index_dir"]), SEED_QUERY))
    seed_dir = _doc_dir(SEED_ID, kb_env["canonical"])
    assert not seed_dir.exists()
    assert (kb_env["data"] / "trash" / r.json()["trash"][0] / "nodes.jsonl").exists()


def test_delete_finds_legacy_twin_catalog(kb_env) -> None:
    """Ревью K13 P1: легаси-каталог-близнец (имя НЕ по текущей формуле _doc_dir,
    document_id в манифесте совпадает) находится фолбэк-сканом и удаляется вместе
    с формульным - без фикса близнец оставался в каноне, и пересборка воскрешала
    «удалённый» документ."""
    import shutil

    client = make_static_client(kb_env["bundle"])
    assert seed_visible(client)

    legacy = kb_env["canonical"] / "legacy_ugolovny_code_catalog"
    shutil.copytree(_doc_dir(SEED_ID, kb_env["canonical"]), legacy)
    assert legacy.exists()

    r = client.delete(f"/v1/documents/{SEED_ID}")
    assert r.status_code == 200
    assert not seed_visible(client)
    assert not legacy.exists()  # близнец уехал в trash вместе с формульным
    trash_names = r.json()["trash"]
    assert len(trash_names) == 2  # оба каталога
    for name in trash_names:
        assert (kb_env["data"] / "trash" / name / "manifest.json").exists()


# --------------------------------------------------------- квоты и re-upload


def test_delete_frees_quota(kb_env, monkeypatch) -> None:
    """Квоты задеты только в хорошую сторону: освободившееся место доступно
    сразу после удаления (rate-limit/очередь не изменены)."""
    monkeypatch.setenv("VELES_QUOTA_DOCS", "1")
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    second = upload_json(client, "Текст про командировки и авансовые отчёты", title="Командировки")
    assert second.status_code == 403 and "квота" in second.json()["detail"]

    assert client.delete(f"/v1/documents/{DOC_ID}").status_code == 200
    third = upload_json(client, "Текст про командировки и авансовые отчёты", title="Командировки")
    assert third.status_code == 202
    assert wait_done(client, third.json()["job_id"])["status"] == "done"


def test_reupload_after_delete_visible_and_survives_restart(kb_env) -> None:
    """Перезагрузка того же текста после удаления: node_version_id совпадают со
    старыми (r1 + тот же текст), stale tombstone-патч overlay не должен убить
    переизданный документ после рестарта (фикс сброса патчей в append_overlay)."""
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    assert client.delete(f"/v1/documents/{DOC_ID}").status_code == 200
    assert doc_gone_from_search(client)

    j = wait_done(client, upload_json(client, DOC_V1).json()["job_id"])
    assert (j["action"], j["revision_id"]) == ("created", "r1")
    # сразу ищется
    assert any(DOC_MARK_V1 in t for t in query_texts(client, DOC_QUERY))
    # и после рестарта (tombstone-патч не всплывает поверх новой меты)
    fresh = load_index(kb_env["index_dir"])
    assert any(DOC_MARK_V1 in t for t in search_bundle(fresh, DOC_QUERY))


# ---------------------------------------------------------------------- CLI


def test_cli_delete_plan_then_delete(kb_env, monkeypatch, capsys) -> None:
    """CLI veles delete: без --yes - только план (канон и поиск не тронуты),
    с --yes - канон в trash, L0 удалён, tombstone с диска (bundle=None) -
    после load_index документ не ищется."""
    from veles.cli import main

    monkeypatch.setattr("veles.cli.INDEX_DIR", kb_env["index_dir"])
    client = make_static_client(kb_env["bundle"])
    wait_done(client, upload_json(client, DOC_V1).json()["job_id"])

    monkeypatch.setattr(sys, "argv", ["veles", "delete", DOC_ID])
    assert main() == 1
    out = capsys.readouterr().out
    assert "будет удалён" in out and "--yes" in out
    assert (_doc_dir(DOC_ID, kb_env["canonical"]) / "manifest.json").exists()
    assert any(DOC_MARK_V1 in t for t in query_texts(client, DOC_QUERY))

    monkeypatch.setattr(sys, "argv", ["veles", "delete", DOC_ID, "--yes"])
    assert main() == 0
    out = capsys.readouterr().out
    assert f"удалён: {DOC_ID}" in out and "trash" in out
    assert not (_doc_dir(DOC_ID, kb_env["canonical"])).exists()
    assert not (uploads_root() / DOC_ID).exists()
    assert list((kb_env["data"] / "trash").iterdir())
    fresh = load_index(kb_env["index_dir"])
    assert not any(DOC_MARK_V1 in t for t in search_bundle(fresh, DOC_QUERY))
    assert seed_visible(client)  # CLI-tombstone не задел соседний документ

    # CLI по несуществующему - код 1
    monkeypatch.setattr(sys, "argv", ["veles", "delete", "owner/нет-такого.md", "--yes"])
    assert main() == 1
    assert "не найден" in capsys.readouterr().out
