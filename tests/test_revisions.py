"""K3: версии документов - reingest/архивация ревизий, ревизионный фильтр
retrieval, API-параметр revision, stats.superseded_nodes, warn_conflict.

Чек-лист брифа docs/plan/briefs/2026-09-29-k3-document-revisions.md.
Совместимость с прод-каноном до 29.09 имитируется tmp-структурой (манифесты без
полей ревизий). Индекс в retrieval-тестах собирается настоящим build_index;
эмбеддер - детерминированный hashing (канон: тесты без загрузки моделей).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_api import FakeEmbedder
from veles.api.app import VelesPipeline, create_app, render_chat_content
from veles.canonical.model import CanonicalDocument, Provenance
from veles.canonical.store import _doc_dir, ingest_document, iter_documents
from veles.config import EMBED_MODEL
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.answer import decide, edition_signals
from veles.policy.identity import StaticOwnerResolver
from veles.retrieval.retriever import Citation, Evidence, hybrid_search

HASH_DIM = 512
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

POLICY_V1 = """\
# Парольная политика

Минимальная длина пароля составляет 8 символов.
"""
POLICY_V2 = """\
# Парольная политика

Минимальная длина пароля составляет 12 символов.
"""
POLICY_REL = "05_vyshka_academic/lectures/Политики/парольная_политика.md"
POLICY_QUERY = "какая минимальная длина пароля"


class HashingEmbedder:
    """Тот же стаб, что в e2e: sublinear tf по хеш-бакетам слов, cosine."""

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


def _parse(tmp_path, rel: str, text: str) -> CanonicalDocument:
    corpus = tmp_path / "corpus"
    path = corpus / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return TxtMdParserAdapter().parse_file(path, corpus_root=corpus)


def _manifest(doc_dir) -> dict:
    return json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- store: ревизии


def test_reingest_changed_file_archives_r1_and_writes_r2(tmp_path) -> None:
    canon = tmp_path / "canonical"
    o1 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    assert (o1.action, o1.revision_id, o1.previous_revision) == ("created", "r1", None)

    o2 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    assert o2.action == "updated"
    assert o2.revision_id == "r2" and o2.previous_revision == "r1"

    doc_dir = _doc_dir(POLICY_REL, canon)
    # старый канон - в revision-r1/ со статусом superseded
    arch = doc_dir / "revision-r1"
    assert (arch / "manifest.json").exists() and (arch / "nodes.jsonl").exists()
    am = _manifest(arch)
    assert am["status"] == "superseded" and am["revision_id"] == "r1"
    assert "8 символов" in (arch / "nodes.jsonl").read_text(encoding="utf-8")
    # новый канон - на месте, r2/current с previous_revision
    m = _manifest(doc_dir)
    assert m["revision_id"] == "r2" and m["status"] == "current"
    assert m["previous_revision"] == "r1"
    assert "12 символов" in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8")
    # node_version_id пересчитаны: ключи r1 и r2 не пересекаются
    r1_ids = {
        json.loads(l)["node_version_id"]
        for l in (arch / "nodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    }
    r2_ids = {
        json.loads(l)["node_version_id"]
        for l in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    }
    assert r1_ids.isdisjoint(r2_ids)

    # итераторы: без ревизий - только текущая, с ревизиями - обе
    cur = list(iter_documents(root=canon))
    assert [d.revision_id for d in cur] == ["r2"]
    allrev = list(iter_documents(root=canon, include_revisions=True))
    assert sorted(d.revision_id for d in allrev) == ["r1", "r2"]
    assert {d.status for d in allrev} == {"current", "superseded"}


def test_reingest_same_sha_is_noop(tmp_path) -> None:
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    doc_dir = _doc_dir(POLICY_REL, canon)
    manifest_before = (doc_dir / "manifest.json").read_bytes()

    o3 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    assert o3.action == "unchanged" and o3.revision_id == "r2"
    assert not (doc_dir / "revision-r2").exists()
    assert (doc_dir / "manifest.json").read_bytes() == manifest_before
    # аудит: событие на каждый реальный ingest, no-op журнал не пишет
    audit = canon / "ingest-audit.jsonl"
    assert audit.exists() and len(audit.read_text(encoding="utf-8").splitlines()) == 2


def test_second_change_creates_r3_chain(tmp_path) -> None:
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    v3 = POLICY_V2.replace("12 символов", "16 символов")
    o3 = ingest_document(_parse(tmp_path, POLICY_REL, v3), root=canon)
    assert (o3.action, o3.revision_id, o3.previous_revision) == ("updated", "r3", "r2")
    doc_dir = _doc_dir(POLICY_REL, canon)
    assert sorted(d.name for d in doc_dir.glob("revision-r*")) == ["revision-r1", "revision-r2"]
    assert all(_manifest(d)["status"] == "superseded" for d in doc_dir.glob("revision-r*"))
    assert _manifest(doc_dir)["previous_revision"] == "r2"


def test_legacy_canon_without_revision_fields_reads_and_upgrades(tmp_path) -> None:
    """Прод-структура до 29.09: манифест вообще без полей ревизий - читается
    без миграции (r1/current), повторный ингест апгрейдит до ревизионной схемы."""
    canon = tmp_path / "canonical"
    corpus = tmp_path / "corpus"
    rel = "docs/старый.md"
    path = corpus / rel
    path.parent.mkdir(parents=True)
    path.write_text("Старый текст нормы.\n", encoding="utf-8", newline="\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    doc_dir = _doc_dir(rel, canon)
    doc_dir.mkdir(parents=True)
    # манифест руками: ни revision_id, ни status, ни previous_revision
    (doc_dir / "manifest.json").write_text(
        json.dumps({
            "document_id": rel, "title": "старый", "sha256": sha,
            "parser_profile": "txt-md:0.1",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    (doc_dir / "nodes.jsonl").write_text(
        json.dumps({
            "logical_node_id": f"{rel}#p0001", "node_version_id": "legacy1",
            "node_type": "paragraph", "text": "Старый текст нормы.",
            "provenance": {"adapter": "txt-md:0.1", "source_id": rel, "lines": [1, 1]},
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    docs = list(iter_documents(root=canon))
    assert len(docs) == 1
    assert docs[0].revision_id == "r1" and docs[0].status == "current"
    assert docs[0].previous_revision is None

    # тот же файл - no-op
    o = ingest_document(_parse(tmp_path, rel, "Старый текст нормы.\n"), root=canon)
    assert o.action == "unchanged"
    # изменённый - r2, безполевой манифест уходит в архив уже со статусом
    o2 = ingest_document(_parse(tmp_path, rel, "Новый текст нормы.\n"), root=canon)
    assert (o2.action, o2.revision_id, o2.previous_revision) == ("updated", "r2", "r1")
    am = _manifest(doc_dir / "revision-r1")
    assert am["status"] == "superseded" and "revision_id" in am


def test_assets_move_into_revision_archive(tmp_path) -> None:
    canon = tmp_path / "canonical"
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG-fake-bytes")

    def make_doc(text: str, sha: str, with_asset: bool = True) -> CanonicalDocument:
        doc = CanonicalDocument(
            document_id="docs/sхемой.md", title="схема", parser_profile="txt-md:0.1",
            sha256=sha, assets={"img.png": str(img)} if with_asset else {},
        )
        doc.add_node(
            logical_node_id="docs/sхемой.md#p0001", text=text,
            provenance=Provenance(adapter="txt-md:0.1", source_id="docs/sхемой.md", lines=[1, 1]),
        )
        return doc

    ingest_document(make_doc("версия один", "a" * 64), root=canon)
    doc_dir = _doc_dir("docs/sхемой.md", canon)
    assert (doc_dir / "assets" / "img.png").exists()

    # вторая ревизия без активов: активы r1 живут только в архиве
    ingest_document(make_doc("версия два", "b" * 64, with_asset=False), root=canon)
    assert (doc_dir / "revision-r1" / "assets" / "img.png").exists()
    assert not (doc_dir / "assets").exists()
    assert (doc_dir / "revision-r1" / "manifest.json").exists()


# --------------------------------------- store: сбои архивации (ревью K3)


def test_recovery_after_crash_between_archive_and_write(tmp_path, monkeypatch) -> None:
    """Сбой между архивацией и записью нового канона: в doc_dir лежит полный
    revision-r1, текущего manifest.json нет. Ингест изменённого документа
    продолжает нумерацию (r2), узлы r1 не теряются, индекс собирается."""
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    doc_dir = _doc_dir(POLICY_REL, canon)
    # ручная имитация завершённой архивации без последующей записи (r1 → архив)
    arch = doc_dir / "revision-r1"
    arch.mkdir()
    (doc_dir / "manifest.json").replace(arch / "manifest.json")
    (doc_dir / "nodes.jsonl").replace(arch / "nodes.jsonl")
    m = json.loads((arch / "manifest.json").read_text(encoding="utf-8"))
    m["status"] = "superseded"
    m["revision_id"] = "r1"
    (arch / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    assert not (doc_dir / "manifest.json").exists()

    o2 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    assert (o2.action, o2.revision_id, o2.previous_revision) == ("updated", "r2", "r1")
    # узлы r1 целы в архиве (не перезаписаны записью r2)
    assert "8 символов" in (arch / "nodes.jsonl").read_text(encoding="utf-8")
    assert "12 символов" in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8")

    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canon)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    assert "8 символов" in " ".join(
        m["text"] for m in bundle.node_meta.values() if m["status"] == "superseded"
    )
    assert "12 символов" in " ".join(
        m["text"] for m in bundle.node_meta.values() if m["status"] == "current"
    )


def test_half_moved_archive_does_not_crash(tmp_path, monkeypatch, capsys) -> None:
    """Обрыв переезда старого порядка «manifest первым»: в revision-r1 лежит
    manifest.json, nodes.jsonl остался в корне. iter_documents/build_index не
    падают (обрывок пропущен с предупреждением), ингест дозавершивает перенос
    узлов в архив и продолжает нумерацию без потери r1."""
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    doc_dir = _doc_dir(POLICY_REL, canon)
    arch = doc_dir / "revision-r1"
    arch.mkdir()
    (doc_dir / "manifest.json").replace(arch / "manifest.json")  # nodes - в корне
    assert (doc_dir / "nodes.jsonl").exists() and not (arch / "nodes.jsonl").exists()

    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canon)
    # обрывок не отдаётся за документ: ни текущей (manifest в корне нет), ни архивной
    assert list(iter_documents(root=canon, include_revisions=True)) == []
    assert "без nodes.jsonl" in capsys.readouterr().err
    assert build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM)).node_meta == {}

    o2 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    assert (o2.action, o2.revision_id, o2.previous_revision) == ("updated", "r2", "r1")
    # единственная копия узлов r1 переехала в архив, а не была удалена как «дубль»
    assert "8 символов" in (arch / "nodes.jsonl").read_text(encoding="utf-8")
    assert _manifest(arch)["status"] == "superseded"
    assert "12 символов" in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8")

    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    texts = " ".join(m["text"] for m in bundle.node_meta.values())
    assert "8 символов" in texts and "12 символов" in texts
    assert {m["status"] for m in bundle.node_meta.values()} == {"current", "superseded"}


def test_sticky_empty_revision_dir_resumes(tmp_path) -> None:
    """«Липкий» пустой каталог ревизии (сбой сразу после mkdir, ни один move не
    дошёл): ингест дозавершивает архивацию вместо вечного RuntimeError, следующая
    ревизия продолжает нумерацию без ручного разбора."""
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    doc_dir = _doc_dir(POLICY_REL, canon)
    (doc_dir / "revision-r1").mkdir()
    assert not (doc_dir / "revision-r1" / "manifest.json").exists()

    o2 = ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    assert (o2.action, o2.revision_id, o2.previous_revision) == ("updated", "r2", "r1")
    am = _manifest(doc_dir / "revision-r1")
    assert am["status"] == "superseded" and am["revision_id"] == "r1"
    assert "8 символов" in (doc_dir / "revision-r1" / "nodes.jsonl").read_text(encoding="utf-8")

    v3 = POLICY_V2.replace("12 символов", "16 символов")
    o3 = ingest_document(_parse(tmp_path, POLICY_REL, v3), root=canon)
    assert (o3.action, o3.revision_id, o3.previous_revision) == ("updated", "r3", "r2")


def test_empty_sha256_not_noop(tmp_path) -> None:
    """Ревью K3: legacy-манифест с sha256='' и входящий doc.sha256='' - НЕ «без
    изменений» (пустой хеш не даёт ложного no-op с молчаливым отбрасыванием)."""
    canon = tmp_path / "canonical"

    def make_doc(text: str) -> CanonicalDocument:
        doc = CanonicalDocument(
            document_id="docs/bez_hash.md", title="без хеша", parser_profile="txt-md:0.1",
            sha256="",
        )
        doc.add_node(
            logical_node_id="docs/bez_hash.md#p0001", text=text,
            provenance=Provenance(adapter="txt-md:0.1", source_id="docs/bez_hash.md", lines=[1, 1]),
        )
        return doc

    o1 = ingest_document(make_doc("версия один"), root=canon)
    assert o1.action == "created"
    doc_dir = _doc_dir("docs/bez_hash.md", canon)
    assert _manifest(doc_dir)["sha256"] == ""

    o2 = ingest_document(make_doc("версия два"), root=canon)
    assert (o2.action, o2.revision_id, o2.previous_revision) == ("updated", "r2", "r1")
    assert "версия два" in (doc_dir / "nodes.jsonl").read_text(encoding="utf-8")
    assert "версия один" in (doc_dir / "revision-r1" / "nodes.jsonl").read_text(encoding="utf-8")


# ------------------------------------------------------- retrieval: фильтр ревизий


@pytest.fixture()
def revision_bundle(tmp_path, monkeypatch):
    """Канон из двух ревизий одного документа + честный build_index по нему."""
    canon = tmp_path / "canonical"
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V1), root=canon)
    ingest_document(_parse(tmp_path, POLICY_REL, POLICY_V2), root=canon)
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canon)
    return build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))


def test_index_meta_carries_revision_and_status(revision_bundle) -> None:
    statuses = {m["status"] for m in revision_bundle.node_meta.values()}
    assert statuses == {"current", "superseded"}
    superseded = [m for m in revision_bundle.node_meta.values() if m["status"] == "superseded"]
    current = [m for m in revision_bundle.node_meta.values() if m["status"] == "current"]
    assert superseded and current
    assert all(m["revision_id"] == "r1" for m in superseded)
    assert all(m["revision_id"] == "r2" for m in current)
    assert "8 символов" in " ".join(m["text"] for m in superseded)


def test_default_search_excludes_superseded_ac6(revision_bundle) -> None:
    """AC6: устаревшая редакция отфильтрована в пользу действующей."""
    embedder = HashingEmbedder()
    qvec = embedder.embed_query(POLICY_QUERY)
    ev = hybrid_search(revision_bundle, qvec, POLICY_QUERY, top_k=8)
    texts = [e.text for e in ev]
    assert any("12 символов" in t for t in texts), "действующая редакция обязана найтись"
    assert all("8 символов" not in t for t in texts), "superseded-узел в дефолтной выдаче"


def test_search_all_and_r1(revision_bundle) -> None:
    embedder = HashingEmbedder()
    qvec = embedder.embed_query(POLICY_QUERY)
    ev_all = hybrid_search(revision_bundle, qvec, POLICY_QUERY, top_k=8, revision="all")
    texts = " ".join(e.text for e in ev_all)
    assert "8 символов" in texts and "12 символов" in texts
    assert {e.citations[0].revision_id for e in ev_all} == {"r1", "r2"}

    ev_r1 = hybrid_search(revision_bundle, qvec, POLICY_QUERY, top_k=8, revision="r1")
    assert ev_r1 and all("8 символов" in e.text for e in ev_r1)
    assert all(e.citations[0].revision_id == "r1" for e in ev_r1)

    assert hybrid_search(revision_bundle, qvec, POLICY_QUERY, top_k=8, revision="r9") == []


def test_citation_human_shows_revision_for_not_r1(revision_bundle) -> None:
    ev = hybrid_search(revision_bundle, HashingEmbedder().embed_query(POLICY_QUERY),
                       POLICY_QUERY, top_k=8)
    human = ev[0].citations[0].human()
    assert "ред. r2" in human


def test_archived_revision_invisible_cross_acl(tmp_path, monkeypatch) -> None:
    """Ревью K3, главный security-комбайн: архивная ревизия закрытого документа
    (группы не пересекаются с группами запросчика) невидима чужому пользователю
    даже с revision='all' - ACL-конъюнкт в authorized() первый и безусловный;
    entitled-группе архив виден, дефолт её не отдаёт."""
    canon = tmp_path / "canonical"

    def restricted(text: str) -> CanonicalDocument:
        doc = _parse(tmp_path, POLICY_REL, text)
        doc.access_groups = ["finance-restricted"]
        for n in doc.nodes:
            n.access_groups = ["finance-restricted"]
        return doc

    ingest_document(restricted(POLICY_V1), root=canon)  # r1 -> уйдёт в архив
    ingest_document(restricted(POLICY_V2), root=canon)  # r2 -> текущая

    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canon)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    embedder = HashingEmbedder()
    qvec = embedder.embed_query(POLICY_QUERY)

    # чужая группа: пусто в дефолте, в 'all' и в 'r1' - архив не исключение
    for rev in (None, "all", "r1"):
        ev = hybrid_search(bundle, qvec, POLICY_QUERY, top_k=8,
                           allowed_groups={"personal"}, revision=rev)
        assert ev == [], f"утечка закрытой ревизии при revision={rev!r}"

    # entitled-группа: 'all' видит ОБЕ ревизии (включая архивную), дефолт - только r2
    ev_all = hybrid_search(bundle, qvec, POLICY_QUERY, top_k=8,
                           allowed_groups={"finance-restricted"}, revision="all")
    texts_all = " ".join(e.text for e in ev_all)
    assert "8 символов" in texts_all and "12 символов" in texts_all
    assert {e.citations[0].revision_id for e in ev_all} == {"r1", "r2"}
    ev_def = hybrid_search(bundle, qvec, POLICY_QUERY, top_k=8,
                           allowed_groups={"finance-restricted"})
    texts_def = " ".join(e.text for e in ev_def)
    assert "12 символов" in texts_def and "8 символов" not in texts_def

    # тот же шов через HTTP: чужой пользователь с revision='all' не получает ничего
    client = make_client(bundle, groups={"personal"})
    d = client.post("/v1/knowledge/query",
                    json={"query": POLICY_QUERY, "use_llm": False, "revision": "all"}).json()
    assert d["evidence"] == []


# ------------------------------------------------------------------- API: revision


def make_client(bundle, groups=None) -> TestClient:
    pipeline = VelesPipeline(bundle, HashingEmbedder(), llm=None)
    app = create_app(pipeline=pipeline,
                     resolver=StaticOwnerResolver(groups or {"personal"}))
    return TestClient(app)


def test_query_revision_param_default_all_r1(revision_bundle) -> None:
    client = make_client(revision_bundle)
    d = client.post("/v1/knowledge/query",
                    json={"query": POLICY_QUERY, "use_llm": False}).json()
    assert d["decision"] == "answer"
    assert "12 символов" in d["answer"]
    assert "8 символов" not in [ev["text"] for ev in d["evidence"]]
    assert d["citations"][0]["revision_id"] == "r2"

    d_all = client.post("/v1/knowledge/query",
                        json={"query": POLICY_QUERY, "use_llm": False, "revision": "all"}).json()
    texts = [ev["text"] for ev in d_all["evidence"]]
    assert any("8 символов" in t for t in texts) and any("12 символов" in t for t in texts)
    assert {c["revision_id"] for c in d_all["citations"]} == {"r1", "r2"}

    d_r1 = client.post("/v1/knowledge/query",
                       json={"query": POLICY_QUERY, "use_llm": False, "revision": "r1"}).json()
    assert all("8 символов" in ev["text"] for ev in d_r1["evidence"])
    assert all(c["revision_id"] == "r1" for c in d_r1["citations"])


@pytest.mark.parametrize("revision", ["r9", "banana", "2", "r"])
def test_query_unknown_revision_is_400(revision_bundle, revision) -> None:
    client = make_client(revision_bundle)
    r = client.post("/v1/knowledge/query",
                    json={"query": POLICY_QUERY, "use_llm": False, "revision": revision})
    assert r.status_code == 400
    assert r.json()["detail"]
    r9 = client.post("/v1/knowledge/query",
                     json={"query": POLICY_QUERY, "use_llm": False, "revision": "r9"})
    assert "неизвестная ревизия" in r9.json()["detail"]


def test_chat_unknown_revision_is_400(revision_bundle) -> None:
    client = make_client(revision_bundle)
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": POLICY_QUERY}], "revision": "r9",
    })
    assert r.status_code == 400


def test_stats_superseded_nodes(revision_bundle) -> None:
    client = make_client(revision_bundle)
    d = client.get("/v1/knowledge/stats").json()
    n_superseded = sum(
        1 for m in revision_bundle.node_meta.values() if m["status"] == "superseded")
    assert d["totals"]["superseded_nodes"] == n_superseded > 0
    assert d["totals"]["nodes"] == len(revision_bundle.node_meta) - n_superseded
    # один документ - листинг только по текущим узлам
    assert len(d["documents"]) == 1


# ------------------------------------------------------------------ warn_conflict


def _ev(nid: str, text: str, rrf: float, doc_id: str) -> Evidence:
    cite = Citation(title=f"t-{doc_id}", document_id=doc_id, heading_path=[],
                    source_id="s.md", lines=[1, 1], node_version_id=nid)
    return Evidence(node_version_id=nid, text=text, rrf=rrf, dense_rank=1, bm25_rank=2,
                    citations=[cite])


def test_edition_signals_extraction() -> None:
    assert edition_signals("ред. 2, версия 1.1, редакция от 01.02.2020") == {
        "ред. 2", "версия 1.1", "01.02.2020", "редакция от 01.02.2020",
    }
    # без маркеров; голый год и «ред» внутри слова - не маркеры
    assert edition_signals("норма от 2020 года, тред 5, редактор 3") == set()


def test_warn_conflict_two_editions_of_norm() -> None:
    a = _ev("a", "Ставка налога 20 процентов. Редакция от 01.02.2020.", 0.0328, "docs/a.md")
    b = _ev("b", "Ставка налога 13 процентов. Редакция от 15.06.2023.", 0.0320, "docs/b.md")
    d = decide([a, b])
    assert d.decision == "warn_conflict"
    assert len(d.conflicts) == 2
    assert {c["document_id"] for c in d.conflicts} == {"docs/a.md", "docs/b.md"}
    assert all(c["revision_id"] == "r1" for c in d.conflicts)


@pytest.mark.parametrize(
    "b_text,b_doc,b_rrf",
    [
        # тот же document_id - это две ревизии одного документа, не конфликт
        ("Ставка налога 13 процентов. Редакция от 15.06.2023.", "docs/a.md", 0.0320),
        # нет маркера у второй стороны - молчание
        ("Ставка налога 13 процентов без даты.", "docs/b.md", 0.0320),
        # общий маркер (одна дата) - не разные редакции
        ("Ставка налога 13 процентов. Редакция от 01.02.2020.", "docs/b.md", 0.0320),
        # большая дельта score - один очевидный лидер
        ("Ставка налога 13 процентов. Редакция от 15.06.2023.", "docs/b.md", 0.0005),
    ],
)
def test_no_false_conflict(b_text, b_doc, b_rrf) -> None:
    """Негативные кейсы AC7: перечисленные ситуации НЕ дают warn_conflict."""
    a = _ev("a", "Ставка налога 20 процентов. Редакция от 01.02.2020.", 0.0328, "docs/a.md")
    b = _ev("b", b_text, b_rrf, b_doc)
    d = decide([a, b])
    assert d.decision == "answer"
    assert d.conflicts == []


def test_warn_conflict_via_api_and_chat_render(revision_bundle) -> None:
    """Фасад: warn_conflict доходит до клиента, chat-рендер содержит блок редакций."""
    meta = {
        "a": {
            "text": "Ставка налога 20 процентов. Редакция от 01.02.2020.",
            "access_groups": ["personal"], "title": "Док А",
            "document_id": "docs/a.md", "heading_path": [],
            "revision_id": "r1", "status": "current",
            "provenance": {"adapter": "txt-md:0.1", "source_id": "a.md", "lines": [1, 1]},
        },
        "b": {
            "text": "Ставка налога 13 процентов. Редакция от 15.06.2023.",
            "access_groups": ["personal"], "title": "Док Б",
            "document_id": "docs/b.md", "heading_path": [],
            "revision_id": "r2", "status": "current",
            "provenance": {"adapter": "txt-md:0.1", "source_id": "b.md", "lines": [2, 2]},
        },
    }

    class TwoVec:
        def search(self, q, top_k=10):
            return [("a", 0.9), ("b", 0.85)]

    class TwoBm25:
        def search(self, q, top_k=10):
            return [("a", 5.0), ("b", 4.9)]

    from veles.indexing.indexer import IndexBundle

    bundle = IndexBundle(vector=TwoVec(), bm25=TwoBm25(), node_meta=meta)
    pipeline = VelesPipeline(bundle, FakeEmbedder(), llm=None)
    result = pipeline.answer("какая ставка налога", top_k=5, use_llm=False)
    assert result["decision"] == "warn_conflict"
    assert len(result["diagnostics"]["conflicts"]) == 2

    content = render_chat_content(result)
    assert "противоречащие редакции" in content
    assert "Док А" in content and "Док Б" in content

    client = make_client(bundle)
    d = client.post("/v1/knowledge/query",
                    json={"query": "какая ставка налога", "use_llm": False}).json()
    assert d["decision"] == "warn_conflict"


# --------------------------------------------------------------------------- CLI


def test_cli_ingest_reports_update_and_noop(tmp_path, monkeypatch, capsys) -> None:
    corpus_root = tmp_path / "corpus"
    canon = tmp_path / "canonical"
    monkeypatch.setattr("veles.cli.CORPUS_ROOT", corpus_root)
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canon)
    rel = "05_vyshka_academic/lectures/doc.md"
    path = corpus_root / rel
    path.parent.mkdir(parents=True)
    args = argparse.Namespace(by="tester", no_manifest_check=True)

    path.write_text(POLICY_V1, encoding="utf-8", newline="\n")
    from veles.cli import cmd_ingest

    cmd_ingest(args)
    out1 = capsys.readouterr().out
    assert "doc.md: 2 узлов" in out1
    assert "doc.md: обновлён" not in out1 and "doc.md: без изменений" not in out1
    assert "новых=1" in out1

    path.write_text(POLICY_V2, encoding="utf-8", newline="\n")
    cmd_ingest(args)
    out2 = capsys.readouterr().out
    assert "doc.md: обновлён до r2, предыдущая r1 сохранена" in out2
    assert "обновлено=1" in out2

    cmd_ingest(args)
    out3 = capsys.readouterr().out
    assert "doc.md: без изменений" in out3
    assert "без изменений=1" in out3
