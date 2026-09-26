"""Юнит-тесты среза Ф1 (без загрузки моделей). Запуск: python -m pytest tests -q"""

from __future__ import annotations

import re
import numpy as np
from pathlib import Path
import pytest

from veles.canonical.model import CanonicalDocument, Provenance
from veles.canonical.store import load_document, save_document
from veles.indexing.bm25 import Bm25Index, tokenize
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.answer import decide
from veles.retrieval.retriever import Citation, Evidence


def _doc() -> CanonicalDocument:
    doc = CanonicalDocument(
        document_id="test/doc-1.txt",
        title="Тестовый документ",
        parser_profile="txt-md:0.1",
        sha256="x",
    )
    doc.add_node(
        logical_node_id="test/doc-1.txt#root",
        node_type="root",
        heading_path=["курс", "лекция"],
        text="root",
        provenance=Provenance(adapter="txt-md:0.1", source_id="test/doc-1.txt"),
    )
    doc.add_node(
        logical_node_id="test/doc-1.txt#p0001",
        node_type="paragraph",
        parent_id="test/doc-1.txt#root",
        heading_path=["курс", "лекция"],
        text="Гибридный поиск объединяет лексическое соответствие и векторную близость.",
        provenance=Provenance(adapter="txt-md:0.1", source_id="test/doc-1.txt", lines=[1, 2]),
    )
    return doc


def test_canonical_roundtrip(tmp_path):
    doc = _doc()
    save_document(doc, root=tmp_path)
    loaded = load_document(doc.document_id, root=tmp_path)
    assert loaded.nodes[1].node_version_id == doc.nodes[1].node_version_id
    assert loaded.nodes[1].provenance.lines == [1, 2]
    # стабильность версии: тот же текст -> тот же id; нормализация пробелов
    from veles.canonical.model import CanonicalNode

    assert CanonicalNode.make_version_id("r1", "n", "текст  с   пробелами") == CanonicalNode.make_version_id("r1", "n", "текст с пробелами")


def test_bm25_positive_and_negative():
    bm = Bm25Index()
    bm.add(["a"], ["Гибридный поиск объединяет лексическое соответствие и векторы"])
    bm.add(["b"], ["Коучинг работает с целями и мотивацией клиента"])
    hits = bm.search("гибридный лексический поиск")
    assert hits and hits[0][0] == "a"
    # негатив: запрос целиком вне словаря -> пусто
    assert bm.search("квантовая запутанность кубитов") == []


def test_vector_store_cosine():
    store = NumpyFlatStore(dim=4)
    v = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    store.add(["x", "y"], v)
    hits = store.search(np.array([1, 0, 0, 0], dtype=np.float32), top_k=2)
    assert hits[0][0] == "x" and hits[0][1] > 0.99
    assert store.search(np.array([0, 0, 1, 0], dtype=np.float32), top_k=2)[0][1] < 0.01


def _ev(rrf: float, dense_rank=None, bm25_rank=None) -> Evidence:
    return Evidence(
        node_version_id="n",
        text="t",
        rrf=rrf,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
        citations=[Citation("t", "d", [], "s", None, "n")],
    )


def test_answer_policy_answer_and_abstain():
    strong = [_ev(rrf=0.030, dense_rank=1, bm25_rank=1)]  # оба канала согласны
    assert decide(strong).decision == "answer"
    weak = [_ev(rrf=0.014, dense_rank=12)]  # один канал, низкий ранг
    d = decide(weak)
    assert d.decision == "abstain"
    assert decide([]).decision == "abstain"


def test_sha256_from_raw_bytes(tmp_path):
    """P1-ревью: sha256 канона = хеш байтов L0 (CRLF не должен теряться)."""
    import hashlib

    from veles.ingest.txt_md import TxtMdParserAdapter

    corpus = tmp_path / "corpus"
    d = corpus / "course" / "лекция1"
    d.mkdir(parents=True)
    raw = b"\ufeffline1\r\nline2 with text\r\n"
    (d / "Пересказ тест от 01.01.2025.txt").write_bytes(raw)
    doc = TxtMdParserAdapter().parse_file(d / "Пересказ тест от 01.01.2025.txt", corpus_root=corpus)
    assert doc.sha256 == hashlib.sha256(raw).hexdigest()


def test_pre_retrieval_acl_filter():
    """P1-ревью: узлы чужих групп не попадают в evidence (инвариант 5)."""
    from veles.indexing.indexer import IndexBundle
    from veles.retrieval.retriever import hybrid_search

    class FakeVector:
        def search(self, q, top_k=10):
            return [("n-personal", 0.9), ("n-legal", 0.8)]

    class FakeBm25:
        def search(self, q, top_k=10):
            return [("n-legal", 2.0), ("n-personal", 1.5)]

    meta = {
        "n-personal": {"access_groups": ["personal"], "title": "t", "document_id": "d",
                        "heading_path": [], "text": "x", "provenance": {"source_id": "s", "adapter": "a"}},
        "n-legal": {"access_groups": ["legal"], "title": "t", "document_id": "d",
                    "heading_path": [], "text": "x", "provenance": {"source_id": "s", "adapter": "a"}},
    }
    bundle = IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta)
    q = np.zeros(4, dtype=np.float32)
    ids_open = [e.node_version_id for e in hybrid_search(bundle, q, "запрос", allowed_groups={"personal"})]
    assert "n-legal" not in ids_open and "n-personal" in ids_open
    ids_legal = [e.node_version_id for e in hybrid_search(bundle, q, "запрос", allowed_groups={"legal"})]
    assert ids_legal == ["n-legal"]


def test_allowlist_scan_idempotent_and_blacklist(tmp_path):
    """P1-ревью: сканер на tmp-фикстурах - паттерны, чёрный список, идемпотентность."""
    from veles.ingest.allowlist import scan_and_copy

    root = tmp_path / "src"
    lec = root / "курс" / "лекция1"
    lec.mkdir(parents=True)
    (lec / "Транскрипция курса.txt").write_text("текст лекции", encoding="utf-8")
    (lec / "Пересказ курса.txt").write_text("пересказ", encoding="utf-8")
    (lec / "чужой файл.txt").write_text("не транскрипция", encoding="utf-8")
    dz = root / "курс" / "лекция2_защита ДЗ"
    dz.mkdir(parents=True)
    (dz / "Транскрипция ДЗ.txt").write_text("домашка", encoding="utf-8")

    spec = {
        "name": "test-corpus",
        "root": root,
        "dest": Path("test/dest"),
        "dir_include": re.compile(r"лекция", re.IGNORECASE),
        "file_include": re.compile(r"(транскрипция|пересказ)", re.IGNORECASE),
        "file_ext": {".txt"},
        "path_exclude": re.compile(r"ДЗ", re.IGNORECASE),
    }
    corpus = tmp_path / "corpus"
    m1 = scan_and_copy(spec=spec, corpus_root=corpus, manifest_dir=tmp_path / "m")
    assert m1.stats["included_files"] == 2
    assert m1.stats["excluded_by_blacklist"] == 1
    assert m1.stats["copied_now"] == 2
    m2 = scan_and_copy(spec=spec, corpus_root=corpus, manifest_dir=tmp_path / "m")
    assert m2.stats["copied_now"] == 0  # идемпотентность: повторный скан ничего не копирует
