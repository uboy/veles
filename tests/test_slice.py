"""Юнит-тесты среза Ф1 (без загрузки моделей). Запуск: python -m pytest tests -q"""

from __future__ import annotations

import numpy as np
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
