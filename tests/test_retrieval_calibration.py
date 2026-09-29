"""K12: калибровка retrieval - юнит-тесты поверхности hybrid_search.

Бриф docs/plan/briefs/2026-09-29-k12-retrieval-calibration.md, eval
evals/results/2026-09-29-k12-retrieval-calibration.md. Три механизма:
identifier-буст документов названного кодекса, фильтр пустышек выдачи,
снятие дубликатов узла внутри канала RRF (дефект прод-индекса 29.09:
241 430 записей на 158 394 id). Дефолты - из veles.config (None = конфиг).

Бандл - синтетический (IndexBundle вручную): вектор узла = вектор запроса
(cosine 1.0), BM25 - настоящий. Так ранги каналов детерминированы и буст/дедуп
проверяются на точных значениях RRF.
"""

from __future__ import annotations

import numpy as np
import pytest

from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import IndexBundle
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import dedupe_pairs, hybrid_search, is_junk_node

RRF_K = 60


def _meta(title: str, text: str, doc_id: str = "docs/doc1.pdf") -> dict:
    return {
        "title": title,
        "document_id": doc_id,
        "heading_path": [title],
        "text": text,
        "status": "current",
        "access_groups": ["personal"],
        "revision_id": "r1",
        "provenance": {"source_id": doc_id, "page": 1},
    }


def _bundle(nodes: dict[str, dict], bm25_adds: int = 1) -> IndexBundle:
    """Бандл: каждый узел - вектор (1,0,..) (первая координата), BM25 текст
    добавляется bm25_adds раз (моделирование дубля прод-индекса)."""
    vec = NumpyFlatStore(dim=4)
    bm25 = Bm25Index()
    ids = []
    for nid, meta in nodes.items():
        vec.add([nid], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        for _ in range(bm25_adds):
            bm25.add([nid], [meta["text"]])
        ids.append(nid)
    return IndexBundle(vector=vec, bm25=bm25, node_meta=dict(nodes))


def _qvec() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


# --- identifier-буст (кодекс, названный в вопросе) ----------------------------

_NODES_BOOST = {
    "nk1": _meta("Налоговый кодекс Российской Федерации",
                 "Плательщиками налога признаются организации и физические лица",
                 "docs/nk.pdf"),
    "gk1": _meta("Гражданский кодекс Российской Федерации",
                 "Плательщиками налога признаются организации и физические лица",
                 "docs/gk.pdf"),
}
# оба узла текстово одинаковы -> каналы равны; вопрос называет НК РФ
_Q_BOOST = "Кто является плательщиком налога согласно НК РФ?"


def test_codex_boost_lifts_named_codex_document() -> None:
    """Буст поднимает узлы НК РФ над равным по каналам узлом ГК РФ
    (позитив: кейс «согласно НК РФ» меняет порядок в пользу названного акта)."""
    bundle = _bundle(_NODES_BOOST)
    base = hybrid_search(bundle, _qvec(), _Q_BOOST, top_k=2, codex_boost=0.0)
    assert [e.node_version_id for e in base] == ["nk1", "gk1"]  # ties: dense первым
    boosted = hybrid_search(bundle, _qvec(), _Q_BOOST, top_k=2, codex_boost=2.0)
    assert [e.node_version_id for e in boosted][0] == "nk1"
    assert boosted[0].rrf > base[0].rrf  # бонус вошёл в скор
    # бонус = codex_boost/(rrf_k+1) поверх равного вклада каналов
    assert boosted[0].rrf == pytest.approx(
        base[0].rrf + 2.0 / (RRF_K + 1.0)
    )


def test_codex_boost_noop_without_codex_mention() -> None:
    """Негатив: вопрос не называет кодекс - буст не меняет ни порядок, ни скоры."""
    bundle = _bundle(_NODES_BOOST)
    q = "Кто является плательщиком налога?"
    plain = hybrid_search(bundle, _qvec(), q, top_k=2, codex_boost=0.0)
    boosted = hybrid_search(bundle, _qvec(), q, top_k=2, codex_boost=2.0)
    assert [e.node_version_id for e in plain] == [e.node_version_id for e in boosted]
    assert [e.rrf for e in plain] == [e.rrf for e in boosted]


# --- фильтр пустышек выдачи ----------------------------------------------------

_NODES_JUNK = {
    "stub": _meta("НК РФ", "Утратил силу.", "docs/nk.pdf"),
    "content": _meta("НК РФ", "Часть вторая утратила силу с 2 июля 2021 г. - ФЗ от 29.11.2020.",
                     "docs/nk.pdf"),
}


def test_filter_junk_drops_stub_keeps_content() -> None:
    """filter_junk=True: пустышка «Утратил силу.» не занимает слот выдачи;
    содержательный узел про утрату силы (не пустышка) остаётся (негатив на
    false-positive фильтра)."""
    bundle = _bundle(_NODES_JUNK)
    q = "утратил силу"
    with_junk = hybrid_search(bundle, _qvec(), q, top_k=5, filter_junk=False)
    assert {e.node_version_id for e in with_junk} == {"stub", "content"}
    filtered = hybrid_search(bundle, _qvec(), q, top_k=5, filter_junk=True)
    assert [e.node_version_id for e in filtered] == ["content"]


def test_is_junk_node_patterns() -> None:
    """Позитивы/негативы шаблонов пустышек: только короткие заглушки целиком."""
    assert is_junk_node("Утратил силу.")
    assert is_junk_node("2. Не применяется.")
    assert is_junk_node("Бесплатная юридическая консультация")
    assert not is_junk_node("Статья 3. Часть вторая утратила силу с 2 июля 2021 г.")
    assert not is_junk_node("")  # пустой - не мусор, просто пустой


# --- дедуп каналов (дефект прод-индекса) ---------------------------------------

def test_dedupe_pairs_keeps_first_best_rank() -> None:
    assert dedupe_pairs([("a", 9.0), ("b", 8.0), ("a", 7.0)]) == [("a", 9.0), ("b", 8.0)]
    assert dedupe_pairs([]) == []


def test_hybrid_search_dedupe_channels_single_contribution() -> None:
    """Дубль узла в BM25-канале (прод-индекс: узел записан дважды) без дедупа
    получает ДВА вклада RRF и вытесняет узел другого канала; с дедупом - один
    вклад (кейс k11-056: BM25-дубли шапок съели top-5)."""
    nodes = {
        "dup": _meta("Гражданский кодекс Российской Федерации", "шапка кодекса",
                     "docs/gk.pdf"),
        "dense1": _meta("Налоговый кодекс Российской Федерации", "норма гл. 28",
                        "docs/nk.pdf"),
        "noise": _meta("Лекции по менеджменту", "совершенно посторонний текст",
                       "docs/lect.pdf"),
    }
    bundle = _bundle(nodes, bm25_adds=2)  # "dup" в BM25 дважды
    q = "шапка кодекса норма"
    depth = 10
    # фактические каналы (как их видит hybrid_search) - для точных ожиданий
    lex_ids = [nid for nid, _ in bundle.bm25.search(q, top_k=depth)]
    dup_lex_ranks = [i + 1 for i, nid in enumerate(lex_ids) if nid == "dup"]
    assert len(dup_lex_ranks) == 2, "прод defect: узел в канале дважды"
    second = dup_lex_ranks[1]  # вклад, который должен снять дедуп

    broken = hybrid_search(bundle, _qvec(), q, top_k=2, dedupe_channels=False)
    dup_broken = next(e for e in broken if e.node_version_id == "dup")
    dense1_broken = next(e for e in broken if e.node_version_id == "dense1")
    # двойной вклад вытащил дубликат на ранг 1, узел другого канала ниже
    assert broken[0].node_version_id == "dup"
    assert dup_broken.rrf > dense1_broken.rrf

    fixed = hybrid_search(bundle, _qvec(), q, top_k=2, dedupe_channels=True)
    dup_fixed = next(e for e in fixed if e.node_version_id == "dup")
    # дедуп снял ровно второй вклад дубля (узлы без дублей не пострадали по
    # вкладам своего места в другом канале); двойной вклад больше не раздувает скор
    assert dup_fixed.rrf == pytest.approx(dup_broken.rrf - 1.0 / (RRF_K + second))
    assert dup_fixed.rrf < dup_broken.rrf
    assert {e.node_version_id for e in fixed} == {e.node_version_id for e in broken}
