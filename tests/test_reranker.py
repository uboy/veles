"""K17: reranker - юнит-тесты опционального пост-этапа hybrid_search.

Бриф docs/plan/briefs/2026-09-30-k17-reranker.md. Проверяются:
- rerank() на фейковом скорере: порядок меняется детерминированно, ties
  сохраняют входной порядок, усечение top_k, диагностика в Evidence;
- выключено по умолчанию (VELES_RERANKER_MODEL пуст) - прежнее поведение
  в точности, модуль реранкера не вызывается (ноль регрессии);
- fail-open: ошибка скорера/загрузки модели - входной порядок, ответ не роняется;
- preload(): no-op при выключенном, не роняет сервис при провале загрузки.

Бандл - синтетический (IndexBundle вручную), как в test_retrieval_calibration:
вектор узла = вектор запроса (cosine 1.0), BM25 настоящий. Скорер подменяется
напрямую (scorer=/monkeypatch _load) - реальная модель в юнит-тесты не тянется.
Тексты узлов - заведомо разной длины: фейковый скорер «скор = длина текста»,
это делает ожидаемый порядок детерминированным без реальной модели.
"""

from __future__ import annotations

import numpy as np
import pytest

import veles.config as config
import veles.retrieval.reranker as reranker
from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import IndexBundle
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import Evidence, hybrid_search


def _meta(nid: str, text: str, doc_id: str = "docs/doc1.pdf") -> dict:
    return {
        "title": f"Документ {nid}",
        "document_id": doc_id,
        "heading_path": [f"Документ {nid}"],
        "text": text,
        "status": "current",
        "access_groups": ["personal"],
        "revision_id": "r1",
        "provenance": {"source_id": doc_id, "page": 1},
    }


def _texts_by_len(prefix: str, n: int) -> dict[str, str]:
    """n текстов строго растущей длины: скор «длина» даёт обратный порядок id."""
    return {f"n{i}": f"{prefix} {i}" + " дополнение" * i for i in range(1, n + 1)}


def _bundle(texts: dict[str, str]) -> IndexBundle:
    """Бандл: каждый узел - вектор (1,0,..) (первая координата), BM25 по тексту.

    Векторы узлов одинаковы -> dense-канал отдаёт их в порядке добавления;
    ранги каналов детерминированы."""
    vec = NumpyFlatStore(dim=4)
    bm25 = Bm25Index()
    nodes: dict[str, dict] = {}
    for nid, text in texts.items():
        vec.add([nid], np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        bm25.add([nid], [text])
        nodes[nid] = _meta(nid, text)
    return IndexBundle(vector=vec, bm25=bm25, node_meta=nodes)


def _qvec() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class LenScorer:
    """Фейковый cross-encoder: скор пары = длина фрагмента (детерминированно)."""

    def predict(self, pairs):
        return [float(len(text)) for _, text in pairs]


class ExplodingScorer:
    def predict(self, pairs):
        raise RuntimeError("модель упала на инференсе")


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """Между кейсами - чистые кэши модели (тесты не зависят от порядка)."""
    reranker.reset_cache()
    yield
    reranker.reset_cache()


def _ev_texts(evidence) -> list[str]:
    return [e.text for e in evidence]


# --- rerank() на фейковом скорере ----------------------------------------------

def test_rerank_reorders_by_scorer_and_truncates() -> None:
    """Позитив: скорер меняет порядок детерминированно, диагностика заполнена,
    органические rrf не тронуты, выдача усечена до top_k."""
    texts = _texts_by_len("фрагмент", 5)
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "фрагмент", top_k=5)
    assert sorted(e.node_version_id for e in ev) == list(texts)
    rrf_before = {e.node_version_id: e.rrf for e in ev}

    out = reranker.rerank("фрагмент", ev, top_k=3, scorer=LenScorer())
    assert len(out) == 3
    # скор = длина текста: наверху самые длинные фрагменты (n5, n4, n3)
    assert _ev_texts(out) == [texts["n5"], texts["n4"], texts["n3"]]
    # диагностика: скор реранкера и позиция в гибридном порядке ДО перестановки
    for e in out:
        assert e.rerank_score == float(len(e.text))
        assert isinstance(e.rerank_rank_hybrid, int) and 1 <= e.rerank_rank_hybrid <= 5
    assert out[0].rerank_rank_hybrid == 5  # n5 был пятым в гибридном порядке
    # органические скоры не изменились (confidence/abstain не зависят от реранка)
    assert {e.node_version_id: e.rrf for e in out} == {
        k: rrf_before[k] for k in ("n5", "n4", "n3")
    }


def test_rerank_ties_keep_input_order() -> None:
    """Равные скоры - порядок не меняется (устойчивая сортировка)."""
    texts = {"a": "одинаковый текст", "b": "одинаковый текст", "c": "одинаковый текст"}
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "одинаковый", top_k=3)

    class ConstScorer:
        def predict(self, pairs):
            return [7.0] * len(pairs)

    out = reranker.rerank("одинаковый", ev, top_k=3, scorer=ConstScorer())
    assert _ev_texts(out) == _ev_texts(ev)


def test_rerank_disabled_empty_model_name_is_noop() -> None:
    """Выключено (пустое имя модели, скорер не подан) - входной порядок,
    усечённый до top_k, без диагностики."""
    texts = _texts_by_len("текст", 3)
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "текст", top_k=3)
    out = reranker.rerank("текст", list(ev), top_k=2, model_name="")
    assert _ev_texts(out) == _ev_texts(ev)[:2]
    assert all(e.rerank_score is None for e in out)


def test_rerank_fail_open_on_scorer_error() -> None:
    """Fail-open: исключение инференса - лог и входной порядок, не raise."""
    texts = _texts_by_len("текст", 3)
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "текст", top_k=3)
    out = reranker.rerank("текст", list(ev), top_k=2, scorer=ExplodingScorer())
    assert _ev_texts(out) == _ev_texts(ev)[:2]
    assert all(e.rerank_score is None for e in out)


def test_rerank_fail_open_on_load_error(monkeypatch) -> None:
    """Fail-open: модель не загрузилась - входной порядок; повторный вызов
    не пытается грузить снова (негативный кэш): CrossEncoder строится один раз."""
    calls = {"n": 0}

    class BrokenCrossEncoder:
        def __init__(self, *a, **kw):
            calls["n"] += 1
            raise RuntimeError("сеть недоступна")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", BrokenCrossEncoder)
    texts = _texts_by_len("текст", 2)
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "текст", top_k=2)
    out1 = reranker.rerank("текст", list(ev), top_k=2, model_name="fake/model")
    out2 = reranker.rerank("текст", list(ev), top_k=2, model_name="fake/model")
    assert _ev_texts(out1) == _ev_texts(ev)
    assert _ev_texts(out2) == _ev_texts(ev)
    assert calls["n"] == 1  # негативный кэш: вторая загрузка не предпринималась


def test_rerank_single_candidate_skips_scoring() -> None:
    """Один кандидат - пересортировка бессмысленна, скорер не вызывается."""
    called = {"n": 0}

    class Spy:
        def predict(self, pairs):
            called["n"] += 1
            return [1.0] * len(pairs)

    stub = Evidence(node_version_id="x", text="единственный текст", rrf=0.5)
    out = reranker.rerank("запрос", [stub], 5, scorer=Spy())
    assert called["n"] == 0
    assert len(out) == 1 and out[0] is stub


# --- интеграция в hybrid_search -------------------------------------------------

_Q = "уникальный формулировка запроса"
_TEXTS_POOL = _texts_by_len("узел содержит формулировку", 6)


def test_hybrid_search_reranker_on_reorders_from_pool(monkeypatch) -> None:
    """Позитив интеграции: включённый реранкер видит ПУЛ (25), а не топ-k, и
    поднимает фрагмент из хвоста гибридного порядка наверх выдачи."""
    bundle = _bundle(_TEXTS_POOL)
    # референс ВЫКЛЮЧЕННОГО реранкера - до патча конфига (дефолт пуст),
    # полная глубина - для сверки органических скоров любого узла пула
    off_full = hybrid_search(bundle, _qvec(), _Q, top_k=len(_TEXTS_POOL))
    off = hybrid_search(bundle, _qvec(), _Q, top_k=2)
    assert len(off) == 2
    assert [e.node_version_id for e in off] == ["n1", "n2"]  # dense-ранги 1,2

    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "fake/reranker")
    monkeypatch.setattr(reranker, "_load", lambda name: LenScorer())
    on = hybrid_search(bundle, _qvec(), _Q, top_k=2)
    assert len(on) == 2
    # скорер (длина текста) поднял самые длинные фрагменты из хвоста пула:
    # n6 был шестым в гибридном порядке - вне топ-2 без реранка
    assert [e.node_version_id for e in on] == ["n6", "n5"]
    assert on[0].rerank_score == float(len(_TEXTS_POOL["n6"]))
    assert on[0].rerank_rank_hybrid == 6
    assert on[1].rerank_rank_hybrid == 5
    # органические скоры при реранке не изменились
    off_rrf = {e.node_version_id: e.rrf for e in off_full}
    on_rrf = {e.node_version_id: e.rrf for e in on}
    assert on_rrf["n6"] == off_rrf["n6"]


def test_hybrid_search_reranker_off_by_default(monkeypatch) -> None:
    """Негатив (дефолт): VELES_RERANKER_MODEL пуст - реранкер не вызывается
    вовсе, выдача совпадает с явным выключением по порядку и скорам."""
    # не читаем живой env: на бенч-хостах переменная бывает экспортирована
    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "")
    assert config.RERANK_CANDIDATES == 25  # бриф K17: пул 25

    def forbidden(*a, **kw):
        raise AssertionError("rerank вызван при выключенном реранкере")

    monkeypatch.setattr(reranker, "rerank", forbidden)
    bundle = _bundle(_TEXTS_POOL)
    default = hybrid_search(bundle, _qvec(), _Q, top_k=3)  # None = дефолт конфига
    explicit = hybrid_search(bundle, _qvec(), _Q, top_k=3, rerank_candidates=10)
    assert [(e.node_version_id, e.rrf) for e in default] == [
        (e.node_version_id, e.rrf) for e in explicit
    ]
    assert len(default) == 3
    assert all(e.rerank_score is None for e in default)


def test_hybrid_search_reranker_fail_open(monkeypatch) -> None:
    """Fail-open интеграции: включён, но модель падает - порядок и скоры
    те же, что при выключенном (деградация к обычной выдаче, не 500)."""
    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "fake/reranker")

    def broken_load(name):
        raise RuntimeError("модель не загрузилась")

    monkeypatch.setattr(reranker, "_load", broken_load)
    bundle = _bundle(_TEXTS_POOL)
    failed = hybrid_search(bundle, _qvec(), _Q, top_k=3)
    off = hybrid_search(bundle, _qvec(), _Q, top_k=3)
    assert [(e.node_version_id, e.rrf) for e in failed] == [
        (e.node_version_id, e.rrf) for e in off
    ]
    assert all(e.rerank_score is None for e in failed)


def test_hybrid_search_rerank_candidates_param(monkeypatch) -> None:
    """rerank_candidates - явная глубина пула реранка (бенч/тесты); None = конфиг.
    Пул 3 = первые три узла гибридного порядка (n1..n3); скорер выбирает из НИХ
    (n3, n2), а не из всех шести - глубина пула уважается. Выдача всегда top_k."""
    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "fake/reranker")
    monkeypatch.setattr(reranker, "_load", lambda name: LenScorer())
    bundle = _bundle(_TEXTS_POOL)
    ev = hybrid_search(bundle, _qvec(), _Q, top_k=2, rerank_candidates=3)
    assert len(ev) == 2
    assert [e.node_version_id for e in ev] == ["n3", "n2"]
    assert ev[0].rerank_rank_hybrid == 3  # n3 был третьим в гибридном пуле


# --- preload ----------------------------------------------------------------------

def test_preload_noop_when_disabled(monkeypatch) -> None:
    """Выключено - preload ничего не грузит (модель не ищется вообще)."""
    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "")
    called = {"n": 0}

    def spy_load(name):
        called["n"] += 1
        return LenScorer()

    monkeypatch.setattr(reranker, "_load", spy_load)
    reranker.preload()
    assert called["n"] == 0


def test_preload_failure_does_not_raise(monkeypatch) -> None:
    """Провал загрузки в прогреве не роняет сервис; негативный кэш заполнен -
    первый rerank деградирует быстро, без повторной загрузки."""
    monkeypatch.setattr(config, "VELES_RERANKER_MODEL", "fake/model")
    calls = {"n": 0}

    def broken_load(name):
        calls["n"] += 1
        raise RuntimeError("упала загрузка")

    monkeypatch.setattr(reranker, "_load", broken_load)
    reranker.preload()  # не raise
    assert calls["n"] == 1
    texts = _texts_by_len("текст", 2)
    bundle = _bundle(texts)
    ev = hybrid_search(bundle, _qvec(), "текст", top_k=2)
    assert [e.node_version_id for e in ev] == ["n1", "n2"]  # fail-open в выдаче
