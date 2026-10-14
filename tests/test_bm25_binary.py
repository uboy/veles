"""Бинарный формат BM25-индекса (Ф1 плана index-format): CSR на numpy + bm25.npz.

Покрытие:
- бит-в-бит идентичность search() с эталонной реализацией v1 (питоновский
  цикл - точная копия старого Bm25Index.search) на синтетическом корпусе
  >= 2000 доков и >= 300 запросов, сравнение (id, score) точно;
- round-trip: индекс -> save bm25.npz -> load -> поиск совпадает;
- чтение старого bm25.json (payload как в старом save) - обратная совместимость;
- приоритет bm25.npz над bm25.json, чистка старого json при save;
- tie-break: при равных скорах порядок стабильный - по индексу дока;
- K4: add() из другого потока во время search() не даёт рассинхрона;
- негативы: битый bm25.npz -> внятная ValueError, не молчаливый мусор.
"""

from __future__ import annotations

import json
import math
import random
import sys
import threading

import numpy as np
import pytest

from veles.indexing.bm25 import Bm25Index, tokenize


# --- эталон и генерация корпуса ---

def reference_search(tfs, df, doc_lens, ids, avgdl, query, top_k=10, k1=1.5, b=0.75):
    """Точная копия Bm25Index.search из v1 (до CSR) - эталон бит-в-бит."""
    if not ids:
        return []
    q_terms = tokenize(query)
    n_docs = len(ids)
    scores: list[tuple[str, float]] = []
    for i in range(n_docs):
        score = 0.0
        for term in q_terms:
            tf = tfs[i].get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1 - b + b * doc_lens[i] / (avgdl or 1))
            score += idf * tf * (k1 + 1) / denom
        if score > 0:
            scores.append((ids[i], score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


def make_corpus(n_docs: int, seed: int = 42) -> list[str]:
    """Синтетический корпус: термин-везде (df == n_docs) + ципфовский словарь
    с повторами слов внутри документа и разной длиной."""
    rng = random.Random(seed)
    vocab = [f"термин{n:03d}" for n in range(180)]
    docs = []
    for _ in range(n_docs):
        length = rng.randint(20, 120)
        words = ["общийслово"]
        words += rng.choices(vocab, weights=[len(vocab) - j for j in range(len(vocab))], k=length - 1)
        # повторы: 1-3 повтора случайного термина (термины с повторами в запросе
        # и документе должны считаться как в v1)
        words += rng.choices(vocab, k=rng.randint(1, 3))
        docs.append(" ".join(words))
    return docs


def make_queries(n: int = 320, seed: int = 7) -> list[str]:
    """>= 300 запросов: 1-5 терминов словаря (с повторами), термин-везде,
    несуществующие слова, пустой запрос."""
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        terms = rng.choices(range(180), k=rng.randint(1, 5))
        queries.append(" ".join(f"термин{t:03d}" for t in terms))
    queries += [
        "общийслово",
        "общийслово общийслово термин005 термин005",
        "несуществующийтермин",
        "термин001 несуществующийтермин",
        "",
    ]
    return queries


def tf_dicts(docs: list[str]) -> list[dict[str, int]]:
    out = []
    for text in docs:
        tf: dict[str, int] = {}
        for t in tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        out.append(tf)
    return out


def df_from(tfs: list[dict[str, int]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for tf in tfs:
        for t in tf:
            df[t] = df.get(t, 0) + 1
    return df


def legacy_payload(tfs, df, ids, doc_lens, avgdl, k1=1.5, b=0.75) -> dict:
    """Payload ровно в формате старого Bm25Index.save (bm25.json)."""
    return {
        "k1": k1,
        "b": b,
        "ids": ids,
        "doc_lens": doc_lens,
        "tfs": tfs,
        "df": df,
        "avgdl": avgdl,
    }


def assert_search_exact(got, expected):
    """(id, score) точно: те же пары в том же порядке, скоры бит-в-бит."""
    assert len(got) == len(expected), f"{len(got)} != {len(expected)}"
    for (g_id, g_s), (e_id, e_s) in zip(got, expected):
        assert g_id == e_id
        assert g_s == e_s, f"{g_id}: {g_s!r} != {e_s!r}"


# --- бит-в-бит ---

def test_bitexact_search_matches_reference_on_big_corpus():
    """>= 2000 доков, >= 300 запросов: CSR-поиск == эталонный цикл v1 точно."""
    docs = make_corpus(2200, seed=42)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)

    idx = Bm25Index()
    idx.add(ids, docs)
    assert len(idx.ids) == len(docs)
    assert idx.avgdl == avgdl  # avgdl пересчитан как в v1, бит-в-бит

    queries = make_queries(320)
    assert len(queries) >= 300
    for q in queries:
        assert_search_exact(idx.search(q, top_k=10), reference_search(tfs, df, doc_lens, ids, avgdl, q))


def test_bitexact_nondefault_k1_b():
    docs = make_corpus(300, seed=5)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    idx = Bm25Index(k1=1.2, b=0.3)
    idx.add(ids, docs)
    for q in make_queries(40, seed=8):
        assert_search_exact(
            idx.search(q, top_k=5),
            reference_search(tfs, df, doc_lens, ids, avgdl, q, top_k=5, k1=1.2, b=0.3),
        )


# --- round-trip и форматы ---

def test_roundtrip_npz_search_identical(tmp_path):
    docs = make_corpus(400, seed=3)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    idx = Bm25Index()
    idx.add(ids, docs)

    out = tmp_path / "index"
    idx.save(out)
    assert (out / "bm25.npz").exists()
    loaded = Bm25Index.load(out)
    assert loaded.ids == ids
    for q in make_queries(60, seed=9):
        expected = reference_search(tfs, df, doc_lens, ids, avgdl, q)
        assert_search_exact(loaded.search(q), expected)
        assert_search_exact(loaded.search(q), idx.search(q))


def test_legacy_json_load_bitexact(tmp_path):
    """Старый bm25.json (tfs/df как в v1) читается, поиск бит-в-бит с v1."""
    docs = make_corpus(300, seed=13)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)

    d = tmp_path / "legacy"
    d.mkdir()
    (d / "bm25.json").write_text(
        json.dumps(legacy_payload(tfs, df, ids, doc_lens, avgdl), ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = Bm25Index.load(d)
    assert loaded.ids == ids
    assert loaded.avgdl == avgdl
    for q in make_queries(60, seed=10):
        assert_search_exact(loaded.search(q), reference_search(tfs, df, doc_lens, ids, avgdl, q))

    # апгрейд на месте: save пишет npz, синхронный bm25.json остаётся рядом
    # (v1-читаемость каталога), после перезагрузки поиск тот же
    loaded.save(d)
    assert (d / "bm25.npz").exists()
    assert (d / "bm25.json").exists()
    reloaded = Bm25Index.load(d)
    for q in make_queries(30, seed=11):
        assert_search_exact(reloaded.search(q), reference_search(tfs, df, doc_lens, ids, avgdl, q))


def test_npz_preferred_over_legacy_json(tmp_path):
    """В каталоге лежат оба файла - читается npz, а не старый json."""
    docs_a = make_corpus(50, seed=21)
    docs_b = make_corpus(50, seed=22)
    d = tmp_path / "both"
    d.mkdir()

    def payload_of(docs, tag):
        ids = [f"{tag}{i}" for i in range(len(docs))]
        tfs = tf_dicts(docs)
        df = df_from(tfs)
        doc_lens = [len(tokenize(t)) for t in docs]
        avgdl = sum(doc_lens) / len(doc_lens)
        return ids, legacy_payload(tfs, df, ids, doc_lens, avgdl)

    ids_b, payload_b = payload_of(docs_b, "b")
    (d / "bm25.json").write_text(json.dumps(payload_b, ensure_ascii=False), encoding="utf-8")
    # npz собираем рядом (save в d удалил бы json) и переносим - оба файла рядом
    side = tmp_path / "side"
    idx_a = Bm25Index()
    ids_a = [f"a{i}" for i in range(len(docs_a))]
    idx_a.add(ids_a, docs_a)
    idx_a.save(side)
    (side / "bm25.npz").replace(d / "bm25.npz")

    loaded = Bm25Index.load(d)
    assert loaded.ids == ids_a
    assert all(nid.startswith("a") for nid in loaded.ids)


def test_empty_index_roundtrip(tmp_path):
    idx = Bm25Index()
    assert idx.search("что угодно") == []
    d = tmp_path / "empty"
    idx.save(d)
    loaded = Bm25Index.load(d)
    assert loaded.ids == []
    assert loaded.search("что угодно") == []


def test_save_keeps_legacy_json_v1_readable(tmp_path):
    """Регрессия ревью (раунд 1, находка 2): save() не имеет права делать каталог
    односторонним. После записи bm25.npz рядом обязан лежать синхронный bm25.json
    в payload v1 - откат кода на v1 (load_index читает json.loads bm25.json)
    читает каталог без пересборки."""
    docs = make_corpus(250, seed=41)
    ids = [f"d{i}" for i in range(len(docs))]
    idx = Bm25Index()
    idx.add(ids, docs)

    d = tmp_path / "idx"
    idx.save(d)
    # payload читается ровно так, как его читает v1 load_index
    payload = json.loads((d / "bm25.json").read_text(encoding="utf-8"))
    assert set(payload) == {"k1", "b", "ids", "doc_lens", "tfs", "df", "avgdl"}
    assert payload["ids"] == ids
    # payload синхронен с данными индекса: tfs/df совпадают с эталонными словарями
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    assert payload["tfs"] == tfs
    assert payload["df"] == df
    assert payload["doc_lens"] == doc_lens
    assert payload["avgdl"] == avgdl
    assert payload["k1"] == 1.5 and payload["b"] == 0.75
    # v1-поиск по записанному payload бит-в-бит равен поиску нового индекса
    for q in make_queries(50, seed=42):
        expected = reference_search(
            payload["tfs"], payload["df"], payload["doc_lens"], payload["ids"],
            payload["avgdl"], q,
        )
        assert_search_exact(idx.search(q), expected)


def test_save_add_after_legacy_load(tmp_path):
    """load старого json -> add (инкремент в CSR) -> поиск с учётом новых доков."""
    docs = make_corpus(100, seed=31)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    d = tmp_path / "inc"
    d.mkdir()
    (d / "bm25.json").write_text(
        json.dumps(legacy_payload(tfs, df, ids, doc_lens, avgdl), ensure_ascii=False),
        encoding="utf-8",
    )
    idx = Bm25Index.load(d)
    new_docs = ["парковка пропуск спецпропуск категория " + "термин003 " * 5,
                "кофе переговорная кухня термин017"]
    new_ids = ["n0", "n1"]
    idx.add(new_ids, new_docs)
    all_docs = docs + new_docs
    all_ids = ids + new_ids
    all_tfs = tf_dicts(all_docs)
    all_df = df_from(all_tfs)
    all_lens = [len(tokenize(t)) for t in all_docs]
    all_avgdl = sum(all_lens) / len(all_lens)
    for q in ["парковка пропуск", "кофе переговорная", "термин003", "общийслово термин009"]:
        assert_search_exact(
            idx.search(q),
            reference_search(all_tfs, all_df, all_lens, all_ids, all_avgdl, q),
        )


# --- tie-break ---

def test_tiebreak_stable_by_doc_index(tmp_path):
    """Равные скоры: порядок - по индексу дока (стабильность list.sort из v1)."""
    same = "одно и то же слово повтор"
    docs = [same, same, same, "другой документ совсем", same]
    ids = ["a", "b", "c", "d", "e"]
    idx = Bm25Index()
    idx.add(ids, docs)

    res = idx.search("одно слово")
    assert [nid for nid, _ in res] == ["a", "b", "c", "e"]
    scores = [s for _, s in res]
    assert scores[0] == scores[1] == scores[2] == scores[3]

    # эталон даёт тот же порядок
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    assert res == reference_search(tfs, df, doc_lens, ids, avgdl, "одно слово")

    # и после round-trip порядок не меняется
    d = tmp_path / "tie"
    idx.save(d)
    assert Bm25Index.load(d).search("одно слово") == res


# --- K4: add из другого потока во время search ---

def test_k4_concurrent_add_during_search():
    docs = make_corpus(700, seed=11)
    ids = [f"d{i}" for i in range(len(docs))]
    idx = Bm25Index()
    idx.add(ids[:500], docs[:500])
    known = set(ids)
    errors: list[Exception] = []

    queries = [
        "термин004 термин011",
        "общийслово",
        "общийслово термин002 термин002",
        "несуществующий термин003",
    ]

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # агрессивная нарезка GIL - ловим гонку
    try:

        def writer():
            try:
                for batch in range(20):
                    lo = 500 + batch * 10
                    idx.add(ids[lo : lo + 10], docs[lo : lo + 10])
            except Exception as e:  # noqa: BLE001 - ошибка воркера = падение теста
                errors.append(e)

        t = threading.Thread(target=writer)
        t.start()
        while t.is_alive():
            for q in queries:
                res = idx.search(q, top_k=10)
                ss = []
                for nid, s in res:
                    assert nid in known, "search вернул неизвестный id"
                    assert s > 0.0
                    ss.append(s)
                assert ss == sorted(ss, reverse=True), "скоры не по убыванию"
        t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert not errors, f"воркер упал: {errors}"
    # финальное состояние полное и бит-в-бит с эталоном
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    assert len(idx.ids) == len(docs)
    assert idx.avgdl == avgdl
    for q in make_queries(100, seed=12):
        assert_search_exact(idx.search(q), reference_search(tfs, df, doc_lens, ids, avgdl, q))


# --- негативы: битый bm25.npz ---

def test_corrupt_npz_garbage_bytes(tmp_path):
    d = tmp_path / "garbage"
    d.mkdir()
    (d / "bm25.npz").write_bytes("это не npz, а мусорные байты".encode("utf-8"))
    with pytest.raises(ValueError, match="bm25.npz"):
        Bm25Index.load(d)


def test_corrupt_npz_missing_fields(tmp_path):
    d = tmp_path / "missing"
    d.mkdir()
    np.savez(d / "bm25.npz", k1=np.float64(1.5), b=np.float64(0.75))
    with pytest.raises(ValueError, match="нет полей"):
        Bm25Index.load(d)


def test_corrupt_npz_wrong_dtype(tmp_path):
    d = tmp_path / "dtype"
    d.mkdir()
    np.savez(
        d / "bm25.npz",
        k1=np.float64(1.5),
        b=np.float64(0.75),
        avgdl=np.float64(1.0),
        doc_lens=np.array([1], dtype=np.int32),  # должен быть int64
        df=np.array([1], dtype=np.int64),
        indptr=np.array([0, 1], dtype=np.int64),
        indices=np.array([0], dtype=np.int32),
        data=np.array([1.0], dtype=np.float64),
        terms=_terms_buf(["a"]),
        ids=_terms_buf(["x"]),
    )
    with pytest.raises(ValueError, match="doc_lens"):
        Bm25Index.load(d)


def test_corrupt_npz_inconsistent_indptr(tmp_path):
    d = tmp_path / "indptr"
    d.mkdir()
    np.savez(
        d / "bm25.npz",
        k1=np.float64(1.5),
        b=np.float64(0.75),
        avgdl=np.float64(1.0),
        doc_lens=np.array([1, 1], dtype=np.int64),
        df=np.array([1], dtype=np.int64),
        indptr=np.array([0, 1], dtype=np.int64),  # 1 строка, а ids две
        indices=np.array([0], dtype=np.int32),
        data=np.array([1.0], dtype=np.float64),
        terms=_terms_buf(["a"]),
        ids=_terms_buf(["x", "y"]),
    )
    with pytest.raises(ValueError, match="indptr"):
        Bm25Index.load(d)


def test_corrupt_npz_indices_out_of_range(tmp_path):
    d = tmp_path / "range"
    d.mkdir()
    np.savez(
        d / "bm25.npz",
        k1=np.float64(1.5),
        b=np.float64(0.75),
        avgdl=np.float64(1.0),
        doc_lens=np.array([1], dtype=np.int64),
        df=np.array([1], dtype=np.int64),
        indptr=np.array([0, 1], dtype=np.int64),
        indices=np.array([5], dtype=np.int32),  # термина с колонкой 5 нет
        data=np.array([1.0], dtype=np.float64),
        terms=_terms_buf(["a"]),
        ids=_terms_buf(["x"]),
    )
    with pytest.raises(ValueError, match="диапазон"):
        Bm25Index.load(d)


def _terms_buf(items: list[str]) -> np.ndarray:
    from veles.indexing.bm25 import _json_buffer

    return _json_buffer(items)


# --- инварианты структуры ---

def test_df_and_search_agree_after_add_growth():
    """Рост ёмкостей (удвоение) не ломает df/поиск: 3000 коротких доков
    выгоняют несколько перераспределений всех массивов."""
    docs = make_corpus(3000, seed=77)
    ids = [f"d{i}" for i in range(len(docs))]
    idx = Bm25Index()
    idx.add(ids, docs)
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    for q in make_queries(40, seed=78):
        assert_search_exact(idx.search(q), reference_search(tfs, df, doc_lens, ids, avgdl, q))
    # df бит-в-бит со словарём эталона
    for term, col in idx.term_to_col.items():
        assert int(idx.df[col]) == df[term]
