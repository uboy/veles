"""mmap базовой матрицы векторов (Ф2 плана index-format).

Покрытие:
- load_index отдаёт NumpyFlatStore с матрицей np.load(mmap_mode='r');
- поиск по memmap-матрице бит-в-бит совпадает с поиском по обычному ndarray
  (векторный канал и bm25-канал);
- add() после mmap-загрузки работает, save пишет полную матрицу,
  после «рестарта» (повторный load_index) новые доки ищутся;
- совместимость с overlay.apply: vstack memmap с оверлейной матрицей;
- обратная совместимость load_index: каталог со СТАРЫМ bm25.json (без npz).
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from tests.test_bm25_binary import (
    df_from,
    make_corpus,
    make_queries,
    reference_search,
    tf_dicts,
)
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.indexing.bm25 import Bm25Index, tokenize
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""


class HashingEmbedder:
    """Детерминированный hashing-эмбеддер (канон: тесты без загрузки моделей),
    тот же стаб, что в test_cli_add/test_delete: dim = EMBED_DIM, sublinear tf."""

    def __init__(self, dim: int = EMBED_DIM):
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
def index_dir(tmp_path, monkeypatch):
    """Индексный каталог от настоящей сборки: numpy_matrix.npy + bm25.npz + meta."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    idx = data / "index"
    corpus = tmp_path / "corpus"
    seed = corpus / "05_vyshka_academic/lectures/коучинг.md"
    seed.parent.mkdir(parents=True)
    seed.write_text(SEED_DOC, encoding="utf-8", newline="\n")
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    from veles.canonical.store import save_document
    from veles.ingest.txt_md import TxtMdParserAdapter

    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=EMBED_DIM))
    idx.mkdir(parents=True)
    bundle.save(idx, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    return idx


BM25_QUERIES = [
    "коучинг цели",
    "мотивация сотрудника",
    "работает с целями",
    "несуществующее слово",
    "",
]


def test_load_index_uses_mmap_and_search_identical(index_dir):
    bundle = load_index(index_dir)
    assert isinstance(bundle.vector.matrix, np.memmap), (
        f"матрица не mmap: {type(bundle.vector.matrix)}"
    )
    # эталон: та же матрица обычным ndarray + тот же bm25
    ref = NumpyFlatStore(dim=EMBED_DIM)
    ref.matrix = np.load(index_dir / "numpy_matrix.npy")
    ref.ids = json.loads((index_dir / "numpy_ids.json").read_text(encoding="utf-8"))
    ref_bm25 = Bm25Index.load(index_dir)

    emb = HashingEmbedder()
    for qtext in BM25_QUERIES + ["коучинг и мотивация", "цели"]:
        qvec = emb.embed_query(qtext)
        assert bundle.vector.search(qvec, top_k=7) == ref.search(qvec, top_k=7), qtext
        assert bundle.bm25.search(qtext, top_k=7) == ref_bm25.search(qtext, top_k=7), qtext


def test_add_after_mmap_and_save_full_matrix(index_dir, tmp_path):
    bundle = load_index(index_dir)
    assert isinstance(bundle.vector.matrix, np.memmap)
    n0 = len(bundle.vector.ids)
    emb = HashingEmbedder()
    texts = [
        "Новый документ про правильную парковку и спецпропуск категории B",
        "Ещё один документ про кофе в переговорных",
    ]
    new_ids = ["new0", "new1"]
    bundle.vector.add(new_ids, emb.embed_passages(texts))
    bundle.bm25.add(new_ids, texts)
    assert len(bundle.vector.matrix) == n0 + 2
    assert bundle.bm25.search("парковка спецпропуск")[0][0] == "new0"

    out = tmp_path / "saved"
    bundle.save(out, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    m = np.load(out / "numpy_matrix.npy")
    assert isinstance(m, np.ndarray) and not isinstance(m, np.memmap)
    assert m.shape == (n0 + 2, EMBED_DIM)
    ids = json.loads((out / "numpy_ids.json").read_text(encoding="utf-8"))
    assert ids == bundle.vector.ids

    # рестарт: всё ищется из пересохранённого каталога
    bundle2 = load_index(out)
    assert bundle2.bm25.search("кофе переговорных")[0][0] == "new1"
    qvec = emb.embed_query("кофе в переговорных")
    assert bundle2.vector.search(qvec, top_k=3)[0][0] == "new1"


def test_overlay_applies_over_mmap_matrix(index_dir):
    """apply_overlay поверх mmap-матрицы: vstack memmap с оверлейной - работает,
    дельта ищется (тот же механизм, что у загрузок фасада)."""
    emb = HashingEmbedder()
    from veles.indexing.overlay import append_overlay

    append_overlay(
        index_dir,
        {},
        {},
        ["ov0"],
        emb.embed_passages(["Оверлейный документ про бег и тренировки"]),
        [("ov0", "Оверлейный документ про бег и тренировки")],
    )
    bundle = load_index(index_dir)  # apply_overlay внутри
    assert bundle.bm25.search("бег тренировки")[0][0] == "ov0"
    qvec = emb.embed_query("бег и тренировки")
    assert bundle.vector.search(qvec, top_k=3)[0][0] == "ov0"


def test_load_index_legacy_json_dir(tmp_path):
    """Каталог старого стенда: bm25.json (без npz) + numpy_matrix.npy -
    load_index работает, bm25-поиск бит-в-бит с эталоном v1."""
    d = tmp_path / "legacy"
    d.mkdir()
    docs = make_corpus(60, seed=51)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    payload = {
        "k1": 1.5,
        "b": 0.75,
        "ids": ids,
        "doc_lens": doc_lens,
        "tfs": tfs,
        "df": df,
        "avgdl": avgdl,
    }
    (d / "bm25.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    rng = np.random.default_rng(51)
    matrix = rng.standard_normal((len(docs), EMBED_DIM)).astype(np.float32)
    np.save(d / "numpy_matrix.npy", matrix)
    (d / "numpy_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (d / "meta.json").write_text(
        json.dumps(
            {"node_meta": {}, "embed": {"model": EMBED_MODEL, "backend": "numpy-flat"},
             "identifiers": []},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    bundle = load_index(d)
    assert isinstance(bundle.vector.matrix, np.memmap)
    for q in make_queries(30, seed=52):
        expected = reference_search(tfs, df, doc_lens, ids, avgdl, q)
        assert len(bundle.bm25.search(q)) == len(expected)
        assert bundle.bm25.search(q) == expected
    # векторный канал по mmap-матрице совпадает с обычной
    ref = NumpyFlatStore(dim=EMBED_DIM)
    ref.matrix = matrix
    ref.ids = ids
    emb = HashingEmbedder()
    qvec = emb.embed_query("коучинг цели мотивация")
    assert bundle.vector.search(qvec, top_k=5) == ref.search(qvec, top_k=5)


def test_tokenize_untouched():
    """Токенизация не менялась - общая точка бит-в-бит эквивалентности."""
    assert tokenize("Кофе в переговорной, ккал-2016") == ["кофе", "в", "переговорной", "ккал", "2016"]
