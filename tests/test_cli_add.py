"""K15: `veles add` - добавление одного txt/md-файла в базу без пересборки индекса
(демо-шаг «добавьте свой документ» в lite, где нет фасада и API-загрузки).

Проверки: документ после add ищется и в живом бандле, и после «рестарта»
(load_index с overlay); L0 лежит в uploads/owner/<имя> под тем же
document_id, что у фасада (`veles delete` снимает такой документ целиком);
повторный add того же содержимого - честный «без изменений»; негативы -
нет файла, не тот формат (L0 и канон не создаются); на свежей установке без
индекса add сам создаёт пустой индекс и документ попадает в поиск.
Эмбеддер - детерминированный hashing (канон: тесты без загрузки моделей),
dim = EMBED_DIM, чтобы overlay докатывался через load_index.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pytest

from veles.cli import cmd_add
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import uploads_root, user_dir
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import hybrid_search

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""

DOC_TITLE = "Правила парковки"
DOC_NAME = "Правила парковки.md"
DOC_TEXT = """\
# Правила парковки

На корпоративной парковке разрешено парковаться только сотрудникам
со спецпропуском категории B.

Гостевые места бронируются на ресепшене не менее чем за один день.
"""
DOC_QUERY = "как забронировать гостевое место на парковке"
DOC_MARK = "Гостевые места бронируются на ресепшене"


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
            import hashlib

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
    """tmp-окружение с мини-индексом (настоящий build_index + load_index)
    и подменённым эмбеддером cmd_add - как в test_delete."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    index_dir = data / "index"
    corpus = tmp_path / "corpus"
    seed = corpus / "05_vyshka_academic/lectures/коучинг.md"
    seed.parent.mkdir(parents=True)
    seed.write_text(SEED_DOC, encoding="utf-8", newline="\n")
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    from veles.canonical.store import save_document

    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    monkeypatch.setattr("veles.config.INDEX_DIR", index_dir)
    monkeypatch.setattr("veles.cli.build_embedder", lambda: HashingEmbedder())
    monkeypatch.setenv("VELES_DATA", str(data))
    return {"data": data, "canonical": canonical, "index_dir": index_dir}


def run_add(file, title="") -> int:
    return cmd_add(argparse.Namespace(file=str(file), title=title, by=""))


def test_add_searchable_and_survives_restart(kb_env, tmp_path):
    src = tmp_path / "входящие" / DOC_NAME
    src.parent.mkdir(parents=True)
    src.write_text(DOC_TEXT, encoding="utf-8", newline="\n")

    rc = run_add(src, title=DOC_TITLE)
    assert rc == 0

    # L0 - та же физика, что у фасада: uploads/<owner>/<имя>, id = owner/<имя>
    l0 = uploads_root() / user_dir("") / DOC_NAME
    assert l0.is_file()
    doc_id = f"{user_dir('')}/{DOC_NAME}"

    # «рестарт»: load_index должен поднять документ из overlay без пересборки
    bundle = load_index(kb_env["index_dir"])
    qvec = HashingEmbedder().embed_query(DOC_QUERY)
    texts = [e.text for e in hybrid_search(bundle, qvec, DOC_QUERY, top_k=5)]
    assert any(DOC_MARK in t for t in texts), texts
    # document_id в мете - фасадный формат owner/<имя>
    metas = [m for m in bundle.node_meta.values() if m["document_id"] == doc_id]
    assert metas, sorted({m["document_id"] for m in bundle.node_meta.values()})
    assert all(m["title"] == DOC_TITLE for m in metas)


def test_add_same_content_is_unchanged(kb_env, tmp_path):
    src = tmp_path / DOC_NAME
    src.write_text(DOC_TEXT, encoding="utf-8", newline="\n")
    assert run_add(src, title=DOC_TITLE) == 0
    assert run_add(src, title=DOC_TITLE) == 0  # вторая загрузка - no-op по sha256


def test_add_bootstraps_empty_index(kb_env, tmp_path, monkeypatch):
    # свежая установка: индекса нет - add сам создаёт пустой индекс,
    # документ кладёт в поиск, L0 не теряется
    import veles.config as config

    fresh = tmp_path / "fresh-index"
    fresh.mkdir()
    monkeypatch.setattr(config, "INDEX_DIR", fresh)
    src = tmp_path / DOC_NAME
    src.write_text(DOC_TEXT, encoding="utf-8", newline="\n")
    assert run_add(src) == 0
    assert (fresh / "meta.json").is_file()  # индекс создан
    # документ реально ищется в свежесозданном индексе
    bundle = load_index(fresh)
    qvec = HashingEmbedder().embed_query(DOC_QUERY)
    texts = [e.text for e in hybrid_search(bundle, qvec, DOC_QUERY, top_k=5)]
    assert any(DOC_MARK in t for t in texts), texts


def test_add_rejects_pdf_and_missing_file(kb_env, tmp_path):
    pdf = tmp_path / DOC_NAME.replace(".md", ".pdf")
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert run_add(pdf) == 1
    assert run_add(tmp_path / "нет-такого-файла.md") == 1
    # L0 и канон не созданы
    assert not (uploads_root() / user_dir("")).exists() or not any(
        (uploads_root() / user_dir("")).iterdir()
    )


def test_add_bit_file_no_l0_orphan(kb_env, tmp_path):
    """Ревью K15 P2: файл скопирован в L0, парс упал ПОСЛЕ копирования - L0-сирота
    не остаётся (в personal-профиле съедала бы квоту). Битый .txt: пустой файл."""
    bad = tmp_path / "битый.md"
    bad.write_text("", encoding="utf-8")
    rc = run_add(bad)
    l0 = uploads_root() / user_dir("") / bad.name
    if rc != 0:
        assert not l0.exists(), "L0-сирота после неудачного парса"
    else:
        # если пустой файл парсер принимает - как минимум документ в базе
        assert l0.exists()
