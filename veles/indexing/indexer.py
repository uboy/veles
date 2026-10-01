"""Сборка L2 из канона: эмбеддинги листовых узлов + BM25; сохранение бандла."""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..canonical.store import iter_documents
from .bm25 import Bm25Index
from .embedder import EmbeddingProvider
from .vector_store import FaissFlatStore, NumpyFlatStore, VectorStore


# --- K18: детектор «утратил силу» по тексту узла ---
# Формы: «утратил/утратила/утратили/утратило силу», причастная официальная
# «признан утратившим силу», «не подлежит применению» (регистр любой).
# Lookbehind «не » отсекает отрицание («не утратило силу и действует») -
# нормализация выше гарантирует одиночные пробелы; на «не подлежит
# применению» не распространяется (там «не» - часть самой фразы утраты).
# Это метка СОДЕРЖИМОГО узла, не документа: официальный
# статус акта на уровне документа (pravo.gov.ru) - будущая фаза. Пустышки-
# заглушки «Утратил силу.» тоже метчатся, но в выдачу их режет K12-фильтр
# (is_junk_node) - механизмы не связаны.
_LOST_FORCE_RES = (
    re.compile(r"(?<!не )(?:утратил[аои]?\s+силу|утративш[а-я]+\s+силу)"),
    re.compile(r"не\s+подлежит\s+применению"),
)


def is_lost_force(text: str) -> bool:
    """True, если текст узла содержит признак утраты силы нормы."""
    t = " ".join((text or "").split()).lower().replace("ё", "е")
    return any(rx.search(t) for rx in _LOST_FORCE_RES)


# K21: узел меты мутируется воркером очереди (upsert расширяет dict) и
# итерируется читателями (stats, _resolve_revision) - без лока возможен редкий
# RuntimeError: dictionary changed size during iteration. Ревизия P2 ревью 01.10.
INDEX_LOCK = threading.RLock()


@dataclass
class IndexBundle:
    vector: VectorStore
    bm25: Bm25Index
    node_meta: dict[str, dict] = field(default_factory=dict)
    identifiers: set[str] = field(default_factory=set)

    def save(self, dir_path: Path, embed_meta: dict) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        self.vector.save(dir_path)
        self.bm25.save(dir_path)
        (dir_path / "meta.json").write_text(
            json.dumps(
                {
                    "node_meta": self.node_meta,
                    "embed": embed_meta,
                    "identifiers": sorted(self.identifiers),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        # полный снапшот снят с канона, куда уже легли все загрузки K4, -
        # накопленный оверлей устарел и должен не дублировать узлы
        from .overlay import overlay_path

        opath = overlay_path(dir_path)
        if opath.exists():
            opath.unlink()


def build_index(
    embedder: EmbeddingProvider, store: VectorStore, batch: int = 256
) -> IndexBundle:
    from ..policy.identifiers import extract_identifiers

    bm25 = Bm25Index()
    bundle = IndexBundle(vector=store, bm25=bm25)
    bundle.identifiers = set()
    texts_batch: list[tuple[str, str]] = []  # (node_version_id, text)

    for doc in iter_documents(include_revisions=True):
        bundle.identifiers |= set(extract_identifiers(doc.title))
        bundle.identifiers |= {
            i for n in doc.nodes if n.node_type != "root" for i in extract_identifiers(n.text)
        }
        for node in doc.nodes:
            if node.node_type == "root":
                continue
            # Детерминированный префикс пути заголовков (инвариант 7 / H4)
            prefix = " > ".join(node.heading_path)
            search_text = f"[{prefix}] {node.text}" if prefix else node.text
            texts_batch.append((node.node_version_id, search_text))
            bundle.node_meta[node.node_version_id] = {
                "document_id": doc.document_id,
                "title": doc.title,
                "logical_node_id": node.logical_node_id,
                "heading_path": node.heading_path,
                "text": node.text,
                "provenance": node.provenance.model_dump(),
                "access_groups": node.access_groups,
                "sha256_doc": doc.sha256,
                "asset": node.asset,
                # ревизии (K3): у меты текущих ревизий status=current, у архивных
                # (revision-r*/ в каноне) - superseded; фильтр - в hybrid_search
                "revision_id": doc.revision_id,
                "status": doc.status,
                # K18: метка «утратил силу» по тексту узла; в выдаче не фильтруется
                # и ранжирование не меняет - только флаг в цитате (плашка в рендере)
                "lost_force": is_lost_force(node.text),
            }
        if len(texts_batch) >= batch:
            _flush(bundle, embedder, texts_batch)
            # прогресс для длинных сборок (union-индекс ~126k узлов идёт часами)
            print(f"[index] {len(bundle.node_meta)} узлов", file=sys.stderr, flush=True)
            texts_batch = []
    if texts_batch:
        _flush(bundle, embedder, texts_batch)
    return bundle


def _flush(bundle: IndexBundle, embedder: EmbeddingProvider, batch: list[tuple[str, str]]) -> None:
    ids = [i for i, _ in batch]
    texts = [t for _, t in batch]
    vecs = embedder.embed_passages(texts)
    bundle.vector.add(ids, vecs)
    bundle.bm25.add(ids, texts)


def load_index(dir_path: Path, store: VectorStore | None = None) -> "IndexBundle":
    """Загрузка бандла. Мета читается первой: валидация модели, выбор бэкенда по мете.

    Несовпадение модели эмбеддингов = ValueError (тихий мусорный retrieval недопустим).
    """
    import numpy as np

    from ..config import EMBED_MODEL

    meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
    embed_meta = meta.get("embed", {})
    if embed_meta.get("model") != EMBED_MODEL:
        raise ValueError(
            f"Индекс собран с моделью '{embed_meta.get('model')}', текущая '{EMBED_MODEL}'. "
            "Пересоберите: python -m veles index"
        )
    if store is None:
        prefer = "faiss" if "faiss" in embed_meta.get("backend", "") else "numpy"
        if prefer == "numpy" or not (dir_path / "faiss.index").exists():
            store = NumpyFlatStore()
        else:
            store = FaissFlatStore()
    bm25_payload = json.loads((dir_path / "bm25.json").read_text(encoding="utf-8"))
    bm25 = Bm25Index(k1=bm25_payload["k1"], b=bm25_payload["b"])
    bm25.ids = bm25_payload["ids"]
    bm25.doc_lens = bm25_payload["doc_lens"]
    bm25.tfs = bm25_payload["tfs"]
    bm25.df = bm25_payload["df"]
    bm25.avgdl = bm25_payload["avgdl"]

    if (dir_path / "numpy_matrix.npy").exists():
        store.matrix = np.load(dir_path / "numpy_matrix.npy")
        store.ids = json.loads((dir_path / "numpy_ids.json").read_text(encoding="utf-8"))
    else:
        import faiss

        store.index = faiss.read_index(str(dir_path / "faiss.index"))
        store.ids = json.loads((dir_path / "faiss_ids.json").read_text(encoding="utf-8"))
    if "identifiers" not in meta:
        raise ValueError(
            "В meta.json нет identifiers - индекс собран старой версией veles. Пересоберите: python -m veles index"
        )
    bundle = IndexBundle(
        vector=store,
        bm25=bm25,
        node_meta=meta["node_meta"],
        identifiers=set(meta["identifiers"]),
    )
    # K4: дельта инкрементальных загрузок поверх снапшота - документ, загруженный
    # пользователем до рестарта, ищется и после него (без пересборки)
    from .overlay import apply_overlay

    apply_overlay(bundle, dir_path)
    return bundle
