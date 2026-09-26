"""Сборка L2 из канона: эмбеддинги листовых узлов + BM25; сохранение бандла."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..canonical.store import iter_documents
from .bm25 import Bm25Index
from .embedder import EmbeddingProvider
from .vector_store import FaissFlatStore, NumpyFlatStore, VectorStore


@dataclass
class IndexBundle:
    vector: VectorStore
    bm25: Bm25Index
    node_meta: dict[str, dict] = field(default_factory=dict)

    def save(self, dir_path: Path, embed_meta: dict) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        self.vector.save(dir_path)
        self.bm25.save(dir_path)
        (dir_path / "meta.json").write_text(
            json.dumps(
                {"node_meta": self.node_meta, "embed": embed_meta},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def build_index(
    embedder: EmbeddingProvider, store: VectorStore, batch: int = 256
) -> IndexBundle:
    bm25 = Bm25Index()
    bundle = IndexBundle(vector=store, bm25=bm25)
    texts_batch: list[tuple[str, str]] = []  # (node_version_id, text)

    for doc in iter_documents():
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
            }
        if len(texts_batch) >= batch:
            _flush(bundle, embedder, texts_batch)
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
    return IndexBundle(vector=store, bm25=bm25, node_meta=meta["node_meta"])
