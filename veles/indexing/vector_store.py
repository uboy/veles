"""VectorStore: один интерфейс, бэкенды numpy (zero-dep) и faiss (IndexFlatIP)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import EMBED_DIM


class VectorStore(Protocol):
    name: str

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]: ...
    def save(self, dir_path: Path) -> None: ...


class NumpyFlatStore:
    """Cosine через нормализованные векторы; честный brute-force (H5-базлайн)."""

    name = "numpy-flat"

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.ids: list[str] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim
        self.ids.extend(ids)
        self.matrix = np.vstack([self.matrix, vectors.astype(np.float32)])

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        q /= np.linalg.norm(q) + 1e-12
        scores = self.matrix @ q.reshape(-1)
        idx = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        np.save(dir_path / "numpy_matrix.npy", self.matrix)
        (dir_path / "numpy_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    def close(self) -> None:
        """Закрыть mmap-карту матрицы (после close() хранилище нечитаемо).

        Windows не даёт перезаписать файл, удерживаемый активной картой:
        пересборка индекса в том же процессе, где load_index отдал mmap
        (CLI-скрипты, тесты), обязана освободить карту до записи
        numpy_matrix.npy. Фасаду не нужно: он матрицу не пишет, очередь
        пишет только overlay (см. бриф 2026-10-14-index-binary-format)."""
        m = getattr(self, "matrix", None)
        if isinstance(m, np.memmap):
            mmap_obj = getattr(m, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()


class FaissFlatStore:
    """Порт донора: faiss IndexFlatIP по L2-нормализованным векторам."""

    name = "faiss-flat"

    def __init__(self, dim: int = EMBED_DIM):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        assert vectors.shape[1] == self.dim
        self.ids.extend(ids)
        self.index.add(np.ascontiguousarray(vectors.astype(np.float32)))

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))
        scores, idx = self.index.search(q, min(top_k, len(self.ids)))
        return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def save(self, dir_path: Path) -> None:
        import faiss

        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        (dir_path / "faiss_ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )


def make_store(prefer: str = "faiss", dim: int = EMBED_DIM) -> VectorStore:
    """Фабрика по имени; prefer='faiss' падает в numpy, если faiss недоступен."""
    if prefer == "faiss":
        try:
            import faiss  # noqa: F401

            return FaissFlatStore(dim)
        except ImportError:
            return NumpyFlatStore(dim)
    return NumpyFlatStore(dim)
