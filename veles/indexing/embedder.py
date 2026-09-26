"""EmbeddingProvider: sentence-transformers (стек донора), e5-префиксы, нормализация."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformersProvider:
    """e5-семейство требует префиксов query:/passage:. Нормализация = cosine по IP."""

    def __init__(self, model_name: str, dim: int, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = dim

    def _enc(self, texts: list[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc(list(texts), "passage: ")

    def embed_query(self, text: str) -> np.ndarray:
        return self._enc([text], "query: ")[0]
