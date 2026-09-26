"""Индексация L2: эмбеддер, векторные сторы, BM25, сборщик."""

from __future__ import annotations

from .bm25 import Bm25Index, tokenize
from .embedder import EmbeddingProvider, SentenceTransformersProvider
from .indexer import IndexBundle, build_index, load_index
from .vector_store import FaissFlatStore, NumpyFlatStore, VectorStore

__all__ = [
    "Bm25Index",
    "tokenize",
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "IndexBundle",
    "build_index",
    "load_index",
    "VectorStore",
    "NumpyFlatStore",
    "FaissFlatStore",
]
