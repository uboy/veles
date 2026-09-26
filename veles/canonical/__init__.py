"""Канонический слой: модель документа и файловое хранилище (manifest.json + nodes.jsonl)."""

from __future__ import annotations

from .model import CanonicalDocument, CanonicalNode, Provenance
from .store import iter_documents, load_document, save_document

__all__ = [
    "CanonicalDocument",
    "CanonicalNode",
    "Provenance",
    "load_document",
    "save_document",
    "iter_documents",
]
