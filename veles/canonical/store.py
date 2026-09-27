"""Физическое хранение канона: canonical/<doc_hash>/manifest.json + nodes.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..config import CANONICAL_DIR
from .model import CanonicalDocument, CanonicalNode


def _doc_dir(document_id: str, root: Path | None = None) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in document_id)
    return (root or CANONICAL_DIR) / safe[:120]


def save_document(doc: CanonicalDocument, root: Path | None = None) -> Path:
    d = _doc_dir(doc.document_id, root)
    d.mkdir(parents=True, exist_ok=True)
    # nodes.jsonl ПЕРВЫМ, manifest последним (ревью-2: убитый процесс не оставит
    # «готовый» документ без узлов)
    with (d / "nodes.jsonl.tmp").open("w", encoding="utf-8") as f:
        for node in doc.nodes:
            f.write(json.dumps(node.model_dump(), ensure_ascii=False) + "\n")
    (d / "nodes.jsonl.tmp").replace(d / "nodes.jsonl")
    manifest = doc.model_dump(exclude={"nodes"})
    (d / "manifest.tmp").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (d / "manifest.tmp").replace(d / "manifest.json")
    return d


def load_document(document_id: str, root: Path | None = None) -> CanonicalDocument:
    d = _doc_dir(document_id, root)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    doc = CanonicalDocument(**manifest, nodes=[])
    for line in (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc.nodes.append(CanonicalNode(**json.loads(line)))
    return doc


def iter_documents(root: Path | None = None) -> Iterator[CanonicalDocument]:
    base = root or CANONICAL_DIR
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if (d / "manifest.json").exists():
            yield load_document(d.name, root)
