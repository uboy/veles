"""Физическое хранение канона: canonical/<doc_hash>/manifest.json + nodes.jsonl."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

from ..config import CANONICAL_DIR
from .model import CanonicalDocument, CanonicalNode


def _doc_dir(document_id: str, root: Path | None = None) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in document_id)
    if len(safe) <= 120:
        return (root or CANONICAL_DIR) / safe
    # длинные id: хвост-хеш устраняет коллизии обрезки (28.09: пересказ/транскрипция
    # одной лекции схлопывались в один каталог - документ молча терялся)
    suffix = hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]
    return (root or CANONICAL_DIR) / f"{safe[:105]}-{suffix}"


def save_document(
    doc: CanonicalDocument,
    root: Path | None = None,
    by: str | None = None,
    audit_log: Path | None = None,
) -> Path:
    """Сохранить документ + дописать событие загрузки в append-only журнал аудита.

    by - кто загрузил (логин/почта из аутентификации; для старых ручных конвейеров -
    значение VELES_INGESTED_BY или "unknown"). audit_log - путь к журналу (по умолчанию
    <корень канона>/../ingest-audit.jsonl). Журнал только дописывается, события -
    единственный полный след «кто/когда/что» по всем загрузкам."""
    from datetime import datetime, timezone

    if not doc.ingested_at:
        doc.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not doc.ingested_by:
        doc.ingested_by = (
            by or os.environ.get("VELES_INGESTED_BY", "").strip() or "unknown"
        )

    d = _doc_dir(doc.document_id, root)
    d.mkdir(parents=True, exist_ok=True)
    # nodes.jsonl ПЕРВЫМ, manifest последним (ревью-2: убитый процесс не оставит
    # «готовый» документ без узлов)
    with (d / "nodes.jsonl.tmp").open("w", encoding="utf-8") as f:
        for node in doc.nodes:
            f.write(json.dumps(node.model_dump(), ensure_ascii=False) + "\n")
    (d / "nodes.jsonl.tmp").replace(d / "nodes.jsonl")
    manifest = doc.model_dump(exclude={"nodes"})
    if doc.assets:
        import shutil

        adir = d / "assets"
        adir.mkdir(exist_ok=True)
        persisted = {}
        for name, src in doc.assets.items():
            safe_name = Path(name).name
            shutil.copyfile(src, adir / safe_name)
            persisted[name] = f"assets/{safe_name}"
        manifest["assets"] = persisted
    (d / "manifest.tmp").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (d / "manifest.tmp").replace(d / "manifest.json")

    log_path = audit_log or (d.parent / "ingest-audit.jsonl")
    try:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "ingest",
            "document_id": doc.document_id,
            "revision_id": doc.revision_id,
            "sha256": doc.sha256,
            "status": doc.status,
            "by": doc.ingested_by,
            "dir": d.name,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:  # noqa: BLE001 - журнал не должен ломать сохранение канона
        pass
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
