"""RRNCB-бенчмарк Ф2: flat baseline vs canonical, Recall@K/MRR, FAR-калибровка.

Датасет: FractalGPT/RRNCBPublic (200 QA: question/answer/document) + PDF-документы
(Google Drive RRNCB). Документы в knowledge-sources/06_benchmarks/rrncb/docs.

Ограничения честности (по ревью Ф1):
- gold-узел = локализация answer-строки в юните (извлекаемая выборка, покрытие
  считается и сообщается; метрики - на общем локализуемом подмножестве обеих конфигов);
- метрики retrieval (H2/H4 - гранулярность чанков и heading-префикс); эффект
  small-to-big расширения (H3) на полноту ответа требует LLM-суждения и в Ф2 не
  измеряется - фиксированный прогон expansion не влияет на ранжирование листьев.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from veles.canonical.model import CanonicalDocument
from veles.ingest.pdf_text import PyMuPdfParserAdapter

RRNCB_DIR = Path(r"<local-pc>\proj\knowledge-sources\06_benchmarks\rrncb")
QA_CSV = RRNCB_DIR / "rrncb_public_dataset.csv"
DOCS_DIR = RRNCB_DIR / "docs"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def load_qa() -> list[dict]:
    rows = list(csv.DictReader(io.open(QA_CSV, encoding="utf-8")))
    return [r for r in rows if r["answer"].strip() and r["answer"].strip() != "-"]


@dataclass
class Unit:
    uid: str
    text: str


@dataclass
class ConfigIndex:
    name: str
    units: list[Unit] = field(default_factory=list)
    bundle: Optional[object] = None


def flat_units(doc: CanonicalDocument, size: int = 800, overlap: int = 100) -> list[Unit]:
    """Базлайн: фиксированные окна по полному тексту документа, без иерархии."""
    full = "\n".join(n.text for n in doc.nodes if n.node_type != "root")
    units = []
    step = size - overlap
    for i, start in enumerate(range(0, len(full), step)):
        chunk = full[start : start + size].strip()
        if len(chunk) < 40:
            break
        units.append(Unit(uid=f"flat:{doc.document_id}:{i:05d}", text=chunk))
    return units


def canonical_units(doc: CanonicalDocument) -> list[Unit]:
    return [
        Unit(uid=n.node_version_id, text=n.text)
        for n in doc.nodes
        if n.node_type == "paragraph"
    ]


def localize_gold(units: list[Unit], answer: str) -> Optional[str]:
    """Узел, содержащий нормализованный префикс ответа. None = не локализован."""
    na = normalize(answer)
    if len(na) < 25:
        return None
    probe = na[:80]
    for u in units:
        if probe in normalize(u.text):
            return u.uid
    # второй проход: первые 40 знаков
    probe = na[:40]
    for u in units:
        if probe in normalize(u.text):
            return u.uid
    return None
