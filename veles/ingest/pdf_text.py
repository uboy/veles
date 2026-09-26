"""Парсер-адаптер PDF (PyMuPDF) -> CanonicalDocument.

Нормативные документы: заголовки по паттернам (Глава/Статья/Раздел/Приложение,
нумерованные пункты), абзацы из текстовых блоков страницы; provenance - страница.
Для бенчмарка Ф2: честно-простая эвристика, Docling-адаптер - фаза 3.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..canonical.model import CanonicalDocument, Provenance

_MAX_NODE = 700
_HEADING_RE = re.compile(
    r"^\s*((Глава|Раздел|Статья|Приложение|Приложения)\s*[№№]?\s*[\dА-ЯA-Z.]+[.{}]?\s+\S|"
    r"\d{1,2}(\.\d{1,3})*\.?\s+\S)",
    re.IGNORECASE,
)
_HEADING_MAX_LEN = 120


class PyMuPdfParserAdapter:
    parser_profile = "pymupdf-heuristic:0.1"

    def parse_file(self, path: Path, corpus_root: Path) -> CanonicalDocument:
        import fitz

        rel = path.relative_to(corpus_root).as_posix()
        raw_bytes = path.read_bytes()
        doc_pdf = fitz.open(stream=raw_bytes, filetype="pdf")

        document = CanonicalDocument(
            document_id=rel,
            title=path.stem,
            document_type="normative",
            sha256=CanonicalDocument.compute_sha256(raw_bytes),
            parser_profile=self.parser_profile,
        )
        document.add_node(
            logical_node_id=f"{rel}#root",
            node_type="root",
            heading_path=[],
            text=path.stem,
            provenance=Provenance(adapter=self.parser_profile, source_id=rel),
        )

        heading_path: list[str] = []
        idx = 0
        for pno in range(doc_pdf.page_count):
            page = doc_pdf[pno]
            for block in page.get_text("blocks"):
                text = (block[3] and " ".join(str(block[4]).split())) or ""
                if len(text) < 25:
                    # короткие строки: кандидат в заголовки
                    line = text.strip()
                    if _HEADING_RE.match(line) and len(line) < _HEADING_MAX_LEN:
                        heading_path = (heading_path + [line])[-4:]
                    continue
                # нумерованные заголовки-абзацы вплотную к тексту: отрезаем как заголовок
                m = _HEADING_RE.match(text)
                if m and len(m.group(0)) < 100 and len(text) < _HEADING_MAX_LEN:
                    heading_path = (heading_path + [text.strip()])[-4:]
                    continue
                for chunk in self._split_long(text):
                    idx += 1
                    document.add_node(
                        logical_node_id=f"{rel}#p{idx:05d}",
                        node_type="paragraph",
                        parent_id=f"{rel}#root",
                        heading_path=list(heading_path),
                        text=chunk,
                        provenance=Provenance(
                            adapter=self.parser_profile, source_id=rel, page=pno + 1
                        ),
                    )
        doc_pdf.close()
        return document

    @staticmethod
    def _split_long(text: str) -> list[str]:
        if len(text) <= _MAX_NODE:
            return [text]
        parts, buf = [], ""
        for sent in re.split(r"(?<=[.!?…])\s+", text):
            if buf and len(buf) + len(sent) + 1 > _MAX_NODE:
                parts.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            parts.append(buf)
        return parts
