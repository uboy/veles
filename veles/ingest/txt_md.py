"""Парсер-адаптер TXT/Markdown -> CanonicalDocument.

Структура TXT (транскрипты/пересказы): абзацы по пустым строкам; длинные абзацы
режутся по границам предложений (~1200 символов). heading_path собирается из
относительного пути корпуса: [курс, лекция, имя файла]. Provenance - строки файла.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..canonical.model import CanonicalDocument, Provenance

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_MAX_PARAGRAPH = 700


def _split_long(text: str) -> list[str]:
    if len(text) <= _MAX_PARAGRAPH:
        return [text]
    parts, buf = [], ""
    for sent in _SENTENCE_SPLIT.split(text):
        if buf and len(buf) + len(sent) + 1 > _MAX_PARAGRAPH:
            parts.append(buf)
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf:
        parts.append(buf)
    return parts


def _parse_markdown(text: str) -> list[tuple[list[str], int, int, str]]:
    """(heading_path, line_start, line_end, text) по заголовкам #..######."""
    units: list[tuple[list[str], int, int, str]] = []
    path: list[str] = []
    buf: list[str] = []
    start = 1
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            if buf:
                units.append((list(path), start, lineno - 1, "\n".join(buf).strip()))
                buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2).strip()]
            start = lineno + 1
        else:
            buf.append(line)
    if buf:
        units.append((list(path), start, len(text.splitlines()), "\n".join(buf).strip()))
    return units


def _parse_plain(text: str) -> list[tuple[list[str], int, int, str]]:
    """Абзацы по пустым строкам с номерами строк."""
    units = []
    lines = text.splitlines()
    buf: list[str] = []
    start = 1
    for lineno, line in enumerate(lines, start=1):
        if line.strip():
            if not buf:
                start = lineno
            buf.append(line)
        elif buf:
            units.append(([], start, lineno - 1, "\n".join(buf).strip()))
            buf = []
    if buf:
        units.append(([], start, len(lines), "\n".join(buf).strip()))
    return units


class TxtMdParserAdapter:
    parser_profile = "txt-md:0.1"

    def parse_file(self, path: Path, corpus_root: Path) -> CanonicalDocument:
        rel = path.relative_to(corpus_root).as_posix()
        raw = path.read_text(encoding="utf-8", errors="replace")
        # BOM/нестандартные пробелы нормализуем, но sha256 считаем от оригинала
        text = raw.replace("\ufeff", "").replace("\r\n", "\n")

        document_id = rel  # относительный путь - стабильная идентичность среза Ф1
        course = path.relative_to(corpus_root).parts[3:5]  # <курс>/<дисциплина>
        lecture = path.stem
        heading_base = [p for p in path.relative_to(corpus_root).parts[:-1]]

        doc = CanonicalDocument(
            document_id=document_id,
            title=f"{path.stem}",
            document_type="transcript" if "транскрипц" in path.name.lower() else "summary",
            sha256=CanonicalDocument.compute_sha256(raw.encode("utf-8")),
            parser_profile=self.parser_profile,
            derived_from=[f"onedev-vyshka:{rel}"],
        )
        doc.add_node(
            logical_node_id=f"{document_id}#root",
            node_type="root",
            heading_path=list(heading_base),
            text=f"{path.stem}",
            provenance=Provenance(adapter=self.parser_profile, source_id=rel),
        )

        if path.suffix.lower() == ".md":
            units = _parse_markdown(text)
        else:
            units = _parse_plain(text)

        idx = 0
        for hpath, lstart, lend, utext in units:
            for chunk in _split_long(utext):
                if not chunk.strip():
                    continue
                idx += 1
                full_path = heading_base + hpath
                doc.add_node(
                    logical_node_id=f"{document_id}#p{idx:04d}",
                    node_type="paragraph",
                    parent_id=f"{document_id}#root",
                    heading_path=full_path,
                    text=chunk.strip(),
                    provenance=Provenance(
                        adapter=self.parser_profile, source_id=rel, lines=[lstart, lend]
                    ),
                )
        return doc
