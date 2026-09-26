"""Идентификаторы нормативных документов (ГОСТ/СП/СНиП/МДС/РД) — детерминированный
сигнал AnswerPolicy: вопрос цитирует номер документа; если этот документ отсутствует
в корпусе, honest-поведение - abstain (Ф2: FAR 0.9 -> 0.25)."""

from __future__ import annotations

import re

_IDENT_RE = re.compile(
    r"(ГОСТ\s*Р?\s*[\d.]+(?:-\d{2,4})?|СП\s+[\d.]+|СНиП\s*[\d.]+(?:-\d{2,4})?|"
    r"МДС\s+[\d.-]+|РД\s+[\d.-]+)",
    re.IGNORECASE,
)


def extract_identifiers(text: str) -> list[str]:
    """Номера нормативных документов в тексте ('ГОСТ 17.2.3.01-86', 'МДС 81-28.2001')."""
    out = []
    for m in _IDENT_RE.finditer(text.upper().replace("Ё", "Е")):
        out.append(re.sub(r"\s+", " ", m.group(1)))
    return out


def identifiers_missed(
    query_identifiers: list[str], known_identifiers: set[str] | None
) -> bool:
    """True, если в вопросе есть номера и НИ ОДИН не найден в корпусе.

    Семантика 'any-found' (калибровка Ф3): вопрос может ссылаться на несколько
    документов - достаточно присутствия хотя бы одного (substring-сверка нормализованных
    строк; 'ГОСТ Р 17.2.3.01' матчит 'ГОСТ 17.2.3.01')."""
    if not query_identifiers:
        return False
    if not known_identifiers:
        return True
    return not any(
        any(i in doc_id for doc_id in known_identifiers) for i in query_identifiers
    )
