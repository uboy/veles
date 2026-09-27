"""Идентификаторы нормативных документов (ГОСТ/СП/СНиП/МДС/РД) — детерминированный
сигнал AnswerPolicy: вопрос цитирует номер документа; если этот документ отсутствует
в корпусе, honest-поведение - abstain (Ф2: FAR 0.9 -> 0.25)."""

from __future__ import annotations

import re

_IDENT_RE = re.compile(
    r"(?<![А-ЯЁA-Z])(ГОСТ\s*Р?\s*[\d.]+(?:-\d{2,4})?|СП\s+[\d.]+|СНиП\s*[\d.]+(?:-\d{2,4})?|"
    r"МДС\s+[\d.-]+|РД\s+[\d.-]+)",
    re.IGNORECASE,
)


def _core(identifier: str) -> str:
    """Числовое ядро номера без префикса и 'Р' ('ГОСТ Р 17.2.3.01-86' -> '17.2.3.01-86')."""
    tail = re.split(r"[А-ЯЁA-Z\s]+", identifier.strip(), maxsplit=1)[-1]
    return tail.strip(" .-")


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
    known_cores = {_core(k) for k in known_identifiers}
    for i in query_identifiers:
        core = _core(i)
        if any(core in k or k in core for k in known_cores if k):
            return False
    return True
