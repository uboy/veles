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


# Кодексы: аббревиатура вопроса -> ожидаемое ядро названия документа в корпусе.
# Ф3: identifier-сигнал (ГОСТ-регэксп) не покрыл кодексные ловушки (FAR 0.8868 на
# НК/ГК/ТК ЕАЭС) - этот словарь закрывает дыру.
_CODEX_MAP = {
    "нк рф": "налоговый кодекс",
    "гк рф": "гражданский кодекс",
    "тк еаэс": "таможенный кодекс",
    "тк рф": "трудовой кодекс",
    "ук рф": "уголовный кодекс",
    "жк рф": "жилищный кодекс",
    "бк рф": "бюджетный кодекс",
    "зк рф": "земельный кодекс",
    "ск рф": "семейный кодекс",
    "гпк рф": "гражданский процессуальный кодекс",
    "апк рф": "арбитражный процессуальный кодекс",
    "упк рф": "уголовно-процессуальный кодекс",
    "кас рф": "кодекс административного судопроизводства",
    "ктм рф": "кодекс торгового мореплавания",
    "коап рф": "кодекс об административных правонарушениях",
    "вк рф": "воздушный кодекс",
    "лк рф": "лесной кодекс",
    "кввт рф": "кодекс внутреннего водного транспорта",
}


def extract_codex_refs(text: str) -> list[str]:
    """Ссылки на кодексы в тексте ('согласно НК РФ') -> ключи _CODEX_MAP."""
    t = re.sub(r"\s+", " ", text.lower().replace("ё", "е"))
    return [k for k in _CODEX_MAP if re.search(rf"(?<![а-я]){re.escape(k)}(?![а-я])", t)]


def codex_missed(codex_refs: list[str], doc_titles: Optional[set[str]]) -> bool:
    """True, если ВСЕ упомянутые кодексы отсутствуют в корпусе (any-found:
    достаточно одного присутствующего). doc_titles=None - проверка отключена."""
    if not codex_refs or doc_titles is None:
        return False
    titles = {re.sub(r"\s+", " ", t.lower().replace("ё", "е")) for t in doc_titles}
    present = sum(
        1 for ref in codex_refs if any(_CODEX_MAP[ref] in t for t in titles)
    )
    return present == 0
