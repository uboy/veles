"""AnswerPolicy (инвариант 8): answer / abstain / warn_conflict.

Срез Ф1: provisional-порог по нормализованному RRF-доверию + сигнал согласия
каналов. Калибровка многосигнальной политики - фаза 2 (RRNCB + traps).
warn_conflict (K3) - реализуемый минимум по брифу: топ-2 доказательства из РАЗНЫХ
документов с сопоставимым score и РАЗНЫМИ маркерами редакции/даты в тексте.
Семантическое «противоречие» не детектится; при любых сомнениях - answer без
конфликта (ложные конфликты запрещены, FAR-ловушка AC7 в силе)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from ..config import ABSTAIN_CONFIDENCE, CONFLICT_SCORE_DELTA, RRF_K
from ..retrieval.retriever import Evidence
from .identifiers import codex_missed, identifiers_missed


@dataclass
class AnswerDecision:
    decision: Literal["answer", "abstain", "warn_conflict"]
    confidence: float  # нормализованное доверие [0..1]
    reason: str
    evidence: list[Evidence]
    # кандидаты в конфликт (для warn_conflict): document_id/revision_id/маркер/score
    conflicts: list[dict] = field(default_factory=list)


def _confidence(evidence: list[Evidence]) -> float:
    """top-RRF, нормированный на максимум двух каналов, бонус/штраф ДО капа
    (фикс ревью Ф1: бонус после капа обнулял различимость)."""
    if not evidence:
        return 0.0
    top = evidence[0].rrf
    both_channels = evidence[0].dense_rank is not None and evidence[0].bm25_rank is not None
    ceiling = 2.0 / (RRF_K + 1)
    factor = 1.15 if both_channels else 0.8
    return round(min(1.0, top / ceiling * factor), 4)


# Маркеры редакции/даты в тексте узла: «ред. 2», «редакция от 01.02.2023»,
# «версия 1.1», даты ДД.ММ.ГГГГ. Голые годы и числа без префикса - не маркер
# (слишком шумно); lookbehind отсекает «ред» внутри слов вроде «тред».
_EDITION_NUM_RE = re.compile(
    r"(?<![а-яёa-z])(?:редакци[яию]\s+(?:от\s+)?|ред\.?\s*|версия\s+)(?:№\s*)?[\d.,]+",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")


def edition_signals(text: str) -> set[str]:
    """Значения identifier-сигнала редакции в тексте (пусто - маркеров нет)."""
    signals = set(_DATE_RE.findall(text))
    # [\d.,]+ жадно съедает запятую/точку до следующего слова - чистим края
    signals |= {m.group(0).strip(" .,").lower() for m in _EDITION_NUM_RE.finditer(text)}
    return {s for s in signals if s}


def _conflict_candidates(evidence: list[Evidence]) -> list[dict]:
    """Реализуемый минимум warn_conflict (бриф K3): топ-2 из разных document_id,
    сопоставимый score и не пересекающиеся маркеры редакции/даты. Сопоставимость -
    дельта нормализованного score (RRF / потолок двух каналов, шкала 0..1) меньше
    CONFLICT_SCORE_DELTA: на сыром RRF порог 0.05 невыполним в принципе (максимум
    RRF 2/(K+1) ~ 0.033, любая дельта меньше 0.05), поэтому «0.05» из брифа осмыслен
    только на нормализованной шкале. Один document_id (две ревизии одного документа)
    конфликтом не считается; нет маркера хотя бы у одной стороны - молчание."""
    if len(evidence) < 2:
        return []
    a, b = evidence[0], evidence[1]
    if not a.citations or not b.citations:
        return []
    if a.citations[0].document_id == b.citations[0].document_id:
        return []
    ceiling = 2.0 / (RRF_K + 1)
    if abs(a.rrf - b.rrf) / ceiling >= CONFLICT_SCORE_DELTA:
        return []
    sig_a, sig_b = edition_signals(a.text), edition_signals(b.text)
    if not sig_a or not sig_b or (sig_a & sig_b):
        return []
    out = []
    for ev, sig in ((a, sig_a), (b, sig_b)):
        c = ev.citations[0]
        out.append({
            "document_id": c.document_id,
            "title": c.title,
            "revision_id": c.revision_id,
            "signal": "; ".join(sorted(sig))[:120],
            "rrf": round(ev.rrf, 4),
        })
    return out


def decide(
    evidence: list[Evidence],
    threshold: float = ABSTAIN_CONFIDENCE,
    query_identifiers: list[str] | None = None,
    known_identifiers: set[str] | None = None,
    codex_refs: list[str] | None = None,
    doc_titles: set[str] | None = None,
) -> AnswerDecision:
    """Identifier-grounding проверяется ПЕРВЫМ (детерминированный сигнал):
    вопрос цитирует номер документа или кодекс, которых нет в корпусе -> abstain
    независимо от скоров (Ф2: скор-гейт слеп на ближний домен, FAR=0.9; Ф3:
    ГОСТ-сигнал не покрыл кодексы - добавлен codex-словарь)."""
    ids = query_identifiers or []
    if identifiers_missed(ids, known_identifiers) or codex_missed(codex_refs or [], doc_titles):
        return AnswerDecision(
            decision="abstain",
            confidence=0.0,
            reason="вопрос ссылается на документ(ы), отсутствующие в корпусе",
            evidence=evidence,
        )
    conf = _confidence(evidence)
    if conf < threshold or not evidence:
        return AnswerDecision(
            decision="abstain",
            confidence=conf,
            reason="нет достаточных подтверждений в корпусе",
            evidence=evidence,
        )
    conflicts = _conflict_candidates(evidence)
    if conflicts:
        return AnswerDecision(
            decision="warn_conflict",
            confidence=conf,
            reason="топ-доказательства из разных документов с разными маркерами "
                   "редакции/даты - возможны противоречащие редакции",
            evidence=evidence,
            conflicts=conflicts,
        )
    return AnswerDecision(
        decision="answer",
        confidence=conf,
        reason="гибридные каналы согласны, порог пройден",
        evidence=evidence,
    )
