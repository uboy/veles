"""AnswerPolicy (инвариант 8): answer / abstain / warn_conflict.

Срез Ф1: provisional-порог по нормализованному RRF-доверию + сигнал согласия
каналов. Калибровка многосигнальной политики - фаза 2 (RRNCB + traps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import ABSTAIN_CONFIDENCE, RRF_K
from ..retrieval.retriever import Evidence
from .identifiers import codex_missed, identifiers_missed


@dataclass
class AnswerDecision:
    decision: Literal["answer", "abstain"]
    confidence: float  # нормализованное доверие [0..1]
    reason: str
    evidence: list[Evidence]


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
    return AnswerDecision(
        decision="answer",
        confidence=conf,
        reason="гибридные каналы согласны, порог пройден",
        evidence=evidence,
    )
