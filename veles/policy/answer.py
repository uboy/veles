"""AnswerPolicy (инвариант 8): answer / abstain / warn_conflict.

Срез Ф1: provisional-порог по нормализованному RRF-доверию + сигнал согласия
каналов. Калибровка многосигнальной политики - фаза 2 (RRNCB + traps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import ABSTAIN_CONFIDENCE, RRF_K
from ..retrieval.retriever import Evidence


@dataclass
class AnswerDecision:
    decision: Literal["answer", "abstain"]
    confidence: float  # нормализованное доверие [0..1]
    reason: str
    evidence: list[Evidence]


def _confidence(evidence: list[Evidence]) -> float:
    """top-RRF, нормированный на максимум двух каналов: (1/(K+1))*2."""
    if not evidence:
        return 0.0
    top = evidence[0].rrf
    both_channels = evidence[0].dense_rank is not None and evidence[0].bm25_rank is not None
    ceiling = 2.0 / (RRF_K + 1)
    conf = min(1.0, top / ceiling)
    # Согласие каналов в топе - бонус к доверию; один канал - штраф
    if both_channels:
        conf = min(1.0, conf * 1.15)
    else:
        conf *= 0.8
    return round(conf, 4)


def decide(evidence: list[Evidence], threshold: float = ABSTAIN_CONFIDENCE) -> AnswerDecision:
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
