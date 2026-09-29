"""Гибридный retrieval: dense + BM25 -> RRF -> evidence pack с цитатами.

Калибровка K12: RRF собирается с весами каналов (channel_weights), после чего
к скору узлов документов, явно упомянутых в вопросе («согласно НК РФ»),
добавляется codex_boost; мусорные узлы («Утратил силу.»-пустышки, телефоны
консультаций) отсекаются из выдачи ПОСЛЕ ранжирования - индекс не меняется.
Все параметры опциональны: None = дефолт из veles.config (поведение прода)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import RRF_K
from ..indexing.indexer import IndexBundle


@dataclass
class Citation:
    title: str
    document_id: str
    heading_path: list[str]
    source_id: str
    lines: list[int] | None
    node_version_id: str
    revision_id: str = "r1"  # ревизия документа (K3); у канона до 29.09 - r1

    def human(self) -> str:
        where = f", строки {self.lines[0]}-{self.lines[1]}" if self.lines else ""
        rev = f", ред. {self.revision_id}" if self.revision_id not in ("", "r1") else ""
        return f"«{self.title}» {self.source_id}{rev}{where}"


@dataclass
class Evidence:
    node_version_id: str
    text: str
    rrf: float
    # ревью K12 P1: скор ДО codex-буста (органика); confidence и дельта конфликтов
    # считаются по нему - бонус равен потолку нормировки и иначе обнулял бы
    # abstain-гейт для вопросов с упоминанием кодекса
    rrf_base: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    citations: list[Citation] = field(default_factory=list)


# --- K12: мусорные узлы выдачи (из индекса НЕ удаляются, только из контекста) ---
# Пустышки парсинга консолидированных актов: строки-заглушки «Утратил силу.»,
# блок телефонов юрконсультаций. Контентные узлы («Часть вторая утратила силу
# с 2 июля 2021 г. - ФЗ...») не матчатся: шаблон обязан покрыть текст ЦЕЛИКОМ.
_JUNK_RES = (
    re.compile(r"^\d*[.\s]*утратил силу\.?$"),
    re.compile(r"^\d*[.\s]*не применяется\.?$"),
    re.compile(r"^бесплатная юридическая консультация"),
    re.compile(r"^8 \(\d{3}\) \d{3}-\d{2}-\d{2}"),
)


def is_junk_node(text: str) -> bool:
    """True для пустышки-узла: короткий текст, целиком совпавший с шаблоном мусора."""
    t = " ".join((text or "").split())
    if not t or len(t) > 200:
        return False
    low = t.lower().replace("ё", "е")
    return any(rx.search(low) for rx in _JUNK_RES)


def dedupe_pairs(pairs: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Снятие дубликатов узла внутри канала (первое вхождение = лучший скор).

    Прод-индекс 29.09: 241 430 записей stores на 158 394 узлов - 83 036 узлов
    записаны в vector/bm25 дважды (артефакт snapshot+overlay загрузки). Без
    снятия дубль узла получает ДВА вклада RRF одного канала и вытесняет узлы
    другого канала (кейс k11-056: BM25-дубли шапки НК РФ/ГК РФ съели top-5,
    хотя dense ставил ответные узлы гл. 28 на ранги 1-6)."""
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for nid, score in pairs:
        if nid in seen:
            continue
        seen.add(nid)
        out.append((nid, score))
    return out


def hybrid_search(
    bundle: IndexBundle,
    query_vec,
    query_text: str,
    top_k: int = 8,
    allowed_groups: set[str] | None = None,
    revision: str | None = None,
    rrf_k: int | None = None,
    channel_weights: tuple[float, float] | None = None,
    candidate_depth: int | None = None,
    codex_boost: float | None = None,
    filter_junk: bool | None = None,
    dedupe_channels: bool | None = None,
) -> list[Evidence]:
    """Pre-retrieval authorization (инвариант 5) + ревизионный фильтр (K3) - один шов:
    узлы без пересечения групп с allowed_groups отсекаются до формирования evidence
    и не попадают в контекст; revision=None (дефолт) - только status=current,
    revision="all" - без ревизионного фильтра, revision="r<n>" - только эта ревизия.
    Мета без полей ревизий (индексы до 29.09) считается current/r1.

    Калибровка K12 (None = дефолт из veles.config, поведение до калибровки):
    rrf_k - константа RRF; channel_weights - веса вкладов (dense, lexical);
    candidate_depth - глубина каналов до RRF (дефолт top_k*5); codex_boost -
    бонус узлам документа, явно упомянутого в вопросе («согласно НК РФ»),
    в единицах вклада ранга 1 одного канала (bonus = codex_boost/(rrf_k+1));
    filter_junk - отсекать мусорные пустышки из выдачи (is_junk_node);
    dedupe_channels - снимать дубликаты узла внутри канала (см. dedupe_pairs;
    с K14 дефолт True: на чистом индексе no-op, защищает от возврата дублей)."""
    from ..config import DEFAULT_ACCESS_GROUPS

    if rrf_k is None:
        rrf_k = RRF_K
    weights = channel_weights if channel_weights is not None else (1.0, 1.0)
    depth = candidate_depth if candidate_depth is not None else top_k * 5
    if codex_boost is None:
        from ..config import CODEX_BOOST

        codex_boost = CODEX_BOOST
    if filter_junk is None:
        from ..config import FILTER_JUNK_NODES

        filter_junk = FILTER_JUNK_NODES
    if dedupe_channels is None:
        from ..config import DEDUPE_CHANNELS

        dedupe_channels = DEDUPE_CHANNELS

    groups = allowed_groups or DEFAULT_ACCESS_GROUPS

    def authorized(nid: str) -> bool:
        meta = bundle.node_meta.get(nid)
        if not meta:
            return False
        # K10: tombstone удаления - безусловный фильтр ДО групп и ревизий:
        # удалённый документ не ищется никому (в т.ч. админу с "*") и ни с каким
        # revision-параметром (all/r<n> включительно)
        if meta.get("status") == "deleted":
            return False
        # K4: "*" в группах запросчика (админ из JwtHeaderResolver) - групповой
        # фильтр обходится, ревизионный остаётся; ревизионный конъюнкт идёт после
        # группового, ACL не ослаблен
        if "*" not in groups and not (set(meta.get("access_groups", [])) & groups):
            return False
        if revision == "all":
            return True
        if revision is None or revision == "current":
            return meta.get("status", "current") == "current"
        return meta.get("revision_id", "r1") == revision

    dense = [
        (nid, s) for nid, s in bundle.vector.search(query_vec, top_k=depth) if authorized(nid)
    ]
    lex = [
        (nid, s) for nid, s in bundle.bm25.search(query_text, top_k=depth) if authorized(nid)
    ]
    if dedupe_channels:
        dense = dedupe_pairs(dense)
        lex = dedupe_pairs(lex)

    rrf: dict[str, float] = {}
    ranks: dict[str, dict] = {}
    for rank, (nid, _) in enumerate(dense, start=1):
        rrf[nid] = rrf.get(nid, 0.0) + weights[0] / (rrf_k + rank)
        ranks.setdefault(nid, {})["dense_rank"] = rank
    for rank, (nid, _) in enumerate(lex, start=1):
        rrf[nid] = rrf.get(nid, 0.0) + weights[1] / (rrf_k + rank)
        ranks.setdefault(nid, {})["bm25_rank"] = rank

    # K12: identifier-буст - вопрос называет акт («согласно НК РФ», «по ГК РФ»):
    # узлы этого документа получают бонус поверх RRF (только кандидаты каналов,
    # authorization уже применён; документ матчится подстрокой названия).
    rrf_base = dict(rrf)  # ревью K12 P1: органические скоры до буста - на них
    if codex_boost > 0.0:  # считаются confidence и дельта конфликтов, иначе
        from ..policy.identifiers import _CODEX_MAP, extract_codex_refs  # буст
        # (= потолку нормировки) обнулял abstain-гейт «согласно НК РФ»

        refs = extract_codex_refs(query_text)
        if refs:
            bonus = codex_boost / (rrf_k + 1.0)
            needles = [_CODEX_MAP[r] for r in refs]
            for nid in rrf:
                meta = bundle.node_meta.get(nid)
                if not meta:
                    continue
                title = " ".join(str(meta.get("title", "")).lower().replace("ё", "е").split())
                if any(n in title for n in needles):
                    rrf[nid] += bonus

    ordered = sorted(rrf.items(), key=lambda x: -x[1])
    evidence: list[Evidence] = []
    for nid, score in ordered:
        if len(evidence) >= top_k:
            break
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        # K12: пустышки («Утратил силу.», телефоны) не занимают слоты выдачи
        if filter_junk and is_junk_node(meta.get("text", "")):
            continue
        prov = meta["provenance"]
        cite = Citation(
            title=meta["title"],
            document_id=meta["document_id"],
            heading_path=meta["heading_path"],
            source_id=prov["source_id"],
            lines=prov.get("lines"),
            node_version_id=nid,
            revision_id=meta.get("revision_id", "r1"),
        )
        evidence.append(
            Evidence(
                node_version_id=nid,
                text=meta["text"],
                rrf=score,
                rrf_base=rrf_base.get(nid, score),
                dense_rank=ranks.get(nid, {}).get("dense_rank"),
                bm25_rank=ranks.get(nid, {}).get("bm25_rank"),
                citations=[cite],
            )
        )
    return evidence
