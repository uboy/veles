"""Гибридный retrieval: dense + BM25 -> RRF -> evidence pack с цитатами.

Калибровка K12: RRF собирается с весами каналов (channel_weights), после чего
к скору узлов документов, явно упомянутых в вопросе («согласно НК РФ»),
добавляется legal_boost; мусорные узлы («Утратил силу.»-пустышки, телефоны
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
    # K18: метка «утратил силу» по тексту узла (node_meta из индекса); ранжирование
    # не меняет и узел не скрывает - только плашка перед цитатой в рендерах
    lost_force: bool = False
    # Видео: границы сегмента речи в секундах (у не-видео узлов None; poisk-po-video)
    t_start: float | None = None
    t_end: float | None = None

    def human(self) -> str:
        where = f", строки {self.lines[0]}-{self.lines[1]}" if self.lines else ""
        if self.t_start is not None:
            tc = f"{format_timecode(self.t_start)}" + (
                f"-{format_timecode(self.t_end)}" if self.t_end is not None else ""
            )
            where = f"{where}, {tc}" if where else f", {tc}"
        rev = f", ред. {self.revision_id}" if self.revision_id not in ("", "r1") else ""
        return f"«{self.title}» {self.source_id}{rev}{where}"


def format_timecode(seconds: float) -> str:
    """Секунды -> «ЧЧ:ММ:СС» (часы всегда; единый формат цитат и рендеров)."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# K18: единый текст плашки lost_force для рендеров фасада и CLI. В tg_format
# (мост) та же строка живёт своей константой: контейнеры фасада и моста не
# делят код, синхронизацию держит тест tests/test_lost_force.py.
LOST_FORCE_PREFIX = "⚠ Норма утратила силу"


def lost_force_prefix(lost_force: bool) -> str:
    """Префикс-плашка перед цитатой lost_force-узла; для обычных - пусто."""
    return f"{LOST_FORCE_PREFIX} - " if lost_force else ""


@dataclass
class Evidence:
    node_version_id: str
    text: str
    rrf: float
    # ревью K12 P1: скор ДО кодекс-буста (органика); confidence и дельта конфликтов
    # считаются по нему - бонус равен потолку нормировки и иначе обнулял бы
    # abstain-гейт для вопросов с упоминанием кодекса
    rrf_base: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    citations: list[Citation] = field(default_factory=list)
    # K17: диагностика reranker'а (None = реранк не применялся). Органические
    # rrf/rrf_base реранкер не трогает; confidence abstain-гейта и
    # warn_conflict считаются по органическим скорам нового порядка топов -
    # после реранка их значения могут отличаться от выключенного режима.
    rerank_score: float | None = None
    rerank_rank_hybrid: int | None = None


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


_QUERY_STOPWORDS = frozenset(
    """the a an and or of in on for to do does how what why when where which is are
    was were be been being this that these those it its as at by with from into about
    как что где когда почему который эта это этот или и в во на с со по для про из под
    над при чтобы есть были быть свой своя""".split()
)


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
    legal_boost: float | None = None,
    title_boost: float | None = None,
    filter_junk: bool | None = None,
    dedupe_channels: bool | None = None,
    rerank_candidates: int | None = None,
) -> list[Evidence]:
    """Pre-retrieval authorization (инвариант 5) + ревизионный фильтр (K3) - один шов:
    узлы без пересечения групп с allowed_groups отсекаются до формирования evidence
    и не попадают в контекст; revision=None (дефолт) - только status=current,
    revision="all" - без ревизионного фильтра, revision="r<n>" - только эта ревизия.
    Мета без полей ревизий (индексы до 29.09) считается current/r1.

    Калибровка K12 (None = дефолт из veles.config, поведение до калибровки):
    rrf_k - константа RRF; channel_weights - веса вкладов (dense, lexical);
    candidate_depth - глубина каналов до RRF (дефолт top_k*5); legal_boost -
    бонус узлам документа, явно упомянутого в вопросе («согласно НК РФ»),
    в единицах вклада ранга 1 одного канала (bonus = legal_boost/(rrf_k+1));
    filter_junk - отсекать мусорные пустышки из выдачи (is_junk_node);
    dedupe_channels - снимать дубликаты узла внутри канала (см. dedupe_pairs;
    с K14 дефолт True: на чистом индексе no-op, защищает от возврата дублей).

    K17: reranker - опциональный пост-этап (env VELES_RERANKER_MODEL, пусто/
    не задано = выключено, поведение в точности прежнее: пул не строится,
    модуль реранкера не импортируется). Включён - из ordered берутся верхние
    rerank_candidates узлов (дефолт config.RERANK_CANDIDATES=25, бриф K17),
    cross-encoder пересортировывает пул, наверх идут топ-k; rrf/rrf_base
    остаются органическими (confidence не меняется), диагностика реранка -
    в Evidence.rerank_score/rerank_rank_hybrid. Ошибка модели - fail-open:
    обычный порядок, ответ не роняется (veles.retrieval.reranker)."""
    from ..config import DEFAULT_ACCESS_GROUPS

    if rrf_k is None:
        rrf_k = RRF_K
    weights = channel_weights if channel_weights is not None else (1.0, 1.0)
    depth = candidate_depth if candidate_depth is not None else top_k * 5
    if legal_boost is None:
        from ..config import LEGAL_BOOST

        legal_boost = LEGAL_BOOST
    if title_boost is None:
        from ..config import TITLE_BOOST

        title_boost = TITLE_BOOST
    if filter_junk is None:
        from ..config import FILTER_JUNK_NODES

        filter_junk = FILTER_JUNK_NODES
    if dedupe_channels is None:
        from ..config import DEDUPE_CHANNELS

        dedupe_channels = DEDUPE_CHANNELS

    # K17: reranker включён только непустым VELES_RERANKER_MODEL (конфиг читается
    # на вызов - тесты подменяют атрибут config без переимпорта модуля). Выключен -
    # пул не строится и модуль реранкера не импортируется (ноль регрессии).
    from ..config import VELES_RERANKER_MODEL

    rerank_on = bool(VELES_RERANKER_MODEL)
    if rerank_on:
        if rerank_candidates is None:
            from ..config import RERANK_CANDIDATES

            rerank_candidates = RERANK_CANDIDATES
        pool_size = max(top_k, rerank_candidates)
    else:
        pool_size = top_k

    groups = allowed_groups or DEFAULT_ACCESS_GROUPS

    # Компактная мета (фаза 5): полный __getitem__ материализует узел с
    # zlib-распаковкой текста (~10 мкс), а горячие циклы ниже читают 1-3
    # скаляра на каждого кандидата (~150-250 чтений на запрос) - на каждом
    # кандидате это давало +250-313% latency запроса (eval
    # evals/results/2026-10-15-meta-compact.md). Скаляры - через get_fields
    # без материализации; полный узел - один раз на запрос (memo: survivor-скан
    # и evidence читают одни и те же nid). Для обычного dict-меты - обёртка с
    # той же семантикой; значения и порядок проверок не меняются.
    node_meta = bundle.node_meta
    _fields_reader = getattr(node_meta, "get_fields", None)

    if _fields_reader is not None:
        def read_fields(nid: str, keys: tuple[str, ...]) -> dict:
            return _fields_reader(nid, keys)
    else:
        def read_fields(nid: str, keys: tuple[str, ...]) -> dict:
            # обычный dict и есть быстрый путь: отдаём узел ЦЕЛИКОМ (надмножество
            # keys) - тот же объект и та же правдивость, что в исходном коде,
            # без лишних аллокаций на каждого кандидата
            return node_meta.get(nid)

    _full_memo: dict[str, dict] = {}

    def full_meta(nid: str):
        m = _full_memo.get(nid)
        if m is None:
            m = node_meta.get(nid)
            if m is not None:
                _full_memo[nid] = m
        return m

    def authorized(nid: str) -> bool:
        meta = read_fields(nid, ("status", "access_groups", "revision_id"))
        if not meta:
            # исходная семантика - проверка правдивости ВСЕГО узла: у живого
            # узла без этих полей scalar-набор пуст (редкий путь - полная
            # материализация через memo)
            if not full_meta(nid):
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

    # Аудит 08.10: группы отрезают неавторизованных ПОСЛЕ отбора каналов -
    # узкая группа получала меньше top_k, хотя ниже в каналах были её документы.
    # Глубина каналов удваивается, пока авторизованных кандидатов меньше пула
    # (или не достигнут кап) - на открытых группах цикл сходится с первой итерации.
    depth_max = max(depth, 250, top_k * 25)
    while True:
        dense = [
            (nid, s) for nid, s in bundle.vector.search(query_vec, top_k=depth)
            if authorized(nid)
        ]
        lex = [
            (nid, s) for nid, s in bundle.bm25.search(query_text, top_k=depth)
            if authorized(nid)
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
        if legal_boost > 0.0:  # считаются confidence и дельта конфликтов, иначе
            from ..policy.identifiers import _LEGAL_CODES_MAP, extract_legal_refs  # буст
            # (= потолку нормировки) обнулял abstain-гейт «согласно НК РФ»

            refs = extract_legal_refs(query_text)
            if refs:
                bonus = legal_boost / (rrf_k + 1.0)
                needles = [_LEGAL_CODES_MAP[r] for r in refs]
                for nid in rrf:
                    meta = read_fields(nid, ("title",))
                    if not meta:
                        continue
                    title = " ".join(str(meta.get("title", "")).lower().replace("ё", "е").split())
                    if any(n in title for n in needles):
                        rrf[nid] += bonus

        # Аудит 08.10: тема вопроса в НАЗВАНИИ документа - сильный сигнал
        # ("Sync Build" по "how to build and sync"): название раньше не участвовало
        # в поиске, и такой документ терялся за страницами, целиком посвящёнными
        # теме. Покрытие >= половины значимых слов вопроса -> бонус, пропорциональный
        # покрытию (стоп-слова вопроса и короткие токены не считаются).
        if title_boost > 0.0 and len(rrf) > 1:
            # ё нормализуется к е с обеих сторон (как в legal-бусте выше):
            # иначе «объем» вопроса не совпадёт с «Объём» названия
            q_words = {
                w for w in re.findall(
                    r"[a-zа-яё0-9]{3,}", query_text.lower().replace("ё", "е"))
                if w not in _QUERY_STOPWORDS
            }
            if len(q_words) >= 2:
                for nid in rrf:
                    meta = read_fields(nid, ("title",))
                    if not meta:
                        continue
                    t_words = set(re.findall(
                        r"[a-zа-яё0-9]{3,}",
                        str(meta.get("title") or "").lower().replace("ё", "е")))
                    if not t_words:
                        continue
                    coverage = len(q_words & t_words) / len(q_words)
                    if coverage >= 0.5:
                        rrf[nid] += title_boost * coverage / (rrf_k + 1.0)

        ordered = sorted(rrf.items(), key=lambda x: -x[1])
        if depth >= depth_max:
            break
        survivors = 0
        for nid, _ in ordered:
            meta = full_meta(nid)
            if not meta:
                continue
            if filter_junk and is_junk_node(meta.get("text", "")):
                continue
            survivors += 1
            if survivors >= pool_size:
                break
        if survivors >= pool_size:
            break
        depth = min(depth * 2, depth_max)
    evidence: list[Evidence] = []
    for nid, score in ordered:
        # без реранка - прежнее усечение top_k; с реранком пул шире (25), усечёт
        # реранкер после пересортировки
        if len(evidence) >= pool_size:
            break
        meta = full_meta(nid)
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
            # Видео: таймкоды из provenance; у старых индексов полей нет - None
            t_start=prov.get("t_start"),
            t_end=prov.get("t_end"),
            node_version_id=nid,
            revision_id=meta.get("revision_id", "r1"),
            # K18: .get с дефолтом False - индексы до метки (старый meta.json
            # без поля lost_force) читаются без пересборки
            lost_force=bool(meta.get("lost_force", False)),
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
    # K17: опциональный пост-этап - cross-encoder пересортировывает пул и
    # усекает до top_k (fail-open внутри rerank: ошибка модели = входной порядок)
    if rerank_on and len(evidence) > 1:
        from . import reranker

        evidence = reranker.rerank(query_text, evidence, top_k)
    return evidence
