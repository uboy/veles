"""Сквозная валидация компактной RAM-меты (фаза 6 части 2 плана index-format,
docs/plan/index-format/plan.md; бриф docs/plan/briefs/2026-10-15-meta-compact.md;
коммит реализации 756c7bf - CompactNodeMeta: zlib-блоб текстов + интернинг строк).

Проверки плана:
  (а) эквивалентность поиска: синтетический индекс >= 2000 узлов с русскими
      текстами; hybrid_search на dict-мете против компактной (перезагрузка
      load_index) - evidence (id, text, rrf, rrf_base, ранги каналов, поля
      цитаты) совпадают бит-в-бит на >= 50 запросах; junk-фильтр отбирает то
      же (прямой обход обеих мет + запросы с filter_junk=True);
  (б) save round-trip: load -> save -> meta.json по-ключевому равен исходному
      payload (тексты байт-в-байт, порядок ключей сохранён); рестарт
      перечитанного каталога снова компактен и бит-в-бит;
  (в) latency: время hybrid_search-запроса на компактной мете vs dict-мете
      (лучшее из 5 прогонов пакета по 30 запросам), деградация < 5%; меряются
      ОБА режима: дефолт прода (filter_junk=False) и горячий путь брифа
      (filter_junk=True - материализация мете-записи на каждого кандидата,
      «junk-фильтр по ~100-500 кандидатам»);
  (г) RAM: синтетика 100к узлов с текстами ~1 КБ - каждый вариант (dict после
      json.loads / CompactNodeMeta) в СВОЁМ свежем подпроцессе; метрика гейта -
      живые аллокации tracemalloc после gc (резидентная мета), RSS-прирост -
      информационная. Ожидание >= 2x ниже и <= 0.5 ГиБ (критерий (г) плана).

Числовые гейты (в)/(г) применяются ПОСЛЕ всех секций: один прогон печатает
замеры всех четырёх пунктов, даже если гейт упал (сверка чисел не обрывается
на первом отказе).

Запуск: cd <корень репо> && python evals/meta-compact-validate.py
Успех: финальная строка VALIDATION-PASS, код возврата 0; ошибка любой
проверки или гейта - AssertionError, VALIDATION-FAIL, код 1. Все проверки -
assertions.

Временные файлы - во временной папке ОС, удаляются в конце; захардкоженных
машинных путей нет. Замер (г) выполняется в подпроцессах того же скрипта
(--child-ram): чистый tracemalloc/RSS без мусора родителя.
"""

from __future__ import annotations

import gc
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- окружение
# Код валидируемого worktree - первым на sys.path: при запуске
# «python evals/...» sys.path[0] - каталог evals/, иначе импортируется
# УСТАНОВЛЕННЫЙ в site-packages veles без коммита 756c7bf.
ROOT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_REPO))

# Режим подпроцесса-замерщика (г): свежий процесс, один вариант меты, одна
# JSON-строка в stdout (чистый tracemalloc/RSS).
_CHILD = len(sys.argv) > 1 and sys.argv[1] == "--child-ram"

# env ДО импортов veles: config читает VELES_DATA на импорте (в подпроцессе
# импорт config безопасен с дефолтами - он там путей не трогает). Каталог НЕ
# создаётся на импорте: сборка модуля pytest'ом (pytest --collect-only) не
# должна оставлять мусор; создаётся в main().
_TMP_ROOT: Path | None = None
if not _CHILD:
    _TMP_ROOT = Path(tempfile.gettempdir()) / f"veles-meta-compact-eval-{os.getpid()}"
    os.environ["VELES_DATA"] = str(_TMP_ROOT / "data")

from veles.config import DEFAULT_ACCESS_GROUPS, EMBED_DIM, EMBED_MODEL  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.indexer import IndexBundle, is_lost_force, load_index  # noqa: E402
from veles.indexing.meta_compact import CompactNodeMeta  # noqa: E402
from veles.indexing.vector_store import NumpyFlatStore  # noqa: E402
from veles.retrieval.retriever import hybrid_search, is_junk_node  # noqa: E402

EMBED_META = {"model": EMBED_MODEL, "backend": "numpy"}


def _current_rss_bytes() -> int | None:
    """Текущий (резидентный) RSS процесса: psutil.rss либо VmRSS (Linux)."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _peak_rss_bytes() -> int | None:
    """Пиковый RSS процесса: psutil.peak_wset (Windows) либо VmHWM (Linux)."""
    try:
        import psutil

        peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
        if peak is not None:
            return int(peak)
    except ImportError:
        pass
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def mb(n_bytes: float) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} МБ"


# --------------------------------------------------- генератор синтетики
# Темы: (название документа, ключевые слова). Ключевые слова встречаются в
# текстах узлов В ТОЙ ЖЕ ФОРМЕ (BM25 без стемминга) - запросы из них дают
# реальные выдачи; часть названий - кодексы (legal_boost, retriever.py:257).

_THEME_SENT_TMPLS = (
    "{kw} оформляется в порядке, установленном настоящим документом, с учётом "
    "требований федерального законодательства и внутренних регламентов организации.",
    "Ответственное подразделение контролирует, как {kw} отражается в учёте и "
    "отчётности за отчётный период.",
    "При нарушении сроков {kw} фиксируется в журнале и передаётся на рассмотрение "
    "комиссии по соблюдению внутреннего распорядка.",
)

_THEMES: list[tuple[str, tuple[str, ...]]] = [
    ("Налоговый кодекс РФ часть вторая",
     ("налог", "ставка", "декларация", "вычет", "ндс", "акциз", "взнос",
      "налогоплательщик")),
    ("Гражданский кодекс РФ часть первая",
     ("сделка", "договор", "обязательство", "иск", "давность", "соглашение")),
    ("Трудовой кодекс РФ",
     ("отпуск", "зарплата", "увольнение", "смена", "отгул", "пособие")),
    ("Положение о закупках",
     ("закупка", "извещение", "поставщик", "заявка", "контракт", "протокол",
      "победитель", "порог")),
    ("Регламент командирования сотрудников",
     ("командирование", "командировка", "аванс", "суточные", "проезд",
      "отчёт")),
    ("Инструкция по оформлению отпусков",
     ("отпуск", "график", "заявление", "выплата", "перенос")),
    ("Регламент финансовой отчётности",
     ("отчётность", "баланс", "счёт", "проводка", "сверка")),
    ("Порядок защиты персональных данных",
     ("данные", "согласие", "обработка", "защита", "реестр")),
    ("Регламент доступа к информационным системам",
     ("доступ", "учётная", "пароль", "система", "роль")),
    ("Инструкция по делопроизводству",
     ("документ", "регистрация", "архив", "номенклатура", "подпись")),
    ("Положение о командировках за рубеж",
     ("валюта", "виза", "загранкомандировка", "граница", "страховка")),
]

# фразы-наполнители (общие для всех тем: повторяющийся «фоновый» русский текст,
# как в продовом корпусе)
_FILLER = (
    "Соблюдение требований настоящего документа обязательно для всех "
    "подразделений организации и их работников.",
    "Контроль исполнения возлагается на руководителя структурного подразделения.",
    "Изменения вносятся приказом с ознакомлением работников под подпись.",
    "Спорные вопросы разрешаются в порядке, установленном законодательством.",
)


def _theme_sentences(kws: tuple[str, ...]) -> list[str]:
    """Пул предложений темы: каждое ключевое слово в 3 шаблонах (та же
    словоформа, что пойдёт в запросы)."""
    out = []
    for kw in kws:
        for tmpl in _THEME_SENT_TMPLS:
            out.append(tmpl.format(kw=kw))
    return out


def _node_text(rng: random.Random, theme: tuple[str, tuple[str, ...]], i: int, d: int,
               nid: str, k_range: tuple[int, int] = (4, 8)) -> str:
    """Текст узла: строка ключевых слов темы (BM25-хук той же словоформы) +
    случайные предложения темы + наполнитель + уникальный хвост узла."""
    title, kws = theme
    pool = _theme_sentences(kws)
    k = rng.randint(*k_range)
    parts = ["Ключевые слова: " + ", ".join(kws) + "."]
    parts += rng.sample(pool, k=min(k, len(pool)))
    parts.append(rng.choice(_FILLER))
    parts.append(f"Уникальный хвост узла {nid} документа {d} ({title}), код {i:05d}.")
    return " ".join(parts)


def make_meta_corpus(n_nodes: int = 2200, seed: int = 42) -> dict[str, dict]:
    """Синтетическая node_meta в форме прода-корпуса (см. build_index):
    дублирующиеся document_id/title/revision_id/status (уровень документа),
    junk-пустышки, tombstone/superseded-доки, видео-узлы с таймкодами,
    asset у каждого пятого узла."""
    n_docs = n_nodes // 50
    assert n_docs * 50 == n_nodes, "синтетика кратно 50 узлам на документ"
    rng = random.Random(seed)
    groups_pool = [["personal"], ["personal", "hr"], ["*"]]
    adapters = ["md", "docling", "video"]
    out: dict[str, dict] = {}
    i = 0
    for d in range(n_docs):
        theme = _THEMES[d % len(_THEMES)]
        did = f"doc-{d:04d}"
        title, _ = theme
        # статусы и ревизии - на уровне документа, как build_index (K3)
        status = "deleted" if d % 22 == 21 else ("superseded" if d % 5 == 4 else "current")
        revision_id = f"r{1 + d % 2}"
        sha = f"{d:064x}"
        for j in range(50):
            nid = f"nv-{i:06d}"
            if j == 25:
                text = "Утратил силу."
            elif j == 38:
                text = ("Бесплатная юридическая консультация: "
                        f"8 (495) {100 + d % 900}-{d % 100:02d}-{d % 90 + 10:02d}")
            else:
                text = _node_text(rng, theme, i, d, nid)
            prov: dict = {
                "source_id": f"src-{d % 6}",
                "adapter": adapters[d % 3],
                "lines": [i % 800 + 1, i % 800 + 8] if i % 2 else None,
            }
            if adapters[d % 3] == "video":
                prov["t_start"] = float(i % 3600)
                prov["t_end"] = float(i % 3600 + 30)
            if i % 7 == 0:
                prov["page"] = i % 300
            node: dict = {
                "document_id": did,
                "title": title,
                "logical_node_id": f"ln-{i}",
                "heading_path": [title, f"Раздел {j % 7}", f"Глава {i % 13}"],
                "text": text,
                "provenance": prov,
                "access_groups": ["hr"] if i % 17 == 0 else groups_pool[i % 3],
                "sha256_doc": sha,
                "revision_id": revision_id,
                "status": status,
                "lost_force": is_lost_force(text),
            }
            if i % 5 == 0:
                node["asset"] = f"pic-{i:05d}.png"
            out[nid] = node
            i += 1
    return out


def make_ram_nodes(n_nodes: int = 100_000, seed: int = 2026) -> dict[str, dict]:
    """Синтетика для RAM-замера: те же темы, тексты ~1 КБ (как в прода-корпусе:
    русский текст с повторяющимися оборотами и уникальным хвостом)."""
    n_docs = max(1, n_nodes // 50)
    rng = random.Random(seed)
    out: dict[str, dict] = {}
    i = 0
    for d in range(n_docs):
        theme = _THEMES[d % len(_THEMES)]
        did = f"doc-{d:05d}"
        title, _ = theme
        sha = f"{d:064x}"
        for j in range(50):
            nid = f"nv-{i:06d}"
            out[nid] = {
                "document_id": did,
                "title": title,
                "logical_node_id": f"ln-{i}",
                "heading_path": [title, f"Раздел {j % 9}", f"Глава {i % 17}"],
                "text": _node_text(rng, theme, i, d, nid, k_range=(6, 12)),
                "provenance": {
                    "source_id": f"src-{d % 6}",
                    "adapter": "docling",
                    "lines": [i % 900 + 1, i % 900 + 9] if i % 2 else None,
                },
                "access_groups": ["personal"],
                "sha256_doc": sha,
                "revision_id": "r1",
                "status": "current",
                "lost_force": False,
            }
            i += 1
    return out


# -------------------------------------------------------- запросы (а)/(в)

def make_queries(seed: int = 7) -> list[tuple[str, dict]]:
    """(текст запроса, kwargs hybrid_search). Темы + кодексы (legal_boost,
    retriever.py:257-275) + тема в названии (title_boost :277-302) + junk +
    граничные случаи; kwargs - варианты ревизий/групп/фильтра (K3/K10/K12)."""
    rng = random.Random(seed)
    queries: list[tuple[str, dict]] = []
    for _ in range(40):
        _, kws = rng.choice(_THEMES)
        queries.append((" ".join(rng.sample(kws, rng.randint(1, 3))), {}))
    queries += [
        ("согласно НК РФ ставка налога", {}),                # legal_boost: налоговый кодекс
        ("порядок по ГК РФ сделка", {}),                     # legal_boost: гражданский кодекс
        ("отпуск по ТК РФ", {}),                             # legal_boost: трудовой кодекс
        ("штраф по КоАП РФ", {}),                            # кодекса в корпусе нет - буст no-op
        ("как оформить командирование", {}),
        ("регламент командирования", {}),                    # title_boost (2 слова из названия)
        ("инструкция по отпускам", {}),                      # title_boost
        ("положение о закупках", {}),                        # title_boost
        ("утратил силу", {"filter_junk": True}),             # junk-пустышки
        ("не применяется", {"filter_junk": True}),
        ("бесплатная юридическая консультация", {"filter_junk": True}),
        ("8 495 123 45 67", {"filter_junk": True}),
        ("несуществующийтермин", {}),                        # пустой BM25, dense даёт топ
        ("", {}),                                            # граничный случай
        ("доступ система пароль", {}),
        ("отчётность баланс счёт", {"revision": "all"}),     # K3: без ревизионного фильтра
        ("персональные данные согласие", {"revision": "all"}),
        ("делопроизводство архив документ", {"revision": "all"}),
        ("закупка извещение заявка", {"filter_junk": True}),
        ("командировка авансовый отчёт", {"filter_junk": True}),
        ("налог ставка вычет ндс", {"filter_junk": True}),
        ("валюта виза загранкомандировка", {"revision": "all"}),
        ("доступ", {"allowed_groups": {"hr"}}),              # узко-групповой запрос
        ("закупка", {"allowed_groups": {"hr"}}),
        ("архив регистрация", {"revision": "r1"}),
        ("подпись номенклатура", {"revision": "r1"}),
    ]
    return queries


def evidence_tuple(ev) -> tuple:
    """Полная распаковка Evidence для бит-в-бит сравнения (id, text, rrf + база,
    ранги каналов и все поля цитаты)."""
    c = ev.citations[0]
    return (
        ev.node_version_id,
        ev.text,
        ev.rrf,
        ev.rrf_base,
        ev.dense_rank,
        ev.bm25_rank,
        c.title,
        c.document_id,
        tuple(c.heading_path),
        c.source_id,
        tuple(c.lines) if c.lines is not None else None,
        c.node_version_id,
        c.revision_id,
        c.lost_force,
        c.t_start,
        c.t_end,
    )


# --------------------------------------------------------- подпроцесс (г)

def _child_ram(mode: str) -> int:
    """Режим --child-ram dict|compact: сборка ОДНОГО payload, перевод в свою
    RAM-форму, живые аллокации tracemalloc после gc + RSS-прирост.
    База RSS - сразу после импортов (до payload/JSON-строки), итог - после gc
    с живой метой: у обоих режимов окно одинаковое, освобождённые промежуточные
    формы (payload, JSON-строка, распарсенный dict) в прирост не попадают
    (возврат страниц ОС после освобождения даёт шум - поэтому метрика гейта -
    tracemalloc, RSS только информационная).
    Печатает В STDOUT ровно одну JSON-строку (ASCII)."""
    import tracemalloc

    rss_before = _current_rss_bytes()
    n = 100_000
    payload = make_ram_nodes(n)
    # выборочный контроль содержимого удерживается вне payload: после
    # построения меты сверяем, что текст реально читается из своей формы
    samples = [(f"nv-{k:06d}", payload[f"nv-{k:06d}"]["text"]) for k in (0, 12345, n - 1)]
    s = json.dumps(payload, ensure_ascii=False)
    json_bytes = len(s.encode("utf-8"))
    del payload
    gc.collect()

    tracemalloc.start()
    if mode == "dict":
        meta = json.loads(s)
    else:
        assert mode == "compact", f"неизвестный режим {mode!r}"
        parsed = json.loads(s)
        meta = CompactNodeMeta.from_dict(parsed)
        del parsed
    del s
    gc.collect()
    traced_live, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(meta) == n, f"режим {mode}: узлов {len(meta)} != {n}"
    sample_ok = all(meta[nid]["text"] == text for nid, text in samples)
    assert sample_ok, f"режим {mode}: текст контрольного узла не совпал"
    gc.collect()
    rss_after = _current_rss_bytes()
    rss_delta = None if None in (rss_before, rss_after) else rss_after - rss_before
    print(
        json.dumps(
            {
                "mode": mode,
                "n_nodes": len(meta),
                "json_bytes": json_bytes,
                "traced_live": traced_live,
                "traced_peak": traced_peak,
                "rss_delta": rss_delta,
                "rss_peak": _peak_rss_bytes(),
                "sample_ok": sample_ok,
            }
        ),
        flush=True,
    )
    return 0


def _run_child(mode: str) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child-ram", mode],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT_REPO),
    )
    assert proc.returncode == 0, (
        f"подпроцесс RAM-замера ({mode}) упал (код {proc.returncode}):\n{proc.stderr[-2000:]}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"подпроцесс ({mode}) не напечатал JSON: stderr={proc.stderr[-500:]}"
    return json.loads(lines[-1])


# ------------------------------------------------------------------- секции

def _dict_bundle_from_disk(src_dir: Path) -> IndexBundle:
    """Эталонный бандл с dict-метой из ТЕХ ЖЕ файлов на диске - прежняя RAM-форма
    (json.loads(meta.json)), вектор/BM25 - обычная загрузка."""
    payload = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
    bundle = IndexBundle(
        vector=NumpyFlatStore(dim=EMBED_DIM),
        bm25=Bm25Index.load(src_dir),
        node_meta=payload["node_meta"],
        identifiers=set(payload["identifiers"]),
    )
    bundle.vector.ids = json.loads(
        (src_dir / "numpy_ids.json").read_text(encoding="utf-8"))
    bundle.vector.matrix = np.load(src_dir / "numpy_matrix.npy")
    return bundle


def _write_index_dir(dir_path: Path, plain: dict, search_texts: list[str],
                     ids: list[str], vectors: np.ndarray, identifiers: set) -> None:
    dir_path.mkdir(parents=True)
    (dir_path / "meta.json").write_text(
        json.dumps(
            {"node_meta": plain, "embed": EMBED_META, "identifiers": sorted(identifiers)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    store = NumpyFlatStore(dim=EMBED_DIM)
    store.add(ids, vectors)
    store.save(dir_path)
    bm25 = Bm25Index()
    bm25.add(ids, search_texts)
    bm25.save(dir_path)


def section_equivalence(checks: list[str]):
    """(а) hybrid_search: dict-мета против компактной (перезагрузка load_index),
    evidence бит-в-бит на >= 50 запросах; junk-фильтр отбирает то же.
    Возвращает (plain, bundle_dict, bundle_compact, queries, qvecs, identifiers)."""
    n_nodes = 2200
    plain = make_meta_corpus(n_nodes)
    assert len(plain) == n_nodes
    ids = list(plain)
    # search_text - как build_index (indexer.py:108-110): префикс пути заголовков
    search_texts = []
    for nid in ids:
        node = plain[nid]
        prefix = " > ".join(node["heading_path"])
        search_texts.append(f"[{prefix}] {node['text']}" if prefix else node["text"])
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n_nodes, EMBED_DIM)).astype(np.float32)
    identifiers = {"ГОСТ 21.301-2014", "СП 42.13330.2019"}

    src_dir = _TMP_ROOT / "index-src"
    _write_index_dir(src_dir, plain, search_texts, ids, vectors, identifiers)

    bundle_dict = _dict_bundle_from_disk(src_dir)
    # валидируемая форма: перезагрузка продовым путём (indexer.py:194)
    bundle_compact = load_index(src_dir)
    assert isinstance(bundle_compact.node_meta, CompactNodeMeta), (
        f"load_index отдал {type(bundle_compact.node_meta).__name__} вместо CompactNodeMeta"
    )
    assert set(iter(bundle_compact.node_meta)) == set(plain), "набор узлов перезагрузки разошёлся"

    queries = make_queries()
    assert len(queries) >= 50, "масштаб проверки (а) ниже требуемого (>= 50 запросов)"
    qrng = np.random.default_rng(7)
    qvecs = [qrng.standard_normal(EMBED_DIM).astype(np.float32) for _ in queries]

    junk_dict = {nid for nid, m in plain.items() if is_junk_node(m.get("text", ""))}
    junk_compact = {
        nid for nid in bundle_compact.node_meta
        if is_junk_node(bundle_compact.node_meta[nid].get("text", ""))
    }
    assert junk_dict == junk_compact, "junk-фильтр отбирает разное на dict- и компактной мете"
    assert len(junk_dict) >= 40, f"junk-узлов в синтетике меньше ожидаемого: {len(junk_dict)}"

    deleted_ids = {nid for nid, m in plain.items() if m.get("status") == "deleted"}
    nonempty = 0
    for qi, (qtext, kwargs) in enumerate(queries):
        got = hybrid_search(bundle_dict, qvecs[qi], qtext, top_k=8, **kwargs)
        comp = hybrid_search(bundle_compact, qvecs[qi], qtext, top_k=8, **kwargs)
        assert len(got) == len(comp), (
            f"(а) запрос #{qi} {qtext!r}: длина выдачи {len(got)} != {len(comp)}"
        )
        for ev_d, ev_c in zip(got, comp):
            t_d, t_c = evidence_tuple(ev_d), evidence_tuple(ev_c)
            assert t_d == t_c, (
                f"(а) запрос #{qi} {qtext!r}: evidence разошёлся:"
                f"\n  dict:    {t_d!r}\n  compact: {t_c!r}"
            )
            assert ev_d.node_version_id not in deleted_ids, (
                f"(а) tombstone-узел {ev_d.node_version_id} в выдаче"
            )
            if kwargs.get("filter_junk"):
                assert not is_junk_node(ev_d.text), (
                    f"(а) filter_junk=True пропустил пустышку: {ev_d.text!r}"
                )
        if got:
            nonempty += 1
    assert nonempty >= len(queries) // 2, (
        f"(а) слишком мало непустых выдач ({nonempty} из {len(queries)}) - "
        "эквивалентность проверена на пустом множестве"
    )
    stats = bundle_compact.node_meta.compact_stats()
    checks.append(
        f"(а) эквивалентность: индекс {n_nodes} узлов / {n_nodes // 50} доков; "
        f"{len(queries)} запросов (непустых {nonempty}); evidence (id, text, rrf, ранги, "
        f"цитаты) бит-в-бит; junk-набор совпал ({len(junk_dict)} пустышек); "
        f"tombstone не в выдаче; компакт: блоб {mb(stats['blob_bytes'])}"
    )
    return plain, bundle_dict, bundle_compact, src_dir, queries, qvecs, identifiers


def section_roundtrip(plain: dict, bundle_compact, identifiers: set,
                      checks: list[str]) -> None:
    """(б) load -> save -> meta.json по-ключевому == исходному; рестарт компактен."""
    out_dir = _TMP_ROOT / "index-saved"
    bundle_compact.save(out_dir, embed_meta=EMBED_META)
    reread = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert reread["embed"] == EMBED_META, "embed-мета разошлась при save"
    assert reread["identifiers"] == sorted(identifiers), "identifiers разошлись при save"
    nodes = reread["node_meta"]
    assert nodes == plain, "meta.json после round-trip != исходному payload"
    assert list(nodes) == list(plain), "порядок ключей node_meta не сохранён при save"
    n_fields = 0
    for nid, expected in plain.items():
        got = nodes[nid]
        assert set(got) == set(expected), f"(б) {nid}: набор полей {set(got)} != {set(expected)}"
        for field in expected:
            assert got[field] == expected[field], f"(б) {nid}: поле {field} разошлось"
            n_fields += 1
    # «рестарт»: перечитанный каталог снова компактен и бит-в-бит
    bundle2 = load_index(out_dir)
    assert isinstance(bundle2.node_meta, CompactNodeMeta)
    assert bundle2.node_meta.to_plain_dict() == plain, "рестарт round-trip разошёлся"
    bundle2.vector.close()  # mmap перед rmtree (Windows)
    checks.append(
        f"(б) round-trip: load -> save -> meta.json по-ключевому равен исходному "
        f"({len(plain)} узлов, {n_fields} полей, тексты байт-в-байт); рестарт "
        f"каталога снова компактен"
    )


def section_latency(bundle_dict, bundle_compact, queries, qvecs,
                    checks: list[str]) -> dict:
    """(в) время hybrid_search-запроса: компакт vs dict, лучшее из 5 пакетов по 30
    запросов, оба режима: дефолт прода и горячий путь filter_junk=True
    (материализация на каждом кандидате). Гейт применяется в main()."""
    lat_vecs = qvecs[:30]

    def run_batch(bundle, filt: bool) -> float:
        t0 = time.perf_counter()
        for qv, (qtext, kw) in zip(lat_vecs, queries[:30]):
            hybrid_search(bundle, qv, qtext, top_k=8, filter_junk=filt, **kw)
        return time.perf_counter() - t0

    out: dict[str, float] = {}
    for label, filt in (("default", False), ("junk", True)):
        # прогрев: mmap-страницы матрицы, дисковый кэш, кодовые пути
        run_batch(bundle_dict, filt)
        run_batch(bundle_compact, filt)
        times: dict[str, list[float]] = {"dict": [], "compact": []}
        for _ in range(5):
            times["dict"].append(run_batch(bundle_dict, filt))
            times["compact"].append(run_batch(bundle_compact, filt))
        t_dict = min(times["dict"])
        t_compact = min(times["compact"])
        ratio = t_compact / t_dict
        mode = "дефолт (filter_junk=False)" if not filt else "горячий путь (filter_junk=True)"
        checks.append(
            f"(в) latency hybrid_search, {mode} (30 запросов, лучшее из 5): "
            f"dict {t_dict * 1000 / 30:.2f} мс/запрос ({t_dict:.3f}s пакет), компакт "
            f"{t_compact * 1000 / 30:.2f} мс/запрос ({t_compact:.3f}s пакет) - "
            f"{(ratio - 1) * 100:+.1f}% (гейт < +5%)"
        )
        out[label] = ratio
        out[f"t_dict_{label}"] = t_dict
        out[f"t_compact_{label}"] = t_compact
    return out


def section_ram(checks: list[str]) -> dict:
    """(г) RAM резидентной меты на 100к узлов: dict (после json.loads) против
    CompactNodeMeta, каждый вариант - в свежем подпроцессе. Гейты - в main()."""
    d = _run_child("dict")
    c = _run_child("compact")
    assert d["n_nodes"] == c["n_nodes"] == 100_000, "ребёнок собрал не 100к узлов"
    assert d["json_bytes"] == c["json_bytes"], "payload детей разошёлся"
    assert d["sample_ok"] and c["sample_ok"], "контрольные тексты не читаются из своей формы"
    assert d["traced_live"] > 0 and c["traced_live"] > 0, "tracemalloc вернул ноль"

    ratio = d["traced_live"] / c["traced_live"]
    checks.append(
        f"(г) RAM меты на 100к узлах (живые аллокации tracemalloc после gc, "
        f"каждый вариант в своём подпроцессе): dict {mb(d['traced_live'])} vs компакт "
        f"{mb(c['traced_live'])} = {ratio:.1f}x ниже (гейты >= 2x и <= 0.5 ГиБ)"
    )
    if d.get("rss_delta") is not None and c.get("rss_delta") is not None:
        checks.append(
            f"(г) RSS-прирост после gc (инфо, не гейт): dict +{mb(d['rss_delta'])} vs "
            f"компакт +{mb(c['rss_delta'])}"
        )
    else:
        checks.append("(г) RSS недоступен (нет psutil и /proc) - прирост не измерен (инфо)")
    checks.append(
        f"(г) пик аллокаций построения (tracemalloc, инфо): dict {mb(d['traced_peak'])} vs "
        f"компакт {mb(c['traced_peak'])}; meta.json в JSON: {mb(d['json_bytes'])}"
    )
    return {"dict": d, "compact": c, "ratio": ratio}


def apply_gates(lat: dict, ram: dict) -> None:
    """Числовые гейты плана после всех замеров: (в) деградация < 5% в обоих
    режимах; (г) компакт >= 2x ниже dict и <= 0.5 ГиБ."""
    for label, mode in (("default", "дефолт"), ("junk", "горячий путь filter_junk=True")):
        ratio = lat[label]
        assert ratio < 1.05, (
            f"(в) latency, {mode}: деградация {(ratio - 1) * 100:+.1f}% "
            f"({lat[f't_compact_{label}']:.3f}s vs {lat[f't_dict_{label}']:.3f}s "
            f"на пакет 30 запросов), гейт < +5%"
        )
    ratio = ram["ratio"]
    d, c = ram["dict"], ram["compact"]
    assert ratio >= 2.0, (
        f"(г) компактная мета всего в {ratio:.2f}x ниже dict "
        f"({mb(d['traced_live'])} vs {mb(c['traced_live'])} живых аллокаций), ожидание >= 2x"
    )
    assert c["traced_live"] <= 0.5 * (1 << 30), (
        f"(г) компактная мета {mb(c['traced_live'])} > 0.5 ГиБ (критерий (г) плана)"
    )


# --------------------------------------------------------------------- main

def main() -> int:
    t_start = time.perf_counter()
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)

    print("veles meta-compact: валидация компактной RAM-меты (коммит 756c7bf)")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, repo {ROOT_REPO}",
          flush=True)
    assert not os.environ.get("VELES_RERANKER_MODEL"), (
        "VELES_RERANKER_MODEL задан - валидация обязана идти с дефолтным поведением"
    )
    checks: list[str] = []
    ram: dict = {}
    lat: dict = {}
    bundle_compact = None
    try:
        print(f"\n=== (а) эквивалентность поиска: dict-мета vs компактная "
              f"(группы по умолчанию: {sorted(DEFAULT_ACCESS_GROUPS)}) ===", flush=True)
        (plain, bundle_dict, bundle_compact, _src_dir,
         queries, qvecs, identifiers) = section_equivalence(checks)

        print("\n=== (б) save round-trip ===", flush=True)
        section_roundtrip(plain, bundle_compact, identifiers, checks)

        print("\n=== (в) latency запроса: компакт vs dict (дефолт + горячий путь) ===", flush=True)
        lat = section_latency(bundle_dict, bundle_compact, queries, qvecs, checks)

        print("\n=== (г) RAM меты: 100к узлов, два подпроцесса ===", flush=True)
        ram = section_ram(checks)

        print("\nИТОГИ")
        for c in checks:
            print(f"  {c}")
        # гейты - после всех замеров и печати: один прогон даёт все числа
        apply_gates(lat, ram)
        d, c = ram["dict"], ram["compact"]
        print(
            f"\nЧисла для отчёта: 100000 узлов, meta.json {mb(d['json_bytes'])}; "
            f"RAM меты (tracemalloc live): dict {mb(d['traced_live'])} -> компакт "
            f"{mb(c['traced_live'])} ({ram['ratio']:.1f}x, гейты >= 2x и <= 0.5 ГиБ); "
            f"latency (лучшее из 5 пакетов по 30): дефолт dict "
            f"{lat['t_dict_default']:.3f}s -> компакт {lat['t_compact_default']:.3f}s "
            f"({(lat['default'] - 1) * 100:+.1f}%), горячий путь dict "
            f"{lat['t_dict_junk']:.3f}s -> компакт {lat['t_compact_junk']:.3f}s "
            f"({(lat['junk'] - 1) * 100:+.1f}%)"
        )
        print(f"\nВсего времени: {time.perf_counter() - t_start:.1f}s")
        print("VALIDATION-PASS", flush=True)
        return 0
    finally:
        if bundle_compact is not None:
            try:
                bundle_compact.vector.close()  # mmap перед rmtree (Windows)
            except Exception:  # noqa: BLE001 - чистка не маскирует результат
                pass
        for _ in range(5):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
            if not _TMP_ROOT.exists():
                break
            time.sleep(1.0)
        if _TMP_ROOT.exists():
            print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось удалить {_TMP_ROOT}", file=sys.stderr)


def test_meta_compact_validation():
    """Точка входа pytest - тот же прогон, что и «python evals/meta-compact-validate.py»
    (прецедент: test_index_format_validation в evals/index-format-validate.py)."""
    assert main() == 0


if __name__ == "__main__":
    if _CHILD:
        raise SystemExit(_child_ram(sys.argv[2]))
    try:
        _code = main()
    except AssertionError as e:
        print(f"VALIDATION-FAIL: {e}", file=sys.stderr, flush=True)
        _code = 1
    raise SystemExit(_code)
