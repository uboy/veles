"""Сквозная валидация бинарного формата индекса (фаза 3 плана index-format,
docs/plan/index-format/plan.md; бриф docs/plan/briefs/2026-10-14-index-binary-format.md;
коммит реализации 62873e1 - bm25 в CSR на numpy + bm25.npz + mmap матрицы).

Проверки плана:
  (а) бит-в-бит: эталонная реализация search питоновским циклом (точная копия
      Bm25Index.search из v1, git 0b44e9d) против нового Bm25Index (CSR):
      синтетический корпус >= 2000 доков, >= 300 запросов, точное равенство
      (id, score) топ-списков (те же пары в том же порядке, скоры бит-в-бит);
  (б) round-trip старого формата: payload старого bm25.json (tfs/df как в v1
      save) -> load -> save npz -> load -> поиск совпадает с эталоном; save
      оставляет рядом СИНХРОННЫЙ bm25.json (payload v1) - каталог после записи
      читается старым кодом без пересборки (ревью раунд 1, находка 2);
  (в) память - критерий (б) плана «RSS при загрузке >= 3x ниже»: меряется
      ПРИРОСТ пикового RSS процесса при загрузке (сэмплы RSS в фоновом потоке
      подпроцесса, каждый формат в своём свежем процессе). Полный пик RSS
      подпроцесса включает одинаковую базу интерпретатора+numpy (~100-130 МБ),
      которая размывает отношение на 100к узлов (см. «Расхождения плана и
      реальности» в plan.md); пик аллокаций tracemalloc - дополнительная
      информационная метрика, ГЕЙТОМ НЕ ЯВЛЯЕТСЯ (ревью раунд 1, находка 1);
  (г) скорость: время загрузки старый vs новый формат на том же наборе
      (лучшее из 3 без трассировки);
  (д) mmap: поиск по матрице, загруженной np.load(mmap_mode='r') через
      load_index, идентичен поиску по обычному ndarray (векторный и bm25
      каналы); add() после mmap работает и save пишет полную матрицу.

Запуск: cd <корень репо> && python evals/index-format-validate.py
Успех: финальная строка VALIDATION-PASS, код возврата 0; любая ошибка -
AssertionError, VALIDATION-FAIL, код 1. Все проверки - assertions.

Временные файлы - во временной папке ОС, удаляются в конце; захардкоженных
машинных путей нет. Замеры (в)/(г) выполняются в подпроцессах того же скрипта
(--child-load): чистый tracemalloc/RSS без мусора родителя.
"""

from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- окружение
# Код валидируемого worktree - первым на sys.path: при запуске
# «python evals/...» sys.path[0] - каталог evals/, иначе импортируется
# УСТАНОВЛЕННЫЙ в site-packages veles без коммита 62873e1.
ROOT_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_REPO))

# Режим подпроцесса-замерщика (в)/(г): без временных каталогов и env-магии,
# только загрузка одного формата и печать одной JSON-строки в stdout.
_CHILD = len(sys.argv) > 1 and sys.argv[1] == "--child-load"

# env ДО импортов veles: config читает VELES_DATA на импорте (в подпроцессе
# импорт config безопасен с дефолтами - он там путей не трогает). Каталог НЕ
# создаётся на импорте: сборка модуля pytest'ом (pytest --collect-only) не
# должна оставлять мусор; создаётся в main().
_TMP_ROOT: Path | None = None
if not _CHILD:
    _TMP_ROOT = Path(tempfile.gettempdir()) / f"veles-idx-fmt-eval-{os.getpid()}"
    os.environ["VELES_DATA"] = str(_TMP_ROOT / "data")

from veles.indexing.bm25 import Bm25Index, tokenize  # noqa: E402


def _peak_rss_bytes() -> int | None:
    """Пиковый RSS процесса: psutil.peak_wset (Windows) либо VmHWM (Linux);
    недоступно - None (вызывающий код печатает, что данных нет)."""
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


# --- эталон v1: точная копия Bm25Index.search ДО CSR (git 0b44e9d) ---

def reference_search(tfs, df, doc_lens, ids, avgdl, query, top_k=10, k1=1.5, b=0.75):
    """Питоновский цикл v1 (tfs - list[dict[str, int]], df - dict) - эталон
    бит-в-бит; формула, арифметика и порядок суммирования не тронуты."""
    if not ids:
        return []
    q_terms = tokenize(query)
    n_docs = len(ids)
    scores: list[tuple[str, float]] = []
    for i in range(n_docs):
        score = 0.0
        for term in q_terms:
            tf = tfs[i].get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf + k1 * (1 - b + b * doc_lens[i] / (avgdl or 1))
            score += idf * tf * (k1 + 1) / denom
        if score > 0:
            scores.append((ids[i], score))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_k]


# --- генераторы синтетики (детерминированные, seed фиксируется) ---

def make_corpus(n_docs: int, seed: int, vocab_size: int = 180) -> list[str]:
    """Термин-везде (общийслово, df == n_docs) + убывающие веса словаря,
    повторы термина внутри документа, разброс длин."""
    rng = random.Random(seed)
    vocab = [f"термин{n:03d}" for n in range(vocab_size)]
    weights = [len(vocab) - j for j in range(len(vocab))]
    docs = []
    for _ in range(n_docs):
        length = rng.randint(20, 120)
        words = ["общийслово"]
        words += rng.choices(vocab, weights=weights, k=length - 1)
        words += rng.choices(vocab, k=rng.randint(1, 3))
        docs.append(" ".join(words))
    return docs


def make_queries(n: int, seed: int, vocab_size: int = 180) -> list[str]:
    """1-5 терминов словаря (с повторами) + термин-везде + несуществующие +
    пустой запрос (граничные случаи эталона и CSR)."""
    rng = random.Random(seed)
    queries = [
        " ".join(f"термин{t:03d}" for t in rng.choices(range(vocab_size), k=rng.randint(1, 5)))
        for _ in range(n)
    ]
    queries += [
        "общийслово",  # все доки, куча равных скоров (tie-break по индексу дока)
        "общийслово общийслово термин005 термин005",
        "несуществующийтермин",
        "термин001 несуществующийтермин",
        "",
    ]
    return queries


def tf_dicts(docs: list[str]) -> list[dict[str, int]]:
    out = []
    for text in docs:
        tf: dict[str, int] = {}
        for t in tokenize(text):
            tf[t] = tf.get(t, 0) + 1
        out.append(tf)
    return out


def df_from(tfs: list[dict[str, int]]) -> dict[str, int]:
    df: dict[str, int] = {}
    for tf in tfs:
        for t in tf:
            df[t] = df.get(t, 0) + 1
    return df


def legacy_payload(tfs, df, ids, doc_lens, avgdl, k1=1.5, b=0.75) -> dict:
    """Payload ровно в формате старого Bm25Index.save (bm25.json, v1)."""
    return {
        "k1": k1,
        "b": b,
        "ids": ids,
        "doc_lens": doc_lens,
        "tfs": tfs,
        "df": df,
        "avgdl": avgdl,
    }


def assert_search_exact(got, expected, where: str) -> None:
    """(id, score) точно: те же пары в том же порядке, скоры бит-в-бит."""
    assert len(got) == len(expected), (
        f"{where}: длина выдачи {len(got)} != {len(expected)}"
    )
    for j, ((g_id, g_s), (e_id, e_s)) in enumerate(zip(got, expected)):
        assert g_id == e_id, f"{where}: позиция {j}: id {g_id!r} != {e_id!r}"
        assert g_s == e_s, f"{where}: позиция {j} ({g_id}): {g_s!r} != {e_s!r}"


def mb(n_bytes: float) -> str:
    return f"{n_bytes / (1024 * 1024):.1f} МБ"


# ------------------------------------------------------------ подпроцесс (в)/(г)

def _child_load(dir_str: str, query: str) -> int:
    """Режим --child-load: загрузка одного формата в свежем процессе.
    Печатает В STDOUT ровно одну JSON-строку с замерами."""
    gc.collect()
    # прогрев кэша ОС - одинаковая процедура у обоих форматов
    warm = Bm25Index.load(Path(dir_str))
    n_docs = len(warm.ids)
    del warm
    gc.collect()
    # пик аллокаций самой загрузки (информационная метрика, не гейт)
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()
    idx = Bm25Index.load(Path(dir_str))
    t_traced = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # RSS-прирост загрузки (гейт критерия (б) плана): ОТДЕЛЬНАЯ загрузка без
    # tracemalloc - её оверхед не должен размечать замер; пик RSS сэмплируется
    # в фоновом потоке (GIL отпускается в numpy/PSO вызовах загрузки)
    rss_before = _current_rss_bytes()
    rss_delta = None
    if rss_before is not None:
        peak_box = [rss_before]
        stop = threading.Event()

        def _sample() -> None:
            while not stop.is_set():
                cur = _current_rss_bytes()
                if cur is not None and cur > peak_box[0]:
                    peak_box[0] = cur
                time.sleep(0.003)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()
        rss_idx = Bm25Index.load(Path(dir_str))
        assert len(rss_idx.ids) == n_docs, "загрузка для RSS-замера потеряла доки"
        stop.set()
        sampler.join()
        del rss_idx
        gc.collect()
        rss_delta = peak_box[0] - rss_before
    # скорость: 3 загрузки без трассировки, лучшее время
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        x = Bm25Index.load(Path(dir_str))
        times.append(time.perf_counter() - t0)
        del x
    res = idx.search(query, top_k=10)
    assert res and all(s > 0.0 for _, s in res), "контрольный поиск пуст"
    nnz = int(idx.indptr[len(idx.ids)])
    del idx
    gc.collect()
    print(
        json.dumps(
            {
                "n_docs": n_docs,
                "nnz": nnz,
                "peak_alloc": peak,
                "rss_delta": rss_delta,
                "rss_peak": _peak_rss_bytes(),
                "t_traced": t_traced,
                "t_best": min(times),
                "search": res,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


def _run_child(index_dir: Path, query: str) -> dict:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child-load", str(index_dir), query],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(ROOT_REPO),
    )
    assert proc.returncode == 0, (
        f"подпроцесс загрузки упал (код {proc.returncode}):\n{proc.stderr[-2000:]}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"подпроцесс не напечатал JSON: stderr={proc.stderr[-500:]}"
    return json.loads(lines[-1])


# ------------------------------------------------------------------- секции

def section_bitexact(checks: list[str]) -> None:
    """(а) бит-в-бит CSR против эталонного цикла v1: >= 2000 доков, >= 300 запросов."""
    docs = make_corpus(2200, seed=42)
    ids = [f"d{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)

    idx = Bm25Index()
    idx.add(ids, docs)
    assert len(idx.ids) == len(docs), "не все документы добавлены"
    assert idx.avgdl == avgdl, f"avgdl разошёлся: {idx.avgdl!r} != {avgdl!r}"
    nnz = int(idx.indptr[len(idx.ids)])
    queries = make_queries(320, seed=7)
    assert len(docs) >= 2000 and len(queries) >= 300, "масштаб проверки (а) ниже требуемого"

    nonempty = 0
    for qi, q in enumerate(queries):
        expected = reference_search(tfs, df, doc_lens, ids, avgdl, q)
        got = idx.search(q, top_k=10)
        assert_search_exact(got, expected, f"(а) запрос #{qi} {q!r}")
        nonempty += 1 if got else 0
    checks.append(
        f"(а) бит-в-бит: корпус {len(docs)} доков (nnz={nnz}), {len(queries)} запросов, "
        f"top-10 (id, score) точны; непустых выдач {nonempty}"
    )

    # не-дефолтные k1/b - та же бит-в-бит гарантия на других параметрах
    docs2 = make_corpus(300, seed=5)
    ids2 = [f"e{i}" for i in range(len(docs2))]
    tfs2 = tf_dicts(docs2)
    df2 = df_from(tfs2)
    lens2 = [len(tokenize(t)) for t in docs2]
    avg2 = sum(lens2) / len(lens2)
    idx2 = Bm25Index(k1=1.2, b=0.3)
    idx2.add(ids2, docs2)
    for qi, q in enumerate(make_queries(40, seed=8)):
        assert_search_exact(
            idx2.search(q, top_k=5),
            reference_search(tfs2, df2, lens2, ids2, avg2, q, top_k=5, k1=1.2, b=0.3),
            f"(а) k1=1.2/b=0.3 запрос #{qi}",
        )
    checks.append("(а) не-дефолтные k1=1.2/b=0.3: 300 доков x 40 запросов точны")


def section_roundtrip(rt_dir: Path, checks: list[str]) -> None:
    """(б) payload старого bm25.json -> load -> save npz -> load == эталон."""
    docs = make_corpus(500, seed=101)
    ids = [f"r{i}" for i in range(len(docs))]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    queries = make_queries(80, seed=102)

    # payload старым форматом v1 (tfs/df-словари), запись как в старом save
    (rt_dir / "bm25.json").write_text(
        json.dumps(legacy_payload(tfs, df, ids, doc_lens, avgdl), ensure_ascii=False),
        encoding="utf-8",
    )
    assert not (rt_dir / "bm25.npz").exists()

    loaded = Bm25Index.load(rt_dir)  # npz нет -> старый json
    assert loaded.ids == ids and loaded.avgdl == avgdl, "ids/avgdl не сохранились"
    for qi, q in enumerate(queries):
        assert_search_exact(
            loaded.search(q),
            reference_search(tfs, df, doc_lens, ids, avgdl, q),
            f"(б) json-load запрос #{qi}",
        )
    checks.append(
        f"(б) старый bm25.json -> load: {len(docs)} доков x {len(queries)} запросов == эталон"
    )

    loaded.save(rt_dir)  # апгрейд на месте: npz пишется, синхронный json остаётся
    assert (rt_dir / "bm25.npz").exists(), "save не написал bm25.npz"
    assert (rt_dir / "bm25.json").exists(), (
        "save не оставил синхронный bm25.json - каталог стал односторонним "
        "(ревью раунд 1, находка 2)"
    )
    # v1-читаемость: json парсится старым кодом (json.loads), payload синхронен
    # с данными, поиск по нему бит-в-бит равен поиску нового индекса
    sync_payload = json.loads((rt_dir / "bm25.json").read_text(encoding="utf-8"))
    assert sync_payload["ids"] == ids and sync_payload["df"] == df
    assert sync_payload["tfs"] == tfs, "синхронный bm25.json разошёлся с tfs"
    reloaded = Bm25Index.load(rt_dir)
    assert reloaded.ids == ids
    for qi, q in enumerate(queries):
        expected = reference_search(
            sync_payload["tfs"], sync_payload["df"], sync_payload["doc_lens"],
            sync_payload["ids"], sync_payload["avgdl"], q,
        )
        assert_search_exact(reloaded.search(q), expected, f"(б) round-trip запрос #{qi}")
        assert_search_exact(
            reference_search(tfs, df, doc_lens, ids, avgdl, q), expected,
            f"(б) синхронный json vs исходный payload, запрос #{qi}",
        )
    npz_bytes = (rt_dir / "bm25.npz").stat().st_size
    checks.append(
        f"(б) round-trip npz: bm25.npz {mb(npz_bytes)}, поиск после перезагрузки == эталон; "
        f"синхронный bm25.json читается v1-форматом и даёт тот же поиск"
    )


def section_memory_speed(old_dir: Path, new_dir: Path, checks: list[str]) -> dict:
    """(в) пик аллокаций (tracemalloc) и RSS подпроцессов; (г) время загрузки."""
    n_big = 100_000
    docs = make_corpus(n_big, seed=2026, vocab_size=600)
    ids = [f"n{i}" for i in range(n_big)]
    tfs = tf_dicts(docs)
    df = df_from(tfs)
    doc_lens = [len(tokenize(t)) for t in docs]
    avgdl = sum(doc_lens) / len(doc_lens)
    nnz = sum(len(tf) for tf in tfs)
    print(f"  генерация: {n_big} доков, nnz={nnz}, avgdl={avgdl:.1f}, "
          f"терминов в словаре={len(df)}", flush=True)

    # старый формат: payload v1 тем же кодировщиком, что в старом save
    with open(old_dir / "bm25.json", "w", encoding="utf-8") as f:
        json.dump(legacy_payload(tfs, df, ids, doc_lens, avgdl), f, ensure_ascii=False)
    del docs, tfs, df, doc_lens, ids
    gc.collect()

    # новый формат из ТЕХ ЖЕ данных: старый json -> CSR -> save npz (+синхронный json)
    big = Bm25Index.load(old_dir)  # заодно легаси-загрузка на масштабе 100к
    assert len(big.ids) == n_big, "легаси-загрузка потеряла документы"
    big.save(new_dir)
    assert (new_dir / "bm25.npz").exists()
    assert (new_dir / "bm25.json").exists(), "save не написал синхронный bm25.json"
    del big
    gc.collect()

    json_bytes = (old_dir / "bm25.json").stat().st_size
    npz_bytes = (new_dir / "bm25.npz").stat().st_size
    print(f"  файлы: bm25.json {mb(json_bytes)}, bm25.npz {mb(npz_bytes)}", flush=True)

    query = "общийслово термин003 термин011"
    old = _run_child(old_dir, query)
    new = _run_child(new_dir, query)
    assert old["n_docs"] == n_big and new["n_docs"] == n_big, "ребёнок загрузил не все доки"
    assert old["nnz"] == new["nnz"] == nnz, "nnz расходится между форматами"
    # тот же поиск на одном наборе в двух форматах - идентичен бит-в-бит
    assert old["search"] == new["search"], "поиск старого и нового формата разошёлся на 100к"

    # ГЕЙТ критерия (б) плана: «RSS при загрузке >= 3x ниже». Метрика - прирост
    # пикового RSS процесса при загрузке (см. «Расхождения плана и реальности»
    # в plan.md: полный пик подпроцесса включает базу интерпретатора+numpy,
    # размывающую отношение на 100к узлов)
    assert old["rss_delta"] is not None and new["rss_delta"] is not None, (
        "RSS недоступен (нет psutil и /proc) - критерий (б) плана не измерен"
    )
    rss_ratio = old["rss_delta"] / new["rss_delta"]
    assert rss_ratio >= 3.0, (
        f"RSS при загрузке новый путь всего в {rss_ratio:.2f}x ниже старого "
        f"(прирост {old['rss_delta']} vs {new['rss_delta']} байт), ожидание >= 3x"
    )
    checks.append(
        f"(в) RSS при загрузке 100к узлов (прирост пикового RSS, критерий (б) плана): "
        f"старый +{mb(old['rss_delta'])} vs новый +{mb(new['rss_delta'])} = "
        f"{rss_ratio:.1f}x ниже (порог 3x)"
    )
    # информационные метрики (не гейт): полный пик RSS подпроцесса и пик аллокаций
    if old.get("rss_peak") and new.get("rss_peak"):
        full_ratio = old["rss_peak"] / new["rss_peak"]
        checks.append(
            f"(в) полный пик RSS подпроцессов (инфо, не гейт): старый {mb(old['rss_peak'])} vs "
            f"новый {mb(new['rss_peak'])} = {full_ratio:.1f}x"
        )
    else:
        checks.append("(в) psutil недоступен - полный пик RSS не измерен")
    alloc_ratio = old["peak_alloc"] / new["peak_alloc"]
    checks.append(
        f"(в) пик аллокаций загрузки (tracemalloc, инфо, не гейт): старый {mb(old['peak_alloc'])} vs "
        f"новый {mb(new['peak_alloc'])} = {alloc_ratio:.1f}x ниже"
    )

    assert new["t_best"] < old["t_best"], (
        f"новый формат грузится не быстрее: {new['t_best']:.2f}s >= {old['t_best']:.2f}s"
    )
    tratio = old["t_best"] / new["t_best"]
    checks.append(
        f"(г) время загрузки (лучшее из 3): старый {old['t_best']:.2f}s vs "
        f"новый {new['t_best']:.2f}s = {tratio:.1f}x быстрее"
    )
    return {
        "n_big": n_big, "nnz": nnz, "json_bytes": json_bytes, "npz_bytes": npz_bytes,
        "old": old, "new": new, "rss_ratio": rss_ratio, "alloc_ratio": alloc_ratio,
        "time_ratio": tratio,
    }


def section_mmap(mm_dir: Path, mm_saved_dir: Path, checks: list[str]) -> list:
    """(д) mmap-матрица: поиск == ndarray; add() после mmap работает."""
    from veles.config import EMBED_DIM, EMBED_MODEL
    from veles.indexing.indexer import load_index
    from veles.indexing.vector_store import NumpyFlatStore

    docs = make_corpus(400, seed=303, vocab_size=120)
    ids = [f"v{i}" for i in range(len(docs))]
    rng = np.random.default_rng(303)
    matrix = rng.standard_normal((len(docs), EMBED_DIM)).astype(np.float32)

    np.save(mm_dir / "numpy_matrix.npy", matrix)
    (mm_dir / "numpy_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    (mm_dir / "meta.json").write_text(
        json.dumps(
            {
                "node_meta": {},
                "embed": {"model": EMBED_MODEL, "backend": "numpy-flat"},
                "identifiers": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bm = Bm25Index()
    bm.add(ids, docs)
    bm.save(mm_dir)

    bundle = load_index(mm_dir)  # apply_overlay внутри: оверлея нет - пусто
    assert isinstance(bundle.vector.matrix, np.memmap), (
        f"матрица после load_index не mmap: {type(bundle.vector.matrix)}"
    )
    ref = NumpyFlatStore(dim=EMBED_DIM)
    ref.matrix = np.load(mm_dir / "numpy_matrix.npy")  # обычный ndarray - эталон
    ref.ids = list(ids)
    ref_bm25 = Bm25Index.load(mm_dir)

    q_vecs = [rng.standard_normal(EMBED_DIM).astype(np.float32) for _ in range(25)]
    for qi, qv in enumerate(q_vecs):
        assert bundle.vector.search(qv, top_k=8) == ref.search(qv, top_k=8), (
            f"(д) векторный запрос #{qi}: mmap != ndarray"
        )
    for qi, qtext in enumerate(make_queries(5, seed=304, vocab_size=120)):
        assert bundle.bm25.search(qtext) == ref_bm25.search(qtext), (
            f"(д) bm25-запрос #{qi} на mmap-бандле разошёлся"
        )
    checks.append(
        f"(д) mmap: load_index отдал np.memmap ({bundle.vector.matrix.shape[0]} строк); "
        f"25 векторных + 5 bm25 запросов == обычный ndarray"
    )

    # add() после mmap: материализация и поиск новых доков
    new_texts = [
        "парковка пропуск спецпропуск категория термин003 термин003",
        "кофе переговорная кухня термин017",
    ]
    new_ids = ["m0", "m1"]
    new_vecs = rng.standard_normal((2, EMBED_DIM)).astype(np.float32)
    bundle.vector.add(new_ids, new_vecs)
    bundle.bm25.add(new_ids, new_texts)
    ref.add(new_ids, new_vecs.copy())
    ref_bm25.add(new_ids, new_texts)
    assert len(bundle.vector.matrix) == len(docs) + 2, "add после mmap потерял строки"
    for qi, qv in enumerate(q_vecs[:10]):
        assert bundle.vector.search(qv, top_k=8) == ref.search(qv, top_k=8), (
            f"(д) после add запрос #{qi}: разошёлся"
        )
    top1 = bundle.bm25.search("парковка спецпропуск")[0][0]
    assert top1 == "m0", f"новый док не найден после mmap+add: top1={top1}"
    checks.append("(д) add() после mmap: 402 строки, новые доки ищутся, выдача == эталону")

    # save полной матрицы и рестарт
    bundle.save(mm_saved_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    m = np.load(mm_saved_dir / "numpy_matrix.npy")
    assert isinstance(m, np.ndarray) and not isinstance(m, np.memmap)
    assert m.shape == (len(docs) + 2, EMBED_DIM), f"save после mmap: shape={m.shape}"
    bundle2 = load_index(mm_saved_dir)
    assert isinstance(bundle2.vector.matrix, np.memmap)
    assert bundle2.bm25.search("кофе переговорная")[0][0] == "m1"
    assert bundle2.vector.search(q_vecs[0], top_k=8) == ref.search(q_vecs[0], top_k=8)
    checks.append(
        f"(д) save после mmap: {m.shape} записан, рестарт находит новые доки"
    )
    return [bundle.vector, bundle2.vector]  # держатели mmap - закрыть перед rmtree


# --------------------------------------------------------------------- main

def main() -> int:
    t_start = time.perf_counter()
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    rt_dir = _TMP_ROOT / "roundtrip"
    rt_dir.mkdir(parents=True)
    mm_dir = _TMP_ROOT / "mmap"
    mm_dir.mkdir(parents=True)
    mm_saved_dir = _TMP_ROOT / "mmap-saved"
    old_dir = _TMP_ROOT / "big-old-json"
    old_dir.mkdir(parents=True)
    new_dir = _TMP_ROOT / "big-new-npz"
    new_dir.mkdir(parents=True)

    print("veles index-format: валидация бинарного формата (коммит 62873e1)")
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, repo {ROOT_REPO}",
          flush=True)
    checks: list[str] = []
    mem: dict = {}
    memmap_holders: list = []
    try:
        print("\n=== (а) бит-в-бит: CSR против эталонного цикла v1 ===", flush=True)
        section_bitexact(checks)

        print("\n=== (б) round-trip старого bm25.json ===", flush=True)
        section_roundtrip(rt_dir, checks)

        print("\n=== (в)/(г) память и скорость на 100 тыс. узлов ===", flush=True)
        mem = section_memory_speed(old_dir, new_dir, checks)

        print("\n=== (д) mmap матрицы векторов ===", flush=True)
        memmap_holders = section_mmap(mm_dir, mm_saved_dir, checks)

        print("\nИТОГИ")
        for c in checks:
            print(f"  {c}")
        if mem:
            print(
                f"\nЧисла для отчёта: bm25.json {mb(mem['json_bytes'])} / "
                f"bm25.npz {mb(mem['npz_bytes'])}; узлов {mem['n_big']}, nnz {mem['nnz']}; "
                f"RSS при загрузке +{mb(mem['old']['rss_delta'])} -> "
                f"+{mb(mem['new']['rss_delta'])} ({mem['rss_ratio']:.1f}x, гейт >= 3x); "
                f"пик аллокаций {mb(mem['old']['peak_alloc'])} -> "
                f"{mb(mem['new']['peak_alloc'])} ({mem['alloc_ratio']:.1f}x, инфо); "
                f"время {mem['old']['t_best']:.2f}s -> {mem['new']['t_best']:.2f}s "
                f"({mem['time_ratio']:.1f}x)"
            )
        print(f"\nВсего времени: {time.perf_counter() - t_start:.1f}s")
        print("VALIDATION-PASS", flush=True)
        return 0
    finally:
        for store in memmap_holders:
            try:
                store.close()
            except Exception:
                pass
        for _ in range(5):
            shutil.rmtree(_TMP_ROOT, ignore_errors=True)
            if not _TMP_ROOT.exists():
                break
            time.sleep(1.0)
        if _TMP_ROOT.exists():
            print(f"ПРЕДУПРЕЖДЕНИЕ: не удалось удалить {_TMP_ROOT}", file=sys.stderr)


def test_index_format_validation():
    """Точка входа pytest - тот же прогон, что и «python evals/index-format-validate.py».

    Повторный запуск скрипта через pytest собирал 0 тестов («no tests ran»,
    exit 5): имя файла не матчит дефолтный python_files, тестовых функций в
    файле не было. AssertionError из проверок main() доходит до pytest как
    падение теста; для evals/ сборку этого файла добавляет evals/conftest.py.
    """
    assert main() == 0


if __name__ == "__main__":
    if _CHILD:
        raise SystemExit(_child_load(sys.argv[2], sys.argv[3]))
    try:
        _code = main()
    except AssertionError as e:
        print(f"VALIDATION-FAIL: {e}", file=sys.stderr, flush=True)
        _code = 1
    raise SystemExit(_code)
