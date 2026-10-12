"""Сквозная валидация коннектора №1 `veles sync` (фаза 2 плана connektor-papka).

Сценарий заказчика (JTBD P0-1) на живом коде, без моков: временные VELES_DATA и
каталог-источник с тремя txt/md-файлами; настоящий эмбеддер
(intfloat/multilingual-e5-small, продовая модель индекса), настоящий faiss-бэкенд,
настоящий CLI-путь cmd_sync (лок + план + применение), гибридный поиск по
СВЕЖЕЙ загрузке индекса с диска (load_index: применяет overlay - семантика
«рестарта фасада», не память процесса).

Проверяемые шаги (все - assertions, финал - строка VALIDATION-PASS):
  (а) первый sync: 3 файла -> 3 документа в каноне, гибридный поиск находит
      различимое содержимое каждого;
  (б) правка файла + sync: дельта обновлено=1, ревизия r2 (r1 в архиве),
      поиск находит НОВЫЙ текст и больше НЕ находит старый;
  (в) удаление файла + sync: дельта удалено=1, канон в trash, документ
      больше НЕ находит (tombstone), аудит-событие delete;
  (г) повторный sync без изменений: ноль дельты, ни одного нового события
      аудита.
Плюс сквозной инвариант: каталог-источник ТОЛЬКО читается (sha256 всех
файлов не меняется sync'ами; меняются только нашей правкой шага б).

Запуск (из корня репозитория): python -u evals/source-sync-validate.py
Скрипт сам создаёт временные каталоги (system temp) и убирает за собой;
машинных путей в коде нет.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Временные VELES_DATA и каталог-источник ДО импорта veles: конфиг читает env
# на импорте (VELES_DATA/INDEX_DIR), sync/store/queue - на вызов.
_TMP = Path(tempfile.mkdtemp(prefix="veles-sync-validate-"))
DATA = _TMP / "veles-data"
SOURCE = _TMP / "источник"
SOURCE.mkdir(parents=True)
DATA.mkdir(parents=True)
os.environ["VELES_DATA"] = str(DATA)

from veles.canonical.store import _doc_dir, iter_documents, trash_root  # noqa: E402
from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import build_index, load_index  # noqa: E402
from veles.indexing.vector_store import make_store  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

# Различимое содержимое трёх файлов (маркерные слова не пересекаются между
# файлами; правка v1->v2 меняет только один узел файла 1)
F1_NAME, F2_NAME, F3_NAME = "otchet-q3.md", "prikaz-otpuska.txt", "instrukciya-vpn.md"
F1_V1 = (
    "# Отчёт за третий квартал\n\n"
    "Продажи медных чайников выросли на двенадцать процентов "
    "благодаря выставке в Сочи.\n"
)
F1_V2 = F1_V1.replace(
    "выросли на двенадцать процентов", "упали на семь процентов"
)
F2 = (
    "Приказ об отпусках\n\n"
    "Ежегодный отпуск оформляется заявкой в кадровой системе "
    "не позднее чем за две недели до начала.\n"
)
F3 = (
    "# Инструкция: удалённый доступ\n\n"
    "Подключение к корпоративному VPN выполняется клиентом WireGuard, "
    "сервер vpn.example.local.\n"
)
Q1 = "как изменились продажи медных чайников"
Q2 = "за сколько дней оформлять ежегодный отпуск"
Q3 = "через какого клиента подключаться к корпоративному vpn"
MARK_1_V1, MARK_1_V2 = "двенадцать процентов", "семь процентов"
MARK_2, MARK_3 = "кадровой системе", "WireGuard"

_DELTA_RE = re.compile(
    r"дельта: добавлено=(\d+), обновлено=(\d+), удалено=(\d+), "
    r"без изменений=(\d+), пропущено=(\d+)"
)
# итоговая строка разбирается по парам «поле=число»: формат отчёта CLI
# обрастает полями (доведено до индекса и т.п.) и не должен ломать eval
_DONE_RE = re.compile(r"синхронизация завершена: (.+)")
_DONE_FIELDS = (
    "добавлено",
    "обновлено",
    "удалено",
    "доведено до индекса",
    "узлов выведено из индекса",
    "ошибок",
)


def step(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


def run_sync(*flags: str) -> tuple[int, str]:
    """Настоящий CLI: veles sync --root <источник> [флаги] (в процессе)."""
    from veles.cli import main as cli_main

    buf = io.StringIO()
    old_argv = sys.argv
    sys.argv = ["veles", "sync", "--root", str(SOURCE), *flags]
    try:
        with contextlib.redirect_stdout(buf):
            rc = cli_main()
    finally:
        sys.argv = old_argv
    out = buf.getvalue()
    print(out, end="", flush=True)
    return rc, out


def delta_of(out: str) -> tuple[int, int, int, int, int]:
    m = _DELTA_RE.search(out)
    assert m, f"в выводе нет строки дельты:\n{out}"
    return tuple(int(g) for g in m.groups())  # type: ignore[return-value]


def done_of(out: str) -> tuple[int, int, int, int, int]:
    """(добавлено, обновлено, удалено, узлов выведено из индекса, ошибок)."""
    m = _DONE_RE.search(out)
    assert m, f"в выводе нет итоговой строки синхронизации:\n{out}"
    pairs = dict(re.findall(rf"({'|'.join(_DONE_FIELDS)})\s*[=:]\s*(\d+)", m.group(1)))
    assert "добавлено" in pairs and "ошибок" in pairs, (
        f"в итоговой строке нет обязательных полей:\n{m.group(1)}"
    )
    return (
        int(pairs["добавлено"]),
        int(pairs["обновлено"]),
        int(pairs["удалено"]),
        int(pairs.get("узлов выведено из индекса", "0")),
        int(pairs["ошибок"]),
    )


_EMBEDDER = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)


def search_hits(query: str) -> list[str]:
    """Гибридный поиск по СВЕЖЕЙ загрузке индекса с диска (рестарт-семантика)."""
    bundle = load_index(INDEX_DIR)
    qvec = _EMBEDDER.embed_query(query)
    return [e.text for e in hybrid_search(bundle, qvec, query, top_k=8)]


def assert_found(mark: str, query: str, what: str) -> None:
    hits = search_hits(query)
    assert any(mark in t for t in hits), f"{what}: маркер {mark!r} не найден: {hits}"
    print(f"OK: {what} - найдено ({mark!r}, запрос {query!r})", flush=True)


def assert_not_found(mark: str, query: str, what: str) -> None:
    hits = search_hits(query)
    assert not any(mark in t for t in hits), f"{what}: маркер {mark!r} найден: {hits}"
    print(f"OK: {what} - не найдено ({mark!r}, запрос {query!r})", flush=True)


def canon_doc_ids() -> set[str]:
    return {d.document_id for d in iter_documents()}


def audit_events() -> list[dict]:
    path = DATA / "canonical" / "ingest-audit.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_sha256() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(SOURCE.iterdir())
        if p.is_file()
    }


def write_source(name: str, text: str) -> None:
    (SOURCE / name).write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    baseline_sha = None
    try:
        step("подготовка: 3 файла источника, пустой индекс (паттерн cmd_add)")
        write_source(F1_NAME, F1_V1)
        write_source(F2_NAME, F2)
        write_source(F3_NAME, F3)
        baseline_sha = source_sha256()
        assert not (INDEX_DIR / "meta.json").exists()
        store = make_store()  # продовый выбор бэкенда (faiss, fallback numpy)
        bundle = build_index(_EMBEDDER, store)
        bundle.save(INDEX_DIR, embed_meta={"model": EMBED_MODEL, "backend": store.name})
        assert (INDEX_DIR / "meta.json").is_file()
        print(f"OK: пустой индекс собран и сохранён -> {INDEX_DIR}", flush=True)

        # ------------------------------------------------ (а) первый sync
        step("(а) первый sync --yes: 3 новых файла")
        rc, out = run_sync("--yes")
        assert rc == 0, f"код выхода {rc}"
        assert delta_of(out) == (3, 0, 0, 0, 0), out
        added, updated, deleted, tomb, errors = done_of(out)
        assert (added, updated, deleted, errors) == (3, 0, 0, 0), out
        assert canon_doc_ids() == {F1_NAME, F2_NAME, F3_NAME}, canon_doc_ids()
        assert_found(MARK_1_V1, Q1, f"(а) новый {F1_NAME}")
        assert_found(MARK_2, Q2, f"(а) новый {F2_NAME}")
        assert_found(MARK_3, Q3, f"(а) новый {F3_NAME}")
        assert source_sha256() == baseline_sha, "источник изменился первым sync'ом"

        # --------------------------------------- (б) правка файла -> ревизия r2
        step("(б) правка файла 1 + sync --yes: ревизия r2")
        write_source(F1_NAME, F1_V2)
        rc, out = run_sync("--yes")
        assert rc == 0, f"код выхода {rc}"
        assert delta_of(out) == (0, 1, 0, 2, 0), out
        added, updated, deleted, tomb, errors = done_of(out)
        assert (added, updated, deleted, errors) == (0, 1, 0, 0), out
        manifest = json.loads(
            (_doc_dir(F1_NAME) / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["revision_id"] == "r2", manifest["revision_id"]
        assert manifest["previous_revision"] == "r1", manifest
        assert (_doc_dir(F1_NAME) / "revision-r1" / "nodes.jsonl").is_file(), (
            "архив r1 неполон"
        )
        assert manifest["sha256"] == hashlib.sha256(F1_V2.encode("utf-8")).hexdigest()
        assert_found(MARK_1_V2, Q1, "(б) новый текст r2")
        assert_not_found(MARK_1_V1, Q1, "(б) старый текст r1 выведен")
        assert_found(MARK_3, Q3, "(б) соседний файл 3 не задет")
        assert source_sha256() == baseline_sha | {F1_NAME: hashlib.sha256(F1_V2.encode("utf-8")).hexdigest()}

        # ------------------------------------- (в) удаление файла -> tombstone
        step("(в) удаление файла 2 + sync --yes: tombstone")
        (SOURCE / F2_NAME).unlink()
        rc, out = run_sync("--yes")
        assert rc == 0, f"код выхода {rc}"
        assert delta_of(out) == (0, 0, 1, 2, 0), out
        added, updated, deleted, tomb, errors = done_of(out)
        assert (added, updated, deleted, errors) == (0, 0, 1, 0), out
        assert tomb > 0, f"узлы не выведены из индекса: {tomb}"
        assert canon_doc_ids() == {F1_NAME, F3_NAME}, canon_doc_ids()
        assert not _doc_dir(F2_NAME).exists(), "канон удалённого остался"
        trash_entries = list(trash_root().iterdir())
        assert len(trash_entries) == 1, trash_entries
        trashed = json.loads(
            (trash_entries[0] / "manifest.json").read_text(encoding="utf-8")
        )
        assert trashed["document_id"] == F2_NAME, trashed
        del_events = [e for e in audit_events() if e.get("event") == "delete"]
        assert del_events and del_events[-1]["document_id"] == F2_NAME, del_events
        assert_not_found(MARK_2, Q2, "(в) удалённый файл 2")
        assert_found(MARK_1_V2, Q1, "(в) файл 1 не задет")
        assert_found(MARK_3, Q3, "(в) файл 3 не задет")
        rest = source_sha256()
        assert rest == {
            F1_NAME: hashlib.sha256(F1_V2.encode("utf-8")).hexdigest(),
            F3_NAME: baseline_sha[F3_NAME],
        }, rest

        # ------------------------------------------- (г) повторный sync: 0 дельты
        # (семантика самолечения: apply запускается и на нулевой дельте -
        # доводит до индекса незакрытые намерения; считаем, что все счётчики
        # по нулям и ни одного нового события в журнале нет)
        step("(г) повторный sync --yes без изменений: ноль дельты")
        events_before = audit_events()
        rc, out = run_sync("--yes")
        assert rc == 0, f"код выхода {rc}"
        assert delta_of(out) == (0, 0, 0, 2, 0), out
        assert done_of(out) == (0, 0, 0, 0, 0), out  # применено ровно ничего
        assert audit_events() == events_before, "аудит дописался на нулевой дельте"
        assert_found(MARK_1_V2, Q1, "(г) файл 1 ищется после холостого sync")
        assert_not_found(MARK_2, Q2, "(г) удалённый не вернулся")
        assert source_sha256() == rest, "источник изменился без правок"

        step("все проверки пройдены")
        print("VALIDATION-PASS", flush=True)
        return 0
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
        print(f"временные каталоги вычищены: {_TMP}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
