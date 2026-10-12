"""Коннектор №1 (JTBD P0-1): синхронизация каталога-источника с каноном.

`veles sync --root <dir>` сверяет каталог-источник (txt/md/pdf) с каноном по
sha256 манифестов; новые файлы ингестятся (ingest_document), изменённые
становятся новой ревизией (там же, K3: старая архивируется), удалённые из
источника - tombstone (delete_document, K10: канон уезжает в trash). Источник
СТРОГО читается: ни один файл под --root не записывается и не удаляется.

Идентичность документа = относительный путь от корня источника (как в
txt_md.py: document_id = rel); цитаты ссылаются в исходное место через
Provenance.source_id.

Границы безопасности:
- симлинки и реперс-точки (junction) не переходятся и не следуются: запись в
  отчёт (skipped), в канон не попадают. На Windows junction - не is_symlink()
  и os.walk(followlinks=False) по ней УХОДИТ В ЦИКЛ, поэтому каталог
  вырезается и при выходе resolve() за корень, и при резолве в УЖЕ ПРОЙДЕННЫЙ
  каталог (junction-петля на собственный корень - бесконечный обход и
  дубли-документы);
- sync распоряжается только своими документами. Владение - маркер
  `fsync:<корень>:<rel>` в derived_from. Обновление чужого документа (у
  корпусных document_id - тоже rel-путь) запрещено так же, как удаление:
  ingest_document архивировал бы корпусную ревизию, затирал бы derived_from,
  а последующее удаление файла унесло бы весь doc_dir чужих ревизий в trash.
  Коллизия _doc_dir (store.py схлопывает 'a/b.md' и 'a_b.md' в один каталог,
  на NTFS ещё и по регистру) отсекается на плане: второй файл ждёт skipped,
  а не молчаливая перезапись чужого каталога;
- файлы, не читающиеся как UTF-8 (частый случай - CP1251), пропускаются с
  записью в отчёт: парсер молча заменил бы байты на U+FFFD (mojibake);
- параллельный sync двух процессов отсечён файловым локом sync_lock()
  (veles/locks.py: msvcrt.locking на Windows / fcntl.flock на POSIX; ОС
  снимает блокировку при смерти процесса - stale-локов не остаётся).

Живой индекс: если рядом с каноном собран индекс (meta.json существует),
добавление/обновление дописываются в живой бандл и overlay (upsert_document +
append_overlay - тот же путь, что у `veles add`), удаление - tombstone
(tombstone_document, как у `veles delete`). Индекса нет - только канон.

Самолечение индекса: дельта канон/индекс не атомарна - падение между
ingest_document и append_overlay (или между delete_document и
tombstone_document) оставляет канон обновлённым, а индекс устаревшим, и
повторный sync по sha-совпадению увидел бы 0 дельты. Поэтому каждое
мутацию канона предваряет запись намерения в журнал <VELES_DATA>/
index-repair.jsonl (write-ahead), успешная догонка индекса закрывает его;
следующий sync доводит незакрытые намерения (repair): недогнанный upsert
повторяется идемпотентно (узлы текущей ревизии уже в индексе - вектор
не дублируется), недогнанный tombstone - повторяется (patch-only).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..locks import file_lock

# Форматы коннектора №1 (бриф connektor-papka); pdf - существующим docling-конвейером
SYNC_EXTENSIONS = {".txt", ".md", ".pdf"}

REASON_SYMLINK = "симлинк"
REASON_OUTSIDE = "вне корня"
REASON_NOT_FILE = "не файл"
REASON_FORMAT = "формат не поддержан"
REASON_LOOP = "цикл ссылки"
REASON_FOREIGN = "чужой документ"
REASON_COLLISION = "коллизия каталога канона"
REASON_ENCODING = "не UTF-8"

# Префикс маркера владения в derived_from (см. докстринг модуля)
SYNC_MARK_PREFIX = "fsync:"

_SYNC_LOCK_NAME = "sync.lock"
_REPAIR_LOG_NAME = "index-repair.jsonl"
_CHUNK = 1024 * 1024


@dataclass
class SourceFile:
    """Файл источника: document_id (rel), путь, sha256 сырых байтов."""

    rel: str
    path: Path
    sha256: str


@dataclass
class SkippedEntry:
    """Пропущенный элемент обхода: относительный путь и причина."""

    rel: str
    reason: str


@dataclass
class SyncPlan:
    """Дельта каталог-источник против канона (до применения)."""

    added: list[SourceFile] = field(default_factory=list)
    updated: list[SourceFile] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)

    @property
    def has_delta(self) -> bool:
        return bool(self.added or self.updated or self.deleted)


@dataclass
class SyncReport:
    """Итог применения дельты (человекочитаемый отчёт CLI + тесты)."""

    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    repaired: list[str] = field(default_factory=list)
    tombstoned_nodes: int = 0
    errors: list[str] = field(default_factory=list)


def _veles_data() -> Path:
    """Корень рантайм-данных: env читается на вызов (как uploads_root/trash_root),
    чтобы тесты подменяли окружение без переимпорта конфига."""
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        first = raw.split(os.pathsep)[0].strip()
        if first:
            return Path(first)
    from ..config import VELES_DATA

    return Path(VELES_DATA)


def sha256_file(path: Path) -> str:
    """sha256 сырых байтов файла, потоково (большие PDF не поднимаются в RAM)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_token(root: Path) -> str:
    """Нормализованный корень для маркера владения: resolve + normcase
    (на Windows регистр и разделители пути не должны менять владельца)."""
    return os.path.normcase(str(Path(root).resolve())).replace(os.sep, "/")


def sync_marker(root: Path, rel: str) -> str:
    """Маркер владения документом: fsync:<корень>:<отн. путь> в derived_from."""
    return f"{SYNC_MARK_PREFIX}{root_token(root)}:{rel}"


def sync_owner_prefix(root: Path) -> str:
    """Префикс, по которому манифест признаётся созданным sync ЭТОГО корня."""
    return f"{SYNC_MARK_PREFIX}{root_token(root)}:"


def _owns_manifest(manifest_like: dict, prefix: str) -> bool:
    """Манифест создан sync ЭТОГО корня (маркер в derived_from)?"""
    return any(
        isinstance(x, str) and x.startswith(prefix)
        for x in manifest_like.get("derived_from") or []
    )


def _sha256_and_utf8(path: Path, ext: str) -> tuple[str, bool]:
    """sha256 сырых байтов + читается ли файл как UTF-8 (только txt/md - у PDF
    кодировкой текста управляет docling). Один проход чтения."""
    digest = hashlib.sha256()
    if ext in {".txt", ".md"}:
        data = path.read_bytes()
        digest.update(data)
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return digest.hexdigest(), False
        return digest.hexdigest(), True
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest(), True


def _classify(root: Path, path: Path) -> SourceFile | SkippedEntry:
    """Файл обхода -> SourceFile или SkippedEntry с причиной."""
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        return SkippedEntry(str(path), REASON_OUTSIDE)
    if path.is_symlink():
        return SkippedEntry(rel, REASON_SYMLINK)
    if not path.is_file():
        return SkippedEntry(rel, REASON_NOT_FILE)
    # defense-in-depth к обходу: даже обычный файл обязан резолвиться внутрь
    # корня (файл-симлинк без атрибута, реперс-точка на родителе и т.п.)
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return SkippedEntry(rel, REASON_OUTSIDE)
    ext = path.suffix.lower()
    if ext not in SYNC_EXTENSIONS:
        return SkippedEntry(rel, REASON_FORMAT)
    sha256, utf8_ok = _sha256_and_utf8(path, ext)
    if not utf8_ok:
        return SkippedEntry(rel, REASON_ENCODING)
    return SourceFile(rel=rel, path=path, sha256=sha256)


def scan_source(root: Path) -> tuple[dict[str, SourceFile], list[SkippedEntry]]:
    """Обход каталога-источника (txt/md/pdf), детерминированный (sorted).

    Симлинки-файлы, каталоги вне корня и каталоги-петли (resolve в уже
    пройденный каталог: junction на собственный корень циклит os.walk и
    плодит дубли) вырезаются из обхода и попадают в skipped: в канон они не
    попадают никогда. Источник только читается."""
    root = Path(root).resolve()
    files: dict[str, SourceFile] = {}
    skipped: list[SkippedEntry] = []
    seen_dirs: set[str] = {os.path.normcase(str(root))}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        keep: list[str] = []
        for name in sorted(dirnames):
            child = here / name
            entry = _classify(root, child) if child.is_symlink() else None
            if entry is None:
                try:
                    child.resolve().relative_to(root)
                except ValueError:
                    entry = SkippedEntry(
                        child.relative_to(root).as_posix(), REASON_OUTSIDE
                    )
            if entry is not None:
                skipped.append(entry)  # симлинк/вне корня: не разворачиваем
            else:
                token = os.path.normcase(str(child.resolve()))
                if token in seen_dirs:
                    skipped.append(
                        SkippedEntry(
                            child.relative_to(root).as_posix(), REASON_LOOP
                        )
                    )
                else:
                    seen_dirs.add(token)
                    keep.append(name)
        dirnames[:] = keep  # обрезка: os.walk не спустится в вырезанное
        for name in sorted(filenames):
            result = _classify(root, here / name)
            if isinstance(result, SkippedEntry):
                skipped.append(result)
            else:
                files[result.rel] = result
    return files, skipped


def canonical_state() -> tuple[dict[str, dict], set[str]]:
    """(document_id -> срез текущего манифеста, занятые имена каталогов канона)
    по всем корням канона (первый каталог побеждает - как у iter_documents).
    Имена каталогов - ВСЕ каталоги канона, включая нечитаемые манифесты:
    коллизия с ними тоже опасна (save_document перезаписал бы молча).
    Архивные ревизии (revision-r*/) не читаются: дельта по текущим манифестам."""
    from ..canonical.store import canonical_roots

    state: dict[str, dict] = {}
    occupied: set[str] = set()
    for canon in canonical_roots():
        canon = Path(canon)
        if not canon.is_dir():
            continue
        for d in sorted(canon.iterdir()):
            if not d.is_dir():
                continue
            occupied.add(os.path.normcase(d.name))
            mpath = d / "manifest.json"
            if not mpath.exists():
                continue
            try:
                manifest = json.loads(mpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # noqa: BLE001 - битый манифест пропускаем
                continue
            doc_id = str(manifest.get("document_id") or "")
            if not doc_id or doc_id in state:
                continue
            state[doc_id] = {
                "sha256": str(manifest.get("sha256") or ""),
                "derived_from": list(manifest.get("derived_from") or []),
                "revision_id": str(manifest.get("revision_id") or "r1"),
                "dir": d.name,
            }
    return state, occupied


def _dir_key(name: str) -> str:
    """Ключ сравнения имён каталогов канона: NTFS регистронезависим."""
    return os.path.normcase(name)


def build_plan(root: Path) -> SyncPlan:
    """Дельта каталог-источник против канона.

    Добавление/обновление - по sha256 текущих манифестов (идентичность =
    относительный путь). Дополнительные заслоны:
    - обновление чужого документа (без маркера владения ЭТОГО корня) запрещено:
      уходит в skipped (см. докстринг модуля - почему обновление опасно так же,
      как удаление);
    - добавление с каталогом канона, занятым другим документом (_doc_dir
      схлопывает 'a/b.md' и 'a_b.md', на NTFS и по регистру), уходит в skipped.
    Удаление - ТОЛЬКО документы с маркером владения этого корня."""
    from ..canonical.store import _doc_dir

    root = Path(root).resolve()
    files, skipped = scan_source(root)
    plan = SyncPlan(skipped=skipped)
    state, occupied = canonical_state()
    occupied_owner: dict[str, str] = {
        _dir_key(m["dir"]): doc_id for doc_id, m in state.items()
    }
    owned_dirs: dict[str, str] = {}  # ключ каталога -> rel, добавляемый этим sync
    for rel in sorted(files):
        sf = files[rel]
        known = state.get(rel)
        if known is None:
            key = _dir_key(_doc_dir(rel).name)
            other = owned_dirs.get(key) or occupied_owner.get(key)
            if other is not None:
                plan.skipped.append(
                    SkippedEntry(rel, f"{REASON_COLLISION}: каталог занят {other!r}")
                )
                continue
            owned_dirs[key] = rel
            plan.added.append(sf)
        elif known["sha256"] == sf.sha256:
            plan.unchanged.append(rel)
        elif _owns_manifest(known, sync_owner_prefix(root)):
            plan.updated.append(sf)
        else:
            plan.skipped.append(SkippedEntry(rel, REASON_FOREIGN))
    prefix = sync_owner_prefix(root)
    for doc_id in sorted(state):
        if doc_id in files:
            continue
        if not _owns_manifest(state[doc_id], prefix):
            continue
        plan.deleted.append(doc_id)
    return plan


# ------------------- журнал намерений «канон обновлён - догони индекс» --------


def _repair_log_path() -> Path:
    return _veles_data() / _REPAIR_LOG_NAME


def _append_repair(document_id: str, op: str, done: bool,
                   root: Path, error: str = "") -> None:
    """Намерение в журнал (write-ahead, best-effort: отказ журнала не валит
    sync - в худшем случае расхождение найдёт полная пересборка)."""
    try:
        path = _repair_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "document_id": document_id,
            "op": op,
            "root": root_token(root),
            "done": done,
            "error": error[:300],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:  # noqa: BLE001
        pass


def _repair_events() -> list[dict]:
    try:
        raw = _repair_log_path().read_text(encoding="utf-8")
    except OSError:
        return []
    events: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("document_id") and event.get("op"):
            events.append(event)
    return events


def _pending_repairs(root: Path) -> dict[tuple[str, str], dict]:
    """Незакрытые намерения ЭТОГО корня: последнее событие на (document_id, op)."""
    token = root_token(root)
    last: dict[tuple[str, str], dict] = {}
    for event in _repair_events():
        if event.get("root") != token:
            continue
        last[(str(event["document_id"]), str(event["op"]))] = event
    return {k: v for k, v in last.items() if not v.get("done")}


def _compact_repair_log() -> None:
    """Переписать журнал, оставив по последнему событию на (document_id, op)
    и только незакрытые: закрытые намерения и история исчезают."""
    last: dict[tuple[str, str], dict] = {}
    for event in _repair_events():
        last[(str(event["document_id"]), str(event["op"]))] = event
    keep = [v for v in last.values() if not v.get("done")]
    try:
        path = _repair_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for event in keep:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        tmp.replace(path)
    except OSError:  # noqa: BLE001 - журнал не критичен
        pass


def apply_plan(
    plan: SyncPlan,
    root: Path,
    *,
    by: str = "",
    index_dir: Path | None = None,
    embedder_factory=None,
) -> SyncReport:
    """Применить дельту к канону и живому индексу.

    Добавление/обновление: parse (parse_document_file: txt/md/pdf) ->
    ingest_document (создание или новая ревизия, K3) -> апсерт живого бандла +
    overlay (upsert_document + append_overlay - как cmd_add). Удаление:
    delete_document (канон -> trash, K10) -> tombstone_document (узлы вон из
    поиска, как cmd_delete без bundle). Индекса нет - только канон. Источник
    только читается. Ошибка одного файла уходит в report.errors, sync
    продолжается. Каждая мутация канона под журналом намерений; незакрытые
    намерения прошлых прогонов доводятся до индекса в начале применения
    (падение между каноном и индексом самолечится следующим sync)."""
    from ..canonical.store import delete_document, ingest_document
    from .queue import parse_document_file, upsert_document

    root = Path(root).resolve()
    report = SyncReport()
    index_dir = Path(index_dir) if index_dir is not None else None
    index_live = bool(index_dir and (index_dir / "meta.json").exists())
    bundle = embedder = None

    if index_live:
        pending = _pending_repairs(root)
        live_docs = (
            {sf.rel for sf in plan.added}
            | {sf.rel for sf in plan.updated}
            | set(plan.deleted)
        )
        repairable = {k: v for k, v in pending.items() if k[0] not in live_docs}
        if any(op == "upsert" for op, _ in repairable) and bundle is None:
            from ..indexing.indexer import load_index

            bundle = load_index(index_dir)
        bundle_ref = [bundle]
        report.errors.extend(
            _repair_pending_index(
                root, index_dir, repairable, bundle_ref,
                embedder_factory, report,
            )
        )
        bundle = bundle_ref[0]

    if index_live and (plan.added or plan.updated):
        if bundle is None:
            from ..indexing.indexer import load_index

            bundle = load_index(index_dir)

    added_rels = {sf.rel for sf in plan.added}
    upserts = sorted(plan.added, key=lambda s: s.rel) + sorted(
        plan.updated, key=lambda s: s.rel
    )
    for sf in upserts:
        if index_live:
            # намерение ДО мутации канона: убил процесс между ingest_document
            # и append_overlay - следующий sync доведёт индекс по журналу
            _append_repair(sf.rel, "upsert", False, root)
        try:
            doc = parse_document_file(sf.path, root, sf.path.suffix.lower())
            if sf.path.suffix.lower() in {".txt", ".md"}:
                try:  # гонка: файл сменился после скана на не-UTF-8
                    sf.path.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(REASON_ENCODING) from None
            # владелец дельты удаления - этот sync (см. build_plan): маркер
            # перезаписывает парсерский derived_from осознанно
            doc.derived_from = [sync_marker(root, sf.rel)]
            outcome = ingest_document(doc, by=by or None)
        except Exception as e:  # noqa: BLE001 - один битый файл не валит sync
            if index_live:
                # канон не изменился - ремонтировать нечего, намерение закрываем
                _append_repair(sf.rel, "upsert", True, root, error=f"канон не изменён: {e}")
            report.errors.append(f"{sf.rel}: не удалось синхронизировать: {e}")
            continue
        if outcome.action != "unchanged" and bundle is not None:
            try:
                if embedder is None:
                    if embedder_factory is None:
                        raise RuntimeError(
                            "индекс жив, но фабрика эмбеддера не передана"
                        )
                    embedder = embedder_factory()
                delta = upsert_document(bundle, doc, outcome, embedder)
                if delta is not None:
                    from ..indexing.overlay import append_overlay

                    append_overlay(
                        index_dir, delta.nodes, delta.patches,
                        delta.ids, delta.matrix, delta.bm25,
                    )
            except Exception as e:  # noqa: BLE001 - канон уже обновлён
                report.errors.append(
                    f"{sf.rel}: документ в каноне, но индекс не обновлён: {e}"
                )
                continue  # намерение остаётся открытым - ремонт следующим sync
        if index_live:
            _append_repair(sf.rel, "upsert", True, root)
        if sf.rel in added_rels:
            report.added.append(sf.rel)
        else:
            report.updated.append(sf.rel)

    for doc_id in sorted(plan.deleted):
        if index_live:
            _append_repair(doc_id, "tombstone", False, root)
        try:
            info = delete_document(doc_id, by=by or None)
        except Exception as e:  # noqa: BLE001 - неудача удаления не валит sync
            report.errors.append(f"{doc_id}: удаление не удалось: {e}")
            continue  # намерение открыто: если канон всё же переехал - сотрут
        if info is None:  # манифест видели, канон исчез (гонка)
            if index_live:
                _append_repair(doc_id, "tombstone", True, root, error="уже удалён (гонка)")
            report.errors.append(f"{doc_id}: документа в каноне уже нет")
            continue
        if index_live:
            try:
                from ..indexing.overlay import tombstone_document

                report.tombstoned_nodes += tombstone_document(index_dir, doc_id)
            except Exception as e:  # noqa: BLE001 - канон уже удалён
                report.errors.append(
                    f"{doc_id}: канон удалён, tombstone индекса не записан: {e}"
                )
                continue  # намерение остаётся открытым - ремонт следующим sync
            _append_repair(doc_id, "tombstone", True, root)
        report.deleted.append(doc_id)

    _compact_repair_log()
    return report


def _repair_pending_index(
    root: Path,
    index_dir: Path,
    repairable: dict[tuple[str, str], dict],
    bundle_ref: list,
    embedder_factory,
    report: SyncReport,
) -> list[str]:
    """Довести незакрытые намерения прошлых sync до живого индекса (сценарий
    падения «канон обновлён, индекс нет»). upsert повторяется идемпотентно:
    если узлы текущей ревизии уже в индексе (мета+overlay), векторы не
    дублируются; tombstone - patch-only, безопасен к повтору. Документы из
    текущей дельты не ремонтируются - их закроет свежий путь применения."""
    from ..canonical.store import IngestOutcome, find_document_manifest
    from ..indexing.overlay import _index_node_ids
    from .queue import parse_document_file, upsert_document

    errors: list[str] = []
    bundle = bundle_ref[0]
    embedder = None
    for (doc_id, op) in sorted(repairable):
        manifest = find_document_manifest(doc_id)
        if op == "upsert":
            if manifest is None:
                # канона нет: либо убили ДО ingest_document (создавать нечего),
                # либо документ потом удалили - tombstone-намерение это закроет
                _append_repair(doc_id, "upsert", True, root, "документа нет в каноне")
                continue
            src = root / doc_id
            if not src.is_file():
                _append_repair(doc_id, "upsert", True, root, "файл источника исчез - закроет дельта")
                continue
            try:
                doc = parse_document_file(src, root, src.suffix.lower())
            except Exception as e:  # noqa: BLE001 - намерение остаётся открытым
                errors.append(f"ремонт индекса {doc_id}: разбор не удался: {e}")
                continue
            if doc.sha256 != (manifest.get("sha256") or ""):
                _append_repair(doc_id, "upsert", True, root, "файл изменился - закроет свежая дельта")
                continue
            expected = {n.node_version_id for n in doc.nodes if n.node_type != "root"}
            if expected and expected <= set(_index_node_ids(index_dir, doc_id)):
                _append_repair(doc_id, "upsert", True, root, "узлы уже в индексе")
                continue
            try:
                if bundle is None:
                    from ..indexing.indexer import load_index

                    bundle = load_index(index_dir)
                    bundle_ref[0] = bundle
                if embedder is None:
                    if embedder_factory is None:
                        raise RuntimeError("фабрика эмбеддера не передана")
                    embedder = embedder_factory()
                rev = str(manifest.get("revision_id") or "r1")
                prev = manifest.get("previous_revision") or None
                # node_version_id считается от ревизии - приводим парс к канону
                doc.retag_revision(rev)
                doc.status = "current"
                doc.previous_revision = prev
                doc.derived_from = [sync_marker(root, doc_id)]
                outcome = IngestOutcome("updated" if prev else "created", rev, prev)
                delta = upsert_document(bundle, doc, outcome, embedder)
                if delta is not None:
                    from ..indexing.overlay import append_overlay

                    append_overlay(
                        index_dir, delta.nodes, delta.patches,
                        delta.ids, delta.matrix, delta.bm25,
                    )
            except Exception as e:  # noqa: BLE001 - намерение остаётся открытым
                errors.append(f"ремонт индекса {doc_id}: не удалось: {e}")
                continue
            _append_repair(doc_id, "upsert", True, root)
            report.repaired.append(doc_id)
        elif op == "tombstone":
            if manifest is not None:
                # документ снова в каноне (переиздан) - текущее состояние прав
                _append_repair(doc_id, "tombstone", True, root, "документ вернулся в канон")
                continue
            try:
                from ..indexing.overlay import tombstone_document

                report.tombstoned_nodes += tombstone_document(index_dir, doc_id)
            except Exception as e:  # noqa: BLE001 - намерение остаётся открытым
                errors.append(f"ремонт индекса {doc_id}: tombstone не удался: {e}")
                continue
            _append_repair(doc_id, "tombstone", True, root)
            report.repaired.append(doc_id)
    return errors


@contextlib.contextmanager
def sync_lock(timeout_s: float = 0.0):
    """Межпроцессный файловый лок от параллельного `veles sync`.

    <VELES_DATA>/sync.lock эксклюзивно; занят - TimeoutError (CLI честно
    откажется: busy - это отказ, а не ожидание). Механизм общий с overlay
    (veles/locks.py): блокировка уровня ОС, при смерти процесса снимается
    сама. В lock-файле остаётся pid владельца - только диагностика."""
    path = _veles_data() / _SYNC_LOCK_NAME
    with file_lock(path, timeout_s=timeout_s):
        try:
            with path.open("r+b") as f:
                f.truncate()
                f.write(str(os.getpid()).encode("ascii"))
        except OSError:  # noqa: BLE001 - диагностика не обязательна
            pass
        yield True
