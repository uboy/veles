"""Очередь пользовательской загрузки документов (K4): джобы, воркер, апсерт.

Одна последовательная фоновая очередь ВНУТРИ фасада (парс тяжёлый, OMP-треды):
POST /v1/documents ставит джоб, воркер делает parse -> canonical (K3
ingest_document: повторная загрузка под тем же document_id сама архивирует
ревизию) -> инкрементальный апсерт в живой индекс + overlay (рестарт фасада
не нужен, полной пересборки нет).

Джобы живут в памяти, но каждое переключение статуса дописывается в
append-журнал <VELES_DATA>/uploads/ingest-jobs.jsonl с boot-id процесса.
Рестарт фасада теряет память, но не врёт: чтение статуса недошедшего джоба
(чужой boot_id, статус queued/running) честно переводит его в failed «сервис
перезапущен». Повторная проверка возвращает failed из журнала.

Квоты (VELES_QUOTA_DOCS / VELES_QUOTA_MB, 0 = без лимита) проверяются ДО
постановки в очередь: документы - 403, суммарный размер L0 - 413. Кап очереди
(K8, VELES_INGEST_QUEUE_MAX, дефолт 50, 0 = без лимита) - 429 «очередь занята»,
постановка под замком. Журнал ротируется при 5 МБ (активный -> .1, 2 файла).

Аудит: событие upload в ingest-audit.jsonl (by, размер, document_id, job_id,
исход) рядом с событиями ingest из canonical store.
"""

from __future__ import annotations

import json
import logging
import os
import queue as pyqueue
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..canonical.model import CanonicalDocument
from ..canonical.store import ingest_document
from ..config import DEFAULT_ACCESS_GROUPS
from ..policy.identifiers import extract_identifiers

logger = logging.getLogger("veles.ingest.queue")

JOB_STATUSES = ("queued", "running", "done", "failed")

# Поддерживаемые форматы первого шага (бриф K4); pdf - существующим docling-конвейером
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

_LEDGER_NAME = "ingest-jobs.jsonl"

# Ротация журнала (K8): при достижении размера активный файл переименовывается
# в .1 (старый .1 заменяется), держатся 2 файла - активный + одна архивная
# генерация; история джобов старше ротации в list_jobs/lookup не читается
_LEDGER_ROTATE_BYTES = 5 * 1024 * 1024

# имя файла: без разделителей пути и обхода каталогов; первая буква/цифра,
# дальше буквы/цифры/пробел/точка/подчёркивание/дефис/@()+[] (\\w - и кириллица)
_FILENAME_RE = re.compile(r"[\w][\w .@()+\[\]-]*", re.UNICODE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uploads_root() -> Path:
    """Корень L0-загрузок: <VELES_DATA>/uploads. Env читается на вызов (как в
    asset_file/source_view фасада), чтобы тесты и кит могли подменить корень."""
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        first = raw.split(os.pathsep)[0].strip()
        if first:
            return Path(first) / "uploads"
    from ..config import VELES_DATA

    return Path(VELES_DATA) / "uploads"


def user_dir(user_id: str) -> str:
    """Каталог пользователя внутри uploads (user_id уже нормализован identity;
    static-режим - владелец без user_id)."""
    return user_id.strip() or "owner"


def quota_limits() -> tuple[int, int]:
    """(лимит документов, лимит МБ); 0 = без лимита. Мусор в env -> дефолт."""
    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return max(0, int(raw))
        except ValueError:
            return default

    return _int_env("VELES_QUOTA_DOCS", 1000), _int_env("VELES_QUOTA_MB", 2048)


class QuotaExceeded(Exception):
    """Превышение квоты пользователя: статус и текст для ответа клиента."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class QueueCapacityExceeded(Exception):
    """Кап очереди загрузки (K8): queued-джобов больше VELES_INGEST_QUEUE_MAX -
    фасад отвечает 429 тем же контрактом, что QuotaExceeded."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def queue_max_queued() -> int:
    """Лимит queued-джобов очереди (VELES_INGEST_QUEUE_MAX, дефолт 50;
    0 = без лимита). Мусор в env -> дефолт. Читается на вызове (как квоты),
    чтобы тесты и кит подменяли лимит без пересоздания фасада."""
    raw = os.environ.get("VELES_INGEST_QUEUE_MAX", "").strip()
    if not raw:
        return 50
    try:
        return max(0, int(raw))
    except ValueError:
        return 50


def check_quota(user_directory: str, incoming_bytes: int, replacing: bool) -> None:
    """Квоты пользователя по его текущему L0 (uploads/<user>/): число файлов +
    суммарный размер. replacing=True - перезапись существующего документа не
    увеличивает число документов. Превышение -> QuotaExceeded (403/413) ДО очереди."""
    docs_limit, mb_limit = quota_limits()
    if not docs_limit and not mb_limit:
        return
    directory = uploads_root() / user_directory
    docs_used, bytes_used = 0, 0
    if directory.is_dir():
        for p in directory.iterdir():
            if p.is_file():
                docs_used += 1
                try:
                    bytes_used += p.stat().st_size
                except OSError:  # noqa: PERF101 - квота не должна падать на гонке fs
                    pass
    if docs_limit and not replacing and docs_used + 1 > docs_limit:
        raise QuotaExceeded(
            403,
            f"квота исчерпана: уже {docs_used} из {docs_limit} документов; "
            "загрузка отклонена до очереди",
        )
    if mb_limit and bytes_used + incoming_bytes > mb_limit * 1024 * 1024:
        raise QuotaExceeded(
            413,
            f"квота исчерпана: занято {bytes_used // (1024 * 1024)} МБ "
            f"из {mb_limit} МБ, нужно ещё {incoming_bytes // 1024} КБ; "
            "загрузка отклонена до очереди",
        )


def upload_max_bytes() -> int:
    """Жёсткий кап размера одной загрузки, байт (VELES_UPLOAD_MAX_MB, дефолт 50;
    0 = без капа). Мусор в env -> дефолт. Проверяется по Content-Length ДО чтения
    тела (P1 ревью K4): заведомо неподъёмный запрос не должен подниматься в RAM."""
    default_mb = 50

    def _default() -> int:
        return default_mb * 1024 * 1024

    raw = os.environ.get("VELES_UPLOAD_MAX_MB", "").strip()
    if not raw:
        return _default()
    try:
        mb = max(0, int(raw))
    except ValueError:
        return _default()
    return mb * 1024 * 1024 if mb else 0


def quota_free_bytes(user_directory: str) -> int | None:
    """Свободный остаток квоты размера, байт; None - лимит не задан (0).

    Оценка с запасом на перезапись: к остатку добавлен размер крупнейшего
    существующего L0-файла пользователя - если даже перезаписать его, больше
    не влезет. Точная проверка всё равно в check_quota после разбора тела
    (ранний гейт не знает имя файла)."""
    docs_limit, mb_limit = quota_limits()
    if not mb_limit:
        return None
    directory = uploads_root() / user_directory
    used, largest = 0, 0
    if directory.is_dir():
        for p in directory.iterdir():
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:  # noqa: PERF101 - оценка не должна падать на гонке fs
                continue
            used += size
            largest = max(largest, size)
    return max(0, mb_limit * 1024 * 1024 - used) + largest


_USER_QUOTA_LOCKS: dict[str, threading.Lock] = {}
_USER_QUOTA_LOCKS_GUARD = threading.Lock()


def user_quota_lock(user_directory: str) -> threading.Lock:
    """Per-user замок квоты (P2 ревью K4, TOCTOU): check_quota + запись L0 под
    одним замком - два одновременных POST одного пользователя не могут оба
    пройти проверку до записи. Замки живут до конца процесса, по одному на
    каталог пользователя."""
    with _USER_QUOTA_LOCKS_GUARD:
        lock = _USER_QUOTA_LOCKS.get(user_directory)
        if lock is None:
            lock = threading.Lock()
            _USER_QUOTA_LOCKS[user_directory] = lock
        return lock


def safe_filename(name: str) -> str | None:
    """Валидация имени файла загрузки: плоское имя без обхода каталогов."""
    name = (name or "").strip()
    if not name or len(name) > 200:
        return None
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    return name if _FILENAME_RE.fullmatch(name) else None


def slugify_title(title: str) -> str:
    """Имя документа для JSON-загрузки (text, title): безопасный слаг."""
    s = re.sub(r"[^\w -]+", "", (title or "").strip(), flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s).strip("_")[:80]
    return s or "document"


@dataclass
class Job:
    """Один джоб ингеста. В журнале - полный снимок на каждое переключение."""

    job_id: str
    boot_id: str
    user_id: str
    filename: str  # имя L0-файла (для JSON-загрузки - слаг из title + .md)
    title: str
    document_id: str
    size_bytes: int
    groups: list[str] = field(default_factory=list)  # access_groups документа
    status: str = "queued"
    error: str = ""
    action: str = ""  # created|updated|unchanged (исход ingest_document)
    revision_id: str = ""
    sha256: str = ""
    nodes: int = 0
    created_at: str = field(default_factory=_now_iso)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "boot_id": self.boot_id,
            "user_id": self.user_id,
            "status": self.status,
            "filename": self.filename,
            "title": self.title,
            "document_id": self.document_id,
            "size_bytes": self.size_bytes,
            "groups": self.groups,
            "action": self.action,
            "revision_id": self.revision_id,
            "sha256": self.sha256,
            "nodes": self.nodes,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Job":
        known = {f for f in cls.__dataclass_fields__ if f in raw}
        return cls(**{k: raw[k] for k in known})


@dataclass
class UpsertDelta:
    """Дельта одного апсерта: и в живой бандл применена, и в overlay уходит."""

    nodes: dict  # node_version_id -> мета (как у build_index)
    patches: dict  # node_version_id -> {"status": "superseded"} для старой ревизии
    ids: list[str]  # новые node_version_id в порядке матрицы
    matrix: np.ndarray  # векторы passages (нормализованные)
    bm25: list  # [(node_version_id, search_text)] в порядке применения


def upsert_document(bundle, doc: CanonicalDocument, outcome, embedder) -> UpsertDelta | None:
    """Инкрементальный апсерт спарсенного документа в ЖИВОЙ индекс (без rebuild).

    Существующие node_version_id не меняются: у старой ревизии патчится только
    status=superseded в мете (тот же итог, что дал бы полный rebuild - K3),
    новые узлы добавляются в конец векторного хранилища и BM25. Порядок мутаций
    безопасен для поиска из других потоков: мета (атомарные присваивания) ->
    векторы (ids раньше матрицы: NumpyFlatStore/FaissFlatStore.add так уже
    устроены) -> BM25 (ids последними в add, см. bm25.py). Между шагами поиск
    либо видит старое, либо целое новое - исключений чтения нет.
    """
    if outcome.action == "unchanged":
        return None

    patches: dict = {}
    if outcome.previous_revision:
        for nid, meta in bundle.node_meta.items():
            if (
                meta.get("document_id") == doc.document_id
                and meta.get("revision_id", "r1") == outcome.previous_revision
                and meta.get("status", "current") != "superseded"
            ):
                meta["status"] = "superseded"
                patches[nid] = {"status": "superseded"}

    texts: list[str] = []
    new_meta: dict[str, dict] = {}
    identifiers: set[str] = set(extract_identifiers(doc.title))
    for node in doc.nodes:
        if node.node_type == "root":  # как build_index: корень в индекс не идёт
            continue
        identifiers |= set(extract_identifiers(node.text))
        prefix = " > ".join(node.heading_path)
        # детерминированный префикс пути заголовков (инвариант 7 / H4) - тот же,
        # что в build_index, иначе после пересборки ранжирование меняется
        texts.append(f"[{prefix}] {node.text}" if prefix else node.text)
        new_meta[node.node_version_id] = {
            "document_id": doc.document_id,
            "title": doc.title,
            "logical_node_id": node.logical_node_id,
            "heading_path": node.heading_path,
            "text": node.text,
            "provenance": node.provenance.model_dump(),
            "access_groups": node.access_groups,
            "sha256_doc": doc.sha256,
            "asset": node.asset,
            "revision_id": doc.revision_id,
            "status": doc.status,
        }

    ids = list(new_meta)
    matrix = embedder.embed_passages(texts) if texts else np.zeros((0, 1), dtype=np.float32)
    # 1) мета: dict.update - одна атомарная операция; 2) векторы; 3) BM25
    bundle.node_meta.update(new_meta)
    bundle.identifiers |= identifiers
    if ids:
        bundle.vector.add(ids, matrix)
        bundle.bm25.add(ids, texts)
    return UpsertDelta(
        nodes=new_meta, patches=patches, ids=ids, matrix=matrix,
        bm25=[(nid, text) for nid, text in zip(ids, texts)],
    )


class IngestQueue:
    """Последовательная очередь ингеста одного процесса фасада."""

    def __init__(self, pipeline_fn, boot_id: str | None = None, start_worker: bool = True):
        self._pipeline_fn = pipeline_fn
        self.boot_id = boot_id or uuid.uuid4().hex
        self._jobs: dict[str, Job] = {}
        self._q: pyqueue.Queue = pyqueue.Queue()
        self._running_job: str | None = None
        self._stopped = False
        # кап очереди (K8): проверка лимита и постановка атомарны - два
        # одновременных submit не могут переполнить очередь вдвоём
        self._submit_lock = threading.Lock()
        # журнал джобов: append воркера и ротация под одним локом (ревью K8 P2)
        self._ledger_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(
                target=self._worker, name="veles-ingest", daemon=True
            )
            self._thread.start()

    # ------------------------------------------------------------- постановка

    def at_capacity(self) -> tuple[bool, int]:
        """Ненадёжная ранняя проверка капа (ревью K8 P2): кап может отпустить
        между этой проверкой и submit - авторитетна только проверка в submit.
        Нужна, чтобы обычный отказ не доходил до перезаписи L0 существующего
        документа (перезапись необратима, L0 потом удаляется)."""
        max_queued = queue_max_queued()
        return (bool(max_queued) and self._q.qsize() >= max_queued), max_queued

    def submit(
        self, *, user_id: str, filename: str, title: str, document_id: str,
        size_bytes: int, groups: list[str],
    ) -> Job:
        max_queued = queue_max_queued()
        with self._submit_lock:
            # кап только на ждущих в очереди (qsize): взятый воркером джоб уже
            # не queued; превышение - отказ ДО записи L0-джоба в журнал
            if max_queued and self._q.qsize() >= max_queued:
                raise QueueCapacityExceeded(
                    429, "очередь занята, повторите позже "
                         f"(лимит {max_queued}, VELES_INGEST_QUEUE_MAX)")
            job = Job(
                job_id=uuid.uuid4().hex,
                boot_id=self.boot_id,
                user_id=user_id,
                filename=filename,
                title=title,
                document_id=document_id,
                size_bytes=size_bytes,
                groups=list(groups),
            )
            self._jobs[job.job_id] = job
            self._append_ledger(job)
            self._q.put(job)
        return job

    def wait_idle(self, timeout_s: float = 60.0) -> bool:
        """Дождаться опустошения очереди (тесты/админ: джобы done/failed)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._q.unfinished_tasks == 0 and self._running_job is None:
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        self._stopped = True

    # ----------------------------------------------------------- чтение статуса

    def lookup(self, job_id: str) -> dict | None:
        """Статус джоба: память -> журнал. Недошедший джоб чужого boot_id
        (рестарт фасада) честно переводится в failed «сервис перезапущен»."""
        in_memory = self._jobs.get(job_id)
        if in_memory is not None:
            return in_memory.to_dict()
        record = self._ledger_last(job_id)
        if record is None:
            return None
        if record.get("status") in ("queued", "running") and record.get("boot_id") != self.boot_id:
            record["status"] = "failed"
            record["error"] = "сервис перезапущен, задание не завершилось - повторите загрузку"
            record["finished_at"] = _now_iso()
            self._append_ledger(Job.from_dict(record))
        return record

    def list_jobs(self, limit: int = 200) -> list[dict]:
        """Все известные джобы (свежие первыми): память поверх журнала.
        Недошедшие джобы чужого boot_id - та же семантика, что и в lookup
        (P2 ревью K4): failed «сервис перезапущен», иначе в списке висит
        вечный running."""
        merged: dict[str, dict] = {}
        for job in self._jobs.values():
            merged[job.job_id] = job.to_dict()
        for record in self._ledger_records():
            merged.setdefault(record["job_id"], record)
        records = []
        for record in merged.values():
            if record.get("status") in ("queued", "running") and record.get("boot_id") != self.boot_id:
                record["status"] = "failed"
                record["error"] = "сервис перезапущен, задание не завершилось - повторите загрузку"
                record["finished_at"] = _now_iso()
                self._append_ledger(Job.from_dict(record))  # персистентно, как в lookup
            records.append(record)
        records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return records[:limit]

    # ------------------------------------------------------------------ журнал

    def _ledger_path(self) -> Path:
        return uploads_root() / _LEDGER_NAME

    def _rotate_ledger_if_needed(self, path: Path) -> None:
        """Ротация (K8): активный журнал достиг _LEDGER_ROTATE_BYTES -
        переименовывается в .1 (прежний .1 заменяется, держатся 2 файла);
        следующий append создаёт активный заново. Неудача не ломает загрузку."""
        try:
            if path.exists() and path.stat().st_size >= _LEDGER_ROTATE_BYTES:
                path.replace(path.with_name(_LEDGER_NAME + ".1"))
        except OSError as e:  # noqa: BLE001 - ротация не должна ломать загрузку
            logger.warning("ротация журнала джобов не удалась: %s", e)

    def _append_ledger(self, job: Job) -> None:
        # ротация+append под одним локом (ревью K8 P2): без него воркер мог
        # держать append в момент rename - запись уезжала в .1 мимо активного
        try:
            with self._ledger_lock:
                path = self._ledger_path()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._rotate_ledger_if_needed(path)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:  # noqa: BLE001 - журнал не должен ломать загрузку
            logger.warning("журнал джобов недоступен: %s", e)

    def _ledger_records(self) -> list[dict]:
        try:
            raw = self._ledger_path().read_text(encoding="utf-8")
        except OSError:
            return []
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("job_id"):
                out.append(record)
        return out

    def _ledger_last(self, job_id: str) -> dict | None:
        found = None
        for record in self._ledger_records():
            if record["job_id"] == job_id:
                found = record  # последнее переключение побеждает
        return found

    # ------------------------------------------------------------------- воркер

    def _worker(self) -> None:
        while not self._stopped:
            try:
                job = self._q.get(timeout=2.0)
            except pyqueue.Empty:
                continue
            self._running_job = job.job_id
            try:
                self._transition(job, "running")
                self._process(job)
                self._transition(job, "failed" if job.error else "done")
                self._audit(job)
            except Exception as e:  # noqa: BLE001 - джоб честно падает, воркер живёт
                job.error = f"внутренняя ошибка очереди: {e}"[:500]
                self._transition(job, "failed")
                self._audit(job)
            finally:
                self._running_job = None
                self._q.task_done()

    def _transition(self, job: Job, status: str) -> None:
        job.status = status
        if status == "running":
            job.started_at = _now_iso()
        elif status in ("done", "failed"):
            job.finished_at = _now_iso()
        self._append_ledger(job)

    def _parse(self, job: Job) -> CanonicalDocument:
        """L0-файл -> CanonicalDocument существующими парсер-адаптерами."""
        path = uploads_root() / user_dir(job.user_id) / job.filename
        ext = path.suffix.lower()
        if ext == ".pdf":
            from .docling_parser import DoclingParserAdapter

            doc = DoclingParserAdapter().parse_file(path, corpus_root=uploads_root())
        else:
            from .txt_md import TxtMdParserAdapter

            doc = TxtMdParserAdapter().parse_file(path, corpus_root=uploads_root())
        return doc

    def _discard_l0(self, job: Job) -> None:
        """P2 ревью K4: упавший разбор не должен навсегда съедать квоту -
        L0-файл удаляется; не удалось удалить - громко в лог (WARN)."""
        path = uploads_root() / user_dir(job.user_id) / job.filename
        try:
            path.unlink()
        except OSError as e:  # noqa: BLE001 - неудача удаления не ломает воркер
            logger.warning("L0 %s не удалён после провала джоба %s: %s",
                           path, job.job_id, e)

    def _process(self, job: Job) -> None:
        """parse -> canonical (K3) -> апсерт живого индекса -> overlay -> аудит."""
        from .. import config
        from ..indexing.overlay import append_overlay

        try:
            pipeline = self._pipeline_fn()
        except Exception as e:  # noqa: BLE001 - нет индекса/модели: честный failed
            job.error = f"индекс недоступен: {e}"[:500]
            return
        bundle, embedder = pipeline.bundle, pipeline.embedder

        try:
            doc = self._parse(job)
        except Exception as e:  # noqa: BLE001 - битый файл: человеческая причина
            job.error = f"не удалось разобрать документ ({job.filename}): {e}"[:500]
            self._discard_l0(job)  # битый файл не съедает квоту навсегда
            return

        job.sha256 = doc.sha256
        doc.ingested_by = user_dir(job.user_id)
        if job.title:
            doc.title = job.title
        groups = job.groups or sorted(DEFAULT_ACCESS_GROUPS)
        doc.access_groups = groups
        for node in doc.nodes:
            node.access_groups = groups

        try:
            outcome = ingest_document(doc, by=doc.ingested_by)
        except Exception as e:  # noqa: BLE001 - конфликт архивации и прочее
            job.error = f"канон не принял документ: {e}"[:500]
            return

        job.action = outcome.action
        job.revision_id = outcome.revision_id
        job.nodes = len(doc.nodes)
        job.document_id = doc.document_id

        delta = upsert_document(bundle, doc, outcome, embedder)
        if delta is not None:
            try:
                append_overlay(
                    config.INDEX_DIR, delta.nodes, delta.patches,
                    delta.ids, delta.matrix, delta.bm25,
                )
            except Exception as e:  # noqa: BLE001 - поиск уже работает, но после
                # рестарта документ доедет только с пересборкой - громко в лог
                logger.error("overlay не записан (%s): %s", doc.document_id, e)

    def _audit(self, job: Job) -> None:
        """Событие загрузки в ingest-audit.jsonl (рядом с ingest-событиями K3)."""
        from ..canonical import store as store_mod

        event = {
            "ts": _now_iso(),
            "event": "upload",
            "by": user_dir(job.user_id),
            "document_id": job.document_id,
            "job_id": job.job_id,
            "filename": job.filename,
            "size_bytes": job.size_bytes,
            "sha256": job.sha256,
            "outcome": job.action if job.status == "done" and job.action else "failed",
            "revision_id": job.revision_id,
            "source": "api",
        }
        try:
            log_path = store_mod.CANONICAL_DIR / "ingest-audit.jsonl"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as e:  # noqa: BLE001 - аудит не ломает загрузку
            logger.warning("журнал аудита недоступен: %s", e)


def discard_l0_file(document_id: str) -> bool:
    """Удалить L0 загрузки по document_id (K10 DELETE: канон ушёл в trash, L0 -
    за ним; корпусные документы L0 не имеют - False без шума). Не удалось
    удалить - False с WARN: удаление документа не отменяется."""
    path = uploads_root() / document_id
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as e:  # noqa: BLE001 - неудача не ломает удаление документа
        logger.warning("L0 %s не удалён при удалении документа: %s", path, e)
        return False
