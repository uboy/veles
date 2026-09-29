"""Физическое хранение канона: canonical/<doc_hash>/manifest.json + nodes.jsonl.

Ревизии (K3): каталог <doc_dir> - рабочая (текущая) ревизия; при повторном ингесте
того же document_id с другим sha256 старые manifest/nodes/assets переезжают в
<doc_dir>/revision-r{n}/ и получают status=superseded, новый канон пишется на место
с revision_id=r{n+1}. Отсутствие revision-r*/ подкаталогов (весь канон до 29.09) =
одна ревизия, читается без миграции.

Удаление (K10): delete_document переезжает doc_dir целиком (все ревизии и assets)
в корзину <VELES_DATA>/trash/<timestamp>-<имя>/ - не безвозвратно, восстановление
вручную админом. Корзина - сосед каталога canonical, вне glob("canonical*") чтения
и вне iter_documents полной пересборки."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from ..config import CANONICAL_DIR
from .model import CanonicalDocument, CanonicalNode


def _doc_dir(document_id: str, root: Path | None = None) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in document_id)
    if len(safe) <= 120:
        return (root or CANONICAL_DIR) / safe
    # длинные id: хвост-хеш устраняет коллизии обрезки (28.09: пересказ/транскрипция
    # одной лекции схлопывались в один каталог - документ молча терялся)
    suffix = hashlib.sha1(document_id.encode("utf-8")).hexdigest()[:10]
    return (root or CANONICAL_DIR) / f"{safe[:105]}-{suffix}"


def save_document(
    doc: CanonicalDocument,
    root: Path | None = None,
    by: str | None = None,
    audit_log: Path | None = None,
) -> Path:
    """Записать документ в канон как есть (без ревизионной логики) + дописать
    событие загрузки в append-only журнал аудита.

    Для повторных загрузок существующего document_id используйте ingest_document -
    эта функция перезаписывает текущую ревизию молча.

    by - кто загрузил (логин/почта из аутентификации; для старых ручных конвейеров -
    значение VELES_INGESTED_BY или "unknown"). audit_log - путь к журналу (по умолчанию
    <корень канона>/../ingest-audit.jsonl). Журнал только дописывается, события -
    единственный полный след «кто/когда/что» по всем загрузкам."""
    from datetime import datetime, timezone

    if not doc.ingested_at:
        doc.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not doc.ingested_by:
        doc.ingested_by = (
            by or os.environ.get("VELES_INGESTED_BY", "").strip() or "unknown"
        )

    d = _doc_dir(doc.document_id, root)
    d.mkdir(parents=True, exist_ok=True)
    # nodes.jsonl ПЕРВЫМ, manifest последним (ревью-2: убитый процесс не оставит
    # «готовый» документ без узлов)
    with (d / "nodes.jsonl.tmp").open("w", encoding="utf-8") as f:
        for node in doc.nodes:
            f.write(json.dumps(node.model_dump(), ensure_ascii=False) + "\n")
    (d / "nodes.jsonl.tmp").replace(d / "nodes.jsonl")
    manifest = doc.model_dump(exclude={"nodes"})
    if doc.assets:
        adir = d / "assets"
        adir.mkdir(exist_ok=True)
        persisted = {}
        for name, src in doc.assets.items():
            safe_name = Path(name).name
            shutil.copyfile(src, adir / safe_name)
            persisted[name] = f"assets/{safe_name}"
        manifest["assets"] = persisted
    (d / "manifest.tmp").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (d / "manifest.tmp").replace(d / "manifest.json")

    log_path = audit_log or (d.parent / "ingest-audit.jsonl")
    try:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": "ingest",
            "document_id": doc.document_id,
            "revision_id": doc.revision_id,
            "sha256": doc.sha256,
            "status": doc.status,
            "by": doc.ingested_by,
            "dir": d.name,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:  # noqa: BLE001 - журнал не должен ломать сохранение канона
        pass
    return d


def load_document(document_id: str, root: Path | None = None) -> CanonicalDocument:
    d = _doc_dir(document_id, root)
    return _load_document_dir(d)


def _load_document_dir(d: Path, archived: bool = False) -> CanonicalDocument:
    """Чтение документа из каталога. archived=True - каталог revision-r*/ внутри
    doc_dir: манифест без status (рукодельная структура) считается superseded -
    архивная ревизия по определению не текущая."""
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    if archived and "status" not in manifest:
        manifest["status"] = "superseded"
    doc = CanonicalDocument(**manifest, nodes=[])
    for line in (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc.nodes.append(CanonicalNode(**json.loads(line)))
    return doc


def canonical_roots() -> list[Path]:
    """Каталоги канона по всем корням VELES_DATA - тот же обход, что у чтения
    фасада (source/asset ищут документ в каждом canonical* каждого корня).
    Env не задан (тесты/встраивание) - один CANONICAL_DIR."""
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        roots = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
        dirs = [c for root in roots for c in sorted(root.glob("canonical*"))]
        if dirs:
            return dirs
    return [CANONICAL_DIR]


def trash_root() -> Path:
    """Корень корзины удалённого: <первый корень VELES_DATA>/trash - сосед
    canonical, вне glob("canonical*") чтения фасада и вне iter_documents
    пересборки (удалённый канон не должен ни читаться, ни индексироваться)."""
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        first = raw.split(os.pathsep)[0].strip()
        if first:
            return Path(first) / "trash"
    from ..config import VELES_DATA

    return Path(VELES_DATA) / "trash"


def _find_doc_dirs(document_id: str, root: Path | None = None) -> list[tuple[Path, dict]]:
    """(doc_dir, manifest) документа по всем корням канона. Guard обхода каталогов:
    _doc_dir пропускает '.'/'..' в id (допустимые для имени символы), а удаление -
    разрушительная операция, поэтому doc_dir обязан лежать НЕ глубже корня канона.

    Ревью K13 P1: формульный путь (_doc_dir по текущей формуле имени) не находит
    легаси-каталоги-близнецы (имена до фикса 28.09, без/с другим хеш-хвостом) -
    удаление оставляло близнеца в каноне, и пересборка воскрешала «удалённый»
    документ. Скан всех каталогов канона со сверкой document_id из манифеста
    выполняется ВСЕГДА (сотни каталогов, путь только DELETE - не горячий):
    формульный каталог существует или нет - близнецы ловятся в обоих случаях."""
    found: list[tuple[Path, dict]] = []
    seen: set[Path] = set()
    roots = [root] if root is not None else canonical_roots()
    for canon in roots:
        canon = Path(canon)
        if not canon.is_dir():
            continue
        d = _doc_dir(document_id, canon)
        if d.parent == canon:
            mpath = d / "manifest.json"
            if mpath.exists():
                try:
                    manifest = json.loads(mpath.read_text(encoding="utf-8"))
                except (OSError, ValueError):  # noqa: BLE001 - битый манифест не для удаления
                    manifest = None
                if manifest is not None:
                    found.append((d, manifest))
                    seen.add(d.resolve())
        for child in sorted(canon.iterdir()):
            if not child.is_dir() or child.resolve() in seen:
                continue
            mpath = child / "manifest.json"
            if not mpath.exists():
                continue
            try:
                manifest = json.loads(mpath.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # noqa: BLE001 - битый манифест не документ
                continue
            if str(manifest.get("document_id") or "") == document_id:
                found.append((child, manifest))
                seen.add(child.resolve())
    return found


def find_document_manifest(document_id: str, root: Path | None = None) -> dict | None:
    """manifest.json документа (первый найденный по корням) - ACL-проверка права
    удаления выполняется вызывающим ДО delete_document; None - документа нет."""
    for _d, manifest in _find_doc_dirs(document_id, root):
        return manifest
    return None


def delete_document(
    document_id: str,
    root: Path | None = None,
    by: str | None = None,
    audit_log: Path | None = None,
) -> dict | None:
    """Удалить документ целиком (K10): doc_dir со ВСЕМИ ревизиями (revision-r*/
    и assets) переезжает в корзину trash/<timestamp>-<имя>/; возвращает сведения
    об удалённом (ответ API/CLI) или None, если документа нет.

    Право удаления (владелец/админ) проверяется вызывающим до вызова - здесь
    только физика: канон -> trash + событие delete в аудит. Узлы живого индекса
    выводит tombstone_document (veles/indexing/overlay.py), L0 - discard_l0_file
    (veles/ingest/queue.py); все три вызывает и API, и CLI.
    """
    from datetime import datetime, timezone

    found = _find_doc_dirs(document_id, root)
    if not found:
        return None
    audit_dir = found[0][0].parent  # каталог канона, из которого удаляем (для аудита)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    troot = trash_root()
    troot.mkdir(parents=True, exist_ok=True)

    total_nodes = 0
    trash_names: list[str] = []
    revisions: set[str] = set()
    title = ""
    for d, manifest in found:
        target = troot / f"{ts}-{d.name}"
        n = 2
        while target.exists():  # два корня/два удаления за одну секунду - суффикс
            target = troot / f"{ts}-{n}-{d.name}"
            n += 1
        try:
            shutil.move(str(d), str(target))
        except (FileNotFoundError, OSError) as e:
            # ревью K10 P2: double-delete от двух инженеров - проигравший получает
            # честный None (-> 404), а не сырой 500; Windows: файл мог быть открыт
            if not d.exists():
                return None
            raise RuntimeError(f"переезд в trash не удался: {e}") from e
        trash_names.append(target.name)
        try:
            total_nodes += sum(
                1 for line in (target / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:  # noqa: BLE001 - подсчёт узлов не ломает удаление
            pass
        rev = str(manifest.get("revision_id") or "r1")
        revisions.add(rev)
        for rdir in sorted(target.glob("revision-r*")):
            if (rdir / "manifest.json").exists():
                revisions.add(rdir.name[len("revision-"):])
        title = title or str(manifest.get("title") or "")
        # мета корзины для ручного восстановления админом (document_id и так
        # внутри manifest.json, здесь - кто и когда удалил); best-effort
        try:
            (target / "trash-meta.json").write_text(
                json.dumps(
                    {
                        "document_id": document_id,
                        "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "deleted_by": by or "unknown",
                        "title": title,
                        "revisions": sorted(revisions),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": "delete",
        "document_id": document_id,
        "by": by or "unknown",
        "revisions": sorted(revisions),
        "nodes_current": total_nodes,
        "trash": trash_names,
    }
    try:
        log_path = audit_log or (audit_dir / "ingest-audit.jsonl")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:  # noqa: BLE001 - журнал не должен ломать удаление
        pass
    return {
        "document_id": document_id,
        "title": title,
        "revisions": sorted(revisions),
        "nodes_current": total_nodes,
        "trash": trash_names,
    }


def iter_documents(
    root: Path | None = None, include_revisions: bool = False
) -> Iterator[CanonicalDocument]:
    """Текущие ревизии всех документов; с include_revisions=True - и архивные
    (revision-r*/ внутри doc_dir, status=superseded) - для сборки полного индекса.

    Один document_id отдаётся ОДИН раз (K13): легаси-каталоги-близнецы (до фикса
    имени 28.09 каталог обрезался без хеш-суффикса) с тем же document_id
    пропускаются с предупреждением - иначе build_index индексировал обе копии
    как один документ (двойной контент в поиске; прецедент прода 29.09: НК РФ
    46 355 узлов в индексе при ~24 000 в каноне). Первый каталог в sorted-порядке
    побеждает; удаление (delete_document) по-прежнему убирает ВСЕ каталоги
    документа. (Предупреждение - не ошибка: содержимое близнецов на проде
    идентично, разбирает владелец по отчёту scripts/ingest_audit.py.)"""
    base = root or CANONICAL_DIR
    if not base.exists():
        return
    seen_ids: set[str] = set()
    for d in sorted(base.iterdir()):
        if (d / "manifest.json").exists():
            doc_id = _manifest_document_id(d)
            if doc_id and doc_id in seen_ids:
                print(
                    f"предупреждение: {d} - дубль document_id {doc_id!r} в {base}, "
                    "пропущен (первый каталог побеждает; см. scripts/ingest_audit.py)",
                    file=sys.stderr,
                )
            else:
                if doc_id:
                    seen_ids.add(doc_id)
                yield _load_document_dir(d)
        if include_revisions:
            for rdir in sorted(d.glob("revision-r*")):
                if _archive_complete(rdir):
                    yield _load_document_dir(rdir, archived=True)


def _manifest_document_id(d: Path) -> str:
    """document_id из манифеста каталога; '' - манифест не читается/без поля."""
    try:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(manifest.get("document_id") or "")


@dataclass
class IngestOutcome:
    """Результат ревизионной загрузки (CLI-сообщения и аудит)."""

    action: Literal["created", "updated", "unchanged"]
    revision_id: str  # ревизия, ставшая/оставшаяся текущей
    previous_revision: str | None = None  # для action=updated


_REV_RE = re.compile(r"^r(\d+)$")


def _revision_number(rev: str) -> int:
    m = _REV_RE.match(rev)
    return int(m.group(1)) if m else 0


def _archive_complete(rdir: Path) -> bool:
    """Архив полон, если есть и manifest.json, и nodes.jsonl (ревью K3: manifest
    в каталоге архива - маркер завершённости переезда; manifest без nodes - обрыв,
    узлов у него нет). Обрывок пропускается с предупреждением в stderr, а не роняет
    iter_documents/build_index/cmd_status."""
    if not (rdir / "manifest.json").exists():
        return False
    if (rdir / "nodes.jsonl").exists():
        return True
    print(
        f"предупреждение: архив {rdir} без nodes.jsonl пропущен (обрыв архивации)",
        file=sys.stderr,
    )
    return False


def _archived_revisions(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    out = []
    for rdir in d.glob("revision-r*"):
        m = re.fullmatch(r"revision-(r\d+)", rdir.name)
        if m and _archive_complete(rdir):
            out.append(m.group(1))
    return out


def _unfinished_archives(d: Path) -> list[str]:
    """Архивы-обрывки: revision-r* с manifest.json, но без nodes.jsonl (ревью K3).
    Документами не отдаются (узлов нет), но в нумерацию восстановления входят -
    ревизия была, и её узлы могли остаться в корне doc_dir."""
    if not d.is_dir():
        return []
    out = []
    for rdir in d.glob("revision-r*"):
        m = re.fullmatch(r"revision-(r\d+)", rdir.name)
        if m and (rdir / "manifest.json").exists() and not (rdir / "nodes.jsonl").exists():
            out.append(m.group(1))
    return out


def _read_current_manifest(d: Path) -> dict | None:
    mpath = d / "manifest.json"
    if not mpath.exists():
        return None
    return json.loads(mpath.read_text(encoding="utf-8"))


def _move_into(dst_dir: Path, src: Path) -> None:
    """Идемпотентный переезд src внутрь dst_dir: источника нет - пусто; в приёмнике
    уже лежит такой же обрывок прошлой попытки - источник-дубль убирается (узлы
    приёмника не перезаписываются: это уже переехавшая единственная копия)."""
    if not src.exists():
        return
    dst = dst_dir / src.name
    if dst.exists():
        if src.is_dir():
            shutil.rmtree(src)
        else:
            src.unlink()
    else:
        shutil.move(str(src), str(dst))


def _archive_current_revision(d: Path, rev_id: str) -> None:
    """Переезд текущих manifest/nodes/assets в <doc_dir>/revision-<rev_id>/ и
    смена статуса в манифесте архива на superseded.

    Порядок переезда (ревью K3): nodes.jsonl и ассеты ПЕРВЫМИ, manifest.json
    ПОСЛЕДНИМ - manifest в каталоге архива становится маркером завершённости:
    сбой посреди переезда не оставит «полного» архива без единственной копии
    узлов. Каталог архива без manifest - незавершённая архивация, дозавершается
    идемпотентно; RuntimeError - только для полного архива с другим содержимым."""
    rdir = d / f"revision-{rev_id}"
    if not rdir.exists():
        rdir.mkdir()
    elif (rdir / "manifest.json").exists():
        # Архив с манифестом занят. Текущего манифеста нет (восстановление после
        # сбоя между архивацией и записью) - хвост ниже дозавершит архив
        # идемпотентно. Есть - сверяем sha256: совпал, архив полон - корень дубль
        # архива, убираем; совпал, а nodes в архиве нет - узлы из корня
        # (единственная копия) доезжают хвостом, а не удаляются как «дубль»;
        # не совпал - честный отказ.
        current = _read_current_manifest(d)
        archived = _read_current_manifest(rdir) or {}
        if current is not None:
            if not (
                current.get("sha256") and current.get("sha256") == archived.get("sha256")
            ):
                raise RuntimeError(
                    f"{rdir} уже существует с другим содержимым - требуется ручной разбор"
                )
            if (rdir / "nodes.jsonl").exists():
                for name in ("manifest.json", "nodes.jsonl"):
                    f = d / name
                    if f.exists():
                        f.unlink()
                adir = d / "assets"
                if adir.is_dir():
                    shutil.rmtree(adir)
                return
    _move_into(rdir, d / "nodes.jsonl")
    _move_into(rdir, d / "assets")
    _move_into(rdir, d / "manifest.json")
    # статус старой ревизии меняется в ЕЁ каталоге (superseded_at не нужен);
    # заодно самодостаточность архива: у legacy-манифестов ревизионных полей нет
    manifest = _read_current_manifest(rdir) or {}
    manifest["status"] = "superseded"
    manifest.setdefault("revision_id", rev_id)
    (rdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def ingest_document(
    doc: CanonicalDocument,
    root: Path | None = None,
    by: str | None = None,
    audit_log: Path | None = None,
) -> IngestOutcome:
    """Ревизионная загрузка (K3) - используйте её для повторных ингестов.

    document_id новый -> created (r1, поведение прежних версий). sha256 совпал с
    текущим -> unchanged, канон не трогается. Другой sha256 -> текущая ревизия
    переезжает в revision-<n>/ (status=superseded), документ записывается заново
    как r{n+1}/current с previous_revision=<n>; node_version_id пересчитываются.
    Падение между архивацией и записью нового манифеста самолечится: ревизия
    продолжается от максимальной заархивированной; недокатившаяся архивация
    (каталог ревизии без manifest - «липкий» пустой или полупереезжий) тоже
    дозавершается идемпотентно, RuntimeError - только полный архив с другим
    содержимым."""
    d = _doc_dir(doc.document_id, root)
    current = _read_current_manifest(d)
    if current is None:
        pool = _archived_revisions(d) + _unfinished_archives(d)
        if not pool:
            save_document(doc, root=root, by=by, audit_log=audit_log)
            return IngestOutcome("created", doc.revision_id)
        # восстановление: текущего манифеста нет, архивы есть - продолжаем нумерацию
        # от максимальной; полупереезжий архив (manifest в архиве, nodes в корне)
        # сначала дозавершается - иначе save_document стёр бы корневые узлы,
        # единственную копию узлов этой ревизии (ревью K3)
        prev = max(pool, key=_revision_number)
        _archive_current_revision(d, prev)
        new_rev = f"r{_revision_number(prev) + 1}"
        doc.revision_id = new_rev
        doc.status = "current"
        doc.previous_revision = prev
        doc.retag_revision(new_rev)
        save_document(doc, root=root, by=by, audit_log=audit_log)
        return IngestOutcome("updated", new_rev, prev)

    cur_rev = current.get("revision_id") or "r1"
    # ревью K3: пустой sha256 (legacy-манифест без поля) не даёт ложный no-op -
    # иначе входящий документ с sha256="" молча отбрасывался как «без изменений»
    if current.get("sha256") and current.get("sha256") == doc.sha256:
        return IngestOutcome("unchanged", cur_rev)

    nums = [_revision_number(cur_rev)] + [_revision_number(r) for r in _archived_revisions(d)]
    prev = cur_rev
    # ревью K3: архивный каталог формируется из revision_id - мусор в манифесте
    # («abc») ушёл бы в revision-abc/, невидимый glob revision-r*; честный отказ
    if not _REV_RE.match(prev):
        raise RuntimeError(
            f"неожиданный revision_id в {d / 'manifest.json'}: {prev!r} "
            "(ожидался вида r<номер>) - архивация прервана"
        )
    new_rev = f"r{max(nums) + 1}"
    _archive_current_revision(d, prev)
    doc.revision_id = new_rev
    doc.status = "current"
    doc.previous_revision = prev
    doc.retag_revision(new_rev)
    save_document(doc, root=root, by=by, audit_log=audit_log)
    return IngestOutcome("updated", new_rev, prev)
