"""Журнал аудита действий (кто, когда, что) - append-only JSONL.

Заказ владельца 12.10: «аудит и логирование в админке должно быть - что когда
и кем произошло, включая все действия в админке и все действия пользователей,
включая пользовательский поиск» (docs/plan/admin-audit/plan.md, бриф
docs/plan/briefs/2026-10-12-admin-audit.md).

Файл: <VELES_DATA>/audit/audit.jsonl; событие - одна строка:
    {ts, actor, admin, action, target, details, ip}
ts - ISO-8601 UTC; actor - user_id личности (пустой = "owner" - bearer-
владелец/static, личности нет = "anonymous"); admin - флаг админа;
action - код действия (каталог - в плане); target - над чем действовали;
details - подробности (текст запроса обрезается до MAX_DETAIL_CHARS);
ip - клиент запроса.

Гарантии:
- never-throw: любая ошибка записи (нет каталога, права, диск) - warning в
  лог "veles.audit" (один на период поломки, не на каждое событие), запрос
  не ломается; чтение битых строк JSONL их молча пропускает;
- потокобезопасно: append и ротация под замком; файл открывается на каждую
  запись и закрывается - ротация переименованием не встречает открытых
  дескрипторов (Windows блокирует rename открытого файла);
- запись отключается VELES_AUDIT=0 (чтение работает всегда);
- ротация по размеру VELES_AUDIT_ROTATE_MB (дефолт 50, 0=выкл): перед
  записью текущий файл переезжает в audit.jsonl.1, хранится VELES_AUDIT_KEEP
  ротированных файлов (дефолт 5, старшие удаляются);
- заголовки и токены в журнал не попадают: точки записи передают только
  поля события; текст запроса в details - явное решение владельца.

Чтение с фильтрами (actor, action, подстрока q по target/details, since,
until, limit, offset) - для GET /v1/admin/audit.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("veles.audit")

MAX_DETAIL_CHARS = 500
MAX_ACTOR_CHARS = 254
DEFAULT_ROTATE_MB = 50
DEFAULT_KEEP = 5

# синглтон на процесс (как очередь ингеста app.state.veles_ingest): env
# читается при создании; тесты сбрасывают reset_audit()
_INSTANCE_LOCK = threading.Lock()
_instance: Optional["AuditLog"] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _env_int(name: str, default: int) -> int:
    """int из env без роняния создания журнала (мусор - дефолт)."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("%s=%r не целое - беру %d", name, raw, default)
        return default


def _clip(value: str, limit: int) -> str:
    return str(value or "")[:limit]


def _as_utc(value: datetime) -> datetime:
    """Наивное время считаем UTC (границы фильтра без пояса) - иначе
    сравнение aware/naive даёт TypeError вместо фильтрации."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_bound(value) -> datetime:
    """Граница since/until: datetime или строка ISO-8601; ValueError - наверх
    (API отдаёт 400). Строка обязана быть датой - пустоту вызывающий убирает."""
    if isinstance(value, datetime):
        return _as_utc(value)
    return _as_utc(datetime.fromisoformat(str(value).strip()))


def actor_of(ident) -> str:
    """actor события из личности запроса: user_id; пустой - "owner"
    (bearer-владелец/static), личности нет (подписанная ссылка) - "anonymous"."""
    if ident is None:
        return "anonymous"
    return _clip(ident.user_id, MAX_ACTOR_CHARS) or "owner"


def client_ip(request) -> str:
    """IP клиента запроса; нет соединения (тесты/внутренние вызовы) - пусто."""
    return request.client.host if request.client is not None else ""


class AuditLog:
    """Append-only JSONL-журнал в <data_dir>/audit/audit.jsonl.

    Параметры rotate_mb/keep/enabled по None читаются из окружения
    (VELES_AUDIT_ROTATE_MB / VELES_AUDIT_KEEP / VELES_AUDIT); явные значения -
    для тестов и встраивания.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        enabled: Optional[bool] = None,
        rotate_mb: Optional[int] = None,
        keep: Optional[int] = None,
    ):
        self.dir = Path(data_dir) / "audit"
        self.path = self.dir / "audit.jsonl"
        if enabled is None:
            enabled = (os.environ.get("VELES_AUDIT") or "").strip() != "0"
        if rotate_mb is None:
            rotate_mb = _env_int("VELES_AUDIT_ROTATE_MB", DEFAULT_ROTATE_MB)
        if keep is None:
            keep = _env_int("VELES_AUDIT_KEEP", DEFAULT_KEEP)
        self.enabled = enabled
        self.rotate_mb = max(0, rotate_mb)
        self.keep = max(0, keep)
        self._lock = threading.Lock()
        # warning о поломке - один на период поломки, а не на каждое событие:
        # журнал только-чтение при горячем поиске не превращается в спам логов
        self._broken = False

    # --------------------------------------------------------------- запись

    def log(
        self,
        action: str,
        *,
        actor: str = "",
        admin: bool = False,
        target: str = "",
        details: str = "",
        ip: str = "",
    ) -> None:
        """Записать событие; никогда не бросает (never-throw). Выключено
        (VELES_AUDIT=0) - no-op. Вызывающий запрос не ломается никогда."""
        if not self.enabled:
            return
        event = {
            "ts": _now_iso(),
            "actor": _clip(actor, MAX_ACTOR_CHARS) or "owner",
            "admin": bool(admin),
            "action": str(action or ""),
            "target": _clip(target, MAX_DETAIL_CHARS),
            # обрезка 500 - решение плана (текст запроса не раздувает журнал)
            "details": _clip(details, MAX_DETAIL_CHARS),
            "ip": _clip(ip, MAX_ACTOR_CHARS),
        }
        try:
            line = json.dumps(event, ensure_ascii=False)
            with self._lock:
                self._rotate_if_needed(len(line) + 1)
                self.dir.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:  # noqa: BLE001 - аудит не ломает запрос
            if not self._broken:
                self._broken = True
                logger.warning("событие %r не записано: %s", action, e)
            return
        if self._broken:
            self._broken = False
            logger.info("запись журнала аудита восстановилась")

    def _rotate_if_needed(self, incoming: int) -> None:
        """Ротация по размеру ПЕРЕД записью (вызывается под замком): текущий
        файл -> audit.jsonl.1, старшие сдвигаются, за KEEP - удаляются.
        rotate_mb=0 - выключено; keep=0 - история не хранится вовсе."""
        limit = self.rotate_mb * 1024 * 1024
        if limit <= 0:
            return
        try:
            size = self.path.stat().st_size
        except OSError:
            return  # файла нет (или не прочитался) - писать в новый
        if size + incoming <= limit:
            return
        if self.keep <= 0:
            self.path.unlink(missing_ok=True)
            return
        for k in range(self.keep - 1, 0, -1):
            src = self.dir / f"audit.jsonl.{k}"
            if src.exists():
                src.replace(self.dir / f"audit.jsonl.{k + 1}")
        self.path.replace(self.dir / "audit.jsonl.1")

    # ---------------------------------------------------------------- чтение

    def read(
        self,
        *,
        actor: str = "",
        action: str = "",
        q: str = "",
        since=None,
        until=None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """Фильтрованное чтение для GET /v1/admin/audit: actor/action - точное
        совпадение, q - подстрока по target/details без регистра, since/until -
        границы ISO-8601 (включительно), limit/offset - страница выборки.
        Возвращает {events, total, limit, offset}; total - ДО постраничного
        среза (кормит «Показать ещё» панели). Битые строки JSONL пропускаются.
        Некорректная дата - ValueError (API переведёт в 400)."""
        since_dt = _parse_bound(since) if since else None
        until_dt = _parse_bound(until) if until else None
        limit = max(0, min(1000, int(limit)))
        offset = max(0, int(offset))
        events: list[dict] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # недописанная/битая строка - не повод падать
                    if not self._matches(
                        event, actor, action, q, since_dt, until_dt
                    ):
                        continue
                    events.append(event)
        except FileNotFoundError:
            pass
        except OSError as e:  # noqa: PERF203 - чтение тоже never-throw
            logger.warning("журнал %s не прочитан: %s", self.path, e)
            return {"events": [], "total": 0, "limit": limit, "offset": offset}
        return {
            "events": events[offset:offset + limit],
            "total": len(events),
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _matches(
        event: dict,
        actor: str,
        action: str,
        q: str,
        since_dt: Optional[datetime],
        until_dt: Optional[datetime],
    ) -> bool:
        if actor and event.get("actor") != actor:
            return False
        if action and event.get("action") != action:
            return False
        if q:
            haystack = (
                str(event.get("target") or "") + "\n" + str(event.get("details") or "")
            ).lower()
            if q.lower() not in haystack:
                return False
        if since_dt is not None or until_dt is not None:
            try:
                ts = _as_utc(datetime.fromisoformat(str(event.get("ts") or "")))
            except ValueError:
                return False  # с временным фильтром несравнимое событие - мимо
            if since_dt is not None and ts < since_dt:
                return False
            if until_dt is not None and ts > until_dt:
                return False
        return True


def from_env(data_dir: Optional[Path] = None) -> AuditLog:
    """Журнал из окружения: VELES_DATA читается на вызов (как uploads_root
    в ingest/queue.py), чтобы тесты и кит подменяли корень подменой env."""
    if data_dir is not None:
        return AuditLog(Path(data_dir))
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        first = raw.split(os.pathsep)[0].strip()
        if first:
            return AuditLog(Path(first))
    # audit.py лежит прямо в пакете veles - сосед config импортируется
    # ОДНОЙ точкой (..config ушёл бы выше top-level, ImportError)
    from .config import VELES_DATA

    return AuditLog(Path(VELES_DATA))


def get_audit() -> AuditLog:
    """Синглтон журнала на процесс (лениво, под двойной проверкой) - тот же
    шов, что у очереди ингеста: вызов из любого запроса без прокидывания
    состояния; env зафиксирован на момент первого вызова."""
    global _instance
    if _instance is None:
        with _INSTANCE_LOCK:
            if _instance is None:
                _instance = from_env()
    return _instance


def reset_audit() -> None:
    """Сброс синглтона (тесты и горячая перезагрузка конфига)."""
    global _instance
    with _INSTANCE_LOCK:
        _instance = None
