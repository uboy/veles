"""Серверные группы доступа (K7): определения, состав, membership.

Зачем: при загрузке (POST /v1/documents) пользователь выбирает группу
(group=<id>), документ несёт её в access_groups, и существующий ACL-фильтр
(K3/K4) отдаёт документ только тем, у кого группа есть в identity. Состав
не-all групп правит админ правкой конфига (админ-API групп - не в этой фазе;
«видно по запросу» - процесс вне системы). Группа ``all`` («видно всем») -
неявно все аутентифицированные: её добавляет в groups identity-резолвер,
members у неё в конфиге не задаются.

Хранение: JSON-файл <VELES_DATA>/groups.json (переопределяется
VELES_GROUPS_FILE). Почему файл, а не env JSON: состав - многострочный список
identity, который админ правит руками; .env кита содержит сгенерированные
секреты (риск задеть их при правке), а docker env_file не переносит
многострочные значения и не снимает кавычки - JSON в env приходил бы с
литеральными кавычками. Чтение - на вызов (как uploads_root/admin_emails):
правка состава действует без рестарта фасада.

Fail-closed: файла нет - действуют дефолтные три группы; файл битый - он НЕ
перезаписывается (правка админа не затирается), действуют дефолты - membership
пуст, не-all документы только сужают видимость, расширения нет. Файл-шаблон с
дефолтами создаётся лениво (GET /v1/groups), чтобы чтения identity из тестов
и разных окружений не плодили файлы. Невалидные записи пропускаются с
предупреждением в лог.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("veles.policy.groups")

# «видно всем»: identity-резолвер добавляет всем аутентифицированным,
# поэтому в конфиге members не задаётся, а при загрузке валидна всегда
GROUP_ALL = "all"

# id группы: ascii-слаг (попадает в access_groups манифеста и в group= запроса);
# двоеточие исключено - служебные личные группы user:<uid> через конфиг не создать
GROUP_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")

# Дефолтные группы (решение владельца 29.09); действуют и без конфига
DEFAULT_GROUPS: tuple[tuple[str, str], ...] = (
    (GROUP_ALL, "Видно всем"),
    ("on-request", "Видно по запросу"),
    ("test-dept", "Тестовый отдел"),
)

_TITLE_MAX = 200


@dataclass(frozen=True)
class GroupDef:
    """Одна группа: id, человекочитаемое название, состав (identity lower/strip)."""

    id: str
    title: str
    members: frozenset[str]


def _norm_member(raw) -> str:
    """member нормализуется так же, как user_id в identity (lower/strip)."""
    return str(raw or "").strip().lower()


def groups_file() -> Path:
    """Путь конфига групп: VELES_GROUPS_FILE, иначе <VELES_DATA>/groups.json.
    Env читается на вызов (как uploads_root) - тесты и кит подменяют окружение."""
    raw = (os.environ.get("VELES_GROUPS_FILE") or "").strip()
    if raw:
        return Path(raw.split(os.pathsep)[0].strip())
    raw = (os.environ.get("VELES_DATA") or "").strip()
    if raw:
        first = raw.split(os.pathsep)[0].strip()
        if first:
            return Path(first) / "groups.json"
    from ..config import VELES_DATA

    return Path(VELES_DATA) / "groups.json"


def _defaults() -> list[GroupDef]:
    return [GroupDef(gid, title, frozenset()) for gid, title in DEFAULT_GROUPS]


def _parse(raw_text: str) -> list[GroupDef]:
    """Разбор конфига: {"groups": [{id, title, members: [...]}]}.

    Невалидная запись пропускается с предупреждением; структурно битый файл
    бросает ValueError (load_groups тогда уходит в дефолт, файл не трогается).
    """
    data = json.loads(raw_text)
    if not isinstance(data, dict) or not isinstance(data.get("groups"), list):
        raise ValueError("ожидается объект с ключом groups: [...]")
    groups: list[GroupDef] = []
    seen: set[str] = set()
    for i, entry in enumerate(data["groups"]):
        if not isinstance(entry, dict):
            logger.warning("groups.json: запись %d не объект - пропущена", i)
            continue
        gid = str(entry.get("id") or "").strip()
        if not GROUP_ID_RE.fullmatch(gid):
            logger.warning(
                "groups.json: id %r не ascii-слаг ([a-z0-9][a-z0-9_-]*) - запись %d пропущена",
                gid, i,
            )
            continue
        if gid in seen:
            logger.warning("groups.json: дубликат id %r - первая запись побеждает", gid)
            continue
        title = str(entry.get("title") or "").strip()[:_TITLE_MAX] or gid
        raw_members = entry.get("members")
        if raw_members is not None and not isinstance(raw_members, list):
            logger.warning("groups.json: members группы %r не список - состав пуст", gid)
            raw_members = []
        members = frozenset(m for m in map(_norm_member, raw_members or []) if m)
        seen.add(gid)
        groups.append(GroupDef(gid, title, members))
    return groups


def load_groups() -> list[GroupDef]:
    """Группы из конфига; файла нет или он не читается - дефолтные три
    (fail-closed: membership пуст, видимость только сужается). Чтение на
    вызов: правка состава админом действует без рестарта фасада."""
    try:
        return _parse(groups_file().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _defaults()
    except (OSError, ValueError) as e:  # noqa: BLE001 - битый конфиг не роняет фасад
        logger.warning(
            "конфиг групп %s не прочитан (%s) - действуют дефолтные группы",
            groups_file(), e,
        )
        return _defaults()


def defined_group_ids() -> set[str]:
    """Все определённые id; all входит всегда (неявно у всех аутентифицированных,
    даже если админ не перечислил его в конфиге)."""
    return {g.id for g in load_groups()} | {GROUP_ALL}


def group_exists(group_id: str) -> bool:
    """Валидация group= при загрузке: группа из конфига или all."""
    return group_id == GROUP_ALL or any(g.id == group_id for g in load_groups())


def membership_of(uid: str) -> set[str]:
    """Не-all группы, в которых состоит uid (all резолвер добавляет всем)."""
    uid = _norm_member(uid)
    if not uid:
        return set()
    return {g.id for g in load_groups() if uid in g.members}


def ensure_groups_file() -> Path:
    """Создать файл-шаблон с дефолтными группами, если конфига ещё нет
    (best-effort, атомарно tmp+replace). Существующий (и битый) файл не
    трогается. Вызывается из GET /v1/groups: админ видит, где править состав."""
    path = groups_file()
    try:
        if path.exists():
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "groups": [
                # members - identity (email) через запятую в JSON-списке;
                # состав правит админ, фасад перечитывает файл на каждый запрос
                {"id": gid, "title": title, "members": []}
                for gid, title in DEFAULT_GROUPS
            ]
        }
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp.replace(path)
    except OSError as e:  # noqa: BLE001 - шаблон не критичен, группы уже действуют
        logger.warning("шаблон %s не создан (%s) - дефолтные группы действуют без файла", path, e)
    return path
