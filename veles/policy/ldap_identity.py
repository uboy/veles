"""LDAP/AD-аутентификация: корпоративный вход логином/паролем (эпик ldap-identity).

Поток: POST /v1/auth/ldap (veles/api/app.py) -> bind в каталоге по DN-шаблону ->
чтение групп пользователя (memberOf записи, fallback posixGroup RFC 2307) ->
маппинг VELES_LDAP_GROUP_MAP (DN группы LDAP -> id группы veles) -> выпуск
короткоживущего HS256-jwt (только sub+groups+exp, identity.sign_hs256).
Дальше существующий контур identity работает сам: клиент ходит с токеном как
X-OpenWebUI-User-Jwt, JwtHeaderResolver резолвит личность вместе с группами
из claims (подпись уже проверена), фильтр по группам в retrieval не менялся.

Выключено по умолчанию (fail-closed): нет VELES_LDAP_URL - маршрут не
регистрируется (404), существующие режимы (bearer/headers/jwt) не затронуты.

Безопасность:
- username экранируется в DN (RFC 4514) и в LDAP-фильтрах (RFC 4515) - инъекция
  спецсимволами не меняет ни запись bind, ни множество групп;
- неверный пароль и неизвестный пользователь неотличимы (401, одно и то же
  сообщение) - перечисление пользователей через эндпоинт невозможно;
- пароль не логируется: в лог при сбое каталога пишется только тип исключения;
- jwt несёт только sub/groups/exp - без email/имени/роли.

Клиент - ldap3, импортируется лениво (зависимость veles[ldap]); при настроенном
VELES_LDAP_URL без установленного пакета - честная ошибка конфигурации (500),
а не молчаливое отключение.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass

from .identity import normalize_user_id

logger = logging.getLogger("veles.policy.ldap_identity")

# логин становится user_id (sub, группа user:<uid>): '\' ':' '/' '..' и
# ведущая точка запрещены с той же семантикой, что identity._UNSAFE_USER_ID -
# иначе резолв личности позднее молча отверг бы токен (fail-closed здесь)
_UNSAFE_LOGIN = re.compile(r"[\\/:]|\.\.|^\.|\x00")

# время жизни выпущенного jwt по умолчанию (короткоживущий по плану эпика)
DEFAULT_JWT_TTL_S = 600

# единое сообщение 401: неверный пароль и неизвестный пользователь неразличимы
AUTH_FAILED = "неверные учётные данные"


class LdapConfigError(Exception):
    """Настройка LDAP неполна/битая - честная ошибка конфигурации (500)."""


class LdapAuthError(Exception):
    """Неверные учётные данные - 401 без деталей (см. AUTH_FAILED)."""


class LdapUnavailableError(Exception):
    """Каталог недоступен или не ответил в таймаут - 503."""


@dataclass(frozen=True)
class LdapIdentity:
    """Результат входа: нормализованный логин и группы veles после маппинга."""

    user_id: str
    groups: frozenset[str]


def ldap_url() -> str:
    """VELES_LDAP_URL (ldap://host:389, ldaps:// - средствами ldap3)."""
    return os.environ.get("VELES_LDAP_URL", "").strip()


def is_configured() -> bool:
    """Включён ли корпоративный вход: маршрут регистрируется только при нём."""
    return bool(ldap_url())


def bind_dn_template() -> str:
    """VELES_LDAP_BIND_DN_TEMPLATE с плейсхолдером {username}: DN пользователя
    ('uid={username},ou=people,{base}') либо UPN/NTLM-форма ('{username}@corp.ru')
    - в последнем случае memberOf ищется поиском по sAMAccountName/uid."""
    tpl = os.environ.get("VELES_LDAP_BIND_DN_TEMPLATE", "").strip()
    if not tpl:
        raise LdapConfigError("VELES_LDAP_BIND_DN_TEMPLATE не задан")
    if "{username}" not in tpl:
        raise LdapConfigError(
            "VELES_LDAP_BIND_DN_TEMPLATE: ожидается плейсхолдер {username}"
        )
    return tpl


def base_dn() -> str:
    """VELES_LDAP_BASE_DN - суффикс каталога (поиск пользователя без DN-формы)."""
    dn = os.environ.get("VELES_LDAP_BASE_DN", "").strip()
    if not dn:
        raise LdapConfigError("VELES_LDAP_BASE_DN не задан")
    return dn


def group_base_dn() -> str:
    """VELES_LDAP_GROUP_BASE_DN (posixGroup); пусто - базовый суффикс каталога."""
    raw = os.environ.get("VELES_LDAP_GROUP_BASE_DN", "").strip()
    return raw or base_dn()


def timeout_s() -> float:
    """VELES_LDAP_TIMEOUT_S - таймаут соединения/ответа каталога (сек, >0)."""
    raw = os.environ.get("VELES_LDAP_TIMEOUT_S", "").strip()
    if not raw:
        return 5.0
    try:
        t = float(raw)
    except ValueError as e:
        raise LdapConfigError("VELES_LDAP_TIMEOUT_S: ожидается число секунд") from e
    if t <= 0:
        raise LdapConfigError("VELES_LDAP_TIMEOUT_S: должен быть положительным")
    return t


def jwt_ttl_s() -> int:
    """Время жизни выпущенного jwt (VELES_LDAP_JWT_TTL_S, дефолт 600с).
    exp проверяется существующим verify_hs256 на каждом запросе - «повторный
    вход после отзыва» не полагается на время жизни токена."""
    raw = os.environ.get("VELES_LDAP_JWT_TTL_S", "").strip()
    if not raw:
        return DEFAULT_JWT_TTL_S
    try:
        ttl = int(raw)
    except ValueError as e:
        raise LdapConfigError("VELES_LDAP_JWT_TTL_S: ожидается целое число секунд") from e
    if ttl <= 0:
        raise LdapConfigError("VELES_LDAP_JWT_TTL_S: должен быть положительным")
    return ttl


def group_map() -> dict[str, str]:
    """VELES_LDAP_GROUP_MAP - JSON «DN группы LDAP -> id группы veles».
    Ключи нормализуются (lower/strip): DN в каталоге регистронезависимы.
    Пусто/не задано - маппинга нет (все пользователи получают пустые группы).
    Битый JSON - LdapConfigError (500), а не молчаливое игнорирование."""
    raw = os.environ.get("VELES_LDAP_GROUP_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise LdapConfigError(f"VELES_LDAP_GROUP_MAP: не JSON ({e})") from e
    if not isinstance(data, dict):
        raise LdapConfigError(
            "VELES_LDAP_GROUP_MAP: ожидается объект {DN группы LDAP: id группы veles}"
        )
    out: dict[str, str] = {}
    for k, v in data.items():
        dn, gid = str(k).strip().lower(), str(v).strip()
        if dn and gid:
            out[dn] = gid
    return out


def escape_dn(value: str) -> str:
    """Экранирование username в DN по RFC 4514 §2.4: \\ " # + , ; < = >,
    ведущий # и ведущий/замыкающий пробел, управляющие символы - \\XX.
    Инъекция «ivanov,uid=admin» не выходит за пределы своего RDN."""
    out: list[str] = []
    for i, ch in enumerate(value):
        if ch in '\\"#+,;<=>' or (ch == " " and (i == 0 or i == len(value) - 1)):
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\{ord(ch):02x}")
        else:
            out.append(ch)
    return "".join(out)


def escape_filter(value: str) -> str:
    """Экранирование значения в LDAP-фильтре по RFC 4515: \\ ( ) * и NUL ->
    \\XX. Слэш экранируется первым, чтобы не испортить готовые \\XX-последовательности."""
    return (
        value.replace("\\", "\\5c")
        .replace("*", "\\2a")
        .replace("(", "\\28")
        .replace(")", "\\29")
        .replace("\x00", "\\00")
    )


def _ldap3():
    """Ленивый импорт ldap3: пакет отсутствует при настроенном VELES_LDAP_URL -
    честная ошибка конфигурации, а не молчаливое отключение входа."""
    try:
        import ldap3
    except ImportError as e:
        raise LdapConfigError(
            "VELES_LDAP_URL настроен, но пакет ldap3 не установлен "
            "(pip install 'veles[ldap]')"
        ) from e
    return ldap3


def authenticate(username: str, password: str) -> LdapIdentity:
    """Вход: bind по DN-шаблону -> группы -> маппинг в группы veles.

    LdapAuthError - неверный пароль/неизвестный пользователь/пустые поля
    (один ответ 401); LdapUnavailableError - каталог недоступен (503);
    LdapConfigError - неполная/битая настройка (500). Пароль никуда не
    логируется и в исключения не попадает (в логе только тип исключения).
    """
    if not ldap_url():
        raise LdapConfigError("VELES_LDAP_URL не задан")
    if not username or not password:
        raise LdapAuthError()
    uid = normalize_user_id(username)
    if not uid or _UNSAFE_LOGIN.search(uid):
        # не пытаемся даже bind: такой uid не резолвится в identity позднее
        raise LdapAuthError()

    ldap3 = _ldap3()
    user_dn = bind_dn_template().format(username=escape_dn(uid))
    bdn = base_dn()
    gbase = group_base_dn()
    try:
        server = ldap3.Server(ldap_url(), connect_timeout=timeout_s(), get_info=None)
        conn = ldap3.Connection(
            server, user=user_dn, password=password,
            auto_bind=True, receive_timeout=timeout_s(),
        )
    except ldap3.core.exceptions.LDAPBindError:
        # сервер ответил отказом в bind: неверный пароль или записи нет -
        # снаружи оба случая выглядят одинаково (перечисления пользователей нет)
        raise LdapAuthError() from None
    except ldap3.core.exceptions.LDAPException as e:
        logger.warning("LDAP недоступен (%s)", type(e).__name__)
        raise LdapUnavailableError("каталог пользователей недоступен") from None
    try:
        found = _collect_groups(conn, ldap3, uid, user_dn, bdn, gbase)
    finally:
        conn.unbind()
    mapping = group_map()
    mapped = frozenset(mapping[dn] for dn in found if dn in mapping)
    return LdapIdentity(uid, mapped)


def _collect_groups(
    conn, ldap3, uid: str, user_dn: str, bdn: str, gbase: str
) -> list[str]:
    """DN групп пользователя: memberOf записи (AD/389ds) + posixGroup с
    memberUid=логин (RFC 2307). Чтение - правами самого пользователя (bind
    без сервисной учётки). Fail-closed: сбой чтения групп - пустой список
    (вход валиден, привилегий нет) и предупреждение в лог с типом исключения."""
    found: list[str] = []
    esc = escape_filter(uid)
    # memberOf: DN-форма шаблона - точечное чтение записи; UPN/NTLM-форма -
    # поиск пользователя по учётному имени
    try:
        if "=" in user_dn:
            ok = conn.search(
                user_dn, "(objectClass=*)", search_scope="BASE",
                attributes=["memberOf"],
            )
            entries = conn.entries if ok else []
        else:
            ok = conn.search(
                bdn, f"(|(sAMAccountName={esc})(uid={esc}))", search_scope="SUBTREE",
                attributes=["memberOf"],
            )
            entries = conn.entries if ok else []
        for entry in entries:
            for dn in entry.entry_attributes_as_dict.get("memberOf", []) or []:
                if dn:
                    found.append(str(dn))
    except ldap3.core.exceptions.LDAPException as e:
        logger.warning("memberOf не прочитан (%s)", type(e).__name__)
    # posixGroup: memberUid - сам логин, экранированный в фильтре
    try:
        ok = conn.search(
            gbase, f"(&(objectClass=posixGroup)(memberUid={esc}))",
            search_scope="SUBTREE", attributes=[],
        )
        for entry in conn.entries if ok else []:
            if entry.entry_dn:
                found.append(entry.entry_dn)
    except ldap3.core.exceptions.LDAPException as e:
        logger.warning("posixGroup не прочитан (%s)", type(e).__name__)
    return found
