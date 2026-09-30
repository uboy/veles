"""IdentityResolver: источник личности и allowed_groups запроса (шов Ф5, инвариант 5).

Группы доступа НЕ приходят от клиента (инвариант 5) - их резолвит сервер из
аутентифицированной личности до retrieval (pre-retrieval trimming).

Личный профиль (static): один владелец, статические группы из окружения,
заголовки запроса игнорируются.

Корпоративный профиль (K1): личность приходит заголовками Open WebUI
(ENABLE_FORWARD_USER_INFO_HEADERS) в одном из двух вариантов:
- ``headers``: plain заголовки X-OpenWebUI-User-Email (fallback Name) -
  для сети, где фасад недоступен в обход доверенного прокси;
- ``jwt``: заголовок X-OpenWebUI-User-Jwt содержит HS256-JWT
  (payload: sub/email/name/role/iat/exp), секрет - VELES_IDENTITY_JWT_SECRET,
  тот же, что FORWARD_USER_INFO_HEADER_JWT_SECRET в Open WebUI.

Верификация JWT - stdlib (hmac/base64/json), новых зависимостей нет.

Группы (K7): каждый аутентифицированный получает ``all`` («видно всем»)
и membership не-all групп из конфига групп (veles/policy/groups.py,
<VELES_DATA>/groups.json); группы из заголовков/JWT WebUI НЕ резолвятся -
Open WebUI 0.11.4 их в ENABLE_FORWARD_USER_INFO_HEADERS не передаёт.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol

from ..config import DEFAULT_ACCESS_GROUPS
from .groups import GROUP_ALL, defined_group_ids, membership_of

# Заголовки Open WebUI 0.11.4 (backend/open_webui/env.py, utils/headers.py)
HEADER_USER_NAME = "X-OpenWebUI-User-Name"
HEADER_USER_ID = "X-OpenWebUI-User-Id"
HEADER_USER_EMAIL = "X-OpenWebUI-User-Email"
HEADER_USER_ROLE = "X-OpenWebUI-User-Role"
HEADER_USER_JWT = "X-OpenWebUI-User-Jwt"

# user_id становится частью пути трекера: запрет обхода каталогов и
# drive-relative путей Windows ('c:'), '..' в любой позиции
_UNSAFE_USER_ID = re.compile(r"[\\/:]|\.\.|^\.|\x00")


def normalize_user_id(raw: str) -> str:
    """user_id нормализуется: email lower/strip."""
    return (raw or "").strip().lower()


def validated_user_id(raw: str) -> Optional[str]:
    """Нормализует и проверяет uid (email) из доверенного источника
    (K16: заголовок X-Veles-On-Behalf-Of при владельческом токене).
    None - пустой или небезопасный (обход каталогов, разделители пути):
    такой uid не должен попадать ни в document_id, ни в каталог L0."""
    uid = normalize_user_id(raw)
    if not uid or _UNSAFE_USER_ID.search(uid):
        return None
    return uid


def admin_emails() -> set[str]:
    """VELES_ADMIN_EMAILS - список админов через запятую (сравнение нормализованное)."""
    raw = os.environ.get("VELES_ADMIN_EMAILS", "")
    return {normalize_user_id(x) for x in raw.split(",") if x.strip()}


@dataclass
class RequestIdentity:
    """Личность одного запроса. user_id="" - владелец (bearer/token/static)."""

    user_id: str
    admin: bool
    groups: set[str]


def verify_hs256(token: str, secret: str, now: Optional[float] = None) -> Optional[dict]:
    """Проверка HS256-JWT stdlib-средствами: структура, alg, подпись, exp.

    Возвращает payload или None (битый/подделанный/просроченный/чужой секрет).
    """
    if not token or not secret:
        return None
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None

    def _b64url(seg: str) -> Optional[bytes]:
        try:
            return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))
        except Exception:
            return None

    head_raw, payload_raw, sig_raw = _b64url(parts[0]), _b64url(parts[1]), _b64url(parts[2])
    if head_raw is None or payload_raw is None or sig_raw is None:
        return None
    try:
        header = json.loads(head_raw)
        payload = json.loads(payload_raw)
    except Exception:
        return None
    if not isinstance(header, dict) or header.get("alg") != "HS256":
        return None
    expected = hmac.new(
        secret.encode("utf-8"), f"{parts[0]}.{parts[1]}".encode("utf-8"), hashlib.sha256
    ).digest()
    if not hmac.compare_digest(sig_raw, expected):
        return None
    exp = payload.get("exp")
    if exp is not None:
        try:
            expired = float(exp) <= float(time.time() if now is None else now)
        except (TypeError, ValueError):
            return None
        if expired:
            return None
    return payload if isinstance(payload, dict) else None


class IdentityResolver(Protocol):
    """Шов Ф5: резолв личности запроса из заголовков (инвариант 5)."""

    def resolve(self, headers: Mapping[str, str]) -> Optional[RequestIdentity]: ...


class StaticOwnerResolver:
    """Личный профиль: все запросы от владельца, группы фиксированы.

    VELES_ALLOWED_GROUPS="a,b" переопределяет дефолт (config.DEFAULT_ACCESS_GROUPS).
    """

    def __init__(self, groups: set[str] | None = None):
        self._groups = groups or DEFAULT_ACCESS_GROUPS

    @classmethod
    def from_env(cls) -> "StaticOwnerResolver":
        raw = os.environ.get("VELES_ALLOWED_GROUPS", "")
        groups = {g.strip() for g in raw.split(",") if g.strip()}
        return cls(groups or None)

    def allowed_groups(self) -> set[str]:
        """Группы владельца (K7): настроенные + all (неявно) + все определённые
        серверные - владелец админ, состав групп ему виден целиком. Произвольные
        access_groups вне конфига - как раньше: только с правом (группа в
        allowed_groups), не автоматически."""
        return set(self._groups) | {GROUP_ALL} | defined_group_ids()

    def resolve(self, headers: Mapping[str, str]) -> Optional[RequestIdentity]:
        return RequestIdentity(user_id="", admin=True, groups=self.allowed_groups())


class JwtHeaderResolver:
    """Корпоративный профиль: личность из заголовков Open WebUI (K1).

    mode="headers": X-OpenWebUI-User-Email (fallback X-OpenWebUI-User-Name).
    mode="jwt": X-OpenWebUI-User-Jwt, HS256, секрет обязателен.

    K7: у пользователя all (неявно все аутентифицированные) + membership не-all
    групп из конфига групп (<VELES_DATA>/groups.json, admin правит файлом);
    admin - по VELES_ADMIN_EMAILS, с wildcard "*".
    """

    def __init__(self, mode: str, secret: str = "", groups: set[str] | None = None):
        if mode not in ("headers", "jwt"):
            raise ValueError(f"неизвестный identity-режим: {mode!r}")
        if mode == "jwt" and not secret:
            raise ValueError("режим jwt требует VELES_IDENTITY_JWT_SECRET")
        self.mode = mode
        self.secret = secret
        self._groups = groups or DEFAULT_ACCESS_GROUPS

    @classmethod
    def from_env(cls, mode: str) -> "JwtHeaderResolver":
        raw = os.environ.get("VELES_IDENTITY_GROUPS", "")
        groups = {g.strip() for g in raw.split(",") if g.strip()}
        return cls(
            mode,
            secret=os.environ.get("VELES_IDENTITY_JWT_SECRET", ""),
            groups=groups or None,
        )

    def allowed_groups(self) -> set[str]:
        """Базовые группы владельца/bearer-доступа (K7): настроенные + all +
        все определённые серверные. Пер-пользовательский membership в это
        множество НЕ входит - он срезается по uid в _identity."""
        return set(self._groups) | {GROUP_ALL} | defined_group_ids()

    def resolve(self, headers: Mapping[str, str]) -> Optional[RequestIdentity]:
        if self.mode == "jwt":
            return self._resolve_jwt(headers)
        raw = headers.get(HEADER_USER_EMAIL) or headers.get(HEADER_USER_NAME) or ""
        # имя WebUI передаёт URL-процитанным (quote(name, safe=" "))
        uid = normalize_user_id(urllib.parse.unquote(raw))
        return self._identity(uid)

    def _resolve_jwt(self, headers: Mapping[str, str]) -> Optional[RequestIdentity]:
        candidates = [headers.get(HEADER_USER_JWT, "")]
        # защита от расхождений версий WebUI: JWT мог прийти в другом user-заголовке
        candidates += [
            v for k, v in headers.items()
            if k.lower().startswith("x-openwebui-user-")
            and k.lower() != HEADER_USER_JWT.lower()
        ]
        for value in candidates:
            payload = verify_hs256(value, self.secret)
            if payload is None:
                continue
            raw = payload.get("email") or payload.get("sub") or payload.get("id") or ""
            ident = self._identity(normalize_user_id(str(raw)))
            if ident is not None:
                return ident
        return None

    def _identity(self, uid: str) -> Optional[RequestIdentity]:
        if not uid or _UNSAFE_USER_ID.search(uid):
            return None
        admin = uid in admin_emails()
        # K4 (user ingest): у каждого пользователя личная группа user:<uid> -
        # его загрузки несут только её, поэтому чужие документы невидимы в поиске
        # даже при общих корпоративных группах; админ получает wildcard "*"
        # (hybrid_search пропускает групповой фильтр) - видит всё.
        # K7 (группы): all неявно у всех аутентифицированных; membership не-all
        # групп - из конфига групп; настроенные VELES_IDENTITY_GROUPS остаются
        # общими (обратная совместимость K1).
        groups = set(self._groups) | {GROUP_ALL} | membership_of(uid) | {f"user:{uid}"}
        if admin:
            groups.add("*")
        return RequestIdentity(uid, admin, groups)
