"""IdentityResolver: источник allowed_groups запроса (шов Ф5, инвариант 5).

Группы доступа НЕ приходят от клиента (инвариант 5) - их резолвит сервер из
аутентифицированной личности до retrieval (pre-retrieval trimming). Личный
профиль: один владелец, статические группы из окружения. Корпоративный
профиль (Ф5): JWT/OIDC-резолвер реализует тот же Protocol без изменения
кода retrieval и API.
"""

from __future__ import annotations

import os
from typing import Protocol

from ..config import DEFAULT_ACCESS_GROUPS


class IdentityResolver(Protocol):
    def allowed_groups(self) -> set[str]: ...


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
        return set(self._groups)
