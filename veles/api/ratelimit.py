"""Rate-limit фасада (K6): скользящее окно в памяти, per-identity/per-token.

VELES_RATE_LIMIT_RPM - лимит запросов в минуту: в identity-режимах (headers/jwt)
ключ - личность пользователя, в static - токен (bearer/token=); без всего -
IP клиента. 0 (и неразборчивое значение) = выключено, пусто = щедрый дефолт.
Превышение - 429 с Retry-After (секунды до освобождения окна).

Ключ собирается только из ВАЛИДИРОВАННОГО (ревью K6 P1-2): bearer/token=
сравнивается с настроенным токеном фасада - совпал, стабильный бакет владельца;
не совпал, общий бакет tok:bad (иначе брутфорс/ротация токенов давали бы новый
бакет на каждый запрос и обходили лимит); sig= подписанной ссылки в ключ не
входит по той же причине - это случайная строка из запроса. Фолбэк - IP клиента.
Известное ограничение: за reverse-proxy (caddy) ip-фолбэк видит один IP - все
анонимы делят один бакет; XFF не доверяем (спуфинг).

Память: dict бакетов ограничен вытеснением - при достижении MAX_KEYS свипаются
бакеты, чьё окно истекло (не чаще раза за окно; между свипами dict может
превышать порог).

Состояние в памяти процесса: рестарт сбрасывает счётчики (осознанно - это
грубое ограничение "один клиент не забивает фасад", а не точный учёт; для
жёстких квот нужен общий стор). /healthz и /readyz лимитом не накрываются:
проба мониторинга, отвалившаяся по 429, читается как сбой сервиса.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

WINDOW_S = 60.0
DEFAULT_RPM = 240  # щедрый дефолт: обычные клиенты (чат, мост) не упираются
# порог dict бакетов: выше - свип бакетов с истёкшим окном (ревью K6 P1-2)
MAX_KEYS = 10_000
# пробы мониторинга не лимитируются (см. докстринг модуля)
EXEMPT_PATHS = frozenset({"/healthz", "/readyz"})


class SlidingWindowLimiter:
    """Скользящее окно на ключ: не более rpm попаданий за window_s.

    hit() возвращает (True, 0), если запрос пропущен, иначе (False, retry_after).
    """

    def __init__(self, rpm: int, window_s: float = WINDOW_S):
        self.rpm = rpm
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def _sweep_expired(self, border: float) -> None:
        """Вытеснение бакетов, чьё окно истекло: последнее попадание старше border
        (в пустом deque попаданий нет - вытесняется сразу)."""
        dead = [k for k, q in self._hits.items() if not q or q[-1] <= border]
        for k in dead:
            del self._hits[k]

    def hit(self, key: str, now: float | None = None) -> tuple[bool, int]:
        ts = time.time() if now is None else now
        with self._lock:
            border = ts - self.window_s
            if len(self._hits) >= MAX_KEYS and ts - self._last_sweep >= self.window_s:
                # пустые deque сами не вытесняются - флуд уникальными ключами
                # не должен расти без границы (ревью K6 P1-2)
                self._last_sweep = ts
                self._sweep_expired(border)
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= border:
                q.popleft()
            if len(q) >= self.rpm:
                # окно освободится, когда из него уйдёт самое старое попадание
                retry_after = max(1, math.ceil(q[0] + self.window_s - ts))
                return False, retry_after
            q.append(ts)
            return True, 0


def limiter_from_env() -> SlidingWindowLimiter | None:
    """VELES_RATE_LIMIT_RPM: пусто - щедрый дефолт; 0/мусор - None (выключено)."""
    raw = os.environ.get("VELES_RATE_LIMIT_RPM", "").strip()
    rpm = DEFAULT_RPM
    if raw:
        try:
            rpm = int(raw)
        except ValueError:
            return None
    return SlidingWindowLimiter(rpm) if rpm > 0 else None


def _bucket_key(state: dict, request: Request) -> str:
    """Ключ бакета: личность (identity-режимы) -> валидный bearer/token= -> ip.

    В ключ берётся только валидированное (детали - докстринг модуля): токен
    сравнивается с настроенным токеном фасада, случайные строки из запроса
    (ротация ?sig=/невалидные токены) новых бакетов не создают. Токены не
    хранятся в открытом виде: ключ - короткий sha256.
    """
    if state["identity_mode"] != "static":
        ident = state["resolver"].resolve(request.headers)
        if ident is not None and ident.user_id:
            return f"id:{ident.user_id}"
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else auth
    if not token:
        token = request.query_params.get("token", "")
    if token:
        state_token = state["token"]
        if state_token and token == state_token:
            return "tok:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
        # невалидный токен - один общий бакет, а не ключ из данных запроса
        return "tok:bad"
    host = request.client.host if request.client is not None else "unknown"
    return f"ip:{host}"


def install_rate_limit(
    app: FastAPI, state: dict, limiter: SlidingWindowLimiter | None = None
) -> None:
    """Подключение лимита к приложению (вызывается из create_app одной строкой).

    limiter=None - берётся из окружения (VELES_RATE_LIMIT_RPM); выключено - no-op.
    """
    lim = limiter if limiter is not None else limiter_from_env()
    if lim is None:
        return

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        ok, retry_after = lim.hit(_bucket_key(state, request))
        if not ok:
            return JSONResponse(
                {"detail": "слишком много запросов, повторите позже"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        return await call_next(request)
