"""Rate-limit фасада (K6): скользящее окно в памяти, per-identity/per-token.

VELES_RATE_LIMIT_RPM - лимит запросов в минуту: в identity-режимах (headers/jwt)
ключ - личность пользователя, в static - токен (bearer/token=); без всего -
IP клиента. 0 (и неразборчивое значение) = выключено, пусто = щедрый дефолт.
Превышение - 429 с Retry-After (секунды до освобождения окна).

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

    def hit(self, key: str, now: float | None = None) -> tuple[bool, int]:
        ts = time.time() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            border = ts - self.window_s
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
    """Ключ бакета: личность (identity-режимы) -> bearer/token= -> подпись ссылки -> ip.

    Токены не хранятся в открытом виде: ключ - короткий sha256 (память процесса,
    но и ключей незачем быть секретами).
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
        return "tok:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    sig = request.query_params.get("sig", "")
    if sig:
        return "sig:" + hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]
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
