"""Тесты K6 (ops hardening): rate-limit фасада, security-заголовки HTML, readiness индекса.

Без моделей (как в test_api): фейковый пайплайн; окружение ставится monkeypatch.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_api import make_pipeline
from veles.api.app import create_app
from veles.api.ratelimit import SlidingWindowLimiter, limiter_from_env


# --- rate-limit: скользящее окно ---


def test_limiter_window_slides():
    """Скользящее окно: сверх лимита - отказ с Retry-After, после окна - пропуск."""
    lim = SlidingWindowLimiter(rpm=3, window_s=60.0)
    t0 = 1000.0
    assert lim.hit("k", now=t0) == (True, 0)
    assert lim.hit("k", now=t0 + 1) == (True, 0)
    assert lim.hit("k", now=t0 + 2) == (True, 0)
    # 4-й в окне - отказ; окно освободится, когда уйдёт самое старое попадание
    ok, retry = lim.hit("k", now=t0 + 2.5)
    assert ok is False
    assert retry == 58  # ceil(1000 + 60 - 1002.5), падает по своей причине
    # ключи независимы (per-identity/per-token, не глобальный счётчик)
    assert lim.hit("other", now=t0 + 3) == (True, 0)
    # окно ушло - снова пропуск
    assert lim.hit("k", now=t0 + 61) == (True, 0)


def test_ratelimit_env_zero_or_garbage_disables(monkeypatch):
    """VELES_RATE_LIMIT_RPM=0 (и мусор) = выключено; пусто = щедрый дефолт."""
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "0")
    assert limiter_from_env() is None
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "много")
    assert limiter_from_env() is None
    monkeypatch.delenv("VELES_RATE_LIMIT_RPM", raising=False)
    lim = limiter_from_env()
    assert lim is not None and lim.rpm == 240


def test_ratelimit_over_limit_429_with_retry_after(monkeypatch):
    """Негатив: N+1-й запрос - 429 с целочисленным Retry-After."""
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "2")
    client = TestClient(create_app(pipeline=None))
    assert client.get("/v1/models").status_code == 200
    assert client.get("/v1/models").status_code == 200
    r = client.get("/v1/models")
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) >= 1


def test_ratelimit_zero_disables_endpoint(monkeypatch):
    """Лимит 0 = выкл: серия запросов не упирается в 429."""
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "0")
    client = TestClient(create_app(pipeline=None))
    for _ in range(10):
        assert client.get("/v1/models").status_code == 200


def test_ratelimit_per_identity(monkeypatch):
    """Identity-режим: бакет на личность - лимит одного не блокирует другого."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "1")
    client = TestClient(create_app(pipeline=None))
    ha = {"X-OpenWebUI-User-Email": "a@corp.ru"}
    hb = {"X-OpenWebUI-User-Email": "b@corp.ru"}
    assert client.get("/v1/models", headers=ha).status_code == 200
    assert client.get("/v1/models", headers=ha).status_code == 429
    assert client.get("/v1/models", headers=hb).status_code == 200


def test_healthz_and_readyz_not_rate_limited(monkeypatch):
    """Пробы мониторинга не лимитируются: 429 на /healthz читается как сбой."""
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "1")
    client = TestClient(create_app(pipeline=None))
    for _ in range(3):
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503  # индекса нет, но не 429


# --- security-заголовки на HTML /tracker ---


def test_tracker_security_headers(monkeypatch, tmp_path):
    """HTML-страница трекера отдаёт nosniff/DENY/no-referrer и строгий CSP."""
    (tmp_path / "owner" / "task").mkdir(parents=True)
    monkeypatch.setenv("TRACKER_ROOT", str(tmp_path))
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    client = TestClient(create_app(pipeline=None))
    r = client.get("/tracker")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["Referrer-Policy"] == "no-referrer"
    csp = r.headers["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "unsafe-inline" not in csp
    # HEAD (curl -I) отдаёт те же заголовки - FastAPI сам HEAD к GET не добавляет
    rh = client.head("/tracker")
    assert rh.status_code == 200
    assert rh.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in rh.headers


def test_json_endpoints_have_no_security_headers(monkeypatch, tmp_path):
    """JSON-эндпоинты не тронуты: security-заголовков на них нет."""
    (tmp_path / "owner" / "task").mkdir(parents=True)
    monkeypatch.setenv("TRACKER_ROOT", str(tmp_path))
    client = TestClient(create_app(pipeline=None))
    for path in ("/healthz", "/v1/models"):
        r = client.get(path)
        assert r.status_code == 200
        for h in ("X-Content-Type-Options", "X-Frame-Options",
                  "Referrer-Policy", "Content-Security-Policy"):
            assert h not in r.headers, f"{path}: {h} не должен ставиться на JSON"


# --- readiness: liveness vs готовность индекса ---


def test_readiness_without_index_is_honest():
    """Без индекса /healthz остаётся живостью (200), а /readyz честно 503."""
    client = TestClient(create_app(pipeline=None))
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"  # старый контракт /healthz сохранён
    assert body["index_loaded"] is False
    rz = client.get("/readyz")
    assert rz.status_code == 503
    assert rz.json()["index_loaded"] is False


def test_readiness_with_pipeline_is_ready():
    client = TestClient(create_app(pipeline=make_pipeline()))
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["index_loaded"] is True
    rz = client.get("/readyz")
    assert rz.status_code == 200
    assert rz.json() == {"status": "ready", "index_loaded": True}
