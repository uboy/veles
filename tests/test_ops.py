"""Тесты K6 (ops hardening): rate-limit фасада, security-заголовки HTML, readiness индекса.

Без моделей (как в test_api): фейковый пайплайн; окружение ставится monkeypatch.
"""

from __future__ import annotations

import threading
import time

from fastapi import Request
from fastapi.testclient import TestClient

from tests.test_api import make_pipeline
from veles.api import app as app_module
from veles.api.app import create_app
from veles.api.ratelimit import MAX_KEYS, SlidingWindowLimiter, _bucket_key, limiter_from_env


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


def _request(headers: dict | None = None, query: str = "", host: str = "10.0.0.1") -> Request:
    h = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({
        "type": "http", "method": "GET", "path": "/v1/models",
        "headers": h, "query_string": query.encode(), "client": (host, 12345),
    })


def test_bucket_key_ignores_sig_rotation():
    """Ревью K6 P1-2(а): sig= - случайная строка из запроса, в ключ не входит;
    ротация ?sig= не даёт нового бакета (фолбэк ip:)."""
    state = {"identity_mode": "static", "token": "", "resolver": None}
    k1 = _bucket_key(state, _request(query="sig=aaa"))
    k2 = _bucket_key(state, _request(query="sig=bbb"))
    assert k1 == k2 == "ip:10.0.0.1"


def test_bucket_key_invalid_tokens_share_one_bucket():
    """Ревью K6 P1-2(а): невалидные bearer/token= делят один бакет (брутфорс
    не получает бакет на попытку); валидный токен - стабильный бакет владельца."""
    state = {"identity_mode": "static", "token": "real-token", "resolver": None}
    junk = {_bucket_key(state, _request(headers={"authorization": f"Bearer junk{i}"}))
            for i in range(10)}
    junk.add(_bucket_key(state, _request(query="token=other-junk")))
    assert junk == {"tok:bad"}
    owner = _bucket_key(state, _request(headers={"authorization": "Bearer real-token"}))
    assert owner.startswith("tok:") and owner != "tok:bad"
    assert owner == _bucket_key(state, _request(query="token=real-token"))


def test_ratelimit_sig_rotation_does_not_bypass(monkeypatch):
    """HTTP-негатив: ротация ?sig= не обходит лимит - бакет общий, 429 наступает."""
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "2")
    client = TestClient(create_app(pipeline=None))
    assert client.get("/v1/models", params={"sig": "a"}).status_code == 200
    assert client.get("/v1/models", params={"sig": "b"}).status_code == 200
    assert client.get("/v1/models", params={"sig": "c"}).status_code == 429


def test_ratelimit_invalid_bearers_share_bucket(monkeypatch):
    """HTTP-негатив: 4 разных невалидных bearer'а - один бакет, 4-й запрос 429."""
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    monkeypatch.setenv("VELES_RATE_LIMIT_RPM", "3")
    client = TestClient(create_app(pipeline=None))
    codes = [client.get("/v1/models", headers={"Authorization": f"Bearer junk{i}"}).status_code
             for i in range(4)]
    assert codes == [200, 200, 200, 429]


def test_limiter_sweeps_expired_buckets():
    """Ревью K6 P1-2(б): при достижении MAX_KEYS бакеты с истёкшим окном
    вытесняются - флуд уникальными ключами не растит память без границы."""
    lim = SlidingWindowLimiter(rpm=2, window_s=60.0)
    for i in range(MAX_KEYS):
        lim.hit(f"k{i}", now=1000.0)
    assert len(lim._hits) == MAX_KEYS
    lim.hit("fresh", now=2000.0)  # порог достигнут: свипает истёкшие k*
    assert "k0" not in lim._hits and f"k{MAX_KEYS - 1}" not in lim._hits
    assert "fresh" in lim._hits


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


def test_healthz_not_blocked_while_index_loads(monkeypatch):
    """Ревью K6 P1-1: пока индекс грузится под локом (warmup из lifespan),
    /healthz и /readyz отвечают сразу с index_loaded=false, а не висят минуты.

    TestClient без `with` не запускает lifespan - warmup-поток не стартует и
    лок никогда не удержан; поэтому строго `with TestClient(app)`.
    """
    started = threading.Event()
    release = threading.Event()

    def slow_load(_index_dir):
        started.set()  # лок _pipeline() удержан, загрузка "висит"
        release.wait(timeout=10)
        raise RuntimeError("имитация долгой загрузки индекса")

    monkeypatch.setattr(app_module, "load_index", slow_load)
    monkeypatch.setattr(app_module, "SentenceTransformersProvider", lambda *a, **k: object())
    monkeypatch.setattr(app_module, "llm_config_from_env", lambda: None)
    monkeypatch.setattr(app_module, "OpenAICompatibleProvider", lambda cfg: None)
    app = create_app(pipeline=None)
    with TestClient(app) as client:  # with: запускает lifespan -> warmup держит лок
        assert started.wait(timeout=5)  # загрузка пошла, лок занят
        t0 = time.monotonic()
        r = client.get("/healthz")
        dt = time.monotonic() - t0
        assert r.status_code == 200
        assert r.json()["index_loaded"] is False  # честный ответ без ожидания лока
        # регрессия (лок в _index_loaded) висела бы до окончания загрузки (~10 c)
        assert dt < 2.0
        assert client.get("/readyz").status_code == 503
    release.set()
