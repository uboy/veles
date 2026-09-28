"""Регрессия K1 (корпоративная идентичность): JwtHeaderResolver, изоляция трекера,
подписанные ссылки. Static-режим обратно совместим."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from veles.api.app import create_app
from veles.policy.identity import (
    JwtHeaderResolver,
    normalize_user_id,
    verify_hs256,
)

SECRET = "jwt-secret-тест"


def make_jwt(payload: dict, secret: str = SECRET) -> str:
    def b64(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    head = b64({"alg": "HS256", "typ": "JWT"})
    body = b64(payload)
    sig = hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest()
    return f"{head}.{body}." + base64.urlsafe_b64encode(sig).rstrip(b"=").decode()


def test_verify_hs256_roundtrip_and_tamper() -> None:
    payload = make_jwt({"email": "Ivanov@corp.ru", "exp": time.time() + 600})
    parsed = verify_hs256(payload, SECRET)
    assert parsed and parsed["email"] == "Ivanov@corp.ru"
    assert verify_hs256(payload, "другой-секрет") is None
    tampered = payload[:-2] + ("aa" if not payload.endswith("aa") else "bb")
    assert verify_hs256(tampered, SECRET) is None
    expired = make_jwt({"email": "a@b", "exp": time.time() - 10})
    assert verify_hs256(expired, SECRET) is None


def test_jwt_resolver_headers_and_normalize() -> None:
    r = JwtHeaderResolver("jwt", secret=SECRET)
    ident = r.resolve({"X-OpenWebUI-User-Jwt": make_jwt({"email": "Ivanov@corp.ru"})})
    assert ident is not None and ident.user_id == "ivanov@corp.ru"
    assert ident.groups  # аутентифицированному - группы по умолчанию
    assert r.resolve({}) is None
    assert normalize_user_id(" A@B.C ") == "a@b.c"


def test_headers_resolver_email_fallback() -> None:
    r = JwtHeaderResolver("headers")
    ident = r.resolve({"X-OpenWebUI-User-Email": "Petrova@corp.ru"})
    assert ident and ident.user_id == "petrova@corp.ru"
    ident2 = r.resolve({"X-OpenWebUI-User-Name": "Петров П."})
    assert ident2 and ident2.user_id.startswith("петров")
    assert r.resolve({}) is None


def test_jwt_mode_requires_secret() -> None:
    with pytest.raises(ValueError):
        JwtHeaderResolver("jwt")


def test_identity_mode_headers_requires_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("TRACKER_ROOT", str(tmp_path))
    app = create_app(pipeline=None)
    client = TestClient(app)
    # без заголовков личности - 401 ещё до похода в индекс (fail-closed)
    assert client.get("/v1/knowledge/stats").status_code == 401
    # с личностью - 200 на /v1/models (индекс для stats в тестовом окружении не собран)
    r = client.get("/v1/models", headers={"X-OpenWebUI-User-Email": "a@corp.ru"})
    assert r.status_code == 200 and r.json()["data"][0]["id"].startswith("veles-")
    monkeypatch.setenv("VELES_IDENTITY_MODE", "static")
    app2 = create_app(pipeline=None)  # режим фиксируется при создании приложения
    # static без настроенного токена = личный профиль без пароля
    assert TestClient(app2).get("/v1/models").status_code == 200


def test_tracker_isolation_in_identity_mode(monkeypatch, tmp_path) -> None:
    (tmp_path / "ivanov@corp.ru" / "task").mkdir(parents=True)
    (tmp_path / "petrov@corp.ru" / "task").mkdir(parents=True)
    (tmp_path / "petrov@corp.ru" / "task" / "task-1-secret.md").write_text(
        "секрет Петрова", encoding="utf-8"
    )
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("TRACKER_ROOT", str(tmp_path))
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")
    client = TestClient(create_app(pipeline=None))

    # Иванов не видит каталог Петрова ни списком, ни файлом
    r = client.get("/tracker", headers={"X-OpenWebUI-User-Email": "ivanov@corp.ru"})
    assert "Петрова" not in r.text and "секрет" not in r.text
    r2 = client.get(
        "/tracker/file/task/task-1-secret.md",
        headers={"X-OpenWebUI-User-Email": "ivanov@corp.ru"},
    )
    assert r2.status_code == 404

    # админ видит по chat=
    r3 = client.get(
        "/tracker", params={"chat": "petrov@corp.ru"},
        headers={"X-OpenWebUI-User-Email": "boss@corp.ru"},
    )
    assert r3.status_code == 200 and "petrov@corp.ru" in r3.text
    r4 = client.get(
        "/tracker/file/task/task-1-secret.md",
        headers={"X-OpenWebUI-User-Email": "boss@corp.ru"},
    )
    assert r4.status_code == 200 and "секрет Петрова" in r4.text


def test_signed_source_link(tmp_path, monkeypatch) -> None:
    import io as _io
    import json as _json

    canon = tmp_path / "data" / "canonical-test"
    doc_dir = canon / "lectures_x.txt"
    doc_dir.mkdir(parents=True)
    (doc_dir / "manifest.json").write_text(_json.dumps(
        {"title": "Т", "document_id": "lectures/x.txt", "revision_id": "r1",
         "parser_profile": "txt-md:0.1", "access_groups": ["personal"],
         "sha256": "a", "status": "current", "derived_from": []}), encoding="utf-8")
    (doc_dir / "nodes.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("VELES_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("VELES_LINK_SECRET", "link-secret")

    app = create_app(pipeline=None, api_token="tok")
    client = TestClient(app)
    exp = int(time.time()) + 600
    sig = hmac.new(b"link-secret", b"lectures/x.txt|nv1|" + str(exp).encode(),
                   hashlib.sha256).hexdigest()
    r = client.get("/v1/source/lectures/x.txt",
                   params={"node": "nv1", "exp": str(exp), "sig": sig})
    assert r.status_code == 200
    assert client.get("/v1/source/lectures/x.txt",
                      params={"node": "nv1", "exp": str(exp), "sig": "bad"}).status_code == 401
    old = int(time.time()) - 10
    sig_old = hmac.new(b"link-secret", b"lectures/x.txt|nv1|" + str(old).encode(),
                       hashlib.sha256).hexdigest()
    assert client.get("/v1/source/lectures/x.txt",
                      params={"node": "nv1", "exp": str(old), "sig": sig_old}).status_code == 401
