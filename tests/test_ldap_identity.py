"""Регрессия корпоративного входа LDAP/AD (эпик ldap-identity, фаза 1).

In-process LDAP-сервер на ldaptor (Twisted reactor в фоновом потоке): bind по
userPassword, чтение memberOf записи и posixGroup (memberUid), живая мутация
атрибутов - отзыв группы виден при следующем входе. Негативные кейсы: неверный
пароль 401 неотличим от неизвестного пользователя (и от DN-инъекции), пароль
не в логах/ответе, маршрут отключён без VELES_LDAP_URL (404), отсутствие
пакета ldap3 - честная ошибка конфигурации (500), пользователь вне маппинга
получает пустые группы.

Квирк ldaptor, учтённый в дереве: фильтр (objectClass=*) у него матчит только
записи, у которых атрибут objectClass заполнен (в RFC-каталоге он структурный,
у всех записей) - поэтому objectClass задан у каждой записи дерева.
"""

from __future__ import annotations

import json
import os
import logging
import sys
import threading

import pytest
from fastapi.testclient import TestClient

from veles.api.app import create_app
from veles.policy.identity import JwtHeaderResolver, sign_hs256, verify_hs256
from veles.policy.ldap_identity import (
    LdapConfigError,
    escape_dn,
    escape_filter,
    group_map,
    jwt_ttl_s,
    timeout_s,
)

SECRET = "jwt-secret-ldap-тест"

BASE_DN = "dc=corp,dc=ru"
BIND_TEMPLATE = "uid={username},ou=people,dc=corp,dc=ru"
GROUP_BASE_DN = "ou=groups,dc=corp,dc=ru"

# первый ключ в другом регистре: DN каталога регистронезависимы, маппинг обязан
# сходиться и на это (нормализация lower/strip в group_map)
GROUP_MAP = {
    "CN=analysts,OU=groups,DC=corp,DC=ru": "analysts",
    "cn=devs,ou=groups,dc=corp,dc=ru": "devs",
}

IVANOV = "ivanov"
IVANOV_PASSWORD = "пароль-иванова"
PETROV = "petrov"
PETROV_PASSWORD = "пароль-петрова"


def _build_tree():
    """Каталог «корпорация»: ivanov (memberOf analysts + posixGroup devs),
    petrov (без групп ни в одной форме); запись groups с posixGroup."""
    from ldaptor.inmemory import ReadOnlyInMemoryLDAPEntry

    root = ReadOnlyInMemoryLDAPEntry(
        BASE_DN, {"dc": ["corp", "ru"], "objectClass": ["dcObject", "organization"]}
    )
    people = root.addChild(
        "ou=people", {"ou": ["people"], "objectClass": ["organizationalUnit"]}
    )
    people.addChild(
        "uid=ivanov",
        {
            "uid": [IVANOV],
            "cn": ["Иванов И."],
            "objectClass": ["inetOrgPerson"],
            "userPassword": [IVANOV_PASSWORD],
            "memberOf": ["cn=analysts,ou=groups,dc=corp,dc=ru"],
        },
    )
    people.addChild(
        "uid=petrov",
        {
            "uid": [PETROV],
            "cn": ["Петров П."],
            "objectClass": ["inetOrgPerson"],
            "userPassword": [PETROV_PASSWORD],
        },
    )
    groups = root.addChild(
        "ou=groups", {"ou": ["groups"], "objectClass": ["organizationalUnit"]}
    )
    groups.addChild(
        "cn=devs",
        {
            "cn": ["devs"],
            "objectClass": ["posixGroup"],
            "memberUid": [IVANOV],
        },
    )
    return root


@pytest.fixture(scope="module")
def ldap_reactor():
    """Один Twisted reactor на модуль - фоновый daemon-поток; порты каждого
    теста закрываются, reactor останавливается в конце модуля."""
    from twisted.internet import reactor

    started = threading.Event()

    def _run():
        started.set()
        reactor.run(installSignalHandlers=False)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    assert started.wait(10), "reactor не поднялся"
    yield reactor
    reactor.callFromThread(reactor.stop)
    thread.join(timeout=10)


@pytest.fixture()
def ldap_server(ldap_reactor, monkeypatch):
    """Свежее LDAP-дерево на 127.0.0.1:<ephemeral> + env конфигурации входа.
    Возвращает root дерева: атрибуты живые, отзыв группы мутацией виден
    следующему входу (как в настоящем каталоге после правки админом)."""
    from ldaptor import interfaces as ldap_ifaces
    from ldaptor.protocols.ldap.ldapserver import LDAPServer
    from twisted.internet import protocol as tprotocol
    from twisted.python.components import registerAdapter

    class Factory(tprotocol.ServerFactory):
        protocol = LDAPServer

        def __init__(self, root):
            self.root = root

    registerAdapter(lambda x: x.root, Factory, ldap_ifaces.IConnectedLDAPEntry)

    root = _build_tree()
    box: list = []
    done = threading.Event()

    def _listen():
        try:
            box.append(ldap_reactor.listenTCP(0, Factory(root), interface="127.0.0.1"))
        finally:
            done.set()

    ldap_reactor.callFromThread(_listen)
    assert done.wait(10), "listenTCP не завершился"
    port = box[0]

    monkeypatch.setenv("VELES_LDAP_URL", f"ldap://127.0.0.1:{port.getHost().port}")
    monkeypatch.setenv("VELES_LDAP_BIND_DN_TEMPLATE", BIND_TEMPLATE)
    monkeypatch.setenv("VELES_LDAP_BASE_DN", BASE_DN)
    monkeypatch.setenv("VELES_LDAP_GROUP_BASE_DN", GROUP_BASE_DN)
    monkeypatch.setenv("VELES_LDAP_GROUP_MAP", json.dumps(GROUP_MAP))
    monkeypatch.setenv("VELES_LDAP_TIMEOUT_S", "5")
    monkeypatch.setenv("VELES_IDENTITY_JWT_SECRET", SECRET)
    monkeypatch.setenv("VELES_IDENTITY_MODE", "jwt")
    yield root
    ldap_reactor.callFromThread(lambda: port.stopListening())


@pytest.fixture()
def client(ldap_server):
    return TestClient(create_app(pipeline=None))


def _login(client: TestClient, username: str, password: str):
    return client.post("/v1/auth/ldap", json={"username": username, "password": password})


def test_login_success_groups_from_mapping(client) -> None:
    """Успешный вход: bind ок, обе формы membership (memberOf + posixGroup)
    прошли маппинг; jwt несёт ровно sub+groups+exp, группы совпадают."""
    r = _login(client, IVANOV, IVANOV_PASSWORD)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    payload = verify_hs256(body["access_token"], SECRET)
    assert payload is not None, "выпущенный токен обязан проходить verify_hs256"
    assert set(payload) == {"sub", "groups", "exp"}
    assert payload["sub"] == IVANOV
    assert set(payload["groups"]) == {"analysts", "devs"}
    assert set(body["groups"]) == {"analysts", "devs"}


def test_issued_token_resolves_to_identity_with_groups(client) -> None:
    """Токен входа работает в существующем контуре: JwtHeaderResolver
    принимает его как X-OpenWebUI-User-Jwt, группы из claims попадают в
    identity; чужим секретом токен не резолвится."""
    token = _login(client, IVANOV, IVANOV_PASSWORD).json()["access_token"]
    resolver = JwtHeaderResolver("jwt", secret=SECRET)
    ident = resolver.resolve({"X-OpenWebUI-User-Jwt": token})
    assert ident is not None and ident.user_id == IVANOV
    assert {"analysts", "devs"} <= ident.groups
    assert "all" in ident.groups and f"user:{IVANOV}" in ident.groups
    assert resolver.resolve({"X-OpenWebUI-User-Jwt": token + "подмена"}) is None
    # и через API: личность принимается без 401
    r = client.get("/v1/models", headers={"X-OpenWebUI-User-Jwt": token})
    assert r.status_code == 200


def test_wrong_password_and_unknown_user_indistinguishable(client) -> None:
    """401 на неверный пароль и на несуществующего пользователя - одно и то же
    сообщение и одинаковое тело (перечисление пользователей невозможно)."""
    wrong = _login(client, PETROV, "неверный-пароль-иванова")
    unknown = _login(client, "такого-нет-в-каталоге", "любой-пароль")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "неверные учётные данные"}


def test_username_injection_does_not_bypass(client) -> None:
    """Инъекция спецсимволами LDAP не пробивает: ни DN-шаблон (своим RDN
    пользователь не расширяется), ни фильтры групп. Даже с ВЕРНЫМ паролем
    иванова - bind проходит только на его записи, инъекционные логины дают 401."""
    injections = [
        "*",  # wildcard в DN
        "*)(uid=ivanov",  # попытка сломать DN/фильтр
        f"{IVANOV})(|(memberOf=*))",  # расширение фильтра групп
        "uid=ivanov,ou=people,dc=corp,dc=ru",  # полный DN вместо uid (DIT-обход)
        f"{IVANOV},uid=admin",  # добавка RDN
        f"{IVANOV})(",  # обрыв фильтра
        f"{IVANOV}\\2a",  # escape-байты как литералы
    ]
    for name in injections:
        r = _login(client, name, IVANOV_PASSWORD)
        assert r.status_code == 401, f"инъекция {name!r} не должна проходить"


def test_password_not_in_logs_or_response(client, caplog) -> None:
    """Пароль не попадает ни в логи (на любом уровне), ни в тело ответа -
    ни при отказе, ни при успешном входе."""
    with caplog.at_level(logging.DEBUG):
        bad = _login(client, IVANOV, "пароль-ошибочный-секрет")
        ok = _login(client, PETROV, PETROV_PASSWORD)
    assert bad.status_code == 401 and ok.status_code == 200
    assert "пароль-ошибочный-секрет" not in caplog.text
    assert "пароль-ошибочный-секрет" not in bad.text
    assert PETROV_PASSWORD not in caplog.text
    assert PETROV_PASSWORD not in ok.text


def test_user_outside_mapping_gets_empty_groups(client) -> None:
    """Пользователь есть в каталоге, но ни одна его группа не в маппинге:
    вход валиден, группы пустые; в identity остаются только общие (all +
    личная user:<uid>), ни одной группы из маппинга."""
    body = _login(client, PETROV, PETROV_PASSWORD).json()
    assert body["groups"] == []
    payload = verify_hs256(body["access_token"], SECRET)
    assert payload is not None and payload["groups"] == []
    ident = JwtHeaderResolver("jwt", secret=SECRET).resolve(
        {"X-OpenWebUI-User-Jwt": body["access_token"]}
    )
    assert ident is not None
    assert "all" in ident.groups and f"user:{PETROV}" in ident.groups
    assert not ({"analysts", "devs"} & ident.groups)


def test_group_revocation_invalidates_on_next_login(ldap_server, client) -> None:
    """Отзыв из группы в каталоге виден при следующем входе: старые группы
    не возвращаются (токен перевыпускается из живого состояния каталога)."""
    first = _login(client, IVANOV, IVANOV_PASSWORD)
    assert set(first.json()["groups"]) == {"analysts", "devs"}

    entry = ldap_server.lookup(f"uid={IVANOV},ou=people,dc=corp,dc=ru").result
    entry.__delitem__("memberOf")  # админ снял memberOf
    second = _login(client, IVANOV, IVANOV_PASSWORD)
    assert second.status_code == 200
    assert "analysts" not in second.json()["groups"], "отозванная memberOf-группа вернулась"
    assert second.json()["groups"] == ["devs"]

    devs = ldap_server.lookup("cn=devs,ou=groups,dc=corp,dc=ru").result
    devs.__delitem__("memberUid")  # админ вывел из posixGroup
    third = _login(client, IVANOV, IVANOV_PASSWORD)
    assert third.status_code == 200
    assert third.json()["groups"] == []
    # в identity тоже пусто, кроме общих
    ident = JwtHeaderResolver("jwt", secret=SECRET).resolve(
        {"X-OpenWebUI-User-Jwt": third.json()["access_token"]}
    )
    assert ident is not None
    assert not ({"analysts", "devs"} & ident.groups)


def test_route_disabled_without_env(monkeypatch) -> None:
    """Fail-closed: без VELES_LDAP_URL маршрут не регистрируется - POST
    получает 404, ничего не ломает существующие режимы."""
    monkeypatch.delenv("VELES_LDAP_URL", raising=False)
    client = TestClient(create_app(pipeline=None))
    r = client.post("/v1/auth/ldap", json={"username": IVANOV, "password": "x"})
    assert r.status_code == 404


def test_missing_ldap3_is_honest_config_error(client, monkeypatch) -> None:
    """VELES_LDAP_URL настроен, а пакета ldap3 нет - 500 с честным сообщением
    (какую зависимость поставить), а не молчаливое отключение входа."""
    monkeypatch.setitem(sys.modules, "ldap3", None)
    r = _login(client, IVANOV, IVANOV_PASSWORD)
    assert r.status_code == 500
    assert "ldap3" in r.json()["detail"]


def test_escape_dn_and_filter() -> None:
    """Экранирование RFC 4514 (DN) и RFC 4515 (фильтры): инъекционные
    спецсимволы становятся литералами своего значения."""
    assert escape_dn("ivanov,uid=admin") == "ivanov\\,uid\\=admin"
    assert escape_dn('a"b+c\\d<e>f;g') == 'a\\"b\\+c\\\\d\\<e\\>f\\;g'
    assert escape_dn("#инъекция") == "\\#инъекция"
    assert escape_dn(" пробелы ") == "\\ пробелы\\ "
    # RFC 4515: слэш экранируется первым, готовые \\XX-последовательности
    # не портятся; скобки/звезда/NUL - hex-эскейпы
    assert escape_filter("a(b)c*d\\e") == "a\\28b\\29c\\2ad\\5ce"
    assert escape_filter("\\2a") == "\\5c2a"
    assert escape_filter("x\x00y") == "x\\00y"


def test_config_errors_are_honest(monkeypatch) -> None:
    """Битая конфигурация - LdapConfigError (500 на маршруте), а не тихое
    дефолтное поведение: битый JSON маппинга, нечисловой/отрицательный
    таймаут и TTL."""
    monkeypatch.setenv("VELES_LDAP_GROUP_MAP", "{битый json")
    with pytest.raises(LdapConfigError):
        group_map()
    monkeypatch.setenv("VELES_LDAP_GROUP_MAP", '["не объект"]')
    with pytest.raises(LdapConfigError):
        group_map()
    monkeypatch.setenv("VELES_LDAP_TIMEOUT_S", "abc")
    with pytest.raises(LdapConfigError):
        timeout_s()
    monkeypatch.setenv("VELES_LDAP_TIMEOUT_S", "-1")
    with pytest.raises(LdapConfigError):
        timeout_s()
    monkeypatch.setenv("VELES_LDAP_JWT_TTL_S", "abc")
    with pytest.raises(LdapConfigError):
        jwt_ttl_s()
    monkeypatch.setenv("VELES_LDAP_JWT_TTL_S", "0")
    with pytest.raises(LdapConfigError):
        jwt_ttl_s()


def test_sign_hs256_symmetric_with_verify() -> None:
    """sign_hs256 (выпуск) и verify_hs256 (проверка) симметричны: roundtrip
    сходится, чужой секрет и подмена не проходят."""
    token = sign_hs256({"sub": "a", "groups": [], "exp": 4102444800}, SECRET)
    payload = verify_hs256(token, SECRET)
    assert payload is not None and payload["sub"] == "a"
    assert verify_hs256(token, "чужой-секрет") is None
    with pytest.raises(ValueError):
        sign_hs256({}, "")


def test_DEBUG_ci_candidates(ldap_server):
    """Временная диагностика CI: версии, кандидаты адресов, пошаговый коннект."""
    import sys

    import ldap3
    import twisted
    import ldaptor

    print("PY:", sys.version.split()[0], "| twisted:", twisted.__version__,
          "| ldaptor:", getattr(ldaptor, "__version__", "?"),
          "| ldap3:", ldap3.__version__)
    s = ldap3.Server(os.environ["VELES_LDAP_URL"], connect_timeout=5, get_info=None)
    print("SERVER host/port:", repr(s.host), repr(s.port))
    cands = list(s.candidate_addresses())
    print("CANDIDATES:", [(c[0], c[4]) for c in cands])
    for a in cands:
        try:
            sk = socket.socket(*a[:3])
            sk.settimeout(5)
            sk.connect(a[4])
            sk.close()
            print("CONNECT-OK:", a[4])
        except Exception as e:
            print("CONNECT-FAIL:", a[4], type(e).__name__, e)
    conn = ldap3.Connection(s, user="cn=x", password="y", auto_bind=False)
    try:
        conn.open()
        print("OPEN-OK")
    except Exception as e:
        print("OPEN-FAIL:", type(e).__name__, e)
    assert False, "debug: см. Captured stdout"


def test_receive_timeout_is_int() -> None:
    """Регрессия CI 12.10: receive_timeout=float на Linux ронял bind
    (ldap3 pack('LL', ...) требует int); connect_timeout может быть float."""
    from veles.policy.ldap_identity import receive_timeout_s

    assert isinstance(receive_timeout_s(), int) and receive_timeout_s() >= 1
