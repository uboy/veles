"""K7: группы при загрузке - group= в POST /v1/documents, GET /v1/groups,
membership из конфига групп (<VELES_DATA>/groups.json), all неявно у всех
аутентифицированных, K3-ревизия несёт группу НОВОЙ загрузки. Дефолт
user:<uid> (K4) и static-режим не сломаны.

Индекс - настоящий build_index; эмбеддер - детерминированный hashing
(канон: тесты без загрузки моделей) - общий с test_documents_ingest.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.test_documents_ingest import (
    DOC_MARK_V1,
    DOC_MARK_V2,
    DOC_QUERY,
    DOC_SLUG,
    DOC_TITLE,
    DOC_V1,
    DOC_V2,
    HashingEmbedder,
    kb_env,  # noqa: F401 - pytest-фикстура (tmp VELES_DATA + мини-индекс)
    make_static_client,
    query_texts,
    upload_json,
    wait_done,
)
from tests.test_identity import make_jwt
from veles.api.app import VelesPipeline, create_app
from veles.canonical.store import _doc_dir
from veles.ingest.queue import user_dir
from veles.policy.groups import (
    GROUP_ALL,
    group_exists,
    load_groups,
    membership_of,
)
from veles.policy.identity import JwtHeaderResolver, StaticOwnerResolver

IVANOV = "ivanov@corp.ru"
PETROV = "petrov@corp.ru"
BOSS = "boss@corp.ru"
OWNER_AUTH = {"Authorization": "Bearer owner-token"}

# дефолтный конфиг (тот же, что пишет ensure_groups_file)
DEFAULT_GROUPS = [("all", "Видно всем"), ("on-request", "Видно по запросу"),
                  ("test-dept", "Тестовый отдел")]
# конфиг с составом: иванов - член тестового отдела
DEPT_GROUPS = [
    {"id": "all", "title": "Видно всем", "members": []},
    {"id": "on-request", "title": "Видно по запросу", "members": []},
    {"id": "test-dept", "title": "Тестовый отдел", "members": [IVANOV]},
]


def h(user: str) -> dict:
    return {"X-OpenWebUI-User-Email": user}


@pytest.fixture()
def headers_env(kb_env, monkeypatch):
    """kb_env + корпоративный headers-режим: админ boss, bearer-владелец."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", BOSS)
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    return kb_env


def make_app(bundle) -> TestClient:
    return TestClient(
        create_app(pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None))
    )


def write_groups(data, groups: list[dict]) -> None:
    (data / "groups.json").write_text(
        json.dumps({"groups": groups}, ensure_ascii=False), encoding="utf-8"
    )


def upload(client: TestClient, *, text: str = DOC_V1, title: str = DOC_TITLE,
           group: str | None = None, headers: dict | None = None):
    payload: dict = {"text": text, "title": title}
    if group is not None:
        payload["group"] = group
    return client.post("/v1/documents", json=payload, headers=headers or {})


def doc_groups(canonical, doc_id: str) -> list[str]:
    d = _doc_dir(doc_id, canonical)
    return json.loads((d / "manifest.json").read_text(encoding="utf-8"))["access_groups"]


def stat_doc_ids(client: TestClient, headers: dict) -> set[str]:
    d = client.get("/v1/knowledge/stats", headers=headers).json()
    return {doc["document_id"] for doc in d["documents"]}


# ------------------------------------------------------------ GET /v1/groups


def test_groups_endpoint_auth_and_defaults(headers_env) -> None:
    """Без личности - 401 (та же авторизация, что у остальных API); с личностью -
    дефолтные три группы с человеческими названиями, без members и без user:*;
    вызов создаёт файл-шаблон в VELES_DATA."""
    data = headers_env["data"]
    client = make_app(headers_env["bundle"])
    assert client.get("/v1/groups").status_code == 401

    r = client.get("/v1/groups", headers=h(IVANOV))
    assert r.status_code == 200
    items = r.json()["groups"]
    assert [g["id"] for g in items] == [gid for gid, _ in DEFAULT_GROUPS]
    assert [g["title"] for g in items] == [title for _, title in DEFAULT_GROUPS]
    assert all("members" not in g for g in items)  # состав не раскрывается
    assert not any(g["id"].startswith("user:") for g in items)
    assert (data / "groups.json").exists()  # шаблон материализован


def test_groups_endpoint_custom_config_and_override(headers_env, monkeypatch, tmp_path) -> None:
    """Конфиг админа замещает дефолт; VELES_GROUPS_FILE переопределяет путь
    (дефолтный <VELES_DATA>/groups.json тогда не создаётся)."""
    write_groups(headers_env["data"], DEPT_GROUPS)
    client = make_app(headers_env["bundle"])
    ids = [g["id"] for g in client.get("/v1/groups", headers=h(PETROV)).json()["groups"]]
    assert ids == ["all", "on-request", "test-dept"]

    custom = tmp_path / "custom-groups.json"
    custom.write_text(
        json.dumps({"groups": [{"id": "legal", "title": "Юристы", "members": [PETROV]}]},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("VELES_GROUPS_FILE", str(custom))
    override = make_app(headers_env["bundle"])
    items = override.get("/v1/groups", headers=h(PETROV)).json()["groups"]
    assert [g["id"] for g in items] == ["legal"]
    # membership из переопределённого конфига доезжает до identity
    petrov = JwtHeaderResolver.from_env("headers").resolve(h(PETROV))
    assert "legal" in petrov.groups


# ------------------------------------------------------- загрузка с group=all


def test_upload_group_all_visible_to_everyone(headers_env) -> None:
    """group=all: документ «видно всем» - находят все аутентифицированные
    (поиск, source 200, stats), манифест несёт [all]."""
    ivanov_client = make_app(headers_env["bundle"])
    petrov_client = make_app(headers_env["bundle"])

    r = upload(ivanov_client, group="all", headers=h(IVANOV))
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert wait_done(ivanov_client, r.json()["job_id"], headers=h(IVANOV))["status"] == "done"

    assert doc_groups(headers_env["canonical"], doc_id) == [GROUP_ALL]
    for client, headers in ((ivanov_client, h(IVANOV)), (petrov_client, h(PETROV)),
                            (make_app(headers_env["bundle"]), OWNER_AUTH)):
        texts = query_texts(client, headers=headers)
        assert any(DOC_MARK_V1 in t for t in texts)
        assert client.get(f"/v1/source/{doc_id}", headers=headers).status_code == 200
        assert doc_id in stat_doc_ids(client, headers)


# ---------------------------------------------------- group=test-dept: ACL


def test_upload_group_membership_acl(headers_env) -> None:
    """test-dept: член группы видит документ (поиск, source/asset, stats),
    чужой - 404 на source/asset и пусто в поиске/stats; админ (wildcard) и
    bearer-владелец (админ группы в allowed_groups) видят."""
    write_groups(headers_env["data"], DEPT_GROUPS)
    ivanov_client = make_app(headers_env["bundle"])
    petrov_client = make_app(headers_env["bundle"])

    r = ivanov_client.post(
        "/v1/documents",
        files={"file": ("отчёт_отдела.txt", DOC_V1.encode("utf-8"), "text/plain")},
        data={"group": "test-dept", "title": "Отчёт отдела"},
        headers=h(IVANOV),
    )
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    job_id = r.json()["job_id"]
    assert wait_done(ivanov_client, job_id, headers=h(IVANOV))["status"] == "done"
    assert doc_groups(headers_env["canonical"], doc_id) == ["test-dept"]

    # актив - как в K4: парсер txt активов не создаёт, кладём руками
    doc_dir = _doc_dir(doc_id, headers_env["canonical"])
    (doc_dir / "assets").mkdir()
    (doc_dir / "assets" / "схема.png").write_bytes(b"\x89PNG-fake")

    # чужой: нигде
    assert not any(DOC_MARK_V1 in t for t in query_texts(petrov_client, headers=h(PETROV)))
    assert not any(DOC_MARK_V1 in t
                   for t in query_texts(petrov_client, headers=h(PETROV), revision="all"))
    assert petrov_client.get(f"/v1/source/{doc_id}", headers=h(PETROV)).status_code == 404
    assert petrov_client.get(f"/v1/asset/{doc_id}/схема.png", headers=h(PETROV)).status_code == 404
    assert doc_id not in stat_doc_ids(petrov_client, h(PETROV))
    # чужой джоб не палится
    assert petrov_client.get(f"/v1/documents/jobs/{job_id}", headers=h(PETROV)).status_code == 404

    # член группы: везде
    assert any(DOC_MARK_V1 in t for t in query_texts(ivanov_client, headers=h(IVANOV)))
    assert ivanov_client.get(f"/v1/source/{doc_id}", headers=h(IVANOV)).status_code == 200
    assert ivanov_client.get(f"/v1/asset/{doc_id}/схема.png", headers=h(IVANOV)).status_code == 200
    assert doc_id in stat_doc_ids(ivanov_client, h(IVANOV))

    # админ по email - wildcard; bearer-владелец - группы владельца
    # (определённые серверные группы в allowed_groups)
    boss_client = make_app(headers_env["bundle"])
    assert any(DOC_MARK_V1 in t for t in query_texts(boss_client, headers=h(BOSS)))
    assert boss_client.get(f"/v1/source/{doc_id}", headers=h(BOSS)).status_code == 200
    owner_client = make_app(headers_env["bundle"])
    assert any(DOC_MARK_V1 in t for t in query_texts(owner_client, headers=OWNER_AUTH))
    assert owner_client.get(f"/v1/source/{doc_id}", headers=OWNER_AUTH).status_code == 200


# ------------------------------------------------------- невалидная группа


def test_upload_unknown_group_400_no_job(headers_env) -> None:
    """Неизвестная группа - 400 с понятным текстом, джоба и L0 нет;
    служебную user:* группой выбрать нельзя."""
    client = make_app(headers_env["bundle"])
    r = upload(client, group="нет-такая", headers=h(IVANOV))
    assert r.status_code == 400
    assert "неизвестная группа" in r.json()["detail"]
    assert "all" in r.json()["detail"]  # перечень доступных в тексте

    r2 = client.post(
        "/v1/documents",
        files={"file": ("файл.txt", DOC_V1.encode("utf-8"), "text/plain")},
        data={"group": "user:ivanov@corp.ru"},
        headers=h(IVANOV),
    )
    assert r2.status_code == 400

    r3 = upload(client, group=GROUP_ALL + "-x", headers=h(IVANOV))
    assert r3.status_code == 400  # префикс существующей не проходит

    assert client.get("/v1/documents/jobs", headers=h(IVANOV)).json()["jobs"] == []
    assert not (headers_env["data"] / "uploads" / user_dir(IVANOV) / f"{DOC_SLUG}.md").exists()


def test_upload_without_group_stays_personal(headers_env) -> None:
    """Регрессия K4: без group - личная user:<uid> (не all, не derived-группы)."""
    client = make_app(headers_env["bundle"])
    r = upload(client, headers=h(IVANOV))
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert wait_done(client, r.json()["job_id"], headers=h(IVANOV))["status"] == "done"
    assert doc_groups(headers_env["canonical"], doc_id) == [f"user:{IVANOV}"]
    petrov_client = make_app(headers_env["bundle"])
    assert not any(DOC_MARK_V1 in t for t in query_texts(petrov_client, headers=h(PETROV)))


# --------------------------------------------------- ревизия: группа новой загрузки


def test_revision_group_from_new_upload(headers_env) -> None:
    """K3 + K7: повторная загрузка того же документа с другой группой -> r2
    несёт группу НОВОЙ загрузки; чужой видит архивную r1 (она была all), но не
    текущую r2."""
    write_groups(headers_env["data"], DEPT_GROUPS)
    client = make_app(headers_env["bundle"])
    r1 = upload(client, group="all", headers=h(IVANOV))
    doc_id = r1.json()["document_id"]
    j1 = wait_done(client, r1.json()["job_id"], headers=h(IVANOV))
    assert (j1["status"], j1["action"]) == ("done", "created")
    j2 = wait_done(client, upload(client, text=DOC_V2, group="test-dept",
                                  headers=h(IVANOV)).json()["job_id"], headers=h(IVANOV))
    assert (j2["status"], j2["action"], j2["revision_id"]) == ("done", "updated", "r2")

    assert doc_groups(headers_env["canonical"], doc_id) == ["test-dept"]

    # член: в дефолтном поиске только новая редакция (r2, test-dept)
    texts = query_texts(client, headers=h(IVANOV))
    assert any(DOC_MARK_V2 in t for t in texts)
    assert all(DOC_MARK_V1 not in t for t in texts)
    # чужой: текущую редакцию не видит ни так, ни с revision=all;
    # архивная r1 помечена all - по ней чужой правомочен (revision=all)
    petrov_client = make_app(headers_env["bundle"])
    assert not any(DOC_MARK_V2 in t
                   for t in query_texts(petrov_client, headers=h(PETROV)))
    all_texts = query_texts(petrov_client, headers=h(PETROV), revision="all")
    assert any(DOC_MARK_V1 in t for t in all_texts)
    assert all(DOC_MARK_V2 not in t for t in all_texts)
    assert petrov_client.get(f"/v1/source/{doc_id}", headers=h(PETROV)).status_code == 404
    # член видит и архивную редакцию параметром
    r1_texts = query_texts(client, headers=h(IVANOV), revision="r1")
    assert any(DOC_MARK_V1 in t for t in r1_texts)


# ------------------------------------------------------------ identity-резолвер


def test_resolver_all_and_membership(headers_env) -> None:
    """all есть в groups каждого аутентифицированного; membership не-all групп -
    из конфига; админ дополнительно с wildcard; static-владелец - все
    определённые группы."""
    write_groups(headers_env["data"], DEPT_GROUPS)
    # from_env: VELES_IDENTITY_GROUPS читается при создании резолвера
    resolver = JwtHeaderResolver.from_env("headers")
    ivanov = resolver.resolve(h(IVANOV))
    petrov = resolver.resolve(h(PETROV))
    assert {"all", "test-dept", f"user:{IVANOV}"} <= ivanov.groups
    assert "all" in petrov.groups
    assert "test-dept" not in petrov.groups
    assert "corp" in ivanov.groups  # VELES_IDENTITY_GROUPS сохранены (K1)
    boss = resolver.resolve(h(BOSS))
    # админ видит всё через wildcard "*", а не через membership
    assert "*" in boss.groups and "test-dept" not in boss.groups

    jwt = JwtHeaderResolver("jwt", secret="jwt-secret")
    ident = jwt.resolve({"X-OpenWebUI-User-Jwt": make_jwt({"email": IVANOV}, "jwt-secret")})
    assert ident is not None and {"all", "test-dept"} <= ident.groups

    static = StaticOwnerResolver.from_env()
    assert {"all", "on-request", "test-dept"} <= static.allowed_groups()


# ------------------------------------------------------- устойчивость конфига


def test_corrupt_groups_file_fail_closed(headers_env) -> None:
    """Битый конфиг: файл НЕ перезаписывается, действуют дефолтные группы
    (fail-closed: membership пуст - видимость только сужается), загрузка
    с group= работает."""
    data = headers_env["data"]
    (data / "groups.json").write_text("{битый json", encoding="utf-8")
    client = make_app(headers_env["bundle"])
    items = client.get("/v1/groups", headers=h(IVANOV)).json()["groups"]
    assert [g["id"] for g in items] == [gid for gid, _ in DEFAULT_GROUPS]
    assert (data / "groups.json").read_text(encoding="utf-8") == "{битый json"
    r = upload(client, group="test-dept", headers=h(IVANOV))
    assert r.status_code == 202
    assert wait_done(client, r.json()["job_id"], headers=h(IVANOV))["status"] == "done"


def test_groups_config_invalid_entries_skipped(headers_env) -> None:
    """Невалидные записи конфига пропускаются (не роняют чтение), валидные
    применяются; members нормализуются; служебные проверки group_exists."""
    write_groups(headers_env["data"], [
        {"id": "Видно Всем", "title": "не ascii-слаг"},        # пропущена
        {"id": "sales", "title": "Продажи", "members": ["  A@B.C "]},
        {"id": "sales", "title": "дубль"},                     # дубль - первая побеждает
        {"id": "hr", "title": "HR", "members": "не список"},   # members -> пусто
        "не объект",                                           # пропущена
        {"id": "ops", "title": "Операции", "members": ["", ".."]},
    ])
    groups = load_groups()
    assert [g.id for g in groups] == ["sales", "hr", "ops"]
    assert membership_of("a@b.c") == {"sales"}
    assert membership_of(PETROV) == set()
    assert group_exists("all") and group_exists("sales")
    assert not group_exists("нет-такая")
    assert not group_exists("")


# ---------------------------------------------------------------- static-режим


def test_static_mode_not_broken(kb_env) -> None:
    """Static: дефолтная загрузка несёт personal (derived-группы резолвера
    документу не полагаются), владелец видит документ с group=all, а без права
    не видит произвольные access_groups (семантика K4 сохранена)."""
    client = make_static_client(kb_env["bundle"])
    r = upload_json(client, DOC_V1)
    assert r.status_code == 202
    doc_id = r.json()["document_id"]
    assert wait_done(client, r.json()["job_id"])["status"] == "done"
    assert doc_groups(kb_env["canonical"], doc_id) == ["personal"]

    r_all = upload(client, text=DOC_V2, group="all")
    assert r_all.status_code == 202
    all_id = r_all.json()["document_id"]
    assert wait_done(client, r_all.json()["job_id"])["status"] == "done"
    assert doc_groups(kb_env["canonical"], all_id) == [GROUP_ALL]
    assert any(DOC_MARK_V2 in t for t in query_texts(client))

    r_restricted = client.post(
        "/v1/documents",
        files={"file": ("бюджет.txt", "Бюджет: смета затрат 100".encode("utf-8"),
                        "text/plain")},
        data={"access_groups": "finance-restricted", "title": "Бюджет"},
    )
    assert r_restricted.status_code == 202
    assert wait_done(client, r_restricted.json()["job_id"])["status"] == "done"
    # у владельца нет права finance-restricted (вне конфига групп) - не видно
    assert not any("смета затрат" in t for t in query_texts(client, "смета затрат бюджета"))
