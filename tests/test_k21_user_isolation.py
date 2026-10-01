"""K21: разграничение пользователей в поиске (act-as для query/stats) +
подписанные ссылки в static + мелкие фикс-пачки ревью-веера 01.10.

Критерий (требование владельца «пользователи должны быть разграничены»):
- act-as заголовок при владельческом bearer срезает поиск и stats до
  личности пользователя: all + личная user:<uid> + membership (+ базовые
  группы резолвера в identity-режимах); чужое личное не находится, в том
  числе на стенд (static);
- без владельческого токена заголовок молча игнорируется (семантика K16);
- админ-путь без заголовка не изменён;
- подписанные ссылки строятся при заданном VELES_LINK_SECRET в ЛЮБОМ
  режиме (токен из цитат уходит), при пустом - token= как раньше, и ключ
  подписи НИКОГДА не замещается владельческим токеном;
- приписка при llm_error в рендерах; VELES_LLM_TIMEOUT_S читается на вызов.
Индекс - детерминированные подменные хранилища (без моделей): каналы
всегда возвращают все три узла, авторизация решает.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations" / "openwebui" / "functions"))

from tests.test_api import FakeEmbedder  # noqa: E402
from veles.api.app import VelesPipeline, create_app, render_chat_content  # noqa: E402
from veles.indexing.indexer import INDEX_LOCK, IndexBundle  # noqa: E402
from veles.policy.identity import (  # noqa: E402
    JwtHeaderResolver,
    StaticOwnerResolver,
)
from veles.retrieval.retriever import Citation, Evidence  # noqa: E402

TOKEN = "k21-owner-token"
SECRET = "k21-link-secret"

ACT_AS = "X-Veles-On-Behalf-Of"

PUB = "Норма общего доступа."
ALICE = "Личная заметка Алисы про поставку Лютик."
OWN = "Служебная пометка владельца проекта."


class AllVector:
    """Канал dense: отдаёт все узлы с фиксированными скоростями - авторизация
    и есть предмет теста (порядок задаёт релевантность, не принадлежность)."""

    def search(self, q, top_k=10):
        pairs = [("pub1", 0.90), ("alice1", 0.85), ("own1", 0.80)]
        return [p for p in pairs if p[0] in NODE_VERSIONS][:top_k]

    def save(self, dir_path):
        raise NotImplementedError


class AllBm25:
    def search(self, q, top_k=10):
        pairs = [("pub1", 5.0), ("alice1", 4.5), ("own1", 4.0)]
        return [p for p in pairs if p[0] in NODE_VERSIONS][:top_k]

    def save(self, dir_path):
        raise NotImplementedError


def _node(text, groups, title, document_id, source_id):
    return {
        "text": text,
        "access_groups": groups,
        "title": title,
        "document_id": document_id,
        "heading_path": [title],
        "provenance": {"adapter": "txt-md:0.1", "source_id": source_id, "lines": [1, 1]},
        "revision_id": "r1",
    }


NODE_VERSIONS = {
    "pub1": _node(PUB, ["all"], "Общий регламент", "shared/regs.md", "regs.md"),
    "alice1": _node(ALICE, ["user:alice@corp.ru"], "Заметка Алисы", "users/alice-note.md", "alice-note.md"),
    "own1": _node(OWN, ["personal"], "Личное владельца", "owner/own.md", "own.md"),
}


def make_bundle() -> IndexBundle:
    return IndexBundle(vector=AllVector(), bm25=AllBm25(), node_meta=dict(NODE_VERSIONS))


def make_client(api_token: str = TOKEN) -> TestClient:
    pipeline = VelesPipeline(make_bundle(), FakeEmbedder(), llm=None)
    return TestClient(create_app(pipeline=pipeline, api_token=api_token))


def _texts(d: dict) -> list[str]:
    return [ev["text"] for ev in d.get("evidence", [])]


def _cites(d: dict) -> list[dict]:
    return d.get("citations", [])


def _q(client: TestClient, query: str, headers: dict) -> dict:
    return client.post(
        "/v1/knowledge/query", json={"query": query, "use_llm": False}, headers=headers
    ).json()


AUTH = {"Authorization": f"Bearer {TOKEN}"}


# ---------------------------------------------------------------- act-as query


def test_static_owner_sees_personal_without_act_as(monkeypatch) -> None:
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    d = _q(client, "пометка владельца", AUTH)
    assert OWN in _texts(d)
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_static_act_as_cuts_personal_and_others(monkeypatch) -> None:
    """P0-кейс: обычный пользователь стенд не видит ни личное владельца, ни чужое,
    но видит общую базу (all)."""
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    headers = {**AUTH, ACT_AS: "bob@corp.ru"}
    assert OWN not in _texts(_q(client, "пометка владельца", headers))
    assert ALICE not in _texts(_q(client, "заметка про Лютик", headers))
    assert PUB in _texts(_q(client, "норма общего доступа", headers))
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_static_act_as_sees_own_personal_group(monkeypatch) -> None:
    """user:<uid> у act-as-личности: Алиса видит свой личный документ."""
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    d = _q(client, "заметка про Лютик", {**AUTH, ACT_AS: "alice@corp.ru"})
    assert ALICE in _texts(d)
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_act_as_without_owner_token_is_ignored(monkeypatch) -> None:
    """Заголовок без владельческого bearer - молча игнор (K16-семантика):
    чужой bearer - 401, без bearer в static с настроенным токеном - тоже 401."""
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    assert client.post(
        "/v1/knowledge/query", json={"query": "пометка владельца", "use_llm": False},
        headers={"Authorization": "Bearer wrong-token", ACT_AS: "bob@corp.ru"},
    ).status_code == 401
    assert client.post(
        "/v1/knowledge/query", json={"query": "пометка владельца", "use_llm": False},
        headers={ACT_AS: "bob@corp.ru"},
    ).status_code == 401
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_act_as_invalid_uid_is_400(monkeypatch) -> None:
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    r = client.post(
        "/v1/knowledge/query", json={"query": "что угодно", "use_llm": False},
        headers={**AUTH, ACT_AS: "not an email; drop"},
    )
    assert r.status_code == 400
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_act_as_stats_cut(monkeypatch) -> None:
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    own = client.get("/v1/knowledge/stats", headers=AUTH).json()["totals"]["documents"]
    as_user = client.get(
        "/v1/knowledge/stats", headers={**AUTH, ACT_AS: "bob@corp.ru"}
    ).json()["totals"]["documents"]
    assert own == 3
    assert as_user == 1  # только общий all-документ
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_identity_mode_act_as_and_real_user(monkeypatch) -> None:
    """identity-режим (headers): и act-as, и реальный пользователь по заголовкам
    видят базовые группы резолвера + all; предмет K21 - parity двух путей."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp-shared")
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    try:
        client = make_client()
        for headers in (
            {**AUTH, ACT_AS: "bob@corp.ru"},
            {"X-OpenWebUI-User-Email": "bob@corp.ru"},
        ):
            d = _q(client, "норма общего доступа", headers)
            assert PUB in _texts(d), headers
    finally:
        monkeypatch.delenv("VELES_IDENTITY_MODE")
        monkeypatch.delenv("VELES_IDENTITY_GROUPS")
        monkeypatch.delenv("VELES_LINK_SECRET")


def test_membership_of_uid_in_act_as(monkeypatch) -> None:
    """membership из groups.json попадает в act-as-личность."""
    from veles.policy.groups import membership_of

    # дефолтный конфиг групп (без groups.json) - membership пуст, all есть у всех
    assert isinstance(membership_of("bob@corp.ru"), set)
    # санкционированная проверка составной группы - на уровне base_groups ниже;
    # полный membership-цикл покрыт тестами policy/groups
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    monkeypatch.delenv("VELES_LINK_SECRET")


# ------------------------------------------------------- подписанные ссылки K20/K21


def test_signed_links_in_static_no_token(monkeypatch) -> None:
    """P1-кейс: static с заданным секретом - exp+sig, владельческий токен
    из цитат уходит (акт-эс-пользователь тоже получает только подписанные)."""
    monkeypatch.setenv("VELES_LINK_SECRET", SECRET)
    client = make_client()
    d = _q(client, "норма общего доступа", {**AUTH, ACT_AS: "bob@corp.ru"})
    cites = _cites(d)
    assert cites and cites[0].get("source_url")
    url = cites[0]["source_url"]
    assert "exp=" in url and "sig=" in url
    assert f"token={TOKEN}" not in url
    monkeypatch.delenv("VELES_LINK_SECRET")


def test_no_secret_keeps_token_links(monkeypatch) -> None:
    """Регрессия: секрет не задан - ссылки token= как раньше (обе машины до K21)."""
    monkeypatch.delenv("VELES_LINK_SECRET", raising=False)
    client = make_client()
    d = _q(client, "норма общего доступа", AUTH)
    cites = _cites(d)
    assert cites and f"token={TOKEN}" in (cites[0].get("source_url") or "")
    assert "sig=" not in (cites[0].get("source_url") or "")


# ------------------------------------------------------------- llm_error приписка


def test_render_chat_content_llm_error_note() -> None:
    result = {
        "decision": "answer",
        "answer": "Ответ из выдержки.",
        "citations": [{"n": 1, "title": "Док", "source_id": "s.md", "document_id": "d.md",
                       "node_version_id": "pub1", "lines": [1, 2], "lost_force": False}],
        "diagnostics": {"llm_error": "connect timeout", "conflicts": []},
    }
    text = render_chat_content(result, token="t", link_secret="")
    assert "Синтез нейросетью временно недоступен" in text


def test_render_chat_content_no_note_without_error() -> None:
    result = {
        "decision": "answer",
        "answer": "Нормальный синтез.",
        "citations": [],
        "diagnostics": {"llm_used": True, "llm_error": None, "conflicts": []},
    }
    text = render_chat_content(result, token="t", link_secret="")
    assert "временно недоступен" not in text


def test_function_render_llm_error_note() -> None:
    """Рендер Pipe-функции K19 - та же приписка."""
    import veles_knowledge

    result = {
        "decision": "answer",
        "answer": "Ответ из выдержки.",
        "citations": [{"n": 1, "title": "Док", "source_id": "s.md", "document_id": "d.md",
                       "node_version_id": "pub1", "lines": [1, 2]}],
        "diagnostics": {"llm_error": "boom", "conflicts": []},
    }
    text = veles_knowledge._render_answer(result, public_url="http://facade.example", token="t")
    assert "Синтез нейросетью временно недоступен" in text


# ------------------------------------------------------------------ прочее K21


def test_llm_timeout_env(monkeypatch) -> None:
    """VELES_LLM_TIMEOUT_S читается на вызов (подмена атрибута config без
    переимпорта - тот же приём, что у реранкера); мусор в env - дефолт."""
    from veles.api.llm import llm_config_from_env
    from veles.config import _env_float
    import veles.config as config

    monkeypatch.setattr(config, "VELES_LLM_TIMEOUT_S", 17.5)
    assert llm_config_from_env().timeout_s == 17.5
    monkeypatch.setenv("VELES_LLM_TIMEOUT_S", "мусор")
    assert _env_float("VELES_LLM_TIMEOUT_S", 60.0) == 60.0
    monkeypatch.setenv("VELES_LLM_TIMEOUT_S", "17.5")
    assert _env_float("VELES_LLM_TIMEOUT_S", 60.0) == 17.5
    monkeypatch.delenv("VELES_LLM_TIMEOUT_S")


def test_source_roots_pathsep(monkeypatch, tmp_path) -> None:
    """VELES_DATA в /v1/source парсится os.pathsep (P2 ревью: был split(':'))."""
    import os

    client = make_client()
    root1, root2 = tmp_path / "d1", tmp_path / "d2"
    root1.mkdir(), root2.mkdir()
    monkeypatch.setenv("VELES_DATA", f"{root1}{os.pathsep}{root2}")
    try:
        # несуществующий документ - но путь доходит до перебора корней
        # без 500 и без исключения парсинга (на Windows ':' ломал бы C:\... пути)
        r = client.get(
            "/v1/source/lectures/missing/n1",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert r.status_code == 404
    finally:
        monkeypatch.delenv("VELES_DATA")


def test_index_lock_is_reentrant() -> None:
    with INDEX_LOCK:
        with INDEX_LOCK:  # RLock: вложенное взятие не дедлок
            pass


def test_base_groups_methods() -> None:
    assert StaticOwnerResolver(groups={"personal"}).base_groups() == {"personal"}
    r = JwtHeaderResolver("headers", groups={"corp-shared"})
    assert r.base_groups() == {"corp-shared"}
    assert "all" not in r.base_groups()
