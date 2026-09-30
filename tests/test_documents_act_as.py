"""K16: загрузка «от имени пользователя» - X-Veles-On-Behalf-Of + владельческий токен.

Доверенный путь для Pipe-функции Open WebUI (integrations/openwebui/):
POST /v1/documents с Bearer=VELES_API_TOKEN и заголовком создаёт документ
ЛИЧНО указанного пользователя (user:<email>, каталог uploads/<email>/).
Негативы: без владельческого токена заголовок игнорируется (документ по
обычной семантике: владелец/свой user:, в static без токена - 401),
мусорный email - 400. Индекс и эмбеддер - те же стабы, что в
test_documents_ingest (канон: тесты без загрузки моделей).
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app
from veles.canonical.store import _doc_dir
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.queue import slugify_title, uploads_root
from veles.indexing.indexer import build_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

DOC_TITLE = "Протокол совещания K16"
DOC_SLUG = slugify_title(DOC_TITLE)
DOC_TEXT = """\
# Протокол совещания K16

Регламентный срок согласования договора - пять рабочих дней.
"""
DOC_QUERY = "какой срок согласования договора"
DOC_MARK = "пять рабочих дней"

ACT_AS = {"X-Veles-On-Behalf-Of": "Ivanov@Corp.RU "}
OWNER = {"Authorization": "Bearer owner-token"}


class HashingEmbedder:
    """Тот же стаб, что в test_documents_ingest: sublinear tf по хеш-бакетам."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture()
def kb_env(tmp_path, monkeypatch):
    """tmp-окружение как в test_documents_ingest: канон, индекс, VELES_DATA."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    index_dir = data / "index"
    corpus = tmp_path / "corpus"
    seed = corpus / "01_docs/seed.md"
    seed.parent.mkdir(parents=True)
    seed.write_text("# Сид\n\nСидовый документ для поиска.\n", encoding="utf-8")
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    from veles.ingest.txt_md import TxtMdParserAdapter

    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    from veles.canonical.store import save_document

    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=EMBED_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    monkeypatch.setattr("veles.config.INDEX_DIR", index_dir)
    monkeypatch.setenv("VELES_DATA", str(data))
    monkeypatch.delenv("VELES_IDENTITY_MODE", raising=False)
    monkeypatch.delenv("VELES_IDENTITY_GROUPS", raising=False)
    monkeypatch.delenv("VELES_ADMIN_EMAILS", raising=False)
    monkeypatch.delenv("VELES_QUOTA_DOCS", raising=False)
    monkeypatch.delenv("VELES_QUOTA_MB", raising=False)
    return {"data": data, "canonical": canonical, "bundle": bundle}


def make_client(bundle) -> TestClient:
    app = create_app(pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None))
    return TestClient(app)


def wait_done(client: TestClient, job_id: str, headers: dict | None = None) -> dict:
    queue = client.app.state.veles_ingest
    assert queue is not None and queue.wait_idle(15.0), "очередь не опустела"
    r = client.get(f"/v1/documents/jobs/{job_id}", headers=headers or {})
    assert r.status_code == 200
    return r.json()


def manifest_groups(canonical, document_id: str) -> list[str]:
    doc_dir = _doc_dir(document_id, canonical)
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    return manifest["access_groups"]


def search_texts(client: TestClient, headers: dict | None = None) -> list[str]:
    d = client.post(
        "/v1/knowledge/query",
        json={"query": DOC_QUERY, "use_llm": False},
        headers=headers or {},
    ).json()
    return [ev["text"] for ev in d["evidence"]]


# ------------------------------------------------------------ static-режим


def test_act_as_creates_personal_doc(kb_env, monkeypatch) -> None:
    """Владельческий bearer + заголовок -> личный документ пользователя
    (access_groups user:<email>, каталог L0 uploads/<email>/, аудит by)."""
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**OWNER, **ACT_AS},
    )
    assert r.status_code == 202
    body = r.json()
    # email нормализован (lower/strip) в document_id, как в identity-режимах
    assert body["document_id"] == f"ivanov@corp.ru/{DOC_SLUG}.md"

    job = wait_done(client, body["job_id"])
    assert job["status"] == "done", job
    assert job["user_id"] == "ivanov@corp.ru"
    assert manifest_groups(kb_env["canonical"], body["document_id"]) == [
        "user:ivanov@corp.ru"
    ]
    assert (uploads_root() / "ivanov@corp.ru" / f"{DOC_SLUG}.md").exists()
    events = [
        json.loads(l)
        for l in (kb_env["canonical"] / "ingest-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    up = [e for e in events if e["event"] == "upload" and e["job_id"] == body["job_id"]]
    assert up and up[0]["by"] == "ivanov@corp.ru"


def test_act_as_static_search_and_owner_visibility(kb_env, monkeypatch) -> None:
    """static-режим: документ личный (группа user:), владелец-токен видит его
    в поиске (владелец - админ), L0 лежит в каталоге пользователя."""
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**OWNER, **ACT_AS},
    )
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done"
    texts = search_texts(client, headers=OWNER)
    assert any(DOC_MARK in t for t in texts)


def test_act_as_multipart(kb_env, monkeypatch) -> None:
    """Вариант Pipe-функции: multipart с файлом + act-as."""
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents",
        files={"file": ("регламент.txt", DOC_TEXT.encode("utf-8"), "text/plain")},
        data={"title": DOC_TITLE},
        headers={**OWNER, **ACT_AS},
    )
    assert r.status_code == 202
    assert r.json()["document_id"] == "ivanov@corp.ru/регламент.txt"
    job = wait_done(client, r.json()["job_id"])
    assert job["status"] == "done"
    assert manifest_groups(kb_env["canonical"], "ivanov@corp.ru/регламент.txt") == [
        "user:ivanov@corp.ru"
    ]


def test_act_as_without_token_configured_ignored(kb_env, monkeypatch) -> None:
    """Токен не настроен - доверенного пути нет: заголовок игнорируется,
    документ владельца (семантика static, как до K16)."""
    monkeypatch.delenv("VELES_API_TOKEN", raising=False)
    client = make_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE}, headers=ACT_AS
    )
    assert r.status_code == 202
    assert r.json()["document_id"] == f"owner/{DOC_SLUG}.md"


def test_act_as_wrong_or_missing_bearer_401(kb_env, monkeypatch) -> None:
    """Токен настроен, bearer не владельческий (или нет) - static-семантика:
    401 до всякого act-as, документ не создаётся."""
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    for headers in (ACT_AS, {**ACT_AS, "Authorization": "Bearer someone-else"}):
        r = client.post(
            "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
            headers=headers,
        )
        assert r.status_code == 401
    assert not (uploads_root() / "ivanov@corp.ru").exists()


def test_act_as_invalid_email_400(kb_env, monkeypatch) -> None:
    """Владельческий токен есть, но email мусорный (обход каталога) - честный
    400, а не молчаливый документ владельца."""
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**OWNER, "X-Veles-On-Behalf-Of": "../etc/passwd"},
    )
    assert r.status_code == 400
    assert "X-Veles-On-Behalf-Of" in r.json()["detail"]


# ----------------------------------------------------- identity-режим (K1)


@pytest.fixture()
def corp_env(kb_env, monkeypatch):
    """Корпоративный headers-режим: ivanov/petrov - пользователи, boss - админ."""
    monkeypatch.setenv("VELES_IDENTITY_MODE", "headers")
    monkeypatch.setenv("VELES_IDENTITY_GROUPS", "corp")
    monkeypatch.setenv("VELES_ADMIN_EMAILS", "boss@corp.ru")
    monkeypatch.setenv("VELES_API_TOKEN", "owner-token")
    client = make_client(kb_env["bundle"])
    ivanov = {"X-OpenWebUI-User-Email": "ivanov@corp.ru"}
    petrov = {"X-OpenWebUI-User-Email": "petrov@corp.ru"}
    return {"client": client, "ivanov": ivanov, "petrov": petrov}


def test_act_as_identity_mode_acl(corp_env) -> None:
    """Корпоративный режим: act-as от владельца -> документ личный ivanov;
    ivanov видит его и свой джоб, petrov - нет (ни документ, ни джоб)."""
    client, ivanov, petrov = corp_env["client"], corp_env["ivanov"], corp_env["petrov"]
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**OWNER, **ACT_AS},
    )
    assert r.status_code == 202
    document_id = r.json()["document_id"]
    assert document_id == f"ivanov@corp.ru/{DOC_SLUG}.md"

    job_id = r.json()["job_id"]
    assert wait_done(client, job_id, headers={**OWNER, **ACT_AS})["status"] == "done"

    # ivanov видит документ в поиске и джоб в списке; petrov - ни то, ни другое
    assert any(DOC_MARK in t for t in search_texts(client, headers=ivanov))
    assert not any(DOC_MARK in t for t in search_texts(client, headers=petrov))
    jobs = client.get("/v1/documents/jobs", headers=ivanov).json()["jobs"]
    assert [j["job_id"] for j in jobs] == [job_id]
    petrov_jobs = client.get("/v1/documents/jobs", headers=petrov).json()["jobs"]
    assert petrov_jobs == []
    assert (
        client.get(f"/v1/documents/jobs/{job_id}", headers=petrov).status_code == 404
    )


def test_act_as_identity_mode_manifest(corp_env, kb_env) -> None:
    """Манифест документа act-as несёт только личную группу ivanov."""
    client = corp_env["client"]
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**OWNER, **ACT_AS},
    )
    document_id = r.json()["document_id"]
    assert wait_done(client, r.json()["job_id"], headers=OWNER)["status"] == "done"
    doc_dir = _doc_dir(document_id, kb_env["canonical"])
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access_groups"] == ["user:ivanov@corp.ru"]
    assert manifest["ingested_by"] == "ivanov@corp.ru"


def test_act_as_ignored_without_owner_token_identity_mode(corp_env, kb_env) -> None:
    """Негатив (по брифу): пользователь petrov шлёт заголовок act-as со СВОИМИ
    identity-заголовками и без владельческого токена - заголовок игнорируется,
    документ принадлежит petrov, а не указанному в заголовке boss."""
    client, petrov = corp_env["client"], corp_env["petrov"]
    r = client.post(
        "/v1/documents", json={"text": DOC_TEXT, "title": DOC_TITLE},
        headers={**petrov, "X-Veles-On-Behalf-Of": "boss@corp.ru"},
    )
    assert r.status_code == 202
    document_id = r.json()["document_id"]
    assert document_id == f"petrov@corp.ru/{DOC_SLUG}.md"
    assert wait_done(client, r.json()["job_id"], headers=petrov)["status"] == "done"
    doc_dir = _doc_dir(document_id, kb_env["canonical"])
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["access_groups"] == ["user:petrov@corp.ru"]
