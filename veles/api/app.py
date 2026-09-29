"""Veles API: канонический фасад + OpenAI-совместимость (стандарт v3.0, инвариант 9).

POST /v1/knowledge/query  - native API (decision, answer, citations, diagnostics)
POST /v1/chat/completions - совместимость для Open WebUI и других клиентов
GET  /healthz /readyz (живость / готовность индекса, K6)

Запуск: python -m veles serve [--host] [--port]
Личность запроса (K1): VELES_IDENTITY_MODE=static (дефолт, один владелец,
bearer из VELES_API_TOKEN) | headers | jwt (корпоративный, личность из
заголовков Open WebUI; bearer/token = служебный доступ владельца).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pydantic
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from ..config import INDEX_DIR, VELES_PUBLIC_URL, VELES_SOURCE_ROOTS
from ..indexing.embedder import SentenceTransformersProvider
from ..indexing.indexer import IndexBundle, load_index
from ..policy.answer import decide
from ..policy.identifiers import extract_codex_refs, extract_identifiers
from ..policy.identity import (
    IdentityResolver,
    JwtHeaderResolver,
    RequestIdentity,
    StaticOwnerResolver,
)
from ..retrieval.retriever import hybrid_search
from .llm import OpenAICompatibleProvider, llm_config_from_env, request_id
from .ratelimit import install_rate_limit

EMBED_DIM = 384
EMBED_MODEL = "intfloat/multilingual-e5-small"

LINK_TTL_S = 24 * 3600  # срок подписанных ссылок (exp = now + TTL)


def link_sig(secret: str, document_id: str, name_or_node: str, exp: int) -> str:
    """hex HMAC-SHA256 от f"{document_id}|{name_or_node}|{exp}" (VELES_LINK_SECRET)."""
    msg = f"{document_id}|{name_or_node}|{exp}"
    return hmac.new(secret.encode("utf-8"), msg.encode("utf-8"), hashlib.sha256).hexdigest()


def asset_url(document_id: str, name: str, token: str) -> str:
    return (f"{VELES_PUBLIC_URL}/v1/asset/{quote(document_id, safe='/')}/{quote(name)}"
            f"?token={quote(token)}")


def source_url(document_id: str, node_version_id: str, token: str) -> str:
    """Ссылка на источник для цитаты (открывается в отдельной вкладке)."""
    q = f"token={quote(token)}&node={quote(node_version_id)}" if token else (
        f"node={quote(node_version_id)}")
    return f"{VELES_PUBLIC_URL}/v1/source/{quote(document_id, safe=chr(47))}?{q}"


def signed_asset_url(secret: str, document_id: str, name: str, exp: int) -> str:
    sig = link_sig(secret, document_id, name, exp)
    return (f"{VELES_PUBLIC_URL}/v1/asset/{quote(document_id, safe='/')}/{quote(name)}"
            f"?exp={exp}&sig={sig}")


def signed_source_url(secret: str, document_id: str, node_version_id: str, exp: int) -> str:
    """Подписанная ссылка вместо token= в query: sig от document_id|node|exp."""
    sig = link_sig(secret, document_id, node_version_id, exp)
    return (f"{VELES_PUBLIC_URL}/v1/source/{quote(document_id, safe=chr(47))}"
            f"?node={quote(node_version_id)}&exp={exp}&sig={sig}")


class QueryRequest(pydantic.BaseModel):
    query: str
    top_k: int = pydantic.Field(default=5, ge=1, le=20)
    use_llm: bool = True
    # ревизии (K3): None/"current" - только текущие, "all" - все, "r<n>" - указанная
    revision: Optional[str] = None


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ChatRequest(pydantic.BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    revision: Optional[str] = None


def _resolve_revision(bundle: IndexBundle, revision: Optional[str]) -> Optional[str]:
    """Валидация параметра revision (K3): None/'current' -> дефолт (только current),
    'all' -> без фильтра, 'r<n>' -> только эта ревизия. Синтаксически невалидное
    значение или ревизия, отсутствующая в индексе, - 400."""
    if not revision or revision == "current":
        return None
    if revision == "all":
        return "all"
    if not re.fullmatch(r"r\d+", revision):
        raise HTTPException(
            status_code=400, detail="revision: ожидается 'all', 'current' или 'r<номер>'"
        )
    known = {m.get("revision_id", "r1") for m in bundle.node_meta.values()}
    if revision not in known:
        raise HTTPException(status_code=400, detail=f"неизвестная ревизия: {revision}")
    return revision


class VelesPipeline:
    """Единый пайплайн для всех клиентов (инвариант 6): retrieval один, клиенты разные."""

    def __init__(self, bundle: IndexBundle, embedder, llm: Optional[OpenAICompatibleProvider]):
        self.bundle = bundle
        self.embedder = embedder
        self.llm = llm
        self.doc_titles = {
            m.get("title", "") for m in bundle.node_meta.values() if m.get("title")
        }

    def answer(
        self, query: str, top_k: int = 5, use_llm: bool = True,
        allowed_groups: Optional[set[str]] = None,
        revision: Optional[str] = None,
    ) -> dict:
        qvec = self.embedder.embed_query(query)
        evidence = hybrid_search(
            self.bundle, qvec, query, top_k=top_k, allowed_groups=allowed_groups,
            revision=revision,
        )
        decision = decide(
            evidence,
            query_identifiers=extract_identifiers(query),
            known_identifiers=self.bundle.identifiers,
            codex_refs=extract_codex_refs(query),
            doc_titles=self.doc_titles,
        )
        citations, evidence_texts = [], []
        for i, ev in enumerate(decision.evidence[:top_k]):
            c = ev.citations[0]
            nmeta = self.bundle.node_meta.get(ev.node_version_id) or {}
            citations.append({
                "n": i + 1,
                "title": c.title,
                "source_id": c.source_id,
                "lines": c.lines,
                "heading_path": c.heading_path,
                "document_id": c.document_id,
                "node_version_id": c.node_version_id,
                "revision_id": nmeta.get("revision_id", "r1"),
                "asset": nmeta.get("asset"),
            })
            evidence_texts.append(ev.text)

        answer, llm_used, llm_error = "", False, None
        if decision.decision in ("answer", "warn_conflict"):
            answer = evidence_texts[0]
            if use_llm and self.llm and self.llm.enabled:
                try:
                    answer = self.llm.synthesize(query, evidence_texts)
                    llm_used = True
                except Exception as e:  # noqa: BLE001 - деградация к extractive
                    llm_error = str(e)[:200]
        return {
            "request_id": request_id(),
            "decision": decision.decision,
            "answer": answer,
            "citations": citations,
            "evidence": [
                {"n": i + 1, "text": ev.text, "rrf": ev.rrf}
                for i, ev in enumerate(decision.evidence[:top_k])
            ],
            "diagnostics": {
                "confidence": decision.confidence,
                "reason": decision.reason,
                "llm_used": llm_used,
                "llm_error": llm_error,
                "conflicts": decision.conflicts,
            },
        }


def render_chat_content(result: dict, token: str = "", link_secret: str = "") -> str:
    """Ссылки в ответе: token= (static) или подписанные exp+sig (identity-режимы,
    ссылка живёт LINK_TTL_S - браузер не передаёт заголовки личности)."""
    signed = bool(link_secret)
    exp = int(time.time()) + LINK_TTL_S if signed else 0
    if result["decision"] not in ("answer", "warn_conflict"):
        reason = result["diagnostics"].get("reason") or ""
        return (
            "Ответа на этот вопрос в базе знаний нет."
            + (f"\nПричина: {reason}." if reason else "")
            + "\nУточните вопрос, назовите конкретный документ или попросите добавить источник в базу."
        )

    def _src_link(doc_id: str, node: str) -> str:
        if signed:
            return signed_source_url(link_secret, doc_id, node, exp)
        return source_url(doc_id, node, token)

    def _asset_link(doc_id: str, name: str) -> str:
        if signed:
            return signed_asset_url(link_secret, doc_id, name, exp)
        return asset_url(doc_id, name, token)

    links = {
        c["n"]: _src_link(c["document_id"], c["node_version_id"])
        for c in result.get("citations", [])
        if c.get("document_id") and c.get("node_version_id")
    }
    images = [
        _asset_link(c["document_id"], c["asset"])
        for c in result.get("citations", [])
        if c.get("asset") and c.get("document_id")
    ]
    import re as _re

    answer = result["answer"]
    if links:
        # сноски [n] в тексте синтеза -> кликабельные markdown-ссылки
        answer = _re.sub(
            r"\[(\d{1,2})\]",
            lambda m: f"[{m.group(1)}]({links[int(m.group(1))]})" if int(m.group(1)) in links else m.group(0),
            answer,
        )
    blocks = [answer, "", "Источники:"]
    for c in result["citations"]:
        where = f", строки {c['lines'][0]}-{c['lines'][1]}" if c.get("lines") else ""
        label = f"[{c['n']}] «{c['title']}» {c['source_id']}{where}"
        url = links.get(c["n"])
        blocks.append(f"  [{c['n']}]({url}) «{c['title']}» {c['source_id']}{where}" if url
                      else f"  {label}")
    for url in images:
        blocks.append(f"![изображение]({url})")
    if result["decision"] == "warn_conflict":
        conflicts = result.get("diagnostics", {}).get("conflicts") or []
        if conflicts:
            listing = "; ".join(
                f"«{c.get('title') or c.get('document_id')}» ({c.get('revision_id')}, {c.get('signal')})"
                for c in conflicts
            )
            blocks += [
                "",
                f"⚠ Возможны противоречащие редакции: {listing}.",
                "Уточните, какая редакция вам нужна.",
            ]
    return "\n".join(blocks)


def create_app(
    pipeline: Optional[VelesPipeline] = None,
    api_token: Optional[str] = None,
    resolver: Optional[IdentityResolver] = None,
) -> FastAPI:
    token = api_token or os.environ.get("VELES_API_TOKEN", "")
    mode = os.environ.get("VELES_IDENTITY_MODE", "static").strip().lower()
    if resolver is not None:
        # явный resolver = личный профиль (тесты/встраивание), заголовки не нужны
        mode = "static"
        identity_resolver: IdentityResolver = resolver
    elif mode in ("headers", "jwt"):
        identity_resolver = JwtHeaderResolver.from_env(mode)
    else:
        mode = "static"
        identity_resolver = StaticOwnerResolver.from_env()

    state = {
        "pipeline": pipeline,
        "token": token,
        # allowed_groups резолвит СЕРВЕР из личности (инвариант 5), не клиент
        "resolver": identity_resolver,
        "identity_mode": mode,
        # секрет подписанных ссылок; пусто (и токена нет) - подпись не mint'ится
        "link_secret": os.environ.get("VELES_LINK_SECRET", "") or token,
    }

    _pipeline_lock = __import__("threading").Lock()

    def _pipeline() -> VelesPipeline:
        with _pipeline_lock:
            if state["pipeline"] is None:
                embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
                bundle = load_index(INDEX_DIR)
                llm = OpenAICompatibleProvider(llm_config_from_env())
                state["pipeline"] = VelesPipeline(bundle, embedder, llm)
        return state["pipeline"]

    def _warmup() -> None:
        # Прогрев пайплайна на старте сервиса: без него первый /stats после
        # рестарта фасада может упасть по таймауту клиента (загрузка эмбеддера).
        try:
            _pipeline()
        except Exception as e:  # noqa: BLE001 - прогрев не должен ронять сервис
            logging.getLogger("veles.api").warning("warmup не удался (соберётся лениво): %s", e)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # прогрев в фоне: под нагрузкой загрузка эмбеддера занимает минуты,
        # а uvicorn связывает порт только после lifespan - сервис недоступен
        import threading

        threading.Thread(target=_warmup, daemon=True).start()
        yield

    app = FastAPI(title="Veles Knowledge API", version="0.4.0", lifespan=lifespan)
    # rate-limit фасада (K6): скользящее окно per-identity/per-token (VELES_RATE_LIMIT_RPM)
    install_rate_limit(app, state)

    def _auth(authorization: str) -> None:
        token = state["token"]
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    def _owner_identity() -> RequestIdentity:
        """Владелец/служебный клиент (bearer, token=, static-режим)."""
        return RequestIdentity(user_id="", admin=True, groups=state["resolver"].allowed_groups())

    def _header_identity(request: Request) -> Optional[RequestIdentity]:
        """Корпоративный пользователь из заголовков; None - личности нет."""
        if state["identity_mode"] == "static":
            return None
        ident = state["resolver"].resolve(request.headers)
        if ident is not None and ident.user_id:
            return ident
        return None

    def _token_owner(token: str, authorization: str) -> bool:
        state_token = state["token"]
        return bool(state_token) and (
            token == state_token or authorization == f"Bearer {state_token}"
        )

    def _identity(request: Request, authorization: str) -> RequestIdentity:
        """Личность для API-эндпоинтов (query/chat/stats).

        Identity-режимы: пользователь обязателен - из заголовков; bearer-токен
        владельца = служебный доступ (Telegram-мост, CLI); без того и другого -
        401. Static: владелец без условий (как раньше).
        """
        ident = _header_identity(request)
        if ident is not None:
            return ident
        if state["identity_mode"] == "static":
            # личный профиль: bearer обязателен при настроенном токене (как раньше)
            state_token = state["token"]
            if state_token and authorization != f"Bearer {state_token}":
                raise HTTPException(status_code=401, detail="unauthorized")
            return _owner_identity()
        if _token_owner("", authorization):
            return _owner_identity()
        raise HTTPException(status_code=401, detail="identity required")

    def _sig_ok(document_id: str, name_or_node: str, exp: str, sig: str) -> bool:
        """Проверка подписанной ссылки: HMAC от document_id|name_or_node|exp."""
        secret = state["link_secret"]
        if not secret or not exp or not sig:
            return False
        try:
            exp_i = int(exp)
        except ValueError:
            return False
        if exp_i <= time.time():
            return False
        expected = link_sig(secret, document_id, name_or_node, exp_i)
        # encode: compare_digest на str требует ASCII, не-ASCII sig давал TypeError -> 500
        return hmac.compare_digest(expected.encode(), sig.encode())

    def _link_identity(
        request: Request, authorization: str, token: str,
        document_id: str, name_or_node: str, exp: str, sig: str,
    ) -> Optional[RequestIdentity]:
        """Личность для эндпоинтов, открываемых браузером по ссылке.

        Порядок: bearer/token= (владелец) -> подпись exp+sig (аноним,
        доступ только к этому файлу) -> заголовки личности (identity-режимы).
        None - не установлена; static без настроенного токена = открыто (как
        раньше), identity-режим без всего - 401 у вызывающего.
        """
        if _token_owner(token, authorization):
            return _owner_identity()
        if _sig_ok(document_id, name_or_node, exp, sig):
            return None
        ident = _header_identity(request)
        if ident is not None:
            return ident
        if state["identity_mode"] == "static" and not state["token"]:
            return _owner_identity()
        raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/v1/asset/{document_id:path}/{name}")
    def asset_file(
        request: Request, document_id: str, name: str, token: str = "",
        exp: str = "", sig: str = "", authorization: str = Header(default=""),
    ):
        """Актив документа (картинка) - для показа в ответах веба и Telegram.

        Доступ: bearer/token= ИЛИ подписанная ссылка exp+sig
        (sig от document_id|name|exp) ИЛИ заголовки личности (identity-режимы).
        """
        _link_identity(request, authorization, token, document_id, name, exp, sig)
        if not re.fullmatch(r"[\w.-]+", name):
            raise HTTPException(status_code=400, detail="bad asset name")
        from ..canonical.store import _doc_dir

        roots = [Path(q) for q in os.environ.get("VELES_DATA", "").split(os.pathsep) if q.strip()]
        if not roots:
            roots = [Path(__file__).resolve().parents[2] / "data"]
        for data_root in roots:
            for canon in sorted(data_root.glob("canonical*")):
                f = _doc_dir(document_id, canon) / "assets" / name
                if f.exists():
                    return FileResponse(f, media_type="image/png", filename=name)
        raise HTTPException(status_code=404, detail="актив не найден")

    @app.get("/v1/source/{document_id:path}")
    def source_view(
        request: Request,
        document_id: str,
        token: str = "",
        node: str = "",
        exp: str = "",
        sig: str = "",
        authorization: str = Header(default=""),
    ):
        """Источник в отдельной вкладке: оригинал (txt/md/pdf) из VELES_SOURCE_ROOTS,
        иначе текстовая сборка из канона. Авторизация: bearer, token=<...> в query,
        подписанная ссылка exp+sig (sig от document_id|node|exp) или заголовки
        личности - ссылки из цитат не могут передавать заголовки."""
        _link_identity(request, authorization, token, document_id, node, exp, sig)

        from ..canonical.store import _doc_dir

        roots = [Path(p) for p in os.environ.get("VELES_DATA", "").split(":") if p.strip()]
        if not roots:
            roots = [Path(__file__).resolve().parents[2] / "data"]
        doc = None
        for data_root in roots:
            for canon in sorted(data_root.glob("canonical*")):
                d = _doc_dir(document_id, canon)
                if (d / "manifest.json").exists():
                    import json as _json

                    doc = _json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                    doc["_dir"] = d
                    doc["_nodes_path"] = d / "nodes.jsonl"
                    break
            if doc:
                break
        if doc is None:
            raise HTTPException(status_code=404, detail="источник не найден в каноне")

        basename = document_id.rsplit("/", 1)[-1]
        for root_dir in VELES_SOURCE_ROOTS:
            for candidate in root_dir.rglob(basename):
                if candidate.is_file():
                    media = "application/pdf" if candidate.suffix.lower() == ".pdf" else (
                        "text/plain; charset=utf-8"
                    )
                    return FileResponse(candidate, media_type=media, filename=basename)

        nodes = []
        for line in doc["_nodes_path"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                nodes.append(_json.loads(line))
        anchor = f'n-{node}' if node else ""
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{doc.get('title', document_id)}</title></head><body>",
            f"<h1>{doc.get('title', document_id)}</h1>",
            f"<p><i>{document_id} · {doc.get('parser_profile', '')} · "
            f"фрагментов: {len(nodes)}</i></p>",
            "<p><small>загружен: "
            f"{doc.get('ingested_at') or 'дата не зафиксирована'}"
            + (f" · {doc['ingested_by']}" if doc.get("ingested_by") else "")
            + "</small></p>",
        ]
        for nd in nodes:
            if nd.get("node_type") == "root":
                continue
            parts.append(
                f"<a id='n-{nd['node_version_id']}'></a>"
                f"<p><b>{' > '.join(nd.get('heading_path') or [])}</b><br>{nd.get('text', '')}</p>"
            )
        parts.append(
            f"<script>var a = document.getElementById('{anchor}');"
            "if (a) a.scrollIntoView({block: 'start'});</script></body></html>"
        )
        return HTMLResponse("\n".join(parts), status_code=200)

    def _index_loaded() -> bool:
        # честная готовность (K6): pipeline создаётся только после успешной
        # load_index() в _pipeline() - до неё (упавший warmup, нет индекса)
        # /readyz обязан быть 503, а /healthz оставаться живостью.
        # Без лока (ревью K6 P1-1): чтение dict атомарно под GIL; лок держится
        # минуты на загрузке индекса - заблокированные /healthz//readyz роняют
        # docker healthcheck (timeout 3s) и старт open-webui (service_healthy)
        return state.get("pipeline") is not None

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness: процесс жив (проба systemd/monitoring). Готовность индекса -
        отдельное поле index_loaded и эндпоинт /readyz (K6)."""
        return {"status": "ok", "ts": time.time(), "index_loaded": _index_loaded()}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        """Readiness: 200 только с загруженным индексом, до готовности - 503."""
        if _index_loaded():
            return JSONResponse({"status": "ready", "index_loaded": True})
        return JSONResponse({"status": "starting", "index_loaded": False}, status_code=503)

    @app.get("/v1/models")
    def list_models(request: Request, authorization: str = Header(default="")) -> dict:
        """OpenAI-совместимый список моделей: Open WebUI и другие клиенты опрашивают
        /models у соединения; без него соединение выглядит пустым."""
        _identity(request, authorization)
        return {
            "object": "list",
            "data": [
                {
                    # база знаний: answer с цитатами, вне корпуса - abstain без fallback
                    "id": "veles-knowledge",
                    "object": "model",
                    "created": 1727740800,
                    "owned_by": "veles",
                },
                {
                    # KB-first: поиск в базе, при abstain - общий ответ GLM с пометкой
                    "id": "veles-assistant",
                    "object": "model",
                    "created": 1727740800,
                    "owned_by": "veles",
                },
            ],
        }

    @app.get("/v1/knowledge/stats")
    def knowledge_stats(request: Request, authorization: str = Header(default="")) -> dict:
        """Оглавление базы: источники, верхнеуровневые разделы, объёмы.

        Питает команды /stats, /sources, /toc Telegram-моста и любые другие
        клиенты; строится из мета узлов загруженного индекса, без похода в L2.
        """
        _identity(request, authorization)
        p = _pipeline()
        docs: dict[str, dict] = {}
        superseded_nodes = 0
        for m in p.bundle.node_meta.values():
            # ревизии (K3): superseded-узлы в оглавление и totals.nodes не входят
            if m.get("status", "current") == "superseded":
                superseded_nodes += 1
                continue
            did = m.get("document_id") or "(без документа)"
            d = docs.setdefault(
                did, {"document_id": did, "title": m.get("title") or did, "nodes": 0, "sections": set()}
            )
            d["nodes"] += 1
            heading_path = m.get("heading_path") or []
            if heading_path:
                d["sections"].add(heading_path[0])
        doc_list = sorted(docs.values(), key=lambda d: (-d["nodes"], d["title"]))
        for d in doc_list:
            d["sections"] = sorted(d["sections"])
        return {
            "totals": {
                "documents": len(doc_list),
                "nodes": len(p.bundle.node_meta) - superseded_nodes,
                # число superseded-узлов в индексе (бриф K3)
                "superseded_nodes": superseded_nodes,
            },
            "embed_model": EMBED_MODEL,
            "documents": doc_list,
        }

    @app.post("/v1/knowledge/query")
    def knowledge_query(
        req: QueryRequest, request: Request, authorization: str = Header(default="")
    ) -> dict:
        ident = _identity(request, authorization)
        p = _pipeline()
        revision = _resolve_revision(p.bundle, req.revision)
        return p.answer(
            req.query, req.top_k, req.use_llm,
            allowed_groups=ident.groups, revision=revision,
        )

    @app.post("/v1/chat/completions")
    def chat_completions(
        req: ChatRequest, request: Request, authorization: str = Header(default="")
    ) -> JSONResponse:
        ident = _identity(request, authorization)
        user_msgs = [m for m in req.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="нет сообщения пользователя")
        question = user_msgs[-1].content
        requested = (req.model or "veles-knowledge").strip()
        p = _pipeline()
        revision = _resolve_revision(p.bundle, req.revision)
        # veles-assistant = KB-first c fallback: abstain -> общий ответ GLM с пометкой
        fallback = requested == "veles-assistant"
        result = p.answer(
            question, top_k=5, use_llm=True,
            allowed_groups=ident.groups, revision=revision,
        )
        llm = p.llm
        link_secret = (
            state["link_secret"] if state["identity_mode"] != "static" else ""
        )
        if result["decision"] in ("answer", "warn_conflict"):
            content = render_chat_content(
                result, token=state["token"], link_secret=link_secret
            )
        elif fallback and llm and llm.enabled:
            try:
                general = llm.answer_general(question)
                content = (
                    general
                    + "\n\n---\nВ базе знаний ответа на этот вопрос не нашлось - "
                    "выше общий ответ модели из общих знаний, без источников. "
                    "Вопрос по документам переформулируйте или добавьте источник в базу."
                )
            except Exception:  # noqa: BLE001 - fallback не удался -> честный abstain
                content = render_chat_content(result, token=state["token"])
        else:
            content = render_chat_content(result, token=state["token"])
        rid = result["request_id"]
        return JSONResponse({
            "id": f"chatcmpl-veles-{rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": requested,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "veles": {
                "decision": result["decision"],
                "request_id": rid,
                "citations": result["citations"],
            },
        })

    @app.api_route("/tracker", methods=["GET", "HEAD"])
    def tracker_page(
        request: Request, token: str = "", chat: str = "",
        authorization: str = Header(default=""),
    ):
        """Веб-просмотр персонального трекера (задачи + заметки).

        Личность: bearer/token= (владелец) или заголовки (identity-режимы).
        В identity-режимах пользователь видит ТОЛЬКО свой каталог
        TRACKER_ROOT/<user_id>; параметр chat= - только админ."""
        ident = _link_identity(request, authorization, token, "tracker", "", "", "")
        if ident is None:
            ident = _header_identity(request) or _owner_identity()
        if state["identity_mode"] != "static" and not ident.admin and chat:
            raise HTTPException(status_code=403, detail="chat= только для админа")
        # chat= попадает в путь TRACKER_ROOT/<chat>: только безопасные символы,
        # '..' и ведущие точки исключены (иначе mkdir/чтение за пределами корня)
        if chat and (not re.fullmatch(r"[\w@][\w.@-]*", chat) or ".." in chat):
            raise HTTPException(status_code=400, detail="bad chat")
        admin = ident.admin or state["identity_mode"] == "static"
        tracker_root = os.environ.get("TRACKER_ROOT", "")
        root = Path(tracker_root) if tracker_root else None
        if root is None or not root.exists():
            raise HTTPException(status_code=404, detail="трекер не настроен (TRACKER_ROOT)")
        candidates = [p.name for p in root.iterdir()
                      if p.is_dir() and not p.name.startswith(".")]
        if state["identity_mode"] != "static" and not admin:
            chat_id = ident.user_id
            if chat_id not in candidates:
                chat_id = ""
        else:
            chat_id = chat or (candidates[0] if candidates else "")

        import sys as _sys
        import html as _html
        NEWLINE = chr(10)

        bridge_dir = Path(__file__).resolve().parents[2] / "clients" / "telegram-bridge"
        if str(bridge_dir) not in _sys.path:
            _sys.path.insert(0, str(bridge_dir))
        import tracker as tracker_mod

        chat_url_file = root / "chat-url.txt"
        chat_url = chat_url_file.read_text(encoding="utf-8").strip()             if chat_url_file.exists() else ""

        def _chat_link(r: Path, tok: str) -> list[str]:
            if not chat_url:
                return []
            return [f"<p><a href='{chat_url}'>💬 Открыть чат в Telegram</a></p>"]

        def _block(kind: str, title: str) -> list[str]:
            items = tracker_mod.list_items(root, chat_id, kind, include_done=True)
            out = [f"<h2>{title}</h2>"]
            if not items:
                out.append("<p><i>пусто</i></p>")
            for i in items:
                raw = f"/tracker/file/{kind}/{quote(_html.escape(i['id'], True))}.md?token={quote(token)}"
                mark = "✅" if i["status"] == "done" else "⬜"
                due = f" ⏰ {i['due']}" if i.get("due") else ""
                out.append(
                    f"<p>{mark} <a href='{raw}'><b>{_html.escape(i['title'], True)}</b></a>{due} "
                    f"<small>{_html.escape(i['id'], True)} · {i['status']} · {i['created'][:16]}</small></p>"
                )
            return out

        webui_url = _html.escape(os.environ.get("VELES_WEBUI_URL", ""), True)
        top_links = []
        if webui_url:
            top_links.append(f"<a href='{webui_url}'>🖥 Чат (Open WebUI)</a>")
        if chat_url:
            top_links.append(f"<a href='{chat_url}'>💬 Telegram</a>")
        html = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            f"<title>Трекер - {_html.escape(chat_id, True)}</title></head><body>",
            "<p>" + " · ".join(top_links) + "</p>" if top_links else "<p></p>",
            f"<h1>Личный трекер</h1>" + (f" <small>{_html.escape(chat_id, True)}</small>" if chat_id else ""),
            "<p><i>те же данные лежат markdown-файлами в TRACKER_ROOT - открываются в Obsidian</i></p>",
            *_chat_link(root, token),
            *(_block("task", "Задачи") + _block("note", "Заметки")
              if chat_id else
              ["<p><i>Пока нет данных. Отправьте боту /task текст - задача появится здесь.</i></p>"]),
            "</body></html>",
        ]
        # security-заголовки HTML-страницы трекера (K6); JSON-ответы и ошибки
        # не трогаются. CSP строгий без unsafe-inline: на странице нет
        # инлайн-стилей/скриптов, а переходы по ссылкам CSP не ограничивает.
        html_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": (
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; "
                "form-action 'none'"
            ),
        }
        return HTMLResponse(NEWLINE.join(html), headers=html_headers)

    @app.get("/tracker/file/{kind}/{name}")
    def tracker_file(
        request: Request, kind: str, name: str, token: str = "",
        authorization: str = Header(default=""),
    ):
        if kind not in ("task", "note") or not re.fullmatch(r"[\w.-]+\.md", name):
            raise HTTPException(status_code=400, detail="bad path")
        tracker_root = os.environ.get("TRACKER_ROOT", "")
        root = Path(tracker_root) if tracker_root else None
        if root is None:
            raise HTTPException(status_code=404, detail="трекер не настроен")
        if state["identity_mode"] != "static":
            # identity-режим: владелец (bearer/token=) или личность из заголовков;
            # без всего - 401, а НЕ владелец (анонимный доступ к чужим файлам = P0)
            if _token_owner(token, authorization):
                ident2 = _owner_identity()
            else:
                ident2 = _header_identity(request)
            if ident2 is None:
                raise HTTPException(status_code=401, detail="unauthorized")
            if not ident2.admin:
                mine = root / ident2.user_id / kind / name
                if mine.exists():
                    return PlainTextResponse(mine.read_text(encoding="utf-8"),
                                             media_type="text/plain; charset=utf-8")
                raise HTTPException(status_code=404, detail="не найдено")
            # админ: поиск везде
        else:
            state_token = state["token"]
            if state_token and token != state_token and authorization != f"Bearer {state_token}":
                raise HTTPException(status_code=401, detail="unauthorized")
        users = [u for u in root.iterdir() if u.is_dir() and not u.name.startswith(".")]
        for user in users:
            p = user / kind / name
            if p.exists():
                return PlainTextResponse(p.read_text(encoding="utf-8"),
                                         media_type="text/plain; charset=utf-8")
        raise HTTPException(status_code=404, detail="не найдено")

    return app
