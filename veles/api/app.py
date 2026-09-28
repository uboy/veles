"""Veles API: канонический фасад + OpenAI-совместимость (стандарт v3.0, инвариант 9).

POST /v1/knowledge/query  - native API (decision, answer, citations, diagnostics)
POST /v1/chat/completions - совместимость для Open WebUI и других клиентов
GET  /healthz

Запуск: python -m veles serve [--host] [--port]
Корпоративный контур (JWT/SSO) - Ф5; личный профиль - опциональный bearer-токен
из окружения VELES_API_TOKEN.
"""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pydantic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from ..config import INDEX_DIR, VELES_PUBLIC_URL, VELES_SOURCE_ROOTS
from ..indexing.embedder import SentenceTransformersProvider
from ..indexing.indexer import IndexBundle, load_index
from ..policy.answer import decide
from ..policy.identifiers import extract_codex_refs, extract_identifiers
from ..policy.identity import IdentityResolver, StaticOwnerResolver
from ..retrieval.retriever import hybrid_search
from .llm import OpenAICompatibleProvider, llm_config_from_env, request_id

EMBED_DIM = 384
EMBED_MODEL = "intfloat/multilingual-e5-small"


def source_url(document_id: str, node_version_id: str, token: str) -> str:
    """Ссылка на источник для цитаты (открывается в отдельной вкладке)."""
    q = f"token={quote(token)}&node={quote(node_version_id)}" if token else (
        f"node={quote(node_version_id)}")
    return f"{VELES_PUBLIC_URL}/v1/source/{quote(document_id, safe=chr(47))}?{q}"


class QueryRequest(pydantic.BaseModel):
    query: str
    top_k: int = pydantic.Field(default=5, ge=1, le=20)
    use_llm: bool = True


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str


class ChatRequest(pydantic.BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]


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
    ) -> dict:
        qvec = self.embedder.embed_query(query)
        evidence = hybrid_search(
            self.bundle, qvec, query, top_k=top_k, allowed_groups=allowed_groups,
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
            citations.append({
                "n": i + 1,
                "title": c.title,
                "source_id": c.source_id,
                "lines": c.lines,
                "heading_path": c.heading_path,
                "document_id": c.document_id,
                "node_version_id": c.node_version_id,
            })
            evidence_texts.append(ev.text)

        answer, llm_used, llm_error = "", False, None
        if decision.decision == "answer":
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
            },
        }


def render_chat_content(result: dict, token: str = "") -> str:
    if result["decision"] != "answer":
        reason = result["diagnostics"].get("reason") or ""
        return (
            "Ответа на этот вопрос в базе знаний нет."
            + (f"\nПричина: {reason}." if reason else "")
            + "\nУточните вопрос, назовите конкретный документ или попросите добавить источник в базу."
        )
    links = {
        c["n"]: source_url(c["document_id"], c["node_version_id"], token)
        for c in result.get("citations", [])
        if c.get("document_id") and c.get("node_version_id")
    }
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
    return "\n".join(blocks)


def create_app(
    pipeline: Optional[VelesPipeline] = None,
    api_token: Optional[str] = None,
    resolver: Optional[IdentityResolver] = None,
) -> FastAPI:
    state = {
        "pipeline": pipeline,
        "token": api_token or os.environ.get("VELES_API_TOKEN", ""),
        # allowed_groups резолвит СЕРВЕР из личности (инвариант 5), не клиент
        "resolver": resolver or StaticOwnerResolver.from_env(),
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

    def _auth(authorization: str) -> None:
        token = state["token"]
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

    @app.get("/v1/source/{document_id:path}")
    def source_view(
        document_id: str,
        token: str = "",
        node: str = "",
        authorization: str = Header(default=""),
    ):
        """Источник в отдельной вкладке: оригинал (txt/md/pdf) из VELES_SOURCE_ROOTS,
        иначе текстовая сборка из канона. Авторизация: bearer ИЛИ token=<...> в query
        (ссылки из цитат не могут передавать заголовки)."""
        state_token = state["token"]
        if state_token and token != state_token and authorization != f"Bearer {state_token}":
            raise HTTPException(status_code=401, detail="unauthorized")

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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "ts": time.time()}

    @app.get("/v1/models")
    def list_models(authorization: str = Header(default="")) -> dict:
        """OpenAI-совместимый список моделей: Open WebUI и другие клиенты опрашивают
        /models у соединения; без него соединение выглядит пустым."""
        _auth(authorization)
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
    def knowledge_stats(authorization: str = Header(default="")) -> dict:
        """Оглавление базы: источники, верхнеуровневые разделы, объёмы.

        Питает команды /stats, /sources, /toc Telegram-моста и любые другие
        клиенты; строится из мета узлов загруженного индекса, без похода в L2.
        """
        _auth(authorization)
        p = _pipeline()
        docs: dict[str, dict] = {}
        for m in p.bundle.node_meta.values():
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
            "totals": {"documents": len(doc_list), "nodes": len(p.bundle.node_meta)},
            "embed_model": EMBED_MODEL,
            "documents": doc_list,
        }

    @app.post("/v1/knowledge/query")
    def knowledge_query(req: QueryRequest, authorization: str = Header(default="")) -> dict:
        _auth(authorization)
        return _pipeline().answer(
            req.query, req.top_k, req.use_llm,
            allowed_groups=state["resolver"].allowed_groups(),
        )

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest, authorization: str = Header(default="")) -> JSONResponse:
        _auth(authorization)
        user_msgs = [m for m in req.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="нет сообщения пользователя")
        question = user_msgs[-1].content
        requested = (req.model or "veles-knowledge").strip()
        # veles-assistant = KB-first c fallback: abstain -> общий ответ GLM с пометкой
        fallback = requested == "veles-assistant"
        result = _pipeline().answer(
            question, top_k=5, use_llm=True,
            allowed_groups=state["resolver"].allowed_groups(),
        )
        llm = _pipeline().llm
        if result["decision"] == "answer":
            content = render_chat_content(result, token=state["token"])
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

    @app.get("/tracker")
    def tracker_page(token: str = "", chat: str = "", authorization: str = Header(default="")):
        """Веб-просмотр персонального трекера (задачи + заметки) для чата.
        Без параметра chat берётся единственный каталог пользователя."""
        state_token = state["token"]
        if state_token and token != state_token and authorization != f"Bearer {state_token}":
            raise HTTPException(status_code=401, detail="unauthorized")
        tracker_root = os.environ.get("TRACKER_ROOT", "")
        root = Path(tracker_root) if tracker_root else None
        if root is None or not root.exists():
            raise HTTPException(status_code=404, detail="трекер не настроен (TRACKER_ROOT)")
        candidates = [p.name for p in root.iterdir()
                      if p.is_dir() and not p.name.startswith(".")]
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
                raw = f"/tracker/file/{kind}/{i['id']}.md?token={quote(token)}"
                mark = "✅" if i["status"] == "done" else "⬜"
                due = f" ⏰ {i['due']}" if i.get("due") else ""
                out.append(
                    f"<p>{mark} <a href='{raw}'><b>{i['title']}</b></a>{due} "
                    f"<small>{i['id']} · {i['status']} · {i['created'][:16]}</small></p>"
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
            f"<title>Трекер - {chat_id}</title></head><body>",
            "<p>" + " · ".join(top_links) + "</p>" if top_links else "<p></p>",
            f"<h1>Личный трекер</h1>" + (f" <small>{chat_id}</small>" if chat_id else ""),
            "<p><i>те же данные лежат markdown-файлами в TRACKER_ROOT - открываются в Obsidian</i></p>",
            *_chat_link(root, token),
            *(_block("task", "Задачи") + _block("note", "Заметки")
              if chat_id else
              ["<p><i>Пока нет данных. Отправьте боту /task текст - задача появится здесь.</i></p>"]),
            "</body></html>",
        ]
        return HTMLResponse(NEWLINE.join(html))

    @app.get("/tracker/file/{kind}/{name}")
    def tracker_file(kind: str, name: str, token: str = "", authorization: str = Header(default="")):
        state_token = state["token"]
        if state_token and token != state_token and authorization != f"Bearer {state_token}":
            raise HTTPException(status_code=401, detail="unauthorized")
        if kind not in ("task", "note") or not re.fullmatch(r"[\w.-]+\.md", name):
            raise HTTPException(status_code=400, detail="bad path")
        tracker_root = os.environ.get("TRACKER_ROOT", "")
        root = Path(tracker_root) if tracker_root else None
        if root is None:
            raise HTTPException(status_code=404, detail="трекер не настроен")
        for user in root.iterdir():
            p = user / kind / name
            if p.exists():
                return PlainTextResponse(p.read_text(encoding="utf-8"),
                                         media_type="text/plain; charset=utf-8")
        raise HTTPException(status_code=404, detail="не найдено")

    return app
