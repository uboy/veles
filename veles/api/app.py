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
import time
from contextlib import asynccontextmanager
from typing import Optional

import pydantic
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from ..config import INDEX_DIR
from ..indexing.embedder import SentenceTransformersProvider
from ..indexing.indexer import IndexBundle, load_index
from ..policy.answer import decide
from ..policy.identifiers import extract_codex_refs, extract_identifiers
from ..retrieval.retriever import hybrid_search
from .llm import OpenAICompatibleProvider, llm_config_from_env, request_id

EMBED_DIM = 384
EMBED_MODEL = "intfloat/multilingual-e5-small"


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

    def answer(self, query: str, top_k: int = 5, use_llm: bool = True) -> dict:
        qvec = self.embedder.embed_query(query)
        evidence = hybrid_search(
            self.bundle, qvec, query, top_k=top_k,
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


def render_chat_content(result: dict) -> str:
    if result["decision"] != "answer":
        reason = result["diagnostics"].get("reason") or ""
        return (
            "Ответа на этот вопрос в базе знаний нет."
            + (f"\nПричина: {reason}." if reason else "")
            + "\nУточните вопрос, назовите конкретный документ или попросите добавить источник в базу."
        )
    blocks = [result["answer"], "", "Источники:"]
    for c in result["citations"]:
        where = f", строки {c['lines'][0]}-{c['lines'][1]}" if c["lines"] else ""
        blocks.append(f"  [{c['n']}] «{c['title']}» {c['source_id']}{where}")
    return "\n".join(blocks)


def create_app(
    pipeline: Optional[VelesPipeline] = None,
    api_token: Optional[str] = None,
) -> FastAPI:
    state = {"pipeline": pipeline, "token": api_token or os.environ.get("VELES_API_TOKEN", "")}

    def _pipeline() -> VelesPipeline:
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
        _warmup()
        yield

    app = FastAPI(title="Veles Knowledge API", version="0.4.0", lifespan=lifespan)

    def _auth(authorization: str) -> None:
        token = state["token"]
        if token and authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="unauthorized")

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
            "data": [{
                "id": "veles-knowledge",
                "object": "model",
                "created": 1727740800,
                "owned_by": "veles",
            }],
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
        return _pipeline().answer(req.query, req.top_k, req.use_llm)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest, authorization: str = Header(default="")) -> JSONResponse:
        _auth(authorization)
        user_msgs = [m for m in req.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="нет сообщения пользователя")
        result = _pipeline().answer(user_msgs[-1].content, top_k=5, use_llm=True)
        rid = result["request_id"]
        return JSONResponse({
            "id": f"chatcmpl-veles-{rid}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "veles-knowledge",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": render_chat_content(result)},
                "finish_reason": "stop",
            }],
            "veles": {
                "decision": result["decision"],
                "request_id": rid,
                "citations": result["citations"],
            },
        })

    return app
