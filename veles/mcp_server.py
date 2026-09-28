"""MCP-сервер veles: инструмент knowledge_query для кодинг-агентов (стандарт v3.0 §3).

Запуск: python -m veles mcp   (stdio; конфигурация в клиенте агента:
  {"command": "python", "args": ["-m", "veles", "mcp"], "env": {...}})
"""

from __future__ import annotations

from typing import Any

from .api.app import EMBED_DIM, EMBED_MODEL, VelesPipeline
from .api.llm import OpenAICompatibleProvider, llm_config_from_env
from .config import INDEX_DIR
from .indexing.embedder import SentenceTransformersProvider
from .indexing.indexer import load_index

_pipeline: Any = None


def get_pipeline() -> VelesPipeline:
    global _pipeline
    if _pipeline is None:
        embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
        bundle = load_index(INDEX_DIR)
        llm = OpenAICompatibleProvider(llm_config_from_env())
        _pipeline = VelesPipeline(bundle, embedder, llm)
    return _pipeline


def run() -> None:
    # mcp 1.x: FastMCP в mcp.server.fastmcp; mcp >= 2.0: класс переименован
    # в MCPServer (API декоратора .tool() и run(transport='stdio') сохранён).
    try:
        from mcp.server.fastmcp import FastMCP
        server = FastMCP("veles")
    except ImportError:
        from mcp.server import MCPServer
        server = MCPServer(name="veles")

    @server.tool()
    def knowledge_query(query: str, top_k: int = 5) -> dict:
        """Задать вопрос базе знаний veles. Возвращает решение (answer/abstain),
        ответ с цитатами (файл/строки) и evidence-фрагменты."""
        return get_pipeline().answer(query, top_k=top_k, use_llm=False)

    server.run()


if __name__ == "__main__":
    run()
