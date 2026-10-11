"""Схема CanonicalDocument проекта veles (не DoclingDocument; парсер - адаптер).

Идентификация по стандарту v3.0 (§4):
  document_id / revision_id / logical_node_id / node_version_id.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Provenance(BaseModel):
    """Locator по формату источника; абсолютные пути запрещены (инвариант 4)."""

    adapter: str
    source_id: str  # относительный путь в корпусе, posix-стиль
    lines: Optional[list[int]] = None  # TXT/MD: [start, end], 1-based inclusive
    page: Optional[int] = None  # PDF
    slide: Optional[int] = None  # PPTX
    sheet: Optional[str] = None  # XLSX
    # Видео: границы сегмента речи в секундах от начала файла (адаптер video:0.1).
    # Optional - старые каноны и не-видео узлы читаются как None без миграции.
    t_start: Optional[float] = None
    t_end: Optional[float] = None


class CanonicalNode(BaseModel):
    logical_node_id: str
    node_version_id: str
    node_type: Literal["root", "section", "paragraph", "image"] = "paragraph"
    parent_id: Optional[str] = None
    heading_path: list[str] = Field(default_factory=list)
    text: str
    provenance: Provenance
    access_groups: list[str] = Field(default_factory=lambda: ["personal"])
    asset: Optional[str] = None  # имя файла актива в <doc_dir>/assets/ (узел-картинка)

    @staticmethod
    def make_version_id(revision_id: str, logical_node_id: str, text: str) -> str:
        normalized = " ".join(text.split())
        return _sha256(f"{revision_id}|{logical_node_id}|{normalized}")


class CanonicalDocument(BaseModel):
    document_id: str
    revision_id: str = "r1"
    title: str
    document_type: str = "transcript"
    status: Literal["current", "superseded"] = "current"
    sha256: str = ""
    parser_profile: str
    access_groups: list[str] = Field(default_factory=lambda: ["personal"])
    derived_from: list[str] = Field(default_factory=list)
    # ревизии (K3): id предыдущей ревизии при повторном ингесте изменённого документа;
    # у первой ревизии и у канона до 29.09 - None
    previous_revision: Optional[str] = None
    # кто и когда загрузил документ (аудит загрузки; корпус до 28.09 - без этих полей)
    ingested_at: Optional[str] = None
    ingested_by: Optional[str] = None
    # активы (картинки): имя -> путь. У запарсенного в памяти - временный путь;
    # после save_document в манифесте - относительный "assets/<имя>"
    assets: dict[str, str] = Field(default_factory=dict)
    nodes: list[CanonicalNode] = Field(default_factory=list)

    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        """Хеш сырых байтов L0-файла (верифицируем против allowlist-манифеста)."""
        return hashlib.sha256(raw_bytes).hexdigest()

    def add_node(self, **kwargs: Any) -> CanonicalNode:
        node = CanonicalNode(
            logical_node_id=kwargs["logical_node_id"],
            node_version_id=CanonicalNode.make_version_id(
                self.revision_id, kwargs["logical_node_id"], kwargs["text"]
            ),
            **{k: v for k, v in kwargs.items() if k != "logical_node_id"},
        )
        self.nodes.append(node)
        return node

    def retag_revision(self, revision_id: str) -> None:
        """Смена revision_id у спарсенного документа (повторный ингест, K3).

        node_version_id включает revision_id, поэтому без пересчёта узлы новой
        ревизии столкнулись бы ключами с узлами архивной в общем индексе."""
        self.revision_id = revision_id
        for node in self.nodes:
            node.node_version_id = CanonicalNode.make_version_id(
                revision_id, node.logical_node_id, node.text
            )
