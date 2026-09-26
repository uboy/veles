"""Гибридный retrieval: dense + BM25 -> RRF -> evidence pack с цитатами."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RRF_K
from ..indexing.indexer import IndexBundle


@dataclass
class Citation:
    title: str
    document_id: str
    heading_path: list[str]
    source_id: str
    lines: list[int] | None
    node_version_id: str

    def human(self) -> str:
        where = f", строки {self.lines[0]}-{self.lines[1]}" if self.lines else ""
        return f"«{self.title}» {self.source_id}{where}"


@dataclass
class Evidence:
    node_version_id: str
    text: str
    rrf: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    citations: list[Citation] = field(default_factory=list)


def hybrid_search(
    bundle: IndexBundle,
    query_vec,
    query_text: str,
    top_k: int = 8,
    allowed_groups: set[str] | None = None,
) -> list[Evidence]:
    """Pre-retrieval authorization (инвариант 5): узлы без пересечения групп с
    allowed_groups отсекаются до формирования evidence и не попадают в контекст."""
    from ..config import DEFAULT_ACCESS_GROUPS

    groups = allowed_groups or DEFAULT_ACCESS_GROUPS

    def authorized(nid: str) -> bool:
        meta = bundle.node_meta.get(nid)
        return bool(meta) and bool(set(meta.get("access_groups", [])) & groups)

    dense = [(nid, s) for nid, s in bundle.vector.search(query_vec, top_k=top_k * 5) if authorized(nid)]
    lex = [(nid, s) for nid, s in bundle.bm25.search(query_text, top_k=top_k * 5) if authorized(nid)]

    rrf: dict[str, float] = {}
    ranks: dict[str, dict] = {}
    for rank, (nid, _) in enumerate(dense, start=1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
        ranks.setdefault(nid, {})["dense_rank"] = rank
    for rank, (nid, _) in enumerate(lex, start=1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
        ranks.setdefault(nid, {})["bm25_rank"] = rank

    ordered = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
    evidence: list[Evidence] = []
    for nid, score in ordered:
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        prov = meta["provenance"]
        cite = Citation(
            title=meta["title"],
            document_id=meta["document_id"],
            heading_path=meta["heading_path"],
            source_id=prov["source_id"],
            lines=prov.get("lines"),
            node_version_id=nid,
        )
        evidence.append(
            Evidence(
                node_version_id=nid,
                text=meta["text"],
                rrf=score,
                dense_rank=ranks.get(nid, {}).get("dense_rank"),
                bm25_rank=ranks.get(nid, {}).get("bm25_rank"),
                citations=[cite],
            )
        )
    return evidence
