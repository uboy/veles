"""CLI veles: scan-corpus | ingest | index | ask | status.

Запуск: python -m veles <command> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical.store import iter_documents
from .canonical.store import save_document
from .config import CANONICAL_DIR, CORPUS_ROOT, EMBED_DIM, EMBED_MODEL, INDEX_DIR
from .ingest.allowlist import scan_and_copy
from .ingest.txt_md import TxtMdParserAdapter
from .indexing.bm25 import Bm25Index  # noqa: F401 (реэкспорт для типа)
from .indexing.embedder import SentenceTransformersProvider
from .indexing.indexer import build_index, load_index
from .indexing.vector_store import make_store
from .policy.answer import decide
from .policy.identifiers import extract_identifiers
from .retrieval.retriever import hybrid_search


def cmd_scan_corpus(args: argparse.Namespace) -> int:
    m = scan_and_copy(copy=not args.dry)
    print(f"манифест: {m.name} @ {m.created_at}")
    print(f"включено: {m.stats['included_files']} файлов ({m.stats['included_bytes']} байт), "
          f"скопировано сейчас: {m.stats['copied_now']}, чёрный список: {m.stats['excluded_by_blacklist']}, "
          f"вне allowlist: {m.stats['outside_allowlist']}")
    for e in m.included[:5]:
        print(f"  + {e.dest_rel} [{e.sha256[:12]}]")
    if len(m.included) > 5:
        print(f"  ... и ещё {len(m.included) - 5}")
    return 0


def _manifest_known_relpaths() -> set[str]:
    """Все dest_rel из манифестов allowlist-конвейера (любые корпусы)."""
    import json

    from .config import MANIFESTS_DIR

    known: set[str] = set()
    if MANIFESTS_DIR.exists():
        for mpath in sorted(MANIFESTS_DIR.glob("*.json")):
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - битый манифест не роняет ingest
                continue
            known.update(e["dest_rel"] for e in m.get("included", []))
    return known


def cmd_ingest(args: argparse.Namespace) -> int:
    parser = TxtMdParserAdapter()
    docs = 0
    nodes = 0
    skipped = 0
    corpus_dir = CORPUS_ROOT / "05_vyshka_academic" / "lectures"
    known = set() if args.no_manifest_check else _manifest_known_relpaths()
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        rel = path.relative_to(CORPUS_ROOT).as_posix()
        if known and rel not in known:
            print(f"  ! пропущен (нет в allowlist-манифесте): {rel}")
            skipped += 1
            continue
        doc = parser.parse_file(path, corpus_root=CORPUS_ROOT)
        save_document(doc)
        docs += 1
        nodes += len(doc.nodes)
        print(f"  {doc.document_id}: {len(doc.nodes)} узлов")
    print(f"итого: документов={docs}, узлов={nodes}, пропущено мимо конвейера={skipped} -> {CANONICAL_DIR}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    docs = list(iter_documents())
    if not docs:
        print("канон пуст - сначала `ingest`")
        return 1
    print(f"эмбеддер: {EMBED_MODEL}")
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    store = make_store(prefer=args.backend)
    bundle = build_index(embedder, store)
    bundle.save(INDEX_DIR, embed_meta={"model": EMBED_MODEL, "backend": store.name})
    print(f"проиндексировано узлов: {len(bundle.node_meta)}; backend={store.name}; bm25={len(bundle.bm25.ids)} -> {INDEX_DIR}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not (INDEX_DIR / "meta.json").exists():
        print("индекс не собран - сначала `index`")
        return 1
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)
    qvec = embedder.embed_query(args.query)
    evidence = hybrid_search(bundle, qvec, args.query, top_k=args.top_k)
    decision = decide(
        evidence,
        query_identifiers=extract_identifiers(args.query),
        known_identifiers=bundle.identifiers,
    )
    print(f"решение: {decision.decision.upper()} (доверие {decision.confidence}; {decision.reason})")
    if decision.decision == "answer":
        top = decision.evidence[0]
        print(f"ответ (текст узла, {len(top.text)} знаков):\n{top.text}\n")
        print("цитаты:")
        for ev in decision.evidence[:3]:
            for c in ev.citations:
                hp = " > ".join(c.heading_path[-3:]) if c.heading_path else ""
                print(f"  - {c.human()} {('| ' + hp) if hp else ''}")
    print("diagnostics:")
    for ev in decision.evidence[:3]:
        print(f"  {ev.node_version_id[:40]} rrf={ev.rrf:.4f} dense={ev.dense_rank} bm25={ev.bm25_rank}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    docs = list(iter_documents())
    print(f"корпус: {CORPUS_ROOT}")
    print(f"канон: {len(docs)} документов, {sum(len(d.nodes) for d in docs)} узлов ({CANONICAL_DIR})")
    print(f"индекс: {'есть' if (INDEX_DIR / 'meta.json').exists() else 'нет'} ({INDEX_DIR})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="veles", description="Knowledge Core CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan-corpus", help="allowlist-скан Вышки -> копии в корпус + манифест")
    p.add_argument("--dry", action="store_true", help="только манифест, без копирования")
    p.set_defaults(func=cmd_scan_corpus)

    p = sub.add_parser("ingest", help="корпус -> канон (manifest.json + nodes.jsonl)")
    p.add_argument("--no-manifest-check", action="store_true",
                   help="не сверять файлы с allowlist-манифестом (по умолчанию сверка включена)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("index", help="канон -> L2 (dense + BM25)")
    p.add_argument("--backend", default="faiss", choices=["faiss", "numpy"])
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="вопрос с цитатами")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=8)
    p.set_defaults(func=cmd_ask)

    sub.add_parser("status", help="состояние корпуса/канона/индекса").set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
