#!/usr/bin/env python3
"""K12: диагностика выдачи поиска по expected-узлам (каналы/ранги/RRF).

Для выбранных вопросов golden показывает:
- ранг КАЖДОГО expected-узла в каждом канале по отдельности (dense, BM25) и
  после RRF - на глубине depth (по умолчанию 400, а не 25 как в проде);
- top-10 каждого канала и после RRF: кто именно занял верх выдачи;
- статистику дубликатов индекса (записи stores против уникальных id).

Дополняет retrieval_bench (тот даёт агрегаты, этот - механику промаха).
Запуск (стенд, read-only):
  VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
  ~/veles-venv/bin/python retrieval_diag.py --qids k11-056 --depth 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VELES_REPO = Path(
    os.environ.get("VELES_REPO", str(Path.home() / "proj" / "veles-bench" / "veles"))
)


def load_golden(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from synth_lib import load_jsonl  # noqa: E402

    ap = argparse.ArgumentParser(description="K12 диагностика каналов/рангов")
    ap.add_argument("--golden", default=str(here / "golden.jsonl"))
    ap.add_argument("--qids", default="", help="запятая-список qid; пусто - авто-выбор")
    ap.add_argument("--worst", type=int, default=0,
                    help="взять N худших answerable по RRF-рангу expected (после авто-прогона)")
    ap.add_argument("--provenance", default="",
                    help="фильтр авто-выбора по provenance (manual/auto/...)")
    ap.add_argument("--depth", type=int, default=400)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rrf-k", type=int, default=None)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-lex", type=float, default=1.0)
    ap.add_argument("--codex-boost", type=float, default=None)
    ap.add_argument("--filter-junk", action="store_true")
    ap.add_argument("--dedupe", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(VELES_REPO))
    from veles.config import DEFAULT_ACCESS_GROUPS, EMBED_DIM, EMBED_MODEL, RRF_K, INDEX_DIR
    from veles.indexing.embedder import SentenceTransformersProvider
    from veles.indexing.indexer import load_index
    from veles.retrieval.retriever import is_junk_node

    rrf_k = args.rrf_k if args.rrf_k is not None else RRF_K
    codex_boost = args.codex_boost or 0.0

    print(f"live: индекс {INDEX_DIR} (read-only), модель {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)

    # статистика дублей индекса (утверждение из калибровки - проверить по артефакту)
    n_vec, n_bm25 = len(bundle.vector.ids), len(bundle.bm25.ids)
    u_vec, u_bm25 = len(set(bundle.vector.ids)), len(set(bundle.bm25.ids))
    print(
        f"индекс: vector {n_vec} записей / {u_vec} уникальных id "
        f"(дублей {n_vec - u_vec}); bm25 {n_bm25}/{u_bm25} (дублей {n_bm25 - u_bm25}); "
        f"node_meta {len(bundle.node_meta)}",
        flush=True,
    )

    golden = load_golden(args.golden)
    answerable = [e for e in golden if not e.get("abstain")]
    groups = DEFAULT_ACCESS_GROUPS

    def authorized(nid: str) -> bool:
        meta = bundle.node_meta.get(nid)
        return bool(meta) and meta.get("status") != "deleted" and (
            "*" in groups or set(meta.get("access_groups", [])) & groups
        )

    def channels(entry: dict):
        vec = embedder.embed_query(entry["question"])
        dense = [(n, s) for n, s in bundle.vector.search(vec, top_k=args.depth)
                 if authorized(n)]
        lex = [(n, s) for n, s in bundle.bm25.search(entry["question"], top_k=args.depth)
               if authorized(n)]
        if args.dedupe:
            dense = _dedupe(dense)
            lex = _dedupe(lex)
        return dense, lex

    def rrf_of(dense, lex, entry):
        """RRF c бустом/фильтром (та же семантика, что hybrid_search) -> список id."""
        from veles.policy.identifiers import _CODEX_MAP, extract_codex_refs

        score: dict[str, float] = {}
        for rank, (nid, _) in enumerate(dense, start=1):
            score[nid] = score.get(nid, 0.0) + args.w_dense / (rrf_k + rank)
        for rank, (nid, _) in enumerate(lex, start=1):
            score[nid] = score.get(nid, 0.0) + args.w_lex / (rrf_k + rank)
        if codex_boost > 0.0:
            refs = extract_codex_refs(entry["question"])
            if refs:
                bonus = codex_boost / (rrf_k + 1.0)
                needles = [_CODEX_MAP[r] for r in refs]
                for nid in score:
                    meta = bundle.node_meta.get(nid) or {}
                    title = " ".join(str(meta.get("title", "")).lower().replace("ё", "е").split())
                    if any(n in title for n in needles):
                        score[nid] += bonus
        ordered = sorted(score.items(), key=lambda x: -x[1])
        out = []
        for nid, s in ordered:
            meta = bundle.node_meta.get(nid)
            if not meta:
                continue
            if args.filter_junk and is_junk_node(meta.get("text", "")):
                continue
            out.append((nid, s))
        return out

    def first_ranks(pairs: list[tuple[str, float]], wanted: set[str]) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for i, (nid, _) in enumerate(pairs, start=1):
            if nid in wanted and nid not in ranks:
                ranks[nid] = i
        return ranks

    def report(entry: dict) -> float:
        expected = set(entry.get("expected_nodes") or [])
        dense, lex = channels(entry)
        merged = rrf_of(dense, lex, entry)
        rd, rl, rm = (first_ranks(p, expected) for p in (dense, lex, merged))
        best = min((rm.get(n, 10**9) for n in expected), default=10**9)
        print(f"\n## {entry['id']} [{entry.get('provenance')}] rr={round(1/best,3) if best<10**9 else 0} "
              f"depth={args.depth} top_k={args.top_k}")
        print(f"Q: {entry['question']}")
        print(f"expected узлов: {len(expected)}; найдено на depth: "
              f"dense={sum(1 for n in expected if n in rd)}/{len(expected)}, "
              f"bm25={sum(1 for n in expected if n in rl)}/{len(expected)}, "
              f"rrf={sum(1 for n in expected if n in rm)}/{len(expected)}")
        for name, pairs, ranks in (("DENSE", dense, rd), ("BM25", lex, rl), ("RRF", merged, rm)):
            print(f"\n-- {name} top-{min(10, len(pairs))} --")
            for i, (nid, s) in enumerate(pairs[:10], start=1):
                meta = bundle.node_meta.get(nid) or {}
                mark = " <EXPECTED>" if nid in expected else ""
                junk = " [junk]" if is_junk_node(meta.get("text", "")) else ""
                print(f"  {i:3d} rrf={s:.5f} {meta.get('title', '?')[:48]} :: "
                      f"{meta.get('document_id', '?')[:52]}{mark}{junk}")
        print(f"\n-- expected-узлы (rank dense / bm25 / rrf) --")
        rows = sorted(expected, key=lambda n: min(rd.get(n, 10**9), rl.get(n, 10**9)))
        for nid in rows:
            print(f"  d={rd.get(nid, '-'):>4} b={rl.get(nid, '-'):>4} r={rm.get(nid, '-'):>4} {nid[:12]}…")
        return 1.0 / best if best < 10**9 else 0.0

    if args.qids:
        want = [q.strip() for q in args.qids.split(",") if q.strip()]
        by_id = {e["id"]: e for e in answerable}
        for qid in want:
            if qid not in by_id:
                print(f"нет answerable-вопроса {qid} в golden", flush=True)
                continue
            report(by_id[qid])
        return 0

    # авто-выбор: худшие по rr на текущих параметрах (полный прогон по depth)
    scored = []
    for e in answerable:
        if args.provenance and (e.get("provenance") or "") != args.provenance:
            continue
        expected = set(e.get("expected_nodes") or [])
        dense, lex = channels(e)
        merged = rrf_of(dense, lex, e)
        ranks = first_ranks(merged, expected)
        best = min((ranks.get(n, 10**9) for n in expected), default=10**9)
        scored.append((1.0 / best if best < 10**9 else 0.0, e["id"]))
    scored.sort()
    n = args.worst or 5
    print(f"\n# авто-выбор: {n} худших по 1/ранг"
          f"{' provenance=' + args.provenance if args.provenance else ''}")
    print(", ".join(f"{qid}={rr:.3f}" for rr, qid in scored[:n]))
    by_id = {e["id"]: e for e in answerable}
    for _, qid in scored[:n]:
        report(by_id[qid])
    return 0


def _dedupe(pairs):
    seen, out = set(), []
    for nid, s in pairs:
        if nid not in seen:
            seen.add(nid)
            out.append((nid, s))
    return out


if __name__ == "__main__":
    sys.exit(main())
