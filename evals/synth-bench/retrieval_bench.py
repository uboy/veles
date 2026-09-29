#!/usr/bin/env python3
"""K12: retrieval-bench поверх golden-набора K11 (метрики выдачи ПОИСКА).

Метрики по answerable-вопросам golden (abstain-вопросы не считаются и не
смешиваются - у них нет expected):
- R@5  - доля вопросов с хотя бы одним expected-узлом в top-5;
- top1 - доля вопросов с expected-узлом на ранге 1;
- MRR  - среднее 1/ранг первого expected-узла (промах = 0 в среднем);
- P@5  - средняя доля релевантных фрагментов в top-5; фрагмент релевантен =
  его узел в expected ИЛИ его документ совпадает с expected-документом
  (бриф K12: «принадлежность expected-документу/узлу»).

Разбивки: provenance (rrncb/miniqa/auto/manual) и домен (кодексы/лекции/rrncb).

Режимы:
- кэш (по умолчанию): метрики по замороженной выдаче retrieval-cache.json -
  так зафиксирован бейзлайн ДО и результат ПОСЛЕ перезаморозки;
- --live: in-process retrieval против прод-индекса (стенд, read-only, load_index
  без save) с калибровочными параметрами hybrid_search - изолированная проверка
  фикс-кандидатов до перезаморозки кэша.

Запуск live (стенд):
  VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
  ~/veles-venv/bin/python retrieval_bench.py --golden golden.jsonl --live \
      --codex-boost 2.0 --depth 200 --filter-junk
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

# Канонические домены разбивки (по source_doc golden-записи).
DOMAIN_ABSTAIN = "abstain"


def domain_of(source_doc: str) -> str:
    """Домен вопроса по ожидаемому документу: лекции / кодексы / rrncb / abstain."""
    doc = (source_doc or "").strip()
    if not doc:
        return DOMAIN_ABSTAIN
    if doc.startswith("lectures/"):
        return "лекции"
    if "кодекс" in doc.lower():
        return "кодексы"
    return "rrncb"


def fragment_relevant(entry: dict, frag: dict, expected: set[str]) -> bool:
    """Релевантность фрагмента вопросу: узел из expected или документ совпадает."""
    if frag.get("node_version_id") in expected:
        return True
    src = (entry.get("source_doc") or "").strip()
    return bool(src) and frag.get("source_id") == src


def question_metrics(entry: dict, frags: list[dict]) -> dict | None:
    """Метрики одного вопроса по его выдаче (список фрагментов с рангами 1..k).

    abstain-запись -> None (в метрики не попадает, только в счётчик).
    """
    if entry.get("abstain"):
        return None
    expected = set(entry.get("expected_nodes") or [])
    ranks = sorted(rank for rank in range(1, len(frags) + 1)
                   if frags[rank - 1].get("node_version_id") in expected)
    p5_denominator = min(5, len(frags)) or 1
    relevant = sum(1 for i, frag in enumerate(frags[:5], start=1)
                   if fragment_relevant(entry, frag, expected))
    return {
        "id": entry["id"],
        "provenance": entry.get("provenance") or "?",
        "domain": domain_of(entry.get("source_doc") or ""),
        "hit": 1 if ranks else 0,
        "top1": 1 if ranks and ranks[0] == 1 else 0,
        "rr": (1.0 / ranks[0]) if ranks else 0.0,
        "p5": relevant / p5_denominator,
    }


def aggregate(rows: list[dict]) -> dict:
    """Агрегат по набору question_metrics: N, R@5, top1, MRR, P@5."""
    n = len(rows)
    return {
        "n": n,
        "r_at_5": round(sum(r["hit"] for r in rows) / n, 4) if n else None,
        "top1": round(sum(r["top1"] for r in rows) / n, 4) if n else None,
        "mrr": round(sum(r["rr"] for r in rows) / n, 4) if n else None,
        "p_at_5": round(sum(r["p5"] for r in rows) / n, 4) if n else None,
    }


def breakdowns(rows: list[dict], key: str) -> dict[str, dict]:
    """Агрегаты по группам ключа (provenance/domain); порядок групп - по алфавиту."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {g: aggregate(groups[g]) for g in sorted(groups)}


def render_table(title: str, agg: dict, groups: dict[str, dict]) -> str:
    """Markdown: строка TOTAL + строка на группу."""
    fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    out = [f"**{title}**\n", "| ГРУППА | N | R@5 | top1 | MRR | P@5 |", "|---|---|---|---|---|---|"]
    for name, a in [("TOTAL", agg), *groups.items()]:
        out.append(
            f"| {name} | {a['n']} | {fmt(a['r_at_5'])} | {fmt(a['top1'])} "
            f"| {fmt(a['mrr'])} | {fmt(a['p_at_5'])} |"
        )
    return "\n".join(out) + "\n"


def frags_from_cache(cache: dict, qid: str) -> list[dict]:
    """Фрагменты вопроса из замороженного кэша K11/K12 (порядок = ранг)."""
    entry = cache["questions"].get(qid)
    if not entry:
        raise KeyError(f"в кэше нет вопроса {qid}")
    return entry["fragments"]


# ------------------------------------------------------------------ live (стенд)

def run_live(golden: list[dict], args) -> dict[str, list[dict]]:
    """Один прогон in-process retrieval против прод-индекса (read-only, без save).

    Возвращает {qid: fragments} в формате кэша - метрики считаются теми же
    функциями, что для замороженного кэша."""
    sys.path.insert(0, str(VELES_REPO))
    from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR
    from veles.indexing.embedder import SentenceTransformersProvider
    from veles.indexing.indexer import load_index
    from veles.retrieval.retriever import hybrid_search

    print(f"live: индекс {INDEX_DIR} (read-only), модель {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)
    print(f"live: {len(bundle.node_meta)} узлов", flush=True)

    top_k = args.top_k
    out: dict[str, list[dict]] = {}
    for i, entry in enumerate(golden, 1):
        vec = embedder.embed_query(entry["question"])
        evidence = hybrid_search(
            bundle,
            vec,
            entry["question"],
            top_k=top_k,
            rrf_k=args.rrf_k,
            channel_weights=(args.w_dense, args.w_lex),
            candidate_depth=args.depth,
            codex_boost=args.codex_boost,
            filter_junk=args.filter_junk,
            dedupe_channels=args.dedupe,
        )
        frags = []
        for n, ev in enumerate(evidence, 1):
            c = ev.citations[0]
            frags.append(
                {
                    "n": n,
                    "node_version_id": ev.node_version_id,
                    "document_id": c.document_id,
                    "title": c.title,
                    "source_id": c.source_id,
                    "lines": c.lines,
                    "heading_path": c.heading_path,
                    "text": ev.text,
                    "rrf": round(ev.rrf, 5),
                    "dense_rank": ev.dense_rank,
                    "bm25_rank": ev.bm25_rank,
                }
            )
        out[entry["id"]] = frags
        if i % 20 == 0:
            print(f"  {i}/{len(golden)}", flush=True)
    return out


def main() -> int:
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from synth_lib import load_jsonl  # noqa: E402

    ap = argparse.ArgumentParser(description="K12 retrieval-bench")
    ap.add_argument("--golden", default=str(here / "golden.jsonl"))
    ap.add_argument("--cache", default=str(here / "retrieval-cache.json"))
    ap.add_argument("--live", action="store_true", help="прогон live-retrieval (стенд)")
    ap.add_argument("--top-k", type=int, default=5)
    # калибровочные параметры hybrid_search (None -> дефолты config)
    ap.add_argument("--rrf-k", type=int, default=None)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--w-lex", type=float, default=1.0)
    ap.add_argument("--depth", type=int, default=None, help="глубина каналов (дефолт top_k*5)")
    ap.add_argument("--codex-boost", type=float, default=None)
    ap.add_argument("--filter-junk", action="store_true")
    ap.add_argument("--dedupe", action="store_true",
                    help="снимать дубликаты узла внутри каналов (дефект индекса)")
    ap.add_argument("--out-json", default=None, help="сохранить выдачу/метрики в JSON")
    args = ap.parse_args()

    golden = load_jsonl(args.golden)
    n_answerable = sum(1 for e in golden if not e.get("abstain"))

    if args.live:
        frags_by_qid = run_live(golden, args)
    else:
        with open(args.cache, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"кэш: {args.cache} (created {cache['meta'].get('created')})", flush=True)
        frags_by_qid = {e["id"]: frags_from_cache(cache, e["id"]) for e in golden}

    rows = []
    for entry in golden:
        m = question_metrics(entry, frags_by_qid[entry["id"]])
        if m is not None:
            rows.append(m)

    total = aggregate(rows)
    by_prov = breakdowns(rows, "provenance")
    by_dom = breakdowns(rows, "domain")

    label = "live" if args.live else "cache"
    params = ""
    if args.live:
        params = (
            f" (rrf_k={args.rrf_k}, w=({args.w_dense},{args.w_lex}), depth={args.depth}, "
            f"codex_boost={args.codex_boost}, filter_junk={args.filter_junk}, "
            f"dedupe={args.dedupe})"
        )
    print(f"\n# retrieval-bench [{label}]{params}")
    print(f"golden: {len(golden)} вопросов, answerable {n_answerable}")
    print(render_table("Итого", total, {}))
    print(render_table("По provenance", total, by_prov))
    print(render_table("По домену", total, by_dom))

    misses = [r["id"] for r in rows if not r["hit"]]
    if misses:
        print(f"промахи R@5 ({len(misses)}): {', '.join(misses)}")

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {"mode": label, "params": vars(args), "total": total,
                 "by_provenance": by_prov, "by_domain": by_dom, "misses": misses,
                 "rows": rows},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        print(f"метрики: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
