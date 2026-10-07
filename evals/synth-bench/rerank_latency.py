#!/usr/bin/env python3
"""K17: латентность rerank-этапа (p50/p95) на CPU - локальная машина или стенд.

ЧТО меряется (на каждом вопросе наборов golden 72 + rrncb50 50 + codex-check 6):
- baseline: hybrid_search top_k=5 БЕЗ реранкера (wall, ms);
- пул: hybrid_search top_k=25 БЕЗ реранкера - тот же пул кандидатов (верх-25 RRF),
  который при включённом реранкере уходит в cross-encoder;
- rerank: rerank(вопрос, пул-25, top_k=5) с реальной моделью (wall, ms);
- e2e: hybrid_search top_k=5 С включённым реранкером (wall, ms).

ЧЕГО НЕ меряется здесь: качество (top1/R@5) - для него отдельный retrieval_bench.py
(--live) на индексе, содержащем expected-узлы наборов (прод стенд). Этот скрипт -
только латентность этапа и загрузка модели (бриф K17 п.3: цель < 1.5 c на 25 пар).

Запуск (индекс - из VELES_DATA, read-only, load_index без save; на стенд:
VELES_DATA=$HOME/proj/veles-bench/veles/data CUDA_VISIBLE_DEVICES=):
  VELES_DATA=/path/to/veles/data PYTHONUNBUFFERED=1 python evals/synth-bench/rerank_latency.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import veles.config as config  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import IndexBundle, load_index  # noqa: E402
from veles.retrieval import reranker  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402


def load_bundle() -> IndexBundle:
    """load_index, а для индекса старой версии (без identifiers в meta.json) -
    ручная сборка того же бандла: hybrid_search использует только vector/bm25/
    node_meta, identifiers (политика ответа) для латентности не нужен."""
    try:
        return load_index(config.INDEX_DIR)
    except ValueError as e:
        if "identifiers" not in str(e):
            raise
        import json

        import faiss

        from veles.indexing.vector_store import FaissFlatStore

        d = config.INDEX_DIR
        print(f"load_index: {e}; собираю бандл вручную (identifiers={set()})", flush=True)
        bm25_payload = json.loads((d / "bm25.json").read_text(encoding="utf-8"))
        bm25 = Bm25Index(k1=bm25_payload["k1"], b=bm25_payload["b"])
        bm25.ids = bm25_payload["ids"]
        bm25.doc_lens = bm25_payload["doc_lens"]
        bm25.tfs = bm25_payload["tfs"]
        bm25.df = bm25_payload["df"]
        bm25.avgdl = bm25_payload["avgdl"]
        store = FaissFlatStore(dim=config.EMBED_DIM)
        store.index = faiss.read_index(str(d / "faiss.index"))
        store.ids = json.loads((d / "faiss_ids.json").read_text(encoding="utf-8"))
        node_meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))["node_meta"]
        return IndexBundle(vector=store, bm25=bm25, node_meta=node_meta, identifiers=set())


def load_questions() -> list[str]:
    """Вопросы бенч-наборов K11/K12 (латентность не зависит от expected-узлов)."""
    qs: list[str] = []
    for name in (
        "evals/synth-bench/golden.jsonl",
        "evals/synth-bench/rrncb50.jsonl",
        "evals/synth-bench/legal-boost-check.jsonl",
    ):
        with open(ROOT / name, encoding="utf-8") as f:
            qs.extend(json.loads(line)["question"] for line in f if line.strip())
    return qs


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def fmt_table(rows: list[dict]) -> str:
    head = "| этап | p50, мс | p95, мс | mean, мс | N |"
    sep = "|---|---|---|---|---|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['p50']:.0f} | {r['p95']:.0f} | {r['mean']:.0f} | {r['n']} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="K17 rerank latency (CPU)")
    ap.add_argument(
        "--models", nargs="+",
        default=["BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3"],
        help="HuggingFace id моделей CrossEncoder (порядок = порядок прогонов)",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="ограничить число вопросов (0 = все 128; на медленном CPU - "
             "подмножество, N честно пишется в таблицу)",
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    embedder = SentenceTransformersProvider(config.EMBED_MODEL, dim=config.EMBED_DIM)
    bundle = load_bundle()
    print(f"индекс: {config.INDEX_DIR} ({len(bundle.node_meta)} узлов), "
          f"эмбеддер {config.EMBED_MODEL}, прогрев {time.perf_counter() - t0:.0f} с",
          flush=True)

    questions = load_questions()
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"вопросы: {len(questions)} из 128 (golden 72 + rrncb50 50 + codex-check 6)",
          flush=True)

    # --- baseline (без реранкера) + пулы кандидатов ------------------------
    base_ms, pool_ms, pools, lens = [], [], [], []
    for i, q in enumerate(questions, 1):
        vec = embedder.embed_query(q)
        t = time.perf_counter()
        hybrid_search(bundle, vec, q, top_k=5)
        base_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        pool = hybrid_search(bundle, vec, q, top_k=25)
        pool_ms.append((time.perf_counter() - t) * 1000)
        pools.append(pool)
        lens.extend(len(e.text) for e in pool)
        if i % 32 == 0:
            print(f"  baseline/pool {i}/{len(questions)}", flush=True)
    lens.sort()
    print(f"\nдлины текстов пула: p50={lens[len(lens)//2]} p95={lens[int(len(lens)*0.95)]} "
          f"max={lens[-1]} символов")
    print("baseline (без реранкера):")
    print(fmt_table([
        {"stage": "hybrid_search top5", "p50": pct(base_ms, .5), "p95": pct(base_ms, .95),
         "mean": statistics.mean(base_ms), "n": len(base_ms)},
        {"stage": "пул top25 (без rerank)", "p50": pct(pool_ms, .5), "p95": pct(pool_ms, .95),
         "mean": statistics.mean(pool_ms), "n": len(pool_ms)},
    ]))

    # --- по моделям: загрузка, rerank-этап, e2e ----------------------------
    for name in args.models:
        reranker.reset_cache()
        t = time.perf_counter()
        reranker._load(name)
        load_s = time.perf_counter() - t
        reranker.rerank(questions[0], pools[0], 5, model_name=name)  # прогрев ядра
        rr_ms, e2e_ms = [], []
        for i, q in enumerate(questions, 1):
            t = time.perf_counter()
            reranker.rerank(q, pools[i - 1], 5, model_name=name)
            rr_ms.append((time.perf_counter() - t) * 1000)
            vec = embedder.embed_query(q)
            config.VELES_RERANKER_MODEL = name  # включить пост-этап в hybrid_search
            t = time.perf_counter()
            hybrid_search(bundle, vec, q, top_k=5)
            e2e_ms.append((time.perf_counter() - t) * 1000)
            config.VELES_RERANKER_MODEL = ""  # выключить (baseline следующей модели)
            if i % 32 == 0:
                print(f"  {name}: {i}/{len(questions)}", flush=True)
        print(f"\n## {name}")
        print(f"загрузка модели: {load_s:.1f} с (device=cpu)")
        print(fmt_table([
            {"stage": "rerank 25 пар", "p50": pct(rr_ms, .5), "p95": pct(rr_ms, .95),
             "mean": statistics.mean(rr_ms), "n": len(rr_ms)},
            {"stage": "e2e hybrid_search+rerank top5", "p50": pct(e2e_ms, .5),
             "p95": pct(e2e_ms, .95), "mean": statistics.mean(e2e_ms), "n": len(e2e_ms)},
        ]))
    reranker.reset_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
