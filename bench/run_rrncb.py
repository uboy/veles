"""Прогон RRNCB-бенчмарка Ф2.

Этапы: парс PDF -> выбор подмножества (бюджет узлов) -> два индекса (flat/canonical)
-> gold-локализация -> Recall@5/10, MRR@10 -> held-out FAR + калибровка порога.
Запуск: python bench/run_rrncb.py [--node-budget 12000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.rrncb import (  # noqa: E402
    ConfigIndex,
    DOCS_DIR,
    canonical_units,
    flat_units,
    load_qa,
    localize_gold,
)
from veles.canonical.store import save_document  # noqa: E402
from veles.ingest.pdf_text import PyMuPdfParserAdapter  # noqa: E402
from veles.config import INDEX_DIR, RRF_K  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import IndexBundle  # noqa: E402
from veles.policy.answer import _confidence  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402


def build_bundle(units: list, embedder, batch: int = 256) -> IndexBundle:
    from veles.indexing.vector_store import make_store

    store = make_store(prefer="faiss")
    bm25 = Bm25Index()
    meta = {}
    for i in range(0, len(units), batch):
        chunk = units[i : i + batch]
        vecs = embedder.embed_passages([u.text for u in chunk])
        store.add([u.uid for u in chunk], vecs)
        bm25.add([u.uid for u in chunk], [u.text for u in chunk])
        for u in chunk:
            meta[u.uid] = {"text": u.text, "access_groups": ["personal"]}
        print(f"    проиндексировано {len(meta)}/{len(units)}", flush=True)
    return IndexBundle(vector=store, bm25=bm25, node_meta=meta)


def rank_of_gold(bundle, embedder, question: str, gold_uid: str, k: int = 10) -> tuple[int, float]:
    qvec = embedder.embed_query(question)
    hits = bundle.vector.search(qvec, top_k=k * 4)
    lex = bundle.bm25.search(question, top_k=k * 4)
    rrf = {}
    for rank, (nid, _) in enumerate(hits, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (nid, _) in enumerate(lex, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(rrf.items(), key=lambda x: -x[1])
    for pos, (nid, score) in enumerate(ordered[:k], 1):
        if nid == gold_uid:
            return pos, score
    return 0, (ordered[0][1] if ordered else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-budget", type=int, default=12000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--holdout-docs", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    qa = load_qa()
    print(f"QA пар: {len(qa)}")

    # 1. Парс всех доступных PDF
    parser = PyMuPdfParserAdapter()
    parsed: dict[str, object] = {}
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            doc = parser.parse_file(pdf, corpus_root=DOCS_DIR)
            parsed[pdf.name] = doc
        except Exception as e:  # noqa: BLE001 - бенчмарк не должен падать целиком
            print(f"  PARSE FAIL {pdf.name}: {e}")
    print(f"распарсено документов: {len(parsed)} ({time.time() - t0:.0f}с)")

    # 2. Выбор подмножества по бюджету узлов: приоритет - покрытие QA
    qa_by_doc: dict[str, int] = {}
    for r in qa:
        if r["document"] in parsed:
            qa_by_doc[r["document"]] = qa_by_doc.get(r["document"], 0) + 1
    ranked_docs = sorted(qa_by_doc, key=lambda d: -qa_by_doc[d])
    chosen, budget = [], 0
    for d in ranked_docs:
        n = len(parsed[d].nodes)
        if budget + n > args.node_budget and chosen:
            continue
        chosen.append(d)
        budget += n
    chosen.sort()
    covered_qa = sum(qa_by_doc.get(d, 0) for d in chosen)
    print(f"выбрано документов: {len(chosen)} (узлов {budget}), QA покрыто: {covered_qa}")

    subset_qa = [r for r in qa if r["document"] in chosen]
    print(f"QA в подмножестве: {len(subset_qa)}")

    # 3. Юниты двух конфигураций
    cfg_flat = ConfigIndex(name="flat-800")
    cfg_canon = ConfigIndex(name="canonical-s2b")
    for d in chosen:
        doc = parsed[d]
        cfg_flat.units.extend(flat_units(doc))
        cfg_canon.units.extend(canonical_units(doc))
    print(f"юниты: flat={len(cfg_flat.units)}, canonical={len(cfg_canon.units)}")

    # 4. Gold-локализация на обеих конфигурациях
    gold = {}
    for i, r in enumerate(subset_qa):
        g1 = localize_gold(cfg_flat.units, r["answer"])
        g2 = localize_gold(cfg_canon.units, r["answer"])
        if g1 and g2:
            gold[i] = (g1, g2)
    print(f"gold локализован в обеих конфигурациях: {len(gold)}/{len(subset_qa)}")

    # 5. Индексация и метрики
    embedder = SentenceTransformersProvider(
        "intfloat/multilingual-e5-small", dim=384
    )
    results = {"configs": {}, "meta": {"node_budget": args.node_budget,
                "docs": chosen, "qa_subset": len(subset_qa), "qa_gold": len(gold)}}
    for cfg in (cfg_flat, cfg_canon):
        print(f"индексация {cfg.name}: {len(cfg.units)} юнитов...", flush=True)
        t = time.time()
        cfg.bundle = build_bundle(cfg.units, embedder)
        hits5 = hits10 = 0
        rr_sum = 0.0
        for i, (g1, g2) in gold.items():
            g = g1 if cfg.name == "flat-800" else g2
            pos, _ = rank_of_gold(cfg.bundle, embedder, subset_qa[i]["question"], g, k=args.k)
            hits5 += 1 <= pos <= 5
            hits10 += 1 <= pos <= args.k
            rr_sum += (1.0 / pos) if pos else 0.0
        n = len(gold)
        results["configs"][cfg.name] = {
            "units": len(cfg.units),
            "recall@5": round(hits5 / n, 4),
            "recall@10": round(hits10 / n, 4),
            "mrr@10": round(rr_sum / n, 4),
            "index_seconds": round(time.time() - t),
        }
        print(f"  {cfg.name}: R@5={results['configs'][cfg.name]['recall@5']} "
              f"R@10={results['configs'][cfg.name]['recall@10']} "
              f"MRR={results['configs'][cfg.name]['mrr@10']}")

    # 6. Held-out FAR + калибровка порога
    holdout = sorted(chosen, key=lambda d: -qa_by_doc.get(d, 0))[: args.holdout_docs]
    indocs = [d for d in chosen if d not in holdout]
    print(f"held-out ловушки: {len(holdout)} документов, индекс без них...")
    hold_cfg = ConfigIndex(name="holdout")
    for d in indocs:
        doc = parsed[d]
        hold_cfg.units.extend(canonical_units(doc))
    hold_cfg.bundle = build_bundle(hold_cfg.units, embedder)

    trap_qa = [(i, r) for i, r in enumerate(subset_qa) if r["document"] in holdout]
    print(f"ловушек-вопросов: {len(trap_qa)}")
    sweep = {}
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        far = 0
        confs = []
        for i, r in trap_qa:
            qvec = embedder.embed_query(r["question"])
            ev = hybrid_search(hold_cfg.bundle, qvec, r["question"], top_k=8)
            conf = _confidence(ev)
            confs.append(conf)
            far += conf >= thr
        # на полном индексе: доля ответов на локализуемые in-corpus вопросы
        ans = 0
        for i, (g1, g2) in gold.items():
            qvec = embedder.embed_query(subset_qa[i]["question"])
            ev = hybrid_search(cfg_canon.bundle, qvec, subset_qa[i]["question"], top_k=8)
            ans += _confidence(ev) >= thr
        sweep[str(thr)] = {
            "far_heldout": round(far / max(len(trap_qa), 1), 4),
            "answer_rate_ingold": round(ans / max(len(gold), 1), 4),
            "conf_mean_trap": round(sum(confs) / max(len(confs), 1), 4),
        }
        print(f"  порог {thr}: FAR={sweep[str(thr)]['far_heldout']} answer_rate={sweep[str(thr)]['answer_rate_ingold']}")
    results["threshold_sweep"] = sweep

    out = Path(__file__).parent.parent / "data" / "bench" / "rrncb-v1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"результаты: {out}")
    print(f"полное время: {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
