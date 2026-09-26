"""Прогон RRNCB-бенчмарка Ф2, v2.

Метрика A (H2/H4, ранжирование листьев): Recall@5/10, MRR@10 по gold-МНОЖЕСТВУ
(фрагмент может лежать в двух соседних юнитах - засчитывается лучший ранг).
Метрика B (H3, полнота контекста): покрытие gold-фрагмента собранным evidence-pack
при равном символьном бюджете; canonical расширяет топ-листья родителем и соседями.
Калибровка AnswerPolicy v2: identifier-grounding (номер ГОСТ/СП из вопроса отсутствует
в корпусе -> abstain) + свип порога по held-out ловушкам.
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
    _grams_tokens,
    canonical_units,
    corpus_identifier_set,
    extract_identifiers,
    flat_units,
    load_qa,
    localize_gold_best,
)
from veles.config import RRF_K  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import IndexBundle  # noqa: E402
from veles.ingest.pdf_text import PyMuPdfParserAdapter  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

EVIDENCE_BUDGET = 4000  # символов контекста на вопрос, обе конфигурации


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
            doc_part = u.uid.split(":")[-1] if ":" in u.uid else u.uid
            meta[u.uid] = {
                "text": u.text,
                "access_groups": ["personal"],
                "title": doc_part,
                "document_id": doc_part,
                "heading_path": [],
                "provenance": {"adapter": "bench", "source_id": doc_part, "lines": None},
            }
        print(f"    {len(meta)}/{len(units)}", flush=True)
    return IndexBundle(vector=store, bm25=bm25, node_meta=meta)


def rrf_ranked(bundle, embedder, question: str, k: int = 10) -> list[str]:
    qvec = embedder.embed_query(question)
    hits = bundle.vector.search(qvec, top_k=k * 4)
    lex = bundle.bm25.search(question, top_k=k * 4)
    rrf = {}
    for rank, (nid, _) in enumerate(hits, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (nid, _) in enumerate(lex, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    return [nid for nid, _ in sorted(rrf.items(), key=lambda x: -x[1])]


def best_rank(ranked: list[str], gold_set: set[str], k: int) -> int:
    for pos, nid in enumerate(ranked[:k], 1):
        if nid in gold_set:
            return pos
    return 0


def expand_canonical(bundle: IndexBundle, ranked: list[str], doc_nodes: dict, budget: int) -> str:
    """Small-to-big: топ-листья + родитель + соседние узлы того же родителя."""
    out, total = [], 0
    seen = set()
    for nid in ranked[:5]:
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        out.append(meta["text"])
        total += len(meta["text"])
        seen.add(nid)
        # соседи по документу: индекс узла +-1 (позиционные id фазы 2: doc#pNNNNN)
        if "#p" in nid:
            base, pnum = nid.rsplit("#p", 1)
            try:
                num = int(pnum)
            except ValueError:
                continue
            for j in (num - 1, num + 1):
                sib = doc_nodes.get(f"{base}#p{j:05d}")
                if sib is not None and sib not in seen and total + len(sib) <= budget:
                    out.append(sib)
                    seen.add(f"{base}#p{j:05d}")
                    total += len(sib)
        if total >= budget:
            break
    return "\n".join(out)


def flat_pack(bundle: IndexBundle, ranked: list[str], budget: int) -> str:
    out, total = [], 0
    for nid in ranked[:5]:
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        if total + len(meta["text"]) > budget:
            break
        out.append(meta["text"])
        total += len(meta["text"])
    return "\n".join(out)


def coverage(fragment: str, pack: str) -> float:
    fg, _ = _grams_tokens(fragment)
    pg, _ = _grams_tokens(pack)
    if not fg:
        return 0.0
    return len(fg & pg) / len(fg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-budget", type=int, default=14000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--holdout-docs", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    qa = load_qa()
    parser = PyMuPdfParserAdapter()
    parsed = {}
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            parsed[pdf.name] = parser.parse_file(pdf, corpus_root=DOCS_DIR)
        except Exception as e:  # noqa: BLE001
            print(f"  PARSE FAIL {pdf.name}: {e}")
    print(f"документов: {len(parsed)} ({time.time() - t0:.0f}с)")

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
    subset_qa = [r for r in qa if r["document"] in chosen]
    print(f"документов выбрано {len(chosen)} (узлов {budget}), QA: {len(subset_qa)}")

    cfg_flat = ConfigIndex(name="flat-800")
    cfg_canon = ConfigIndex(name="canonical-s2b")
    doc_node_texts = {}  # document_id -> {logical_node_id: text} для расширения
    for d in chosen:
        doc = parsed[d]
        cfg_flat.units.extend(flat_units(doc))
        units = canonical_units(doc)
        cfg_canon.units.extend(units)
        doc_node_texts[d] = {
            n.logical_node_id: n.text for n in doc.nodes if n.node_type != "root"
        }
    print(f"юниты: flat={len(cfg_flat.units)}, canonical={len(cfg_canon.units)}")

    cache_f = [_grams_tokens(u.text) for u in cfg_flat.units]
    cache_c = [_grams_tokens(u.text) for u in cfg_canon.units]
    gold = {}
    for i, r in enumerate(subset_qa):
        g1, frag1, s1 = localize_gold_best(cfg_flat.units, r["answer"], cache_f)
        g2, frag2, s2 = localize_gold_best(cfg_canon.units, r["answer"], cache_c)
        if g1 and g2 and frag1 and frag2:
            gold[i] = {"flat": g1, "canon": g2, "frag": frag1 if s1 >= s2 else frag2}
    print(f"gold в обеих конфигурациях: {len(gold)}/{len(subset_qa)}")

    embedder = SentenceTransformersProvider("intfloat/multilingual-e5-small", dim=384)
    results = {
        "meta": {"docs": chosen, "qa_subset": len(subset_qa), "qa_gold": len(gold),
                 "evidence_budget": EVIDENCE_BUDGET},
        "configs": {},
    }

    # --- Метрика A: ранжирование листьев; Метрика B: покрытие evidence-pack
    for cfg, key, doc_nodes in ((cfg_flat, "flat", None), (cfg_canon, "canon", doc_node_texts)):
        print(f"индексация {cfg.name}: {len(cfg.units)} юнитов...", flush=True)
        t = time.time()
        cfg.bundle = build_bundle(cfg.units, embedder)
        h5 = h10 = 0
        rr = 0.0
        cov_sum = 0.0
        for i, g in gold.items():
            ranked = rrf_ranked(cfg.bundle, embedder, subset_qa[i]["question"], k=args.k)
            pos = best_rank(ranked, g[key], k=args.k)
            h5 += 1 <= pos <= 5
            h10 += 1 <= pos <= args.k
            rr += (1.0 / pos) if pos else 0.0
            pack = (
                expand_canonical(cfg.bundle, ranked, doc_nodes, EVIDENCE_BUDGET)
                if key == "canon"
                else flat_pack(cfg.bundle, ranked, EVIDENCE_BUDGET)
            )
            cov_sum += coverage(g["frag"], pack) >= 0.6
        n = len(gold)
        results["configs"][cfg.name] = {
            "units": len(cfg.units),
            "recall@5": round(h5 / n, 4),
            "recall@10": round(h10 / n, 4),
            "mrr@10": round(rr / n, 4),
            "evidence_coverage@5": round(cov_sum / n, 4),
            "index_seconds": round(time.time() - t),
        }
        print(f"  {cfg.name}: R@5={results['configs'][cfg.name]['recall@5']} "
              f"R@10={results['configs'][cfg.name]['recall@10']} "
              f"MRR={results['configs'][cfg.name]['mrr@10']} "
              f"cov@5={results['configs'][cfg.name]['evidence_coverage@5']}")

    # --- Калибровка AnswerPolicy v2: identifier-grounding + порог
    holdout = sorted(chosen, key=lambda d: -qa_by_doc.get(d, 0))[: args.holdout_docs]
    indocs = [d for d in chosen if d not in holdout]
    print(f"held-out ловушки: {len(holdout)} документов...")
    hold_cfg = ConfigIndex(name="holdout")
    for d in indocs:
        hold_cfg.units.extend(canonical_units(parsed[d]))
    hold_cfg.bundle = build_bundle(hold_cfg.units, embedder)
    hold_ids = corpus_identifier_set({d: parsed[d] for d in indocs})
    full_ids = corpus_identifier_set({d: parsed[d] for d in chosen})

    from veles.policy.answer import _confidence

    trap_qa = [r for r in subset_qa if r["document"] in holdout]
    sweep = {}
    for thr in [0.5, 0.6, 0.7, 0.8]:
        # Вариант A: только скор (старая политика)
        far_score = 0
        for r in trap_qa:
            ev = hybrid_search(hold_cfg.bundle, embedder.embed_query(r["question"]), r["question"], top_k=8)
            far_score += _confidence(ev) >= thr
        # Вариант B: скор + identifier-grounding (miss -> abstain всегда)
        far_ident = 0
        for r in trap_qa:
            ids = extract_identifiers(r["question"])
            if ids and not all(any(i in doc_id for doc_id in hold_ids) for i in ids):
                far_ident += 0
                continue
            ev = hybrid_search(hold_cfg.bundle, embedder.embed_query(r["question"]), r["question"], top_k=8)
            far_ident += _confidence(ev) >= thr
        ans = 0
        ans_ident = 0
        for i, g in gold.items():
            q = subset_qa[i]["question"]
            ev = hybrid_search(cfg_canon.bundle, embedder.embed_query(q), q, top_k=8)
            ok_score = _confidence(ev) >= thr
            ans += ok_score
            ids = extract_identifiers(q)
            ok_ident = ok_score and (not ids or all(any(x in doc_id for doc_id in full_ids) for x in ids))
            ans_ident += ok_ident
        sweep[str(thr)] = {
            "far_score_only": round(far_score / max(len(trap_qa), 1), 4),
            "far_with_identifier": round(far_ident / max(len(trap_qa), 1), 4),
            "answer_rate_score_only": round(ans / len(gold), 4),
            "answer_rate_with_identifier": round(ans_ident / len(gold), 4),
        }
        print(f"  порог {thr}: FAR скор={sweep[str(thr)]['far_score_only']} "
              f"FAR+идент={sweep[str(thr)]['far_with_identifier']} "
              f"answer={sweep[str(thr)]['answer_rate_with_identifier']}")
    results["threshold_sweep"] = {
        "traps": len(trap_qa),
        "holdout_docs": holdout,
        "sweep": sweep,
    }

    out = Path(__file__).parent.parent / "data" / "bench" / "rrncb-v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"результаты: {out}; время {(time.time() - t0) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
