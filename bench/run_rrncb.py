"""RRNCB v3 (Ф3): flat vs pymupdf-canonical vs docling-structural, policy через ядро.

Конфигурации на ОДНОМ корпусе (одинаковый набор документов):
- flat-800: окна 800 знаков (базлайн Ф2)
- canonical-s2b: pymupdf-абзацы <=700 + heading-префикс + расширение (Ф2-проигравший)
- docling-structural: секции Docling целиком (заголовок+абзацы+таблицы, cap 1800)

Метрики: A - Recall@5/10, MRR@10 (gold-множество); B - покрытие gold-фрагмента
evidence-pack'ом при бюджете 4000 знаков. Policy: veles AnswerPolicy v2
(identifier-grounding 'any-found' + формула с бонусом до капа); FAR-свип на held-out.
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
    docling_structural_units,
    flat_units,
    load_qa,
    localize_gold_best,
)
from veles.canonical.store import iter_documents  # noqa: E402
from veles.config import RRF_K  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import IndexBundle  # noqa: E402
from veles.ingest.pdf_text import PyMuPdfParserAdapter  # noqa: E402
from veles.policy.answer import decide  # noqa: E402
from veles.policy.identifiers import extract_identifiers  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

EVIDENCE_BUDGET = 4000
DOCLING_ROOT = Path(__file__).resolve().parents[1] / "data" / "canonical-docling"


def build_bundle(units: list, embedder, identifiers: set[str], batch: int = 256) -> IndexBundle:
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
    return IndexBundle(vector=store, bm25=bm25, node_meta=meta, identifiers=set(identifiers))


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


def expand_neighbors(bundle: IndexBundle, ranked: list[str], doc_node_texts: dict, budget: int) -> str:
    out, total = [], 0
    seen = set()
    for nid in ranked[:5]:
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        out.append(meta["text"])
        total += len(meta["text"])
        seen.add(nid)
        if "#p" in nid:
            base, pnum = nid.rsplit("#p", 1)
            try:
                num = int(pnum)
            except ValueError:
                continue
            for j in (num - 1, num + 1):
                sib = doc_node_texts.get(f"{base}#p{j:05d}")
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
    return len(fg & pg) / max(len(fg), 1) if fg else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-budget", type=int, default=36000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--holdout-docs", type=int, default=3)
    ap.add_argument("--configs", default="flat,canonical,docling")
    args = ap.parse_args()
    want = set(args.configs.split(","))

    t0 = time.time()
    qa = load_qa()
    parser = PyMuPdfParserAdapter()
    parsed = {}
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            parsed[pdf.name] = parser.parse_file(pdf, corpus_root=DOCS_DIR)
        except Exception as e:  # noqa: BLE001
            print(f"  PARSE FAIL {pdf.name}: {e}")
    print(f"pymupdf документов: {len(parsed)} ({time.time() - t0:.0f}с)", flush=True)

    docling_docs = {}
    if "docling" in want:
        for doc in iter_documents(root=DOCLING_ROOT):
            docling_docs[doc.document_id] = doc
        print(f"docling документов: {len(docling_docs)}")

    # общий набор документов: есть у всех запрошенных конфигураций
    qa_by_doc: dict[str, int] = {}
    for r in qa:
        if r["document"] in parsed and (r["document"] in docling_docs or "docling" not in want):
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
    print(f"документов {len(chosen)} (узлов {budget}), QA: {len(subset_qa)}")

    cfgs: dict[str, ConfigIndex] = {}
    doc_node_texts = {}
    for d in chosen:
        doc = parsed[d]
        if "flat" in want:
            cfgs.setdefault("flat-800", ConfigIndex(name="flat-800")).units.extend(flat_units(doc))
        if "canonical" in want:
            cfgs.setdefault("canonical-s2b", ConfigIndex(name="canonical-s2b")).units.extend(
                canonical_units(doc)
            )
        doc_node_texts[d] = {n.logical_node_id: n.text for n in doc.nodes if n.node_type != "root"}
        if "docling" in want and d in docling_docs:
            cfgs.setdefault("docling-structural", ConfigIndex(name="docling-structural")).units.extend(
                docling_structural_units(docling_docs[d])
            )
    for c in cfgs.values():
        print(f"юниты {c.name}: {len(c.units)}", flush=True)

    # gold-локализация на каждой конфигурации
    caches = {
        c.name: [_grams_tokens(u.text) for u in c.units] for c in cfgs.values()
    }
    keys = {"flat-800": "flat", "canonical-s2b": "canon", "docling-structural": "docl"}
    gold = {}
    for i, r in enumerate(subset_qa):
        per = {}
        frags = {}
        for name, cfg in cfgs.items():
            gs, frag, sc = localize_gold_best(cfg.units, r["answer"], caches[name])
            per[name] = gs
            frags[name] = (frag, sc)
        if all(per.values()) and all(frags[n][0] for n in cfgs):
            gold[i] = {"sets": per, "frag": max(frags.values(), key=lambda x: x[1])[0]}
    print(f"gold во всех конфигурациях: {len(gold)}/{len(subset_qa)}", flush=True)

    embedder = SentenceTransformersProvider("intfloat/multilingual-e5-small", dim=384)
    full_ids = corpus_identifier_set({d: parsed[d] for d in chosen})
    results = {
        "meta": {"docs": chosen, "qa_subset": len(subset_qa), "qa_gold": len(gold),
                 "evidence_budget": EVIDENCE_BUDGET},
        "configs": {},
    }

    for name, cfg in cfgs.items():
        print(f"индексация {name}: {len(cfg.units)} юнитов...", flush=True)
        t = time.time()
        key = keys[name]
        cfg.bundle = build_bundle(cfg.units, embedder, full_ids)
        h5 = h10 = 0
        rr = 0.0
        cov_sum = 0.0
        for i, g in gold.items():
            ranked = rrf_ranked(cfg.bundle, embedder, subset_qa[i]["question"], k=args.k)
            pos = best_rank(ranked, g["sets"][key], k=args.k)
            h5 += 1 <= pos <= 5
            h10 += 1 <= pos <= args.k
            rr += (1.0 / pos) if pos else 0.0
            if name == "flat-800":
                pack = flat_pack(cfg.bundle, ranked, EVIDENCE_BUDGET)
            else:
                pack = expand_neighbors(cfg.bundle, ranked, doc_node_texts, EVIDENCE_BUDGET)
            cov_sum += coverage(g["frag"], pack) >= 0.6
        n = len(gold)
        results["configs"][name] = {
            "units": len(cfg.units),
            "recall@5": round(h5 / n, 4),
            "recall@10": round(h10 / n, 4),
            "mrr@10": round(rr / n, 4),
            "evidence_coverage@5": round(cov_sum / n, 4),
            "index_seconds": round(time.time() - t),
        }
        print(f"  {name}: R@5={results['configs'][name]['recall@5']} "
              f"R@10={results['configs'][name]['recall@10']} "
              f"MRR={results['configs'][name]['mrr@10']} "
              f"cov@5={results['configs'][name]['evidence_coverage@5']}", flush=True)

    # калибровка AnswerPolicy v2 (identifier 'any-found' + формула до капа)
    holdout = sorted(chosen, key=lambda d: -qa_by_doc.get(d, 0))[: args.holdout_docs]
    cand_name = "docling-structural" if "docling-structural" in cfgs else "canonical-s2b"
    hold_cfg = ConfigIndex(name="holdout")
    for d in chosen:
        if d in holdout:
            continue
        if d in docling_docs and cand_name == "docling-structural":
            hold_cfg.units.extend(docling_structural_units(docling_docs[d]))
        else:
            hold_cfg.units.extend(canonical_units(parsed[d]))
    hold_ids = corpus_identifier_set({d: parsed[d] for d in chosen if d not in holdout})
    print(f"held-out ловушки: {len(holdout)} документов; индекс без них: {len(hold_cfg.units)} юнитов", flush=True)
    hold_cfg.bundle = build_bundle(hold_cfg.units, embedder, hold_ids)

    trap_qa = [r for r in subset_qa if r["document"] in holdout]
    sweep = {}
    for thr in [0.5, 0.6, 0.7, 0.8]:
        far_score = far_ident = 0
        for r in trap_qa:
            ev = hybrid_search(hold_cfg.bundle, embedder.embed_query(r["question"]), r["question"], top_k=8)
            qids = extract_identifiers(r["question"])
            far_score += decide(ev, threshold=thr).decision == "answer"
            far_ident += decide(
                ev, threshold=thr, query_identifiers=qids, known_identifiers=hold_ids
            ).decision == "answer"
        ans = ans_ident = 0
        for i, g in gold.items():
            q = subset_qa[i]["question"]
            ev = hybrid_search(cfgs[cand_name].bundle, embedder.embed_query(q), q, top_k=8)
            qids = extract_identifiers(q)
            ans += decide(ev, threshold=thr).decision == "answer"
            ans_ident += decide(
                ev, threshold=thr, query_identifiers=qids, known_identifiers=full_ids
            ).decision == "answer"
        sweep[str(thr)] = {
            "far_score_only": round(far_score / max(len(trap_qa), 1), 4),
            "far_with_identifier": round(far_ident / max(len(trap_qa), 1), 4),
            "answer_rate_score_only": round(ans / len(gold), 4),
            "answer_rate_with_identifier": round(ans_ident / len(gold), 4),
        }
        print(f"  порог {thr}: FAR={sweep[str(thr)]['far_score_only']}/"
              f"{sweep[str(thr)]['far_with_identifier']} "
              f"answer={sweep[str(thr)]['answer_rate_score_only']}/"
              f"{sweep[str(thr)]['answer_rate_with_identifier']}", flush=True)
    results["threshold_sweep"] = {"traps": len(trap_qa), "holdout_docs": holdout, "sweep": sweep}

    out = Path(__file__).parent.parent / "data" / "bench" / "rrncb-v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"результаты: {out}; время {(time.time() - t0) / 60:.1f} мин", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
