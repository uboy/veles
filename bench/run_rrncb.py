"""RRNCB v4 (Ф3): LCS-gold (ревью-2 P0 fix), фиксированный корпус, валидные метрики.

Методика (по эксперименту 27.09, полный корпус 65 док / 200 QA):
- PRIMARY gold = вербатим-LCS >=40 знаков: gold-сеты ~1-3 юнита, база случайного
  top-5 <=0.03% в обеих конфигурациях - сравнение валидно. Покрытие публикуется
  per-конфигурация; метрики считаются на локализуемом подмножестве КАЖДОЙ
  конфигурации И на пересечении.
- SECONDARY = bag-3gram gold с lift-нормировкой (ширина сетов и база публикуются
  как санкция против смещения).
- Метрика A: Recall@5/10, MRR@10. Метрика B: покрытие gold-фрагмента evidence-pack
  при бюджете 4000 знаков (расширение только для canonical-s2b: ±1 сосед через
  logical_node_id; docling-секции уже содержат контекст; бюджет проверяется ДО
  добавления).
- Policy: veles AnswerPolicy v2 (identifier any-found по числовому ядру), FAR-свип.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.rrncb import (  # noqa: E402
    DOCS_DIR,
    _grams_tokens,
    canonical_units,
    docling_structural_units,
    flat_units,
    lcs_gold,
    load_qa,
    norm_text,
    rare_tokens,
)
from veles.config import RRF_K  # noqa: E402
from veles.indexing.bm25 import Bm25Index  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import IndexBundle  # noqa: E402
from veles.ingest.pdf_text import PyMuPdfParserAdapter  # noqa: E402
from veles.policy.answer import decide  # noqa: E402
from veles.policy.identifiers import extract_identifiers  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

EVIDENCE_BUDGET = 4000
LCS_MIN_CHARS = 40
K = 10
DOCLING_ROOT = Path(__file__).resolve().parents[1] / "data" / "canonical-docling"


def build_bundle(units, embedder, identifiers: set[str]) -> IndexBundle:
    from veles.indexing.vector_store import make_store

    store = make_store(prefer="faiss")
    bm25 = Bm25Index()
    meta = {}
    batch = 256
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


def rrf_ranked(bundle, embedder, question: str, k: int = K) -> list[str]:
    qvec = embedder.embed_query(question)
    hits = bundle.vector.search(qvec, top_k=k * 4)
    lex = bundle.bm25.search(question, top_k=k * 4)
    rrf: dict[str, float] = {}
    for rank, (nid, _) in enumerate(hits, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (nid, _) in enumerate(lex, 1):
        rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (RRF_K + rank)
    return [nid for nid, _ in sorted(rrf.items(), key=lambda x: -x[1])]


def best_rank(ranked: list[str], gold_set: set[str], k: int = K) -> int:
    for pos, nid in enumerate(ranked[:k], 1):
        if nid in gold_set:
            return pos
    return 0


def evidence_pack(bundle: IndexBundle, ranked: list[str], mode: str,
                  doc_node_texts: dict, budget: int = EVIDENCE_BUDGET) -> str:
    """Сборка контекста с ЖЁСТКИМ бюджетом (проверка ДО добавления).
    mode: flat = окна как есть; canonical = листья + ±1 сосед (small-to-big);
    docling = секции как есть (уже содержат контекст)."""
    out: list[str] = []
    total = 0
    seen: set[str] = set()
    for nid in ranked[:5]:
        meta = bundle.node_meta.get(nid)
        if not meta:
            continue
        pieces = [meta["text"]]
        if mode == "canonical" and meta.get("logical"):
            base, pnum = meta["logical"].rsplit("#p", 1)
            try:
                num = int(pnum)
            except ValueError:
                num = None
            if num is not None:
                for j in (num - 1, num + 1):
                    sib = doc_node_texts.get(f"{base}#p{j:05d}")
                    if sib is not None and sib not in seen:
                        pieces.append(sib)
                        seen.add(f"{base}#p{j:05d}")
        for piece in pieces:
            if total + len(piece) > budget:
                continue  # бюджет до добавления (ревью-2 P1)
            out.append(piece)
            total += len(piece)
        if total >= budget:
            break
    return "\n".join(out)


def coverage(fragment: str, pack: str) -> float:
    fg = _grams_tokens(fragment)[0]
    pg = _grams_tokens(pack)[0]
    return len(fg & pg) / max(len(fg), 1) if fg else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="flat,canonical,docling")
    ap.add_argument("--holdout-docs", type=int, default=3)
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
            print(f"PARSE FAIL {pdf.name}: {e}")
    docs = sorted(parsed)  # фиксированный список: все 65, без greedy-бюджета
    print(f"документов: {len(docs)} ({time.time() - t0:.0f}с)", flush=True)

    docling_docs = {}
    if "docling" in want:
        from veles.canonical.store import iter_documents

        for doc in iter_documents(root=DOCLING_ROOT):
            # канон строился с корнем на уровень выше: нормализуем к имени файла
            doc.document_id = Path(doc.document_id).name
            docling_docs[doc.document_id] = doc
        missing = set(docs) - set(docling_docs)
        print(f"docling: {len(docling_docs)}, без парса: {len(missing)}")
        if missing:
            docs = [d for d in docs if d in docling_docs]
            print(f"корпус сужен до документов с docling-каноном: {len(docs)}")

    subset_qa = [r for r in qa if r["document"] in docs]
    print(f"QA: {len(subset_qa)}")

    cfgs: dict[str, dict] = {}
    doc_node_texts: dict[str, dict] = {}
    for d in docs:
        doc = parsed[d]
        if "flat" in want:
            cfgs.setdefault("flat-800", {"name": "flat-800", "units": [], "mode": "flat"})["units"].extend(
                flat_units(doc)
            )
        if "canonical" in want:
            cfgs.setdefault("canonical-s2b", {"name": "canonical-s2b", "units": [], "mode": "canonical"})["units"].extend(
                canonical_units(doc)
            )
        doc_node_texts[d] = {n.logical_node_id: n.text for n in doc.nodes if n.node_type != "root"}
        if "docling" in want and d in docling_docs:
            cfgs.setdefault("docling-structural", {"name": "docling-structural", "units": [], "mode": "docling"})["units"].extend(
                docling_structural_units(docling_docs[d])
            )
    for c in cfgs.values():
        print(f"юниты {c['name']}: {len(c['units'])}", flush=True)

    # gold: LCS per-config + bag-ширина (secondary, lift-санкция)
    norm_caches, tok_indexes = {}, {}
    for name, c in cfgs.items():
        norm_caches[name] = [norm_text(u.text) for u in c["units"]]
        idx: dict[str, set[int]] = {}
        for i, u in enumerate(c["units"]):
            for w in rare_tokens(u.text):
                idx.setdefault(w, set()).add(i)
        tok_indexes[name] = idx

    gold_lcs: dict[int, dict[str, set[str]]] = {}
    frag_by_q: dict[int, str] = {}
    bag_stats: dict[str, list[int]] = {name: [] for name in cfgs}
    for i, r in enumerate(subset_qa):
        per: dict[str, set[str]] = {}
        for name, c in cfgs.items():
            gs = lcs_gold(c["units"], r["answer"], norm_caches[name], tok_indexes[name])
            if gs:
                per[name] = gs
            # bag-ширина для санкции (порог как в Ф2)
            cand = set()
            for w in sorted(rare_tokens(r["answer"]), key=lambda w: len(tok_indexes[name].get(w, set())))[:6]:
                cand |= tok_indexes[name].get(w, set())
            fgsets = [
                len(_grams_tokens(f)[0] & _grams_tokens(c["units"][j].text)[0]) / max(len(_grams_tokens(f)[0]), 1)
                for f in [" ".join(norm_text(r["answer"]).split()[i2:i2 + 14]) for i2 in range(0, max(len(r["answer"].split()) - 14, 1), 6)]
                for j in cand
                if len(_grams_tokens(f)[0]) >= 10
            ]
            wide = sum(1 for j in cand if any(
                len(_grams_tokens(f)[0] & _grams_tokens(c["units"][j].text)[0]) / max(len(_grams_tokens(f)[0]), 1) >= 0.55
                for f in [" ".join(r["answer"].split()[i2:i2 + 14]) for i2 in range(0, max(len(r["answer"].split()) - 14, 1), 6)]
                if len(_grams_tokens(f)[0]) >= 10
            ))
            bag_stats[name].append(wide)
        if len(per) == len(cfgs):
            gold_lcs[i] = per
    intersection = len(gold_lcs)
    print(f"LCS gold: per-config локализовано и на пересечении: {intersection}/{len(subset_qa)}", flush=True)
    for name in cfgs:
        sizes = [len(g[name]) for g in gold_lcs.values()]
        per_cov = sum(1 for r_i, r in enumerate(subset_qa)
                      if lcs_gold(cfgs[name]["units"], r["answer"], norm_caches[name], tok_indexes[name]))
        bag_w = statistics.mean(bag_stats[name]) if bag_stats[name] else 0
        base = bag_w / len(cfgs[name]["units"]) * 5
        results_row = {
            "lcs_coverage": per_cov,
            "lcs_gold_mean": round(statistics.mean(sizes), 2) if sizes else 0,
            "bag_gold_mean": round(bag_w, 1),
            "bag_random_base_top5": round(base, 4),
        }
        cfgs[name]["stats"] = results_row
        print(f"  {name}: {results_row}", flush=True)

    embedder = SentenceTransformersProvider("intfloat/multilingual-e5-small", dim=384)
    known_ids = set()
    for d in docs:
        known_ids |= set(extract_identifiers(parsed[d].title))
        known_ids |= {i for n in parsed[d].nodes if n.node_type != "root" for i in extract_identifiers(n.text)}

    results = {
        "meta": {"docs": len(docs), "qa_subset": len(subset_qa), "lcs_intersection": intersection,
                 "lcs_min_chars": LCS_MIN_CHARS, "evidence_budget": EVIDENCE_BUDGET},
        "configs": {},
    }

    for name, c in cfgs.items():
        print(f"индексация {name}: {len(c['units'])} юнитов...", flush=True)
        t = time.time()
        c["bundle"] = build_bundle(c["units"], embedder, known_ids)
        idx_of = {u.uid: u for u in c["units"]}
        h5 = h10 = 0
        rr = 0.0
        cov_sum = 0.0
        n_own = 0
        for i, g in gold_lcs.items():
            if name not in g:
                continue
            n_own += 1
            ranked = rrf_ranked(c["bundle"], embedder, subset_qa[i]["question"])
            pos = best_rank(ranked, g[name])
            h5 += 1 <= pos <= 5
            h10 += 1 <= pos <= K
            rr += (1.0 / pos) if pos else 0.0
            # фрагмент для покрытия: лучший LCS-кусок ответа
            na = norm_text(subset_qa[i]["answer"])
            sm_frag = None
            for uid in g[name]:
                u = idx_of.get(uid)
                if not u:
                    continue
                sm = __import__("difflib").SequenceMatcher(None, na, norm_text(u.text), autojunk=True)
                m = sm.find_longest_match(0, len(na), 0, len(norm_text(u.text)))
                if sm_frag is None or m.size > len(sm_frag):
                    sm_frag = na[m.a : m.a + m.size]
            pack = evidence_pack(c["bundle"], ranked, c["mode"], doc_node_texts)
            cov_sum += coverage(sm_frag or na[:120], pack) >= 0.6
        n = max(n_own, 1)
        results["configs"][name] = {
            "units": len(c["units"]),
            "lcs_coverage": c["stats"]["lcs_coverage"],
            "recall@5_own": round(h5 / n, 4),
            "recall@10_own": round(h10 / n, 4),
            "mrr@10_own": round(rr / n, 4),
            "evidence_coverage@5_own": round(cov_sum / n, 4),
            "n_own": n_own,
            "index_seconds": round(time.time() - t),
        }
        print(f"  {name}: R@5={results['configs'][name]['recall@5_own']} "
              f"MRR={results['configs'][name]['mrr@10_own']} "
              f"cov@5={results['configs'][name]['evidence_coverage@5_own']} (n={n_own})", flush=True)

    # пересечение: одна и та же выборка для всех конфигураций
    inter_ids = sorted(gold_lcs.keys())
    if inter_ids and len(cfgs) > 1:
        print("пересечение...", flush=True)
        inter = {}
        for name, c in cfgs.items():
            h5 = rr = cov_sum = 0
            for i in inter_ids:
                ranked = rrf_ranked(c["bundle"], embedder, subset_qa[i]["question"])
                pos = best_rank(ranked, gold_lcs[i][name])
                h5 += 1 <= pos <= 5
                rr += (1.0 / pos) if pos else 0.0
                na = norm_text(subset_qa[i]["answer"])
                pack = evidence_pack(c["bundle"], ranked, c["mode"], doc_node_texts)
                cov_sum += coverage(na[:120], pack) >= 0.6
            n = len(inter_ids)
            inter[name] = {
                "recall@5": round(h5 / n, 4),
                "mrr@10": round(rr / n, 4),
                "evidence_coverage@5": round(cov_sum / n, 4),
                "n": n,
            }
        results["intersection"] = inter
        for name, m in inter.items():
            print(f"  [inter] {name}: R@5={m['recall@5']} MRR={m['mrr@10']} cov@5={m['evidence_coverage@5']} (n={m['n']})", flush=True)

    # FAR-свип через ядро policy
    holdout = sorted(subset_qa and {r["document"] for r in subset_qa})[:0]  # placeholder
    docs_by_qa: dict[str, int] = {}
    for r in subset_qa:
        docs_by_qa[r["document"]] = docs_by_qa.get(r["document"], 0) + 1
    holdout = sorted(docs_by_qa, key=lambda d: -docs_by_qa[d])[: args.holdout_docs]
    cand_name = "docling-structural" if "docling-structural" in cfgs else "canonical-s2b"
    hold_units: list = []
    for d in docs:
        if d in holdout:
            continue
        if cand_name == "docling-structural" and d in docling_docs:
            hold_units.extend(docling_structural_units(docling_docs[d]))
        else:
            hold_units.extend(canonical_units(parsed[d]))
    hold_ids = set()
    for d in docs:
        if d in holdout:
            continue
        hold_ids |= set(extract_identifiers(parsed[d].title))
        hold_ids |= {i for n in parsed[d].nodes if n.node_type != "root" for i in extract_identifiers(n.text)}
    print(f"held-out ловушки: {len(holdout)} документов; {len(hold_units)} юнитов", flush=True)
    hold_bundle = build_bundle(hold_units, embedder, hold_ids)
    trap_qa = [r for r in subset_qa if r["document"] in holdout]
    sweep = {}
    for thr in [0.5, 0.7, 0.9]:
        far = ident_far = 0
        for r in trap_qa:
            ev = hybrid_search(hold_bundle, embedder.embed_query(r["question"]), r["question"], top_k=8)
            qids = extract_identifiers(r["question"])
            far += decide(ev, threshold=thr).decision == "answer"
            ident_far += decide(ev, threshold=thr, query_identifiers=qids, known_identifiers=hold_ids).decision == "answer"
        sweep[str(thr)] = {
            "far_score_only": round(far / max(len(trap_qa), 1), 4),
            "far_with_identifier": round(ident_far / max(len(trap_qa), 1), 4),
        }
        print(f"  порог {thr}: FAR {sweep[str(thr)]['far_score_only']}/{sweep[str(thr)]['far_with_identifier']} (ловушек {len(trap_qa)})", flush=True)
    results["threshold_sweep"] = {"traps": len(trap_qa), "holdout_docs": holdout, "sweep": sweep}

    out = Path(__file__).parent.parent / "data" / "bench" / "rrncb-v4.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"результаты: {out}; время {(time.time() - t0) / 60:.1f} мин", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
