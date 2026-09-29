#!/usr/bin/env python
"""K14: квантификация abstain-гейта AnswerPolicy на старом (dup) и новом
(чистом) индексах. Решения decide() с продовыми дефолтами по 57 answerable
golden + 6 codex-boost-check. Read-only, load_index без save."""
import json
import sys
from pathlib import Path

REPO = Path("<server-home>/proj/veles-bench/veles")
sys.path.insert(0, str(REPO))
from veles.api.app import VelesPipeline  # noqa: F401 (имитация окружения фасада не нужна)
from veles.config import EMBED_DIM, EMBED_MODEL  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import load_index  # noqa: E402
from veles.policy.answer import decide  # noqa: E402
from veles.policy.identifiers import extract_codex_refs, extract_identifiers  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

SB = REPO / "evals" / "synth-bench"
golden = [json.loads(l) for l in open(SB / "golden.jsonl", encoding="utf-8")]
answerable = [e for e in golden if not e.get("abstain")]
cbc = [json.loads(l) for l in open(SB / "codex-boost-check-id.jsonl", encoding="utf-8")]

embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)

def measure(index_dir: Path) -> dict:
    bundle = load_index(index_dir)
    doc_titles = {m.get("title", "") for m in bundle.node_meta.values() if m.get("title")}
    out = {"golden_abstain": [], "cbc": []}
    for e in answerable:
        q = e["question"]
        ev = hybrid_search(bundle, embedder.embed_query(q), q, top_k=5)
        d = decide(ev, query_identifiers=extract_identifiers(q),
                   known_identifiers=bundle.identifiers,
                   codex_refs=extract_codex_refs(q), doc_titles=doc_titles)
        out["golden_abstain"].append(
            {"id": e["id"], "decision": d.decision, "conf": d.confidence})
    for e in cbc:
        q = e["question"]
        ev = hybrid_search(bundle, embedder.embed_query(q), q, top_k=5)
        d = decide(ev, query_identifiers=extract_identifiers(q),
                   known_identifiers=bundle.identifiers,
                   codex_refs=extract_codex_refs(q), doc_titles=doc_titles)
        out["cbc"].append(
            {"id": e["id"], "abstain_expected": bool(e.get("abstain")),
             "decision": d.decision, "conf": d.confidence})
    return out

results = {}
for name, idx in [
    ("new", REPO / "data" / "index"),
    ("old_dup", REPO / "data" / "index-backup-20260929-2137"),
]:
    print(f"=== {name}: {idx}", flush=True)
    r = measure(idx)
    results[name] = r
    g = r["golden_abstain"]
    abst = [x for x in g if x["decision"] == "abstain"]
    print(f"golden answerable {len(g)}: abstain {len(abst)} ({len(abst)/len(g):.1%})")
    for x in abst:
        print(f"   abstain {x['id']} conf={x['conf']}")
    c = r["cbc"]
    ok = sum(1 for x in c if (x["decision"] == "abstain") == x["abstain_expected"])
    print(f"cbc: верных решений гейта {ok}/{len(c)}; " +
          " ".join(f"{x['id']}:{x['decision']}({x['conf']})" for x in c))

json.dump(results, open("/tmp/k14/gate-old-new.json", "w"), ensure_ascii=False, indent=1)
print("артефакт: /tmp/k14/gate-old-new.json")
