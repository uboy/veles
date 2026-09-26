"""Прогон мини-QA среза Ф1. Запуск: python bench/run_miniqa.py [файл.json]"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import load_index  # noqa: E402
from veles.indexing.vector_store import make_store  # noqa: E402
from veles.policy.answer import decide  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402


def main() -> int:
    qa_path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "miniqa-vyshka-module1-v1.json")
    items = json.loads(qa_path.read_text(encoding="utf-8"))["items"]

    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR, make_store(prefer="faiss"))

    ok_top1 = 0
    ok_top3 = 0
    ok_decision = 0
    for item in items:
        qvec = embedder.embed_query(item["question"])
        evidence = hybrid_search(bundle, qvec, item["question"], top_k=8)
        d = decide(evidence)
        top = evidence[0] if evidence else None
        hint = (item.get("expect_doc_hint") or "").lower()

        def doc_hit(ev) -> bool:
            return bool(hint) and any(
                hint in c.document_id.lower() for c in ev.citations
            )

        decision_ok = d.decision == item["expect_decision"] and (
            item["expect_decision"] == "abstain" or bool(evidence)
        )
        hit1 = bool(top) and decision_ok and (item["expect_decision"] == "abstain" or doc_hit(top))
        hit3 = decision_ok and (
            item["expect_decision"] == "abstain" or any(doc_hit(e) for e in evidence[:3])
        )
        ok_top1 += hit1
        ok_top3 += hit3
        ok_decision += decision_ok
        cite = top.citations[0].human() if top else "-"
        status = "OK " if hit3 else "FAIL"
        print(f"[{status}] {item['id']} {item['question'][:70]}")
        print(
            f"       ожидалось={item['expect_decision']}"
            f"{'/' + item['expect_doc_hint'] if item.get('expect_doc_hint') else ''}"
            f" | получено={d.decision} conf={d.confidence} | top: {cite}"
        )
    n = len(items)
    print(f"\nитог: decision {ok_decision}/{n}; top1 {ok_top1}/{n}; top3 {ok_top3}/{n}")
    return 0 if ok_top3 == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
