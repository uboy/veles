#!/usr/bin/env python3
"""K11: заморозка retrieval для synth-bench (запуск НА стенд, venv python).

ОДИН прогон in-process retrieval против прод-индекса (read-only, НИКАКИХ save):
на каждый вопрос golden-набора собираются top_k фрагменты (те же hybrid_search +
top_k=5, что в фасаде) и кэшируются в JSON. Дальше все модели синтеза отвечают
на ОДНИХ И ТЕХ ЖЕ фрагментах - сравнение синтезаторов честное.

Запуск (стенд):
  VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
  ~/veles-venv/bin/python freeze_retrieval.py --golden golden.jsonl --out retrieval-cache.json

Решение по размеру кэша печатается в конце (коммит в git если < 1 МБ, иначе только
снапшот в evals/results/).
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
sys.path.insert(0, str(VELES_REPO))

from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import load_index  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_lib import load_jsonl  # noqa: E402

TOP_K = 5


def main() -> int:
    ap = argparse.ArgumentParser(description="K11 retrieval freeze (стенд)")
    ap.add_argument("--golden", default="golden.jsonl")
    ap.add_argument("--out", default="retrieval-cache.json")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    golden = load_jsonl(args.golden)
    print(f"golden: {len(golden)} вопросов; индекс: {INDEX_DIR} (read-only)", flush=True)
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)
    print(f"индекс: {len(bundle.node_meta)} узлов", flush=True)

    questions: dict[str, dict] = {}
    n_expected_hit = n_answerable = 0
    for i, entry in enumerate(golden, 1):
        vec = embedder.embed_query(entry["question"])
        evidence = hybrid_search(bundle, vec, entry["question"], top_k=args.top_k)
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
                }
            )
        questions[entry["id"]] = {"question": entry["question"], "fragments": frags}
        if not entry.get("abstain"):
            n_answerable += 1
            expected = set(entry.get("expected_nodes", []))
            got = {fr["node_version_id"] for fr in frags}
            if expected & got:
                n_expected_hit += 1
        if i % 20 == 0:
            print(f"  {i}/{len(golden)}", flush=True)

    payload = {
        "meta": {
            "created": "2026-09-29",
            "golden": str(args.golden),
            "index_dir": str(INDEX_DIR),
            "embed_model": EMBED_MODEL,
            "top_k": args.top_k,
            "note": "замороженный retrieval K11: in-process load_index против прод-данных, read-only",
        },
        "questions": questions,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(
        f"кэш: {out} ({size_kb:.0f} КБ); answerable {n_answerable}, "
        f"expected-node в top-{args.top_k}: {n_expected_hit}/{n_answerable}",
        flush=True,
    )
    print(
        "решение по кэшу: " + (
            "< 1 МБ - коммитится в git (evals/synth-bench/)" if size_kb < 1024
            else ">= 1 МБ - в git НЕ кладётся, снапшот в evals/results/"
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
