#!/usr/bin/env python3
"""K12: RRNCB-выборка (50 вопросов) с expected-узлами для retrieval-bench.

Методика self-check из build_golden (K11): вопрос из rrncb_public_dataset.csv
считается покрытым, если нормализованный ответ имеет общий кусок >=40 знаков
с одним из top-5 фрагментов baseline-выдачи (LCS в духе bench/rrncb.py);
этот узел = expected. Выборка детерминирована (--seed), вопросы без попадания
пропускаются (доля покрытия публикуется рядом с метрикой - канон «баг-gold
только с нормировкой»).

Прогон метрик - retrieval_bench.py --golden rrncb50.jsonl (до/после калибровки,
одинаковый expected). Запуск (стенд):
  VELES_REPO=$HOME/proj/veles-bench/veles-k12 VELES_DATA=<прод data> \
  ~/veles-venv/bin/python rrncb_sample.py --csv ~/proj/veles-bench/rrncb/rrncb_public_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

VELES_REPO = Path(
    os.environ.get("VELES_REPO", str(Path.home() / "proj" / "veles-bench" / "veles"))
)
LCS_MIN_CHARS = 40


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    return " ".join(s.split())


def lcs_size(a: str, b: str) -> int:
    return SequenceMatcher(None, a, b, autojunk=True).find_longest_match(
        0, len(a), 0, len(b)
    ).size


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="K12 rrncb-50 sample builder")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=str(here / "rrncb50.jsonl"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-tries", type=int, default=200,
                    help="максимальный объём выборки при поиске покрытых вопросов")
    args = ap.parse_args()

    sys.path.insert(0, str(VELES_REPO))
    from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR
    from veles.indexing.embedder import SentenceTransformersProvider
    from veles.indexing.indexer import load_index
    from veles.retrieval.retriever import hybrid_search

    print(f"live: индекс {INDEX_DIR} (read-only), модель {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)
    print(f"live: {len(bundle.node_meta)} узлов", flush=True)

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"csv: {len(rows)} вопросов", flush=True)

    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    picked, tried, covered = [], 0, 0
    for i in order:
        if len(picked) >= args.n or tried >= args.max_tries:
            break
        tried += 1
        row = rows[i]
        vec = embedder.embed_query(row["question"])
        evidence = hybrid_search(bundle, vec, row["question"], top_k=args.top_k)
        na = norm_text(row["answer"])
        hit = None
        for ev in evidence:
            if lcs_size(na, norm_text(ev.text)) >= LCS_MIN_CHARS:
                hit = ev
                break
        if hit is None:
            continue
        covered += 1
        c = hit.citations[0]
        picked.append(
            {
                "id": f"rrncb50-{len(picked) + 1:03d}",
                "abstain": False,
                "question": row["question"],
                "expected_nodes": [hit.node_version_id],
                "source_doc": c.source_id,
                "provenance": "rrncb50",
                "csv_document": row.get("document", ""),
                "csv_row": i,
            }
        )
        if len(picked) % 10 == 0:
            print(f"  {len(picked)}/{args.n} (попыток {tried})", flush=True)

    meta = {
        "created": "2026-09-29",
        "seed": args.seed,
        "csv": args.csv,
        "csv_rows": len(rows),
        "tried": tried,
        "picked": len(picked),
        "coverage": round(covered / tried, 3) if tried else 0.0,
        "method": "LCS>=40 self-check на top-5 baseline-выдачи (методика build_golden K11)",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        for p in picked:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"выборка: {len(picked)} вопросов, покрытие {meta['coverage']} "
          f"({covered}/{tried}) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
