"""Кросс-язычный замер Qwen3-Embedding-0.6B на вики-470 (протокол wiki-eval).

Тот же протокол, что у e5-базлайна на bm1 (wiki-eval-bm1.py): снапшот индекса,
гибридный поиск (dense + BM25, топ-5), ожидаемый документ «owner/<page>.md»,
метрики top1 и page@5. Отличие: dense-канал пересобран векторами
Qwen3-Embedding-0.6B (SentenceTransformers, prompt_name="query" для запросов
- официальное использование из карточки модели), BM25 и метаданные - из
снапшота без изменений.

Запуск: python scripts/qwen3_wiki_eval.py [--snap ...] [--qs ...] [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, default=Path("/home/dmazur/qwen3-eval/oh-eval-snap"))
    ap.add_argument("--qs", type=Path,
                    default=Path("/home/dmazur/qwen3-eval/oh-golden/wiki-questions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("/home/dmazur/qwen3-eval/qwen3-wiki-report.json"))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit-nodes", type=int, default=None, help="пробный прогон")
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from veles.indexing.indexer import load_index
    from veles.indexing.vector_store import NumpyFlatStore
    from veles.retrieval.retriever import hybrid_search

    bundle = load_index(args.snap)
    print(f"[snap] узлов в индексе: {len(bundle.node_meta)}", flush=True)

    ids = list(bundle.node_meta)
    if args.limit_nodes:
        ids = ids[: args.limit_nodes]
    texts = []
    for nid in ids:
        m = bundle.node_meta[nid]
        prefix = " > ".join(m.get("heading_path") or [])
        texts.append(f"[{prefix}] {m['text']}" if prefix else m["text"])

    t0 = time.time()
    model = SentenceTransformer(args.model, device="cuda",
                               model_kwargs={"torch_dtype": "float16"})
    # Протокол e5: passages и запросы обрезаются на 512 токенов (у Qwen3
    # контекст 32k - без обрезки длинные узлы корпуса роняют GPU по памяти
    # и ломают сопоставимость с базлайном e5)
    model.max_seq_length = 512
    print(f"[model] {args.model} загружена за {time.time() - t0:.0f}с "
          f"(max_seq_length={model.max_seq_length})", flush=True)

    t1 = time.time()
    vecs = model.encode(texts, batch_size=args.batch,
                        normalize_embeddings=True, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    print(f"[encode] {len(texts)} узлов за {time.time() - t1:.0f}с", flush=True)

    store = NumpyFlatStore(dim=vecs.shape[1])
    store.add(ids, vecs)
    bundle.vector = store

    qs = [json.loads(l) for l in args.qs.read_text(encoding="utf-8").splitlines()
          if l.strip()]
    rows = []
    t2 = time.time()
    for i, q in enumerate(qs, 1):
        vec = model.encode([q["question"]], prompt_name="query",
                           normalize_embeddings=True)[0]
        evs = hybrid_search(bundle, vec, q["question"], top_k=5)
        docs = [str(e.citations[0].document_id) for e in evs]
        exp = "owner/" + q["page"] + ".md"
        rows.append({
            "id": q["id"], "page": q["page"], "q": q["question"],
            "hit_top1": bool(docs) and docs[0] == exp,
            "hit5": exp in docs,
            "retrieved_docs": docs,
        })
        if i % 50 == 0:
            print(f"[eval] {i}/{len(qs)}", flush=True)
    n = len(rows)
    t1c = sum(r["hit_top1"] for r in rows)
    h5 = sum(r["hit5"] for r in rows)
    report = {
        "model": args.model,
        "protocol": "hybrid_search top-5, снапшот " + str(args.snap),
        "nodes": len(ids),
        "queries": n,
        "seconds": {"model_load": round(time.time() - t0, 1),
                    "passage_encode": round(time.time() - t1, 1),
                    "queries": round(time.time() - t2, 1)},
        "summary": {"top1": round(t1c / n, 4), "page_at_5": round(h5 / n, 4),
                    "hits_top1": t1c, "hits_page5": h5},
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"ИТОГ: n={n} top1={t1c}/{n} ({t1c / n:.1%}) "
          f"page@5={h5}/{n} ({h5 / n:.1%})", flush=True)


if __name__ == "__main__":
    main()
