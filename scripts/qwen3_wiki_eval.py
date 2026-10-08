"""Кросс-язычный замер эмбеддеров на вики-470 (протокол wiki-eval).

Гибридный поиск (dense + BM25, топ-5) по снапшоту индекса, ожидаемый
документ «owner/<page>.md», метрики top1 и page@5. Группы запросчика -
как у реального пользователя сервиса: {"all", "personal"} (wiki-узлы
корпуса несут группу "all"; замер с группами {"personal"} отфильтровывает
всю вики - урок 08.10).

Два режима:
  --embedder snapshot  - dense-векторы снапшота (e5), базлайн; запросы
                         кодируются тем же e5;
  --embedder qwen3     - dense-канал пересобран векторами Qwen3-Embedding
                         (SentenceTransformers, prompt_name="query" для
                         запросов - официальное использование), BM25 и
                         метаданные - из снапшота. Матрица кэшируется
                         в npy рядом с отчётом (кодирование 281к узлов -
                         десятки минут, повторные прогоны мгновенны).
Обрезка входа 512 токенов для обеих моделей (сопоставимый протокол).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", type=Path, default=Path("/home/dmazur/qwen3-eval/oh-eval-snap"))
    ap.add_argument("--qs", type=Path,
                    default=Path("/home/dmazur/qwen3-eval/oh-golden/wiki-questions.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--embedder", choices=["qwen3", "snapshot"], default="qwen3")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = args.out or Path(f"/home/dmazur/qwen3-eval/{args.embedder}-wiki-report.json")

    import numpy as np
    from sentence_transformers import SentenceTransformer

    from veles.indexing.indexer import load_index
    from veles.indexing.vector_store import NumpyFlatStore
    from veles.retrieval.retriever import hybrid_search

    bundle = load_index(args.snap)
    print(f"[snap] узлов в индексе: {len(bundle.node_meta)}", flush=True)

    # группы реального пользователя: публичные документы ("all") + личные
    groups = {"all", "personal"}

    t0 = time.time()
    if args.embedder == "qwen3":
        model = SentenceTransformer(args.model, device=args.device,
                                    model_kwargs={"torch_dtype": "float16"})
        model.max_seq_length = 512
        print(f"[model] {args.model} загружена за {time.time() - t0:.0f}с "
              f"(max_seq_length={model.max_seq_length})", flush=True)
        ids = list(bundle.node_meta)
        cache = out.parent / f"vecs-{args.embedder}.npy"
        if cache.exists():
            vecs = np.load(cache)
            print(f"[cache] матрица из {cache}", flush=True)
            assert vecs.shape[0] == len(ids), "кэш не совпадает с снапшотом"
        else:
            texts = []
            for nid in ids:
                m = bundle.node_meta[nid]
                prefix = " > ".join(m.get("heading_path") or [])
                texts.append(f"[{prefix}] {m['text']}" if prefix else m["text"])
            t1 = time.time()
            vecs = np.asarray(model.encode(texts, batch_size=args.batch,
                                           normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)
            np.save(cache, vecs)
            print(f"[encode] {len(texts)} узлов за {time.time() - t1:.0f}с", flush=True)
        store = NumpyFlatStore(dim=vecs.shape[1])
        store.add(ids, vecs)
        bundle.vector = store
    else:
        model = SentenceTransformer("intfloat/multilingual-e5-small", device=args.device)
        model.max_seq_length = 512
        print("[model] e5 из снапшота (dense-векторы уже в индексе)", flush=True)

    def qvec(text: str) -> np.ndarray:
        if args.embedder == "qwen3":
            return np.asarray(model.encode([text], prompt_name="query",
                                           normalize_embeddings=True)[0], dtype=np.float32)
        return np.asarray(model.encode([f"query: {text}"],
                                       normalize_embeddings=True)[0], dtype=np.float32)

    qs = [json.loads(l) for l in args.qs.read_text(encoding="utf-8").splitlines()
          if l.strip()]
    rows = []
    t2 = time.time()
    for i, q in enumerate(qs, 1):
        evs = hybrid_search(bundle, qvec(q["question"]), q["question"], top_k=5,
                            allowed_groups=groups)
        docs = [str(e.citations[0].document_id) for e in evs]
        exp = "owner/" + q["page"] + ".md"
        rows.append({
            "id": q["id"], "page": q["page"], "q": q["question"],
            "hit_top1": bool(docs) and docs[0] == exp,
            "hit5": exp in docs,
            "retrieved_docs": docs,
        })
        if i % 100 == 0:
            print(f"[eval] {i}/{len(qs)}", flush=True)
    n = len(rows)
    t1c = sum(r["hit_top1"] for r in rows)
    h5 = sum(r["hit5"] for r in rows)
    report = {
        "embedder": args.embedder,
        "model": args.model if args.embedder == "qwen3" else "intfloat/multilingual-e5-small (снапшот)",
        "protocol": f"hybrid_search top-5, группы {sorted(groups)}, снапшот {args.snap}",
        "nodes": len(bundle.node_meta),
        "queries": n,
        "seconds": {"model_load_and_encode": round(time.time() - t0, 1),
                    "queries": round(time.time() - t2, 1)},
        "summary": {"top1": round(t1c / n, 4), "page_at_5": round(h5 / n, 4),
                    "hits_top1": t1c, "hits_page5": h5},
        "rows": rows,
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ИТОГ[{args.embedder}]: n={n} top1={t1c}/{n} ({t1c / n:.1%}) "
          f"page@5={h5}/{n} ({h5 / n:.1%})", flush=True)


if __name__ == "__main__":
    main()
