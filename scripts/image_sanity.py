"""Ф0 плана поиска по изображениям: санити мультимодальной модели на книгах корпуса.

Извлекает растровые картинки из двух PDF разных тем, кодирует их вместе с
русскими запросами и меряет, попадают ли картинки правильной книги в топ-3.
Модель грузится в переменной окружения VELES_SANITY_MODEL (по умолчанию
jinaai/jina-clip-v2); устройство --device (cuda/cpu). Отчёт: JSON + таблица
в stdout. Гейт плана: книги запросов в топ-3 не хуже 80% (>=16 из 20).

Запуск:
  python scripts/image_sanity.py --pdf КНИГА1 --name1 ИМЯ1 --pdf КНИГА2 \
      --name2 ИМЯ2 --out-dir evals/sanity-run --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_MODEL = "jinaai/jina-clip-v2"

# По 10 русских запросов на книгу: словарь имён книг -> список запросов.
# Темы запросов покрывают типовое содержание, а не точные подписи рисунков -
# санити проверяет связку «тема вопроса -> тематика картинки».
DEFAULT_QUERIES: dict[str, list[str]] = {
    "kormen": [
        "схема красно-чёрного дерева",
        "диаграмма работы алгоритма сортировки",
        "граф и его представление в памяти",
        "двоичное дерево поиска рисунок",
        "хеш-таблица с цепочками схема",
        "структура данных двоичная куча",
        "пример разбиения быстрой сортировки",
        "B-дерево схема",
        "таблица матриц динамического программирования",
        "дерево рекурсивных вызовов схема",
    ],
    "pmbok": [
        "процессы управления проектом схема",
        "жизненный цикл проекта диаграмма",
        "группы процессов управления проектом",
        "заинтересованные стороны проекта схема",
        "управление рисками проекта диаграмма",
        "организационная структура проекта схема",
        "управление стоимостью проекта процесс",
        "взаимосвязи знаний управления проектом",
        "управление качеством схема",
        "устав проекта и его окружение схема",
    ],
    "systemdesign": [
        "балансировщик нагрузки схема",
        "архитектура кэша схема",
        "шардирование базы данных схема",
        "схема работы CDN",
        "очередь сообщений архитектура",
        "микросервисная архитектура схема",
        "репликация базы данных схема",
        "ограничитель частоты запросов схема",
        "хранение данных архитектура",
        "согласованность данных диаграмма",
    ],
}


def extract_images(
    pdf_path: Path, book: str, out_dir: Path, max_images: int, min_dim: int, sample_pages: int
) -> list[dict]:
    """Растровые картинки из PDF: фильтр мелких иконок, дедупликация по хэшу."""
    import fitz

    dest = out_dir / "images" / book
    dest.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    step = max(1, doc.page_count // sample_pages)
    seen: set[str] = set()
    out: list[dict] = []
    for page_no in range(0, doc.page_count, step):
        if len(out) >= max_images:
            break
        for img in doc[page_no].get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except (RuntimeError, ValueError):
                continue
            if pix.colorspace is None:
                continue
            # PNG пишет только gray/RGB: CMYK, ICC- и индексированные
            # цветности конвертируются безусловно (n-подсчёт их не различает)
            pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width < min_dim or pix.height < min_dim:
                continue
            data = pix.tobytes("png")
            digest = hashlib.md5(data).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            name = f"p{page_no + 1:04d}_{xref}.png"
            (dest / name).write_bytes(data)
            out.append(
                {"book": book, "page": page_no + 1, "file": str(dest / name),
                 "width": pix.width, "height": pix.height}
            )
            if len(out) >= max_images:
                break
    doc.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", action="append", required=True, help="путь к PDF (повторяемо)")
    ap.add_argument("--name", action="append", required=True, help="ключ книги для запросов (повторяемо)")
    ap.add_argument("--out-dir", type=Path, default=Path("evals/sanity-run"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-per-book", type=int, default=15)
    ap.add_argument("--min-dim", type=int, default=200)
    ap.add_argument("--sample-pages", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    if len(args.pdf) != len(args.name) or len(args.pdf) != 2:
        ap.error("нужны ровно две книги: --pdf и --name попарно")

    import os

    import torch

    from transformers import AutoModel

    model_name = os.environ.get("VELES_SANITY_MODEL", args.model)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    books: dict[str, list[dict]] = {}
    for pdf, name in zip(args.pdf, args.name):
        books[name] = extract_images(
            Path(pdf), name, out_dir, args.max_per_book, args.min_dim, args.sample_pages
        )
        print(f"[extract] {name}: {len(books[name])} картинок")

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(args.device).eval()
    load_s = time.time() - t0
    print(f"[model] {model_name} на {args.device}, загрузка {load_s:.1f}с")

    all_images = [im for ims in books.values() for im in ims]
    t1 = time.time()
    img_vecs = []
    with torch.no_grad():
        for i in range(0, len(all_images), args.batch):
            chunk = [im["file"] for im in all_images[i : i + args.batch]]
            vecs = model.encode_image(chunk)
            img_vecs.append(np.asarray(vecs, dtype=np.float32))
    img_vecs = np.vstack(img_vecs)
    img_vecs /= np.linalg.norm(img_vecs, axis=1, keepdims=True) + 1e-9
    img_s = time.time() - t1
    print(f"[encode] {len(all_images)} картинок за {img_s:.1f}с")

    results, hits3, hits1 = [], 0, 0
    t2 = time.time()
    for book, queries in DEFAULT_QUERIES.items():
        if book not in books:
            continue
        vecs = np.asarray(model.encode_text(queries), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        sims = vecs @ img_vecs.T
        for qi, (query, row) in enumerate(zip(queries, sims)):
            top = np.argsort(-row)[:3]
            top_books = [all_images[int(i)]["book"] for i in top]
            hit3 = book in top_books
            hit1 = top_books[0] == book
            hits3 += hit3
            hits1 += hit1
            results.append(
                {"book": book, "query": query,
                 "top3": [{"book": b, "page": all_images[int(i)]["page"],
                           "sim": float(row[int(i)])} for i, b in zip(top, top_books)],
                 "hit_top3": hit3, "hit_top1": hit1}
            )
            mark = "OK " if hit3 else "MISS"
            print(f"[{mark}] {book} | {query} -> {top_books}")
    query_s = time.time() - t2

    total = len(results)
    summary = {
        "model": model_name,
        "device": args.device,
        "images_per_book": {k: len(v) for k, v in books.items()},
        "queries": total,
        "hit_top3": hits3,
        "hit_top3_rate": round(hits3 / total, 3) if total else 0.0,
        "hit_top1": hits1,
        "hit_top1_rate": round(hits1 / total, 3) if total else 0.0,
        "gate_top3_080": bool(total and hits3 / total >= 0.8),
        "seconds": {"model_load": round(load_s, 1), "image_encode": round(img_s, 1),
                    "queries": round(query_s, 2)},
    }
    (out_dir / "sanity-summary.json").write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print("[gate]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
