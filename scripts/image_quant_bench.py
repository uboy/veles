"""Гейт варианта int8: динамическое квантование кодировщика jina-clip-v2.

Квантует Linear-слои в int8 (torch.ao.quantization, динамически), меряет:
пиковую и установившуюся память процесса, задержку запроса, время кодирования
картинок, и качество на ТОЙ ЖЕ выборке, что Ф0 (Кормен + System Design,
20 русских запросов, топ-3/топ-1 по книге). Итог - JSON в stdout/файл.

Ключевая оговорка развертывания: naive-поток «загрузил fp32 - квантовал»
даёт ТРАНЗИЕНТНЫЙ пик полной модели; установившийся RSS показывает, что
остаётся, если веса int8 материализованы при загрузке (ONNX/кэш состояния).
Оба числа идут в отчёт.

Запуск: python scripts/image_quant_bench.py --images-dir <sanity>/images
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

QUERIES = [
    # первые 5 - Кормен, последние 5 - System Design (как в санити Ф0)
    "схема красно-чёрного дерева",
    "диаграмма работы алгоритма сортировки",
    "граф и его представление в памяти",
    "двоичное дерево поиска рисунок",
    "хеш-таблица с цепочками схема",
    "балансировщик нагрузки схема",
    "микросервисная архитектура диаграмма",
    "шардирование базы данных схема",
    "очередь сообщений архитектура",
    "схема работы CDN",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="каталог <sanity>/images с подпапками-книгами")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--model", default="jinaai/jina-clip-v2")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    import gc

    import torch
    from transformers import AutoModel

    t0 = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
    load_fp32_s = time.time() - t0

    t1 = time.time()

    def quant_plain(module) -> None:
        """int8 только для «чистых» nn.Linear: подклассы с параметризацией
        (ParametrizedLinear) динамическому квантованию не поддаются и
        остаются fp32 (их масса мала, на память почти не влияют)."""
        for name, child in module.named_children():
            if type(child) is torch.nn.Linear:
                setattr(module, name, torch.ao.quantization.quantize_dynamic(
                    child, {torch.nn.Linear}, dtype=torch.qint8))
            else:
                quant_plain(child)

    qmodel = model
    quant_plain(qmodel)
    quant_s = time.time() - t1
    del model
    gc.collect()

    # установившийся RSS (без транзиента fp32) и пик процесса
    with open("/proc/self/status", encoding="utf-8") as fh:
        cur_kb = next(int(l.split()[1]) for l in fh if l.startswith("VmRSS"))
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    t2 = time.time()
    n = 0
    for _ in range(3):
        for q in QUERIES[:5]:
            qmodel.encode_text([q])
            n += 1
    lat_ms = (time.time() - t2) / n * 1000

    # качество на выборке Ф0: картинки уже извлечены, раскладка images/<book>/
    books = sorted(p.name for p in args.images_dir.iterdir() if p.is_dir())
    files, labels = [], []
    for b in books:
        for f in sorted((args.images_dir / b).glob("*.png")):
            files.append(str(f))
            labels.append(b)
    t3 = time.time()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(files), args.batch):
            from PIL import Image

            imgs = [Image.open(p).convert("RGB") for p in files[i : i + args.batch]]
            vecs.append(np.asarray(qmodel.encode_image(imgs), dtype=np.float32))
    vecs = np.vstack(vecs)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    img_s = time.time() - t3

    qvecs = np.asarray(qmodel.encode_text(QUERIES), dtype=np.float32)
    qvecs /= np.linalg.norm(qvecs, axis=1, keepdims=True) + 1e-9
    hits3 = hits1 = total = 0
    for qi, qv in zip(range(len(QUERIES)), qvecs):
        book = "kormen" if qi < 5 else "systemdesign"
        sims = qv @ vecs.T
        top = np.argsort(-sims)[:3]
        top_books = [labels[int(i)] for i in top]
        hits3 += book in top_books
        hits1 += top_books[0] == book
        total += 1
    report = {
        "model": args.model,
        "quantization": "torch-dynamic-int8-linear",
        "load_fp32_seconds": round(load_fp32_s, 1),
        "quantize_seconds": round(quant_s, 1),
        "steady_rss_mb": round(cur_kb / 1024),
        "peak_rss_mb": round(peak_mb),
        "query_latency_ms": round(lat_ms),
        "image_encode_seconds": round(img_s, 1),
        "images": len(files),
        "queries": total,
        "hit_top3": hits3,
        "hit_top3_rate": round(hits3 / total, 3),
        "hit_top1": hits1,
        "hit_top1_rate": round(hits1 / total, 3),
    }
    print(json.dumps(report, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                            encoding="utf-8")


if __name__ == "__main__":
    main()
