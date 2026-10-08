"""Сборка индекса изображений из готового текстового индекса.

Читает node_meta текстового индекса (VELES_DATA/index/meta.json), берёт узлы
с активами, кодирует файлы активов мультимодальной моделью и пишет каталог
индекса (records.jsonl + векторное хранилище + image-meta.json). Сборка
тяжёлая - выполняется на машине с GPU; готовый каталог переносится на хост
сервиса и включается переменной VELES_IMAGE_INDEX_DIR.

Запуск:
  VELES_DATA=/data python scripts/build_image_index.py --device cuda
Сводка печатается в stdout и пишется рядом с индексом (build-report.json).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None, help="VELES_DATA (index/ + canonical/)")
    ap.add_argument("--out-dir", type=Path, default=None, help="куда писать индекс картинок")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=None, help="по умолчанию из veles.config.IMAGE_MODEL")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-per-doc", type=int, default=None, help="пробная сборка")
    args = ap.parse_args()

    from veles.config import CANONICAL_DIR, IMAGE_DIM, IMAGE_MODEL, INDEX_DIR, VELES_DATA

    data_dir = args.data_dir or VELES_DATA
    out_dir = args.out_dir or (Path(data_dir) / "image-index")
    meta_path = Path(data_dir) / "index" / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"нет текстового индекса: {meta_path}")
    node_meta = json.loads(meta_path.read_text(encoding="utf-8"))["node_meta"]

    from veles.indexing.image_index import JinaClipEmbedder, build_image_index

    embedder = JinaClipEmbedder(
        args.model or IMAGE_MODEL, device=args.device, dim=args.dim or IMAGE_DIM
    )

    t0 = time.time()
    # Канонические каталоги: и VELES_DATA/canonical, и каталоги рядом с data-dir
    # (сборка может идти на машине с другим набором VELES_DATA)
    canonical_dirs = [Path(data_dir) / "canonical"]
    if CANONICAL_DIR not in canonical_dirs and CANONICAL_DIR.exists():
        canonical_dirs.append(CANONICAL_DIR)

    report = build_image_index(
        node_meta,
        canonical_dirs,
        embedder,
        out_dir,
        batch=args.batch,
        limit_per_doc=args.limit_per_doc,
    )
    report["seconds"] = round(time.time() - t0, 1)
    (Path(report["out_dir"]) / "build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "skipped_detail"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
