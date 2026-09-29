"""Оверлей инкрементальных загрузок (K4): дельта индекса поверх полного снапшота.

Зачем: пользовательская загрузка документа (POST /v1/documents) обязана стать
видимой поиску сразу после джоба, без рестарта фасада и без полной пересборки
(27 мин на 158k узлов). Полный rebuild не используется: новые узлы добавляются
в живой бандл (вектор + BM25 + мета), а для выживания после рестарта та же
дельта дописывается в overlay-файл рядом с индексом; load_index применяет его
после загрузки основного индекса.

Формат: один файл <INDEX_DIR>/overlay.npz - атомарная запись (tmp + replace)
держит JSON-мету и матрицу векторов консистентными:
- meta: json {embed_model, nodes, patches, bm25, updated_at}
  - nodes: {node_version_id: node_meta} - новые узлы загруженных документов
    (финальное состояние меты на момент записи);
  - patches: {node_version_id: {status: superseded}} - патчи статусов узлов
    ОСНОВНОГО индекса (старая ревизия при повторной загрузке);
  - bm25: [[node_version_id, search_text], ...] - в порядке применения
    (Bm25Index.add пересчитывает df/avgdl детерминированно);
- ids + matrix: нормализованные векторы новых узлов (те же, что ушли в
  живой бандл), порядок совпадает с keys(nodes) на момент записи.

node_version_id существующих узлов не меняются никогда - патчится только поле
status в мете (то же, что дал бы полный rebuild: у архивной ревизии узлы те же,
K3). Полный build_index+save стирает оверлей: снапшот снят с канона, в котором
все загрузки уже лежат, - дельта устаревает.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import EMBED_MODEL

OVERLAY_NAME = "overlay.npz"

logger = logging.getLogger("veles.indexing.overlay")


def overlay_path(index_dir: Path) -> Path:
    return Path(index_dir) / OVERLAY_NAME


def read_overlay(index_dir: Path) -> dict | None:
    """Прочитать оверлей как словарь; нет файла или битый файл -> None."""
    path = overlay_path(index_dir)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            meta = json.loads(npz["meta"].tobytes().decode("utf-8"))
            return {
                "meta": meta,
                "ids": [str(x) for x in npz["ids"].tolist()],
                "matrix": np.asarray(npz["matrix"], dtype=np.float32),
            }
    except Exception as e:  # noqa: BLE001 - битый оверлей не должен ронять сервис
        logger.warning("оверлей %s не читается, пропущен: %s", path, e)
        return None


def apply_overlay(bundle, index_dir: Path) -> None:
    """Подмешать оверлей к загруженному бандлу (load_index, после основного индекса).

    Любая проблема (нет файла, битый, другая модель эмбеддингов, расхождение
    размеров) - предупреждение и работа без оверлея: основной индекс остаётся
    рабочим, загрузки пользователя доедут следующей загрузкой/пересборкой.
    """
    data = read_overlay(index_dir)
    if data is None:
        return
    meta = data["meta"]
    if meta.get("embed_model") != EMBED_MODEL:
        logger.warning(
            "оверлей собран с моделью '%s', текущая '%s' - пропущен "
            "(пересоберите индекс: python -m veles index)",
            meta.get("embed_model"), EMBED_MODEL,
        )
        return
    ids, matrix = data["ids"], data["matrix"]
    bm25 = meta.get("bm25") or []
    if len(ids) != len(matrix) or len(bm25) != len(ids):
        logger.warning(
            "оверлей неконсистентен (векторов=%d, bm25=%d, меты=%d) - пропущен",
            len(ids), len(bm25), len(meta.get("nodes") or {}),
        )
        return
    if not ids:
        return
    bundle.node_meta.update(meta.get("nodes") or {})
    for nid, patch in (meta.get("patches") or {}).items():
        if nid in bundle.node_meta:
            bundle.node_meta[nid].update(patch)
    # порядок вставок как в upsert_document: мета -> векторы -> BM25
    bundle.vector.add(ids, matrix)
    if bm25:
        bundle.bm25.add([nid for nid, _ in bm25], [text for _, text in bm25])


def append_overlay(index_dir: Path, new_nodes: dict, patches: dict,
                   ids: list[str], matrix: np.ndarray,
                   bm25_pairs: list[tuple[str, str]]) -> None:
    """Дописать дельту одной загрузки к оверлею и атомарно перезаписать файл.

    Накопительный файл переписывается целиком (объём = только пользовательские
    загрузки, не весь индекс), tmp + replace - читатель всегда видит цельную
    версию либо старую, либо новую.
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    old = read_overlay(index_dir)
    old_meta = (old or {}).get("meta") or {}

    nodes = dict(old_meta.get("nodes") or {})
    all_patches = dict(old_meta.get("patches") or {})
    # патч узла, живущего в самом оверлее, сворачивается в его мету
    for nid in list(all_patches):
        if nid in nodes:
            nodes[nid].update(all_patches.pop(nid))
    nodes.update(new_nodes)
    for nid, patch in patches.items():
        if nid in nodes:
            nodes[nid].update(patch)
        else:
            all_patches[nid] = patch

    all_ids = (old or {}).get("ids") or []
    all_matrix = (old or {}).get("matrix")
    if all_ids and all_matrix is not None and len(all_ids) == len(all_matrix):
        matrix = np.vstack([all_matrix, matrix.astype(np.float32)])
        ids = all_ids + list(ids)
        bm25_all = (old_meta.get("bm25") or []) + [list(p) for p in bm25_pairs]
    else:
        # старого оверлея не было/бит - начинаем с этой дельты (предупреждение
        # уже отдал read_overlay, если файл был)
        matrix = np.asarray(matrix, dtype=np.float32)
        bm25_all = [list(p) for p in bm25_pairs]

    meta = {
        "embed_model": EMBED_MODEL,
        "nodes": nodes,
        "patches": all_patches,
        "bm25": bm25_all,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tmp = index_dir / (OVERLAY_NAME + f".tmp{os.getpid()}")
    np.savez(
        tmp,
        meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
        # строковый dtype (не object) - savez без pickle, чтение c allow_pickle=False
        ids=np.array(ids, dtype=str) if ids else np.empty(0, dtype=str),
        matrix=matrix.astype(np.float32),
    )
    tmp.replace(overlay_path(index_dir))
