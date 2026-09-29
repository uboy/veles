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
    try:
        _apply_checked(bundle, data)
    except Exception as e:  # noqa: BLE001 - оверлей не должен ронять сервис
        logger.warning("оверлей %s не применён: %s", overlay_path(index_dir), e)


def _apply_checked(bundle, data: dict) -> None:
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
    bundle.node_meta.update(meta.get("nodes") or {})
    for nid, patch in (meta.get("patches") or {}).items():
        if nid in bundle.node_meta:
            bundle.node_meta[nid].update(patch)
    if not ids:
        # патч-only оверлей (K10 tombstone удаления, дельта документа без
        # индексируемых узлов): мета и патчи применены выше, векторов нет
        return
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
    # K10: узел переиздан живым (повторная загрузка после удаления) - его старый
    # патч (tombstone status=deleted) недействителен, иначе рестарт применил бы
    # патч поверх свежей меты и переизданный документ навсегда пропал из поиска.
    # Патчи superseded не задеваются: они целятся в id ПРЕДЫДУЩЕЙ ревизии, у новых
    # узлов id другие (пересчитываются от новой ревизии, K3)
    for nid in new_nodes:
        all_patches.pop(nid, None)
    for nid, patch in patches.items():
        if nid in nodes:
            nodes[nid].update(patch)
        else:
            all_patches[nid] = patch

    old_ids = (old or {}).get("ids") or []
    old_matrix = (old or {}).get("matrix")
    old_consistent = bool(old_ids) and old_matrix is not None and len(old_ids) == len(old_matrix)
    old_bm25 = old_meta.get("bm25") or []
    if len(matrix) == 0:
        # дельта без индексируемых узлов (например документ из одного корня):
        # мета/патчи выше всё равно записываются, векторы не меняются
        ids = list(old_ids) if old_consistent else []
        matrix = old_matrix if old_consistent else np.zeros((0, 1), dtype=np.float32)
        bm25_all = old_bm25 if old_consistent else []
    elif old_consistent:
        ids = old_ids + list(ids)
        matrix = np.vstack([old_matrix, matrix.astype(np.float32)])
        bm25_all = old_bm25 + [list(p) for p in bm25_pairs]
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
    # tmp-имя обязано кончаться на .npz: np.savez иначе дописывает расширение
    tmp = index_dir / f"overlay.tmp{os.getpid()}.npz"
    # сирота прошлого аварийного write (обрыв между savez и replace) - мусор
    # рядом с индексом: свои tmp-файлы оверлея чистим в начале записи (записей
    # одновременно не бывает - очередь ингеста последовательная, K8)
    for orphan in index_dir.glob("overlay.tmp*.npz"):
        try:
            orphan.unlink()
        except OSError as e:  # noqa: BLE001 - чистка не должна ломать запись
            logger.warning("tmp-сирота оверлея %s не удалена: %s", orphan, e)
    np.savez(
        tmp,
        meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8),
        # строковый dtype (не object) - savez без pickle, чтение c allow_pickle=False
        ids=np.array(ids, dtype=str) if ids else np.empty(0, dtype=str),
        matrix=matrix.astype(np.float32),
    )
    # fsync перед replace (K8): без сброса страниц файла на диск сбой питания
    # между savez и replace оставляет replace-нутый файл пустым/битым -
    # overlay молча пропускается и загрузки доедут только с пересборкой.
    # r+b: на Windows os.fsync требует дескриптор с правом записи
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    tmp.replace(overlay_path(index_dir))


def _index_node_ids(index_dir: Path, document_id: str) -> list[str]:
    """node_version_id документа из дискового индекса (meta.json + мета overlay)
    - CLI-путь без загрузки векторов и эмбеддера. Союз покрывает всё: meta.json -
    снапшот последней пересборки, overlay - загрузки после неё."""
    ids: list[str] = []
    meta_path = Path(index_dir) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # noqa: BLE001 - битая мета не ломает удаление
            meta = {}
        ids = [
            nid for nid, m in (meta.get("node_meta") or {}).items()
            if isinstance(m, dict) and m.get("document_id") == document_id
        ]
    data = read_overlay(index_dir)
    if data is not None:
        for nid, m in (data["meta"].get("nodes") or {}).items():
            if isinstance(m, dict) and m.get("document_id") == document_id and nid not in ids:
                ids.append(nid)
    return ids


def tombstone_document(index_dir: Path, document_id: str, bundle=None) -> int:
    """Вывести документ из поиска (K10 DELETE): tombstone status=deleted всем его
    узлам. Возвращает число затронутых узлов.

    Механизм - tombstone-патч, не перезапись бандла: векторное хранилище и BM25
    append-only (физическое удаление = пересборка обеих структур), а
    IndexBundle.save стёр бы overlay вместе с дельтами ДРУГИХ документов,
    загруженных после пересборки. Патч статуса - тот же механизм, что у
    superseded-ревизий: дёшев (O(узлов документа)), атомарен для читателей
    (обновление полей мета), персистентен через overlay и самоликвидируется при
    полной пересборке (канон уже в trash, build_index его не видит, save стирает
    overlay). До пересборки узлы остаются в векторных структурах - фильтр
    status=deleted в hybrid_search отсекает их безусловно (и от админа, и от
    revision=all/r<n>).

    bundle живого фасада патчится в памяти - поиск закрывается СРАЗУ; тот же
    патч дописывается в overlay - после рестарта load_index применит его к
    основному индексу. bundle=None (CLI) - узлы собираются с диска
    (_index_node_ids). Индекса нет - вернёт 0 (удалять нечего)."""
    if bundle is not None:
        nids = [
            nid for nid, m in bundle.node_meta.items()
            if m.get("document_id") == document_id
        ]
        for nid in nids:
            bundle.node_meta[nid]["status"] = "deleted"
    else:
        nids = _index_node_ids(index_dir, document_id)
    if not nids:
        return 0
    append_overlay(
        Path(index_dir),
        {},
        {nid: {"status": "deleted"} for nid in nids},
        [],
        np.zeros((0, 1), dtype=np.float32),
        [],
    )
    return len(nids)
