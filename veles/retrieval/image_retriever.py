"""Запрос-путь картинного канала: поиск по индексу изображений активов.

Включается только переменной VELES_IMAGE_INDEX_DIR (пусто/не задано = канал
выключен: ни индекс, ни модель не грузятся, поведение прежнее - ноль
регрессии). Индекс и модель грузятся лениво при первом запросе и кэшируются
на процесс. Ошибка канала (нет каталога, битая модель) - деградация к
текстовой выдаче: пустой список, ответ не роняется.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_cache: dict = {}


def _image_state() -> tuple | None:
    """(ImageIndex, ImageEmbedder) из конфига; None - канал выключен/не собран."""
    from ..config import IMAGE_DEVICE, IMAGE_DIM, IMAGE_INDEX_DIR, IMAGE_MODEL

    if not IMAGE_INDEX_DIR:
        return None
    if "state" in _cache:
        return _cache["state"]
    from pathlib import Path

    from ..indexing.image_index import ImageIndex, JinaClipEmbedder

    root = Path(IMAGE_INDEX_DIR)
    if not (root / "image-meta.json").exists():
        logger.warning("image-index: каталог %s без image-meta.json - канал выключен", root)
        _cache["state"] = None
        return None
    index = ImageIndex.load(root)
    embedder = JinaClipEmbedder(IMAGE_MODEL, device=IMAGE_DEVICE, dim=IMAGE_DIM)
    _cache["state"] = (index, embedder)
    return _cache["state"]


def reset_image_cache() -> None:
    """Сброс кэша (тесты и горячая перезагрузка конфига)."""
    _cache.clear()


def search_images(
    query_text: str,
    top_k: int | None = None,
    allowed_groups: set[str] | None = None,
) -> list[dict]:
    """Топ-K картинок под текстовый запрос; [] если канал выключен/ошибка.

    Каждая запись - ImageRecord.public() плюс score (косинус в пространстве
    модели) и rrf-подобный ранговый скор для диагностики порядка."""
    from ..config import IMAGE_TOPK, RRF_K

    state = _image_state()
    if state is None:
        return []
    index, embedder = state
    k = top_k if top_k is not None else IMAGE_TOPK
    try:
        qvec = embedder.embed_query(query_text)
        hits = index.search(qvec, top_k=k, allowed_groups=allowed_groups)
    except Exception as exc:  # noqa: BLE001 - канал не должен ронять ответ
        logger.warning("image-index: поиск не удался, канал пропущен: %s", exc)
        return []
    out = []
    for rank, hit in enumerate(hits, start=1):
        item = hit["record"].public()
        item["score"] = round(float(hit["score"]), 4)
        item["rrf"] = round(1.0 / (RRF_K + rank), 6)
        out.append(item)
    return out
