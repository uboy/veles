"""Индекс изображений: отдельное векторное хранилище по активам документов.

Картинки PDF уже извлечены при разборе как активы (<doc_dir>/assets/, узлы
node_type="image" в текстовом индексе). Здесь они кодируются мультимодальной
моделью в СВОЁ хранилище векторов: текстовый индекс не меняется, замена
текстовой модели не требуется (план docs/plan/poisk-po-izobrazheniyam).

Инвентарь строится из node_meta готового текстового индекса (записи с
непустым asset); файлы активов ищутся в canonical-каталоге по документу.
Группы доступа картинки = группы исходного документа; на запросе фильтр тот
же, что в hybrid_search. Битые/отсутствующие файлы пропускаются и попадают
в skipped-отчёт сборки.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..config import IMAGE_DIM


class ImageEmbedder(Protocol):
    dim: int

    def embed_images(self, paths: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class JinaClipEmbedder:
    """Мультимодальная модель «текст-картинка» через transformers.

    Модель отдаёт вектора с программируемой размерностью (MRL): до dim
    усекаем и перенормируем; dim=0 - полный вектор модели."""

    def __init__(self, model_name: str, device: str = "cpu", dim: int = IMAGE_DIM):
        import torch

        from transformers import AutoModel

        self._torch = torch
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        # Полная размерность - пробным прогоном (атрибуты модели не гарантированы)
        with self._torch.no_grad():
            probe = self.model.encode_text(["проверка размерности"])
        full_dim = int(np.asarray(probe).shape[-1])
        self.dim = dim if dim and 0 < dim <= full_dim else full_dim

    def _norm_trunc(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)[:, : self.dim]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        from PIL import Image

        imgs = [Image.open(p).convert("RGB") for p in paths]
        with self._torch.no_grad():
            vecs = self.model.encode_image(imgs)
        return self._norm_trunc(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            vecs = self.model.encode_text([text])
        return self._norm_trunc(vecs)[0]


@dataclass
class ImageRecord:
    image_id: str  # node_version_id узла-картинки в текстовом индексе
    document_id: str
    asset: str  # имя файла актива (pic-NNN.png)
    page: int | None
    title: str  # подпись/заголовок раздела из парсера
    access_groups: list[str]
    path: str  # абсолютный путь файла на машине сборки

    def public(self) -> dict:
        """Поля записи для ответа API (путь файла наружу не идёт).

        node_version_id = image_id: узел-картинка существует и в текстовом
        индексе, поэтому ссылочная механика цитат (source/asset) работает
        без исключений."""
        return {
            "kind": "image",
            "image_id": self.image_id,
            "node_version_id": self.image_id,
            "document_id": self.document_id,
            "asset": self.asset,
            "page": self.page,
            "title": self.title,
        }


def inventory_from_meta(
    node_meta: dict, canonical_dirs: list[Path], limit_per_doc: int | None = None
) -> tuple[list[ImageRecord], list[dict]]:
    """Инвентарь картинок из мета текстового индекса.

    Берутся узлы текущих ревизий с непустым asset; файл актива ищется в
    canonical-каталогах (_doc_dir логика из canonical.store). Возвращает
    (записи, пропуски) - пропуски для отчёта сборки (нет файла/нет групп).
    """
    from ..canonical.store import _doc_dir

    records: list[ImageRecord] = []
    skipped: list[dict] = []
    per_doc: dict[str, int] = {}
    for nid, meta in node_meta.items():
        asset = meta.get("asset")
        if not asset or meta.get("status", "current") != "current":
            continue
        doc_id = meta.get("document_id", "")
        if limit_per_doc and per_doc.get(doc_id, 0) >= limit_per_doc:
            continue
        found: Path | None = None
        for cdir in canonical_dirs:
            candidate = _doc_dir(doc_id, cdir) / "assets" / asset
            if candidate.exists():
                found = candidate
                break
        if found is None:
            skipped.append({"image_id": nid, "document_id": doc_id, "asset": asset,
                            "reason": "asset-file-missing"})
            continue
        per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        records.append(
            ImageRecord(
                image_id=nid,
                document_id=doc_id,
                asset=asset,
                page=(meta.get("provenance") or {}).get("page"),
                title=" ".join(str(meta.get("title", "")).split()),
                access_groups=list(meta.get("access_groups") or []),
                path=str(found),
            )
        )
    return records, skipped


class ImageIndex:
    """Хранилище векторов картинок + записи; тот же VectorStore, что у текста."""

    def __init__(self, dim: int, store=None):
        from .vector_store import make_store

        self.dim = dim
        self.store = store or make_store(prefer="faiss", dim=dim)
        self.records: list[ImageRecord] = []

    def add(self, records: list[ImageRecord], vectors: np.ndarray) -> None:
        assert len(records) == len(vectors)
        self.store.add([r.image_id for r in records], vectors)
        self.records.extend(records)

    def search(
        self, query_vec: np.ndarray, top_k: int, allowed_groups: set[str] | None = None
    ) -> list[dict]:
        """Топ-K картинок с групповым фильтром (семантика hybrid_search:
        пересечение групп, "*" = без ограничений). Возвращает (record, score)."""
        from ..config import DEFAULT_ACCESS_GROUPS

        groups = allowed_groups if allowed_groups is not None else DEFAULT_ACCESS_GROUPS
        depth = top_k * 8
        out: list[dict] = []
        for image_id, score in self.store.search(query_vec, top_k=depth):
            rec = self._by_id.get(image_id)
            if rec is None:
                continue
            if "*" not in groups and not (set(rec.access_groups) & groups):
                continue
            out.append({"record": rec, "score": score})
            if len(out) >= top_k:
                break
        return out

    @property
    def _by_id(self) -> dict[str, ImageRecord]:
        if not hasattr(self, "_map"):
            self._map = {r.image_id: r for r in self.records}
        return self._map

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        self.store.save(dir_path)
        (dir_path / "records.jsonl").write_text(
            "\n".join(json.dumps(r.__dict__, ensure_ascii=False) for r in self.records),
            encoding="utf-8",
        )
        (dir_path / "image-meta.json").write_text(
            json.dumps({"dim": self.dim, "count": len(self.records)}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path, store=None) -> "ImageIndex":
        meta = json.loads((dir_path / "image-meta.json").read_text(encoding="utf-8"))
        index = cls(dim=meta["dim"], store=store)
        index.records = [
            ImageRecord(**json.loads(line))
            for line in (dir_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return index


def build_image_index(
    node_meta: dict,
    canonical_dirs: list[Path],
    embedder: ImageEmbedder,
    out_dir: Path,
    batch: int = 8,
    limit_per_doc: int | None = None,
) -> dict:
    """Сборка индекса: инвентарь -> кодирование файлов -> сохранение.

    Возвращает сводку (проиндексировано/пропущено/размерности) - она идёт в
    отчёт сборки, число проиндексированных обязано сойтись с манифестом."""
    records, skipped = inventory_from_meta(node_meta, canonical_dirs, limit_per_doc)
    index = ImageIndex(dim=embedder.dim)
    for i in range(0, len(records), batch):
        chunk = records[i : i + batch]
        vecs = embedder.embed_images([r.path for r in chunk])
        index.add(chunk, vecs)
        print(f"[image-index] {len(index.records)}/{len(records)}", flush=True)
    index.save(out_dir)
    return {
        "indexed": len(index.records),
        "skipped": len(skipped),
        "skip_reasons": sorted({s["reason"] for s in skipped}),
        "dim": index.dim,
        "out_dir": str(out_dir),
        "skipped_detail": skipped,
    }
