"""Тесты картинного канала: инвентарь активов, индекс, ACL, слияние в API.

Без реальных моделей: фейковый кодировщик с детерминированными векторами.
Канал по умолчанию выключен (VELES_IMAGE_INDEX_DIR пуст) - отдельный тест
фиксирует, что текстовая выдача при этом прежняя, а images пуст.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, attach_citation_links, create_app, render_chat_content
from veles.canonical.store import _doc_dir
from veles.indexing.image_index import (
    ImageIndex,
    build_image_index,
    inventory_from_meta,
)
from veles.indexing.indexer import IndexBundle
from veles.retrieval import image_retriever


def _vec(seed: str, dim: int = 8) -> np.ndarray:
    raw = hashlib.md5(seed.encode("utf-8")).digest()
    v = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)[:dim]
    v /= np.linalg.norm(v) + 1e-9
    return v


class FakeImageEmbedder:
    """Вектор = md5(seed); у картинки seed - имя файла, у запроса - текст.

    Поиск «запрос == имя файла» гарантирует top1 этой картинки."""

    def __init__(self, model_name: str | None = None, device: str = "cpu", dim: int = 8):
        # Подпись и семантика размерности совместимы с JinaClipEmbedder:
        # запрошенное MRL-усечение больше полной размерности (16 = длина md5)
        # откатывается к полной. Класс подменяется в ретривере.
        self.dim = dim if 0 < dim <= 16 else 16

    def embed_images(self, paths):
        import os

        return np.stack([_vec(os.path.basename(p), self.dim) for p in paths])

    def embed_query(self, text: str) -> np.ndarray:
        return _vec(text, self.dim)


def _meta_with_images() -> dict:
    """node_meta: два image-узла personal-документа, один restricted, один удалён."""
    return {
        "n_img1": {
            "document_id": "docs/book_a.pdf", "title": "Книга А",
            "logical_node_id": "docs/book_a.pdf#img001",
            "heading_path": ["Глава 1"], "text": "Изображение. Схема дерева",
            "provenance": {"adapter": "docling", "source_id": "docs/book_a.pdf", "page": 12},
            "access_groups": ["personal"], "asset": "pic-001.png",
            "revision_id": "r1", "status": "current", "lost_force": False,
        },
        "n_img2": {
            "document_id": "docs/book_a.pdf", "title": "Книга А",
            "logical_node_id": "docs/book_a.pdf#img002",
            "heading_path": ["Глава 2"], "text": "Изображение. Граф",
            "provenance": {"adapter": "docling", "source_id": "docs/book_a.pdf", "page": 30},
            "access_groups": ["personal"], "asset": "pic-002.png",
            "revision_id": "r1", "status": "current", "lost_force": False,
        },
        "n_img3": {
            "document_id": "docs/secret_b.pdf", "title": "Секретный Б",
            "logical_node_id": "docs/secret_b.pdf#img001",
            "heading_path": [], "text": "Изображение. Схема сети",
            "provenance": {"adapter": "docling", "source_id": "docs/secret_b.pdf", "page": 5},
            "access_groups": ["legal"], "asset": "pic-001.png",
            "revision_id": "r1", "status": "current", "lost_force": False,
        },
        "n_img4": {
            "document_id": "docs/old_c.pdf", "title": "Удалённый В",
            "logical_node_id": "docs/old_c.pdf#img001",
            "heading_path": [], "text": "Изображение. Таблица",
            "provenance": {"adapter": "docling", "source_id": "docs/old_c.pdf", "page": 7},
            "access_groups": ["personal"], "asset": "pic-009.png",
            "revision_id": "r1", "status": "deleted", "lost_force": False,
        },
    }


@pytest.fixture()
def corpus(tmp_path):
    """Canonical-каталог с файлами активов для n_img1/n_img2/n_img3."""
    meta = _meta_with_images()
    cdir = tmp_path / "canonical"
    for nid in ("n_img1", "n_img2", "n_img3"):
        m = meta[nid]
        adir = _doc_dir(m["document_id"], cdir) / "assets"
        adir.mkdir(parents=True, exist_ok=True)
        (adir / m["asset"]).write_bytes(b"fake-png-" + nid.encode())
    return meta, cdir


def test_inventory_skips_missing_and_deleted(corpus):
    meta, cdir = corpus
    records, skipped = inventory_from_meta(meta, [cdir])
    # удалённый документ (n_img4, status=deleted) исключается молча - это норма,
    # не аномалия; в skipped попадают только битые ссылки на файлы
    assert {r.image_id for r in records} == {"n_img1", "n_img2", "n_img3"}
    assert all(r.page for r in records)
    assert skipped == []


def test_inventory_limit_per_doc(corpus):
    meta, cdir = corpus
    records, _ = inventory_from_meta(meta, [cdir], limit_per_doc=1)
    assert sorted(r.image_id for r in records) == ["n_img1", "n_img3"]


def test_inventory_reports_missing_file(corpus, tmp_path):
    meta, cdir = corpus
    # ломаем файл актива n_img1 - сборка обязана отчитаться о пропуске
    f = _doc_dir("docs/book_a.pdf", cdir) / "assets" / "pic-001.png"
    f.unlink()
    records, skipped = inventory_from_meta(meta, [cdir])
    assert "n_img1" not in {r.image_id for r in records}
    assert [s["image_id"] for s in skipped] == ["n_img1"]
    assert skipped[0]["reason"] == "asset-file-missing"


def test_build_and_load_roundtrip_group_filter(corpus, tmp_path):
    meta, cdir = corpus
    out = tmp_path / "image-index"
    report = build_image_index(meta, [cdir], FakeImageEmbedder(), out, batch=2)
    assert report["indexed"] == 3
    # n_img4 (status=deleted) исключён молча; битых ссылок на файлы нет
    assert report["skipped"] == 0
    # число проиндексированных сходится с манифестом
    loaded = ImageIndex.load(out)
    assert len(loaded.records) == report["indexed"]

    q = _vec("pic-001.png")
    hits = loaded.search(q, top_k=3, allowed_groups={"personal"})
    # n_img3 (legal) отсечён; n_img1 топ-1, personal-картинка n_img2 добирается
    assert hits[0]["record"].image_id == "n_img1"
    assert "n_img3" not in [h["record"].image_id for h in hits]
    # чужой группе personal-картинки не видны; legal видит только свою
    legal_hits = loaded.search(q, top_k=3, allowed_groups={"legal"})
    assert [h["record"].image_id for h in legal_hits] == ["n_img3"]
    assert len(loaded.search(q, top_k=10, allowed_groups={"*"})) == 3


@pytest.fixture()
def fake_retriever_embedder(monkeypatch):
    """Реальный JinaClipEmbedder в ретривере подменён фейковым: тесты не
    должны грузить модель. Исходный класс восстанавливается monkeypatch'ем."""
    monkeypatch.setattr(
        "veles.indexing.image_index.JinaClipEmbedder", FakeImageEmbedder, raising=True
    )
    yield
    image_retriever.reset_image_cache()


def test_search_images_disabled_by_default(monkeypatch):
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", "", raising=False)
    image_retriever.reset_image_cache()
    assert image_retriever.search_images("схема дерева") == []


def test_search_images_enabled_and_acl(
    corpus, tmp_path, monkeypatch, fake_retriever_embedder
):
    meta, cdir = corpus
    out = tmp_path / "image-index"
    # в проде сборка и сервис живут с одним IMAGE_DIM/IMAGE_TOPK - в тесте так же
    monkeypatch.setattr("veles.config.IMAGE_DIM", 8, raising=False)
    monkeypatch.setattr("veles.config.IMAGE_TOPK", 1, raising=False)
    build_image_index(meta, [cdir], FakeImageEmbedder(), out)
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", str(out), raising=False)
    image_retriever.reset_image_cache()

    hits = image_retriever.search_images("pic-001.png", allowed_groups={"personal"})
    assert len(hits) == 1
    assert hits[0]["kind"] == "image"
    assert hits[0]["document_id"] == "docs/book_a.pdf"
    assert hits[0]["asset"] == "pic-001.png"
    assert hits[0]["page"] == 12
    assert 0.0 < hits[0]["score"] <= 1.0
    # узкая группа не получает чужих картинок; выключение конфига - пусто
    legal = image_retriever.search_images("pic-001.png", allowed_groups={"legal"})
    assert [h["document_id"] for h in legal] == ["docs/secret_b.pdf"]
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", "", raising=False)
    image_retriever.reset_image_cache()
    assert image_retriever.search_images("pic-001.png", allowed_groups={"personal"}) == []


# --- Слияние в API: текстовая выдача прежняя, картинки отдельным списком ---


class FakeVector:
    def search(self, q, top_k=10):
        return [("n1", 0.95)]


class FakeBm25:
    def search(self, q, top_k=10):
        return [("n1", 5.0)]


def make_pipeline() -> VelesPipeline:
    meta = {
        "n1": {
            "text": "Норма управляемости руководителя - от 5 до 9 пунктов.",
            "access_groups": ["personal"],
            "title": "Пересказ Общий менеджмент",
            "document_id": "lectures/менеджмент/лекция2/пересказ.txt",
            "heading_path": ["Общий менеджмент"],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "пересказ.txt",
                           "lines": [82, 82]},
        }
    }
    class FakeEmbedder:
        dim = 4

        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    return VelesPipeline(IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta),
                         FakeEmbedder(), None)


def test_query_channel_off_text_only(monkeypatch):
    """Регрессия: канал выключен - images пуст, citations/evidence прежние."""
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", "", raising=False)
    image_retriever.reset_image_cache()
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "норма управляемости"})
    d = r.json()
    assert r.status_code == 200
    assert d["images"] == []
    assert d["decision"] == "answer"
    assert "5 до 9" in d["answer"]
    assert d["citations"][0]["title"].startswith("Пересказ")


def test_query_channel_on_images_in_response(
    corpus, tmp_path, monkeypatch, fake_retriever_embedder
):
    meta, cdir = corpus
    out = tmp_path / "image-index"
    monkeypatch.setattr("veles.config.IMAGE_DIM", 8, raising=False)
    monkeypatch.setattr("veles.config.IMAGE_TOPK", 1, raising=False)
    build_image_index(meta, [cdir], FakeImageEmbedder(), out)
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", str(out), raising=False)
    image_retriever.reset_image_cache()
    app = create_app(pipeline=make_pipeline())
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "pic-001.png"})
    d = r.json()
    assert d["decision"] == "answer"
    assert len(d["images"]) == 1
    im = d["images"][0]
    assert im["asset"] == "pic-001.png"
    assert "/v1/asset/docs/book_a.pdf/pic-001.png" in im["asset_url"]
    # текстовая часть не изменилась
    assert d["citations"][0]["title"].startswith("Пересказ")
    monkeypatch.setattr("veles.config.IMAGE_INDEX_DIR", "", raising=False)
    image_retriever.reset_image_cache()


def test_attach_links_adds_asset_url_for_images():
    result = {
        "citations": [],
        "images": [{"kind": "image", "image_id": "x1", "node_version_id": "x1",
                     "document_id": "docs/a.pdf", "asset": "pic-001.png",
                     "page": 3, "title": "Схема"}],
    }
    out = attach_citation_links(result, token="t")
    assert out["images"][0]["asset_url"] == (
        "http://127.0.0.1:8400/v1/asset/docs/a.pdf/pic-001.png?token=t"
    )
    assert "token=t" in out["images"][0]["asset_url"]
    assert out["images"][0]["source_url"]


def test_render_chat_content_shows_channel_images_once():
    result = {
        "decision": "answer",
        "answer": "Норма - от 5 до 9.",
        "citations": [{"n": 1, "title": "Пересказ", "source_id": "пересказ.txt",
                        "document_id": "docs/a.pdf", "node_version_id": "n1",
                        "asset": "pic-001.png", "lines": None, "lost_force": False}],
        "images": [{"kind": "image", "image_id": "x1", "node_version_id": "x1",
                     "document_id": "docs/a.pdf", "asset": "pic-001.png",
                     "page": 3, "title": "Схема дерева"}],
        "diagnostics": {"confidence": 0.5, "reason": "", "conflicts": []},
    }
    text = render_chat_content(result, token="t")
    # актив уже показан из цитаты - дубль канала не добавляется
    assert text.count("![изображение]") == 1
    result["citations"] = []
    text2 = render_chat_content(result, token="t")
    assert "![изображение]" in text2
    assert "Схема дерева" in text2 and "стр. 3" in text2
    assert "/v1/asset/docs/a.pdf/pic-001.png" in text2
