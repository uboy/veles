"""K13: фикс docling-парсера - контент ДО первого заголовка (картинки, абзацы,
таблицы) привязывается к корню документа, а не к несуществующей секции #s0000.

Находка аудита прод-канона 29.09: 83 image-узла docs_Data1_*_index.pdf с
parent_id на секцию #s0000, которая никогда не создаётся (нумерация секций
начинается с s0001). Регресс-тест строит фейковый docling-конвертер (без
реальной зависимости) и требует, чтобы КАЖДЫЙ parent_id разрешался в
существующий logical_node_id документа.
"""

from __future__ import annotations

import sys
import types

import pytest

from veles.ingest.docling_parser import DoclingParserAdapter


# ---------------------------------------------------------------- fake docling

class _FakePil:
    """Стенд-in для PIL-картинки: save() пишет маркерный файл."""

    def save(self, path, format="PNG"):  # noqa: ARG002 - сигнатура PIL
        with open(path, "wb") as f:
            f.write(b"fake-png")


class _Prov(list):
    def __init__(self, page: int):
        super().__init__([types.SimpleNamespace(page_no=page)])


class _FakeItem:
    def __init__(self, page: int = 1):
        self.prov = _Prov(page)


class PictureItem(_FakeItem):
    def __init__(self, page: int = 1):
        super().__init__(page)
        self.image = types.SimpleNamespace(pil_image=_FakePil())


class SectionHeaderItem(_FakeItem):
    def __init__(self, text: str, page: int = 1):
        super().__init__(page)
        self.text = text


class TextItem(_FakeItem):
    def __init__(self, text: str, page: int = 1):
        super().__init__(page)
        self.text = text


class TableItem(_FakeItem):
    def __init__(self, markdown: str, page: int = 1):
        super().__init__(page)
        self._md = markdown

    def export_to_markdown(self, doc):  # noqa: ARG002 - сигнатура docling
        return self._md


class _FakeDoclingDocument:
    def __init__(self, items: list):
        self._items = items

    def iterate_items(self):
        return list(self._items)


class _FakeConvertResult:
    def __init__(self, document: _FakeDoclingDocument):
        self.document = document


class _FakeConverterInstance:
    def __init__(self, document: _FakeDoclingDocument):
        self._document = document

    def convert(self, path, **kwargs):  # noqa: ARG002 - сигнатура docling
        return _FakeConvertResult(self._document)


def install_fake_docling(monkeypatch: pytest.MonkeyPatch, items: list) -> None:
    """Фейковые модули docling в sys.modules (адаптер импортирует их лениво
    внутри parse_file); конвертер отдаёт документ с заданными items."""
    holder: dict = {"document": _FakeDoclingDocument(items)}
    base_models = types.ModuleType("docling.datamodel.base_models")
    base_models.InputFormat = type("InputFormat", (), {"PDF": "pdf"})
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")
    pipeline_options.PdfPipelineOptions = type("PdfPipelineOptions", (), {})
    document_converter = types.ModuleType("docling.document_converter")
    document_converter.PdfFormatOption = type(
        "PdfFormatOption", (), {"__init__": lambda self, **kw: None}
    )
    document_converter.DocumentConverter = lambda *a, **kw: _FakeConverterInstance(
        holder["document"]
    )
    docling = types.ModuleType("docling")
    datamodel = types.ModuleType("docling.datamodel")
    monkeypatch.setitem(sys.modules, "docling", docling)
    monkeypatch.setitem(sys.modules, "docling.datamodel", datamodel)
    monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", base_models)
    monkeypatch.setitem(
        sys.modules, "docling.datamodel.pipeline_options", pipeline_options
    )
    monkeypatch.setitem(sys.modules, "docling.document_converter", document_converter)


# --------------------------------------------------------------------- tests

def _parse(monkeypatch: pytest.MonkeyPatch, tmp_path, items: list):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    pdf = corpus / "docs_Data1_10_1_index.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    install_fake_docling(monkeypatch, items)
    return DoclingParserAdapter().parse_file(pdf, corpus_root=corpus)


def test_prelude_content_attaches_to_root_not_s0000(tmp_path, monkeypatch):
    """Картинка и абзац ДО первого заголовка: parent - корень документа,
    а не несуществующая #s0000; после заголовка - созданная секция s0001."""
    doc = _parse(monkeypatch, tmp_path, [
        PictureItem(page=1),                      # прелюдия: картинка
        TextItem("Абзац прелюдии до всяких заголовков, длинный достаточно.", page=1),
        SectionHeaderItem("ГЛАВА 1. Начало", page=1),
        TextItem("Абзац уже внутри первой секции, тоже достаточно длинный.", page=1),
    ])
    ids = {n.logical_node_id for n in doc.nodes}
    rel = "docs_Data1_10_1_index.pdf"
    assert f"{rel}#s0000" not in ids
    nodes = {n.logical_node_id: n for n in doc.nodes}
    assert nodes[f"{rel}#img001"].parent_id == f"{rel}#root"
    # абзац прелюдии (id сдвинут: картинка тоже расходует счётчик p)
    prelude = [n for n in doc.nodes if "прелюдии" in n.text]
    assert len(prelude) == 1 and prelude[0].parent_id == f"{rel}#root"
    assert nodes[f"{rel}#s0001"].parent_id == f"{rel}#root"       # сама секция
    inside = [n for n in doc.nodes if "внутри первой секции" in n.text]
    assert len(inside) == 1 and inside[0].parent_id == f"{rel}#s0001"


def test_all_parent_ids_resolve(tmp_path, monkeypatch):
    """Инвариант целостности: каждый parent_id указывает на существующий узел
    (или пуст) при произвольном порядке картинок/таблиц/заголовков."""
    doc = _parse(monkeypatch, tmp_path, [
        TextItem("Первый абзац до заголовка, с запасом длины для фильтра.", page=1),
        PictureItem(page=1),
        SectionHeaderItem("Раздел А", page=1),
        TextItem("Абзац раздела А, достаточно длинный для парсера.", page=2),
        TableItem("| а | б |\n|---|---|\n| 1 | 2 |", page=2),
        PictureItem(page=2),
        SectionHeaderItem("Раздел Б", page=2),
        TextItem("Абзац раздела Б, достаточно длинный для парсера.", page=3),
    ])
    ids = {n.logical_node_id for n in doc.nodes}
    broken = [
        n.logical_node_id for n in doc.nodes
        if n.parent_id and n.parent_id not in ids
    ]
    assert broken == []
    # узлы обоих типов в прелюдии и в секциях на месте
    kinds = [n.node_type for n in doc.nodes]
    assert kinds.count("image") == 2
    assert kinds.count("section") == 2


def test_version_id_unaffected_by_parent_fix(tmp_path, monkeypatch):
    """parent_id не входит в node_version_id (sha256 revision|lnid|text):
    фикс привязки прелюдии не меняет версии узлов."""
    from veles.canonical.model import CanonicalNode

    doc = _parse(monkeypatch, tmp_path, [
        PictureItem(page=1),
        SectionHeaderItem("Глава", page=1),
        TextItem("Текст первой секции, длинный enough для фильтра парсера.", page=1),
    ])
    rel = "docs_Data1_10_1_index.pdf"
    by_lnid = {n.logical_node_id: n for n in doc.nodes}
    img = by_lnid[f"{rel}#img001"]
    assert img.node_version_id == CanonicalNode.make_version_id(
        doc.revision_id, img.logical_node_id, img.text
    )
