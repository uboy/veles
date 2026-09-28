"""DoclingParserAdapter: DoclingDocument -> CanonicalDocument со структурными единицами.

Единицы Ф3: секция = заголовок раздела + его контент ДО следующего заголовка того же/высшего
уровня (подобие статьи/пункта нормативного документа); таблицы - отдельные узлы (markdown).
Provenance - страница (page_no из prov Docling). Для судебного прогона Ф3 раннер собирает
из этих узлов секционные юниты (структурная конфигурация) или абзацные (наивная, Ф2).
"""

from __future__ import annotations

from pathlib import Path

from ..canonical.model import CanonicalDocument, Provenance


class DoclingParserAdapter:
    parser_profile = "docling:2.130-structural"

    def parse_file(
        self, path: Path, corpus_root: Path, max_pages: int | None = None
    ) -> CanonicalDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        rel = path.relative_to(corpus_root).as_posix()
        raw_bytes = path.read_bytes()

        opts = PdfPipelineOptions()
        opts.do_ocr = False  # текстовый слой есть; OCR - отдельное решение
        opts.generate_page_images = False
        # картинки документа -> активы (требование владельца 28.09: релевантная
        # картинка показывается в ответе; распознавание её содержимого не требуется)
        opts.generate_picture_images = True
        opts.images_scale = 2  # ~144 dpi
        conv = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
        kwargs = {}
        if max_pages:
            kwargs["page_range"] = (1, max_pages)
        res = conv.convert(str(path), **kwargs)
        d = res.document

        document = CanonicalDocument(
            document_id=rel,
            title=path.stem,
            document_type="normative",
            sha256=CanonicalDocument.compute_sha256(raw_bytes),
            parser_profile=self.parser_profile,
        )
        assets_dir = Path(__import__("tempfile").mkdtemp(prefix="veles-assets-"))
        assets: dict[str, str] = {}
        document.add_node(
            logical_node_id=f"{rel}#root",
            node_type="root",
            heading_path=[],
            text=path.stem,
            provenance=Provenance(adapter=self.parser_profile, source_id=rel),
        )

        idx = 0
        section_idx = 0
        section_path: list[str] = []

        def add_par(text: str, page: int | None) -> None:
            nonlocal idx
            if not text or len(text.strip()) < 25:
                return
            idx += 1
            document.add_node(
                logical_node_id=f"{rel}#p{idx:05d}",
                node_type="paragraph",
                parent_id=f"{rel}#s{section_idx:04d}",
                heading_path=list(section_path),
                text=" ".join(text.split()),
                provenance=Provenance(
                    adapter=self.parser_profile, source_id=rel, page=page
                ),
            )

        last_text_head = ""
        img_idx = 0

        for item in d.iterate_items():
            it = item[0] if isinstance(item, tuple) else item
            kind = type(it).__name__
            prov = getattr(it, "prov", None)
            page = prov[0].page_no if prov and len(prov) else None

            if kind == "PictureItem":
                img_idx += 1
                name = f"pic-{img_idx:03d}.png"
                cap = getattr(it, "caption_text", None)
                if callable(cap):
                    try:
                        cap = cap()
                    except Exception:  # noqa: BLE001
                        cap = None
                caption = (str(cap) if cap else "").strip()
                if not caption:
                    caption = last_text_head or f"Изображение {img_idx}"
                pil = None
                img_ref = getattr(it, "image", None)
                if img_ref is not None:
                    pil = getattr(img_ref, "pil_image", None)
                    if pil is None and hasattr(img_ref, "_pil_image"):
                        pil = img_ref._pil_image
                if pil is None:
                    img_idx -= 1
                    continue
                try:
                    pil.save(assets_dir / name, format="PNG")
                except Exception:  # noqa: BLE001 - битая картинка не роняет парс
                    img_idx -= 1
                    continue
                assets[name] = str(assets_dir / name)
                idx += 1
                document.add_node(
                    logical_node_id=f"{rel}#img{img_idx:03d}",
                    node_type="image",
                    parent_id=f"{rel}#s{section_idx:04d}",
                    heading_path=list(section_path),
                    text=f"Изображение. {caption}"[:500],
                    provenance=Provenance(
                        adapter=self.parser_profile, source_id=rel, page=page
                    ),
                    asset=name,
                )
                continue
            last_text_head = (" ".join((getattr(it, "text", "") or "").split()))[:120]

            if kind == "SectionHeaderItem":
                section_idx += 1
                text = " ".join((getattr(it, "text", "") or "").split())
                document.add_node(
                    logical_node_id=f"{rel}#s{section_idx:04d}",
                    node_type="section",
                    parent_id=f"{rel}#root",
                    heading_path=list(section_path),
                    text=text,
                    provenance=Provenance(
                        adapter=self.parser_profile, source_id=rel, page=page
                    ),
                )
                section_path = (section_path + [text])[-3:]
            elif kind == "TableItem":
                idx += 1
                try:
                    tbl = it.export_to_markdown(d)
                except Exception:  # noqa: BLE001 - таблица без экспорта не роняет парс
                    tbl = getattr(it, "captiontext", "") or ""
                if tbl.strip():
                    document.add_node(
                        logical_node_id=f"{rel}#p{idx:05d}",
                        node_type="paragraph",
                        parent_id=f"{rel}#s{section_idx:04d}",
                        heading_path=list(section_path) + ["Таблица"],
                        text="Таблица. " + tbl.strip()[:3000],
                        provenance=Provenance(
                            adapter=self.parser_profile, source_id=rel, page=page
                        ),
                    )
            elif kind in ("TextItem", "ListItem", "CaptionItem", "FootnoteItem"):
                add_par(getattr(it, "text", "") or "", page)
            #-MetaItem и прочее игнорируется; PictureItem обрабатывается выше

        document.assets = assets
        return document
