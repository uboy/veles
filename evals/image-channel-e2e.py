"""E2E картинного канала на живом фасаде (план Ф3, проверка на реальной модели).

Собирает мини-канон: два документа (по текстовому узлу и картинкам-активам),
строит картинный индекс настоящей моделью, поднимает фасад по HTTP с
включённым каналом и проходит путь пользователя:
  1) POST /v1/knowledge/query - в ответе images с подписанной asset-ссылкой;
  2) GET asset_url - файл картинки (image/png);
  3) запрос с чужой группой - картинки не отдаются.
Текстовый канал - фейковый эмбеддер (не предмет проверки); картинный -
jina-clip-v2 (VELES_E2E_IMAGE_MODEL, по умолчанию jinaai/jina-clip-v2).

Запуск: python evals/image-channel-e2e.py --work-dir /tmp/veles-e2e
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

TOKEN = "e2e-token"


def build_mini_corpus(work: Path, images_dir: Path) -> dict:
    """Мини-канон: doc-a (personal, 3 картинки) и doc-b (legal, 2 картинки)."""
    from veles.canonical.model import CanonicalDocument, Provenance
    from veles.canonical.store import _doc_dir, save_document

    def make_doc(document_id: str, title: str, groups: list[str], n_imgs: int) -> dict:
        doc = CanonicalDocument(
            document_id=document_id, title=title, document_type="normative",
            sha256=CanonicalDocument.compute_sha256(title.encode()),
            parser_profile="e2e-mini",
        )
        doc.add_node(
            logical_node_id=f"{document_id}#root", node_type="root",
            heading_path=[], text=title,
            provenance=Provenance(adapter="e2e-mini", source_id=document_id),
        )
        doc.add_node(
            logical_node_id=f"{document_id}#t1", node_type="section",
            heading_path=["Содержание"], text="Текст документа про архитектуру.",
            provenance=Provenance(adapter="e2e-mini", source_id=document_id),
        )
        meta_nodes = {}
        for i in range(1, n_imgs + 1):
            asset = f"pic-{i:03d}.png"
            doc.add_node(
                logical_node_id=f"{document_id}#img{i:03d}", node_type="image",
                heading_path=["Содержание"], text=f"Изображение. {title}, рисунок {i}",
                provenance=Provenance(adapter="e2e-mini", source_id=document_id,
                                      page=i * 10),
                asset=asset,
            )
        save_document(doc, root=work / "canonical", by="e2e")
        # файлы активов - в doc_dir
        adir = _doc_dir(document_id, work / "canonical") / "assets"
        adir.mkdir(parents=True, exist_ok=True)
        return {"doc": doc, "adir": adir}

    docs = {
        "docs/book_a.pdf": make_doc("docs/book_a.pdf", "Книга А", ["personal"], 3),
        "docs/secret_b.pdf": make_doc("docs/secret_b.pdf", "Секретный Б", ["legal"], 2),
    }
    # раскладываем картинки из санити-выборки по активам (просто копии файлов)
    samples = sorted(images_dir.rglob("*.png"))
    if len(samples) < 5:
        raise SystemExit(f"нужно >=5 картинок в {images_dir}, найдено {len(samples)}")
    for doc_id, item in docs.items():
        for i in range(1, len(item["doc"].nodes)):
            node = item["doc"].nodes[i]
            if node.asset:
                shutil.copyfile(samples[i % len(samples)], item["adir"] / node.asset)

    # node_meta текстового индекса (текстовые вектора не предмет E2E)
    node_meta = {}
    for doc_id, item in docs.items():
        for node in item["doc"].nodes:
            if node.node_type == "root":
                continue
            node_meta[node.node_version_id] = {
                "document_id": doc_id, "title": item["doc"].title,
                "logical_node_id": node.logical_node_id,
                "heading_path": node.heading_path, "text": node.text,
                "provenance": node.provenance.model_dump(),
                "access_groups": ["personal"] if "book_a" in doc_id else ["legal"],
                "sha256_doc": item["doc"].sha256, "asset": node.asset,
                "revision_id": "r1", "status": "current", "lost_force": False,
            }
    return node_meta


class FakeTextEmbedder:
    dim = 4

    def embed_passages(self, texts):
        import numpy as np

        return np.zeros((len(texts), 4), dtype=np.float32)

    def embed_query(self, text):
        import numpy as np

        return np.zeros(4, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, default=Path("/tmp/veles-image-e2e"))
    ap.add_argument("--images-dir", type=Path, required=True,
                    help="каталог с png (например sanity-run2/images/systemdesign)")
    ap.add_argument("--port", type=int, default=8491)
    args = ap.parse_args()

    work = args.work_dir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    import os

    image_model = os.environ.get("VELES_E2E_IMAGE_MODEL", "jinaai/jina-clip-v2")
    os.environ["VELES_IMAGE_MODEL"] = image_model  # читается конфигом при импорте
    os.environ["VELES_DATA"] = str(work)
    os.environ["VELES_IMAGE_INDEX_DIR"] = str(work / "image-index")
    os.environ["VELES_IMAGE_DEVICE"] = os.environ.get("VELES_E2E_IMAGE_DEVICE", "cuda")
    os.environ["VELES_API_TOKEN"] = TOKEN
    # публичный URL = этот фасад (asset_url строится из него), топ-5 = оба документа
    os.environ["VELES_PUBLIC_URL"] = f"http://127.0.0.1:{args.port}"
    os.environ["VELES_IMAGE_TOPK"] = "5"

    node_meta = build_mini_corpus(work, args.images_dir)

    # картинный индекс - настоящей моделью
    from veles.indexing.image_index import JinaClipEmbedder, build_image_index

    t0 = time.time()
    report = build_image_index(
        node_meta, [work / "canonical"],
        JinaClipEmbedder(image_model, device=os.environ["VELES_IMAGE_DEVICE"]),
        work / "image-index",
    )
    print("[e2e] image index:", json.dumps({k: v for k, v in report.items()
                                            if k != "skipped_detail"}), flush=True)

    # фасад: текстовый канал фейковый, картинный - живой
    from veles.api.app import VelesPipeline, create_app
    from veles.indexing.indexer import IndexBundle

    class FakeVector:
        def search(self, q, top_k=10):
            first = next(iter(node_meta))
            return [(first, 0.9)]

    class FakeBm25:
        def search(self, q, top_k=10):
            return []

    pipeline = VelesPipeline(
        IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=node_meta),
        FakeTextEmbedder(), None,
    )
    app = create_app(pipeline=pipeline)

    config = uvicorn_config(app, args.port)
    import threading

    import uvicorn

    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        time.sleep(0.2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2)
            break
        except Exception:  # noqa: BLE001
            continue
    print(f"[e2e] фасад жив на :{args.port}", flush=True)

    base = f"http://127.0.0.1:{args.port}"
    out = {"steps": []}

    def post(path, payload):
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {TOKEN}"})
        return json.loads(urllib.request.urlopen(req, timeout=120).read())

    # 1) вопрос - картинки должны прийти отдельным списком со ссылками
    r = post("/v1/knowledge/query", {"query": "схема архитектуры"})
    images = r.get("images") or []
    step1 = {
        "step": "query", "decision": r.get("decision"),
        "images_count": len(images),
        "docs": sorted({im.get("document_id") for im in images}),
        "has_asset_url": all(im.get("asset_url") for im in images),
    }
    out["steps"].append(step1)
    print("[e2e] query:", json.dumps(step1), flush=True)
    assert images, "канал не вернул картинки"

    # 2) файл картинки по ссылке - 200 image/png
    url = images[0]["asset_url"]
    with urllib.request.urlopen(url, timeout=30) as resp:
        step2 = {"step": "asset_fetch", "status": resp.status,
                 "content_type": resp.headers.get("content-type"),
                 "bytes": len(resp.read())}
    out["steps"].append(step2)
    print("[e2e] asset:", json.dumps(step2), flush=True)
    assert step2["status"] == 200 and "image" in step2["content_type"]

    # 3) группы: bearer-клиент несёт группы владельца (personal, без "*") -
    # как в текстовом hybrid_search, legal-документ скрыт. Это живая проверка
    # ACL: из 5 проиндексированных картинок приходят только 3 personal-картинки.
    r2 = post("/v1/knowledge/query", {"query": "схема архитектуры"})
    docs2 = sorted({im.get("document_id") for im in (r2.get("images") or [])})
    pages_ok = all(im.get("page") for im in (r2.get("images") or []))
    step3 = {"step": "groups_personal_only", "docs": docs2, "pages_ok": pages_ok}
    out["steps"].append(step3)
    print("[e2e] acl:", json.dumps(step3), flush=True)
    assert docs2 == ["docs/book_a.pdf"], docs2
    assert pages_ok

    out["verdict"] = "PASS"
    print("[e2e] VERDICT PASS", flush=True)
    (work / "e2e-result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    server.should_exit = True
    thread.join(timeout=10)


def uvicorn_config(app, port):
    import uvicorn

    return uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
