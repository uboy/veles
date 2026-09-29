"""K13: детекторы scripts/ingest_audit.py на синтетическом каноне.

Чистый канон (реальный save_document) -> ЗЕЛЁНЫЙ, exit 0. Каждый детектор
проверяется на синтетически испорченном каноне: дубли logical_node_id (в
документе и между документами), дубли node_version_id, пустой текст, битый и
пустой heading_path, битый parent_id, orphan-ассет в обе стороны, узел без
node_version_id и с неверным version_id, обрыв ревизии (revision-r*/ без
манифеста и наоборот), previous_revision в никуда, дубли document_id-каталогов,
расхождение counts с индексом (--index) и журналом джобов (--jobs).
Негативные проверки падают по своей причине: тест требует КОНКРЕТНЫЙ детектор.
Прогон - через main() с --json: проверяется и CLI-обвязка, и exit-code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from veles.canonical.model import CanonicalDocument, CanonicalNode, Provenance
from veles.canonical.store import save_document

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ingest_audit", _ROOT / "scripts" / "ingest_audit.py"
)
ingest_audit = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("ingest_audit", ingest_audit)
_SPEC.loader.exec_module(ingest_audit)


# ------------------------------------------------------------------ helpers

def make_doc(document_id: str, paragraphs: list[str], *, title: str = "") -> CanonicalDocument:
    doc = CanonicalDocument(
        document_id=document_id, title=title or document_id, parser_profile="test:audit",
        sha256="0" * 64,
    )
    doc.add_node(
        logical_node_id=f"{document_id}#root", node_type="root",
        heading_path=[], text=document_id,
        provenance=Provenance(adapter="test:audit", source_id=document_id),
    )
    for i, text in enumerate(paragraphs, start=1):
        doc.add_node(
            logical_node_id=f"{document_id}#p{i:04d}", node_type="paragraph",
            parent_id=f"{document_id}#root", heading_path=["Глава 1"], text=text,
            provenance=Provenance(adapter="test:audit", source_id=document_id),
        )
    return doc


def write_canon(tmp_path: Path, docs: list[CanonicalDocument]) -> Path:
    """tmp_path выступает корнем VELES_DATA: канон в tmp_path/canonical."""
    canon = tmp_path / "canonical"
    for doc in docs:
        save_document(doc, root=canon, by="tester")
    return tmp_path


def run_audit(data: Path, *extra: str) -> tuple[int, dict]:
    """Прогон через CLI main() с JSON-отчётом; возвращает (exit-code, отчёт)."""
    out = data / "audit-report.json"
    rc = ingest_audit.main(
        ["--data", str(data), "--json", str(out), "--json-documents", *extra]
    )
    return rc, json.loads(out.read_text(encoding="utf-8"))


def detector(rep: dict, name: str) -> int:
    return rep["detectors"].get(name, 0)


def examples(rep: dict, name: str) -> list[str]:
    return rep["examples"].get(name, [])


def add_raw_node(canon: Path, dir_name: str, node: dict) -> None:
    """Дописать сырой узел в nodes.jsonl (мимо модели - так попадает и мусор)."""
    with (canon / dir_name / "nodes.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(node, ensure_ascii=False) + "\n")


def rewrite_nodes(canon: Path, dir_name: str, nodes: list[dict]) -> None:
    (canon / dir_name / "nodes.jsonl").write_text(
        "".join(json.dumps(n, ensure_ascii=False) + "\n" for n in nodes),
        encoding="utf-8",
    )


def read_nodes(canon: Path, dir_name: str) -> list[dict]:
    lines = (canon / dir_name / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def raw_node(lnid: str, text: str, **extra) -> dict:
    node = {
        "logical_node_id": lnid,
        # честный version_id: чтобы тест бил по СВОЕМУ детектору, а не ловил
        # collateral version_id_mismatch
        "node_version_id": CanonicalNode.make_version_id("r1", lnid, text),
        "node_type": "paragraph",
        "parent_id": "", "heading_path": ["Глава 1"], "text": text,
        "provenance": {"adapter": "t", "source_id": lnid.rsplit("#", 1)[0]},
    }
    node.update(extra)
    return node


GOOD_PARAS = [
    "Первый абзац документа про целостность сохранения данных.",
    "Второй абзац, достаточно длинный, чтобы пройти любые фильтры парсера.",
]


# ------------------------------------------------------------------- tests

def test_clean_canon_green(tmp_path):
    """Позитив: канон из реального save_document - ЗЕЛЁНЫЙ, exit 0, нули."""
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 0, rep["detectors"]
    assert rep["verdict"]["ok"] is True
    assert rep["nodes_total"] == 3  # root + 2 абзаца
    assert all(v == 0 for v in rep["detectors"].values())


def test_duplicate_lnid_in_doc(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    canon = data / "canonical"
    add_raw_node(canon, "owner_doc1.md", read_nodes(canon, "owner_doc1.md")[1])
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "lnid_duplicate_in_doc") == 1  # конкретно этот детектор
    assert "#p0001 x2" in examples(rep, "lnid_duplicate_in_doc")[0]


def test_empty_text_and_threshold(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    canon = data / "canonical"
    add_raw_node(canon, "owner_doc1.md", raw_node("owner/doc1.md#p0003", "   "))
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "empty_text") == 1
    assert "empty_text_frac" in rep["verdict"]["threshold_breaches"]  # 1/4 > 2%


def _doc_with_asset(png: Path) -> CanonicalDocument:
    doc = make_doc("owner/doc1.md", GOOD_PARAS)
    doc.assets["pic-001.png"] = str(png)  # save_document копирует файл в assets/
    doc.add_node(
        logical_node_id="owner/doc1.md#img001", node_type="image",
        parent_id="owner/doc1.md#root", heading_path=[], text="Изображение. схема",
        provenance=Provenance(adapter="t", source_id="owner/doc1.md"),
        asset="pic-001.png",
    )
    return doc


def test_orphan_asset_file_warning(tmp_path):
    png = tmp_path / "src-pic.png"
    png.write_bytes(b"png")
    data = write_canon(tmp_path, [_doc_with_asset(png)])
    adir = data / "canonical" / "owner_doc1.md" / "assets"
    assert (adir / "pic-001.png").exists()  # ассет реально сохранён
    (adir / "pic-002.png").write_bytes(b"png")  # лишний файл без узла
    rc, rep = run_audit(data, "--no-codes")
    assert detector(rep, "asset_file_orphan") == 1
    assert rc == 0  # warning ниже порога --max-orphan-assets=10
    assert rep["verdict"]["ok"] is True


def test_orphan_assets_over_threshold_red(tmp_path):
    png = tmp_path / "src-pic.png"
    png.write_bytes(b"png")
    data = write_canon(tmp_path, [_doc_with_asset(png)])
    adir = data / "canonical" / "owner_doc1.md" / "assets"
    for i in range(12):  # 12 сирот (имена не пересекаются с реальным pic-001) > порога 10
        (adir / f"pic-1{i:02d}.png").write_bytes(b"png")
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "asset_file_orphan") == 12
    assert "orphan_assets" in rep["verdict"]["threshold_breaches"]


def test_asset_file_missing_hard(tmp_path):
    doc = make_doc("owner/doc1.md", GOOD_PARAS)
    doc.add_node(
        logical_node_id="owner/doc1.md#img001", node_type="image",
        parent_id="owner/doc1.md#root", heading_path=[], text="Изображение. схема",
        provenance=Provenance(adapter="t", source_id="owner/doc1.md"),
        asset="pic-099.png",  # узел есть, файла нет
    )
    data = write_canon(tmp_path, [doc])
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "asset_file_missing") == 1
    assert "pic-099.png" in examples(rep, "asset_file_missing")[0]


def test_duplicate_document_dirs_and_nvid_cross_doc(tmp_path):
    """Каталоги-близнецы одного document_id (легаси-имена до фикса 28.09):
    дубли document_id + перекрёстные дубли logical_node_id/node_version_id."""
    data = write_canon(
        tmp_path,
        [make_doc("owner/doc1.md", GOOD_PARAS), make_doc("owner/twin.md", GOOD_PARAS)],
    )
    canon = data / "canonical"
    # второй каталог = копия первого: те же document_id, logical_node_id,
    # пересчитанные node_version_id (как у реальных близнецов на проде)
    twin_nodes = read_nodes(canon, "owner_twin.md")
    for n in twin_nodes:
        n["logical_node_id"] = n["logical_node_id"].replace("owner/twin.md", "owner/doc1.md")
        n["node_version_id"] = CanonicalNode.make_version_id(
            "r1", n["logical_node_id"], n["text"]
        )
    rewrite_nodes(canon, "owner_twin.md", twin_nodes)
    mpath = canon / "owner_twin.md" / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest["document_id"] = "owner/doc1.md"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "duplicate_document_id_dirs") == 1
    assert detector(rep, "lnid_duplicate_cross_doc") >= 1
    assert detector(rep, "nvid_duplicate_cross_doc") >= 1  # коллизия ключа индекса


def test_version_id_missing_and_mismatch(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    canon = data / "canonical"
    nodes = read_nodes(canon, "owner_doc1.md")
    nodes[1]["node_version_id"] = ""  # без version id
    nodes[2]["node_version_id"] = "0" * 64  # не сходится с пересчётом
    rewrite_nodes(canon, "owner_doc1.md", nodes)
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "node_version_id_missing") == 1
    assert detector(rep, "version_id_mismatch") == 1


def test_heading_path_broken_and_parent_missing(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    canon = data / "canonical"
    add_raw_node(canon, "owner_doc1.md",
                 raw_node("owner/doc1.md#p0003",
                          "Третий абзац для проверки битого heading_path в аудите.",
                          heading_path=["Глава", ""]))  # пустой элемент = битый путь
    nodes = read_nodes(canon, "owner_doc1.md")
    nodes[1]["heading_path"] = ["Глава", ""]  # испортили существующий узел
    nodes[2]["parent_id"] = "owner/doc1.md#nope"  # родителя нет
    rewrite_nodes(canon, "owner_doc1.md", nodes)
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "heading_path_broken") == 2
    assert detector(rep, "parent_missing") == 1
    assert "owner/doc1.md#nope" in examples(rep, "parent_missing")[0]


def test_empty_heading_path_is_warning_only(tmp_path):
    """Пустой heading_path у не-root легален (docling: абзацы до первого
    заголовка) - warning; порог 0.30 доли не превышен (1 из 5) -> exit 0."""
    doc = make_doc("owner/doc1.md", [
        *GOOD_PARAS,
        "Третий абзац добавлен, чтобы доля пустых путей была ниже порога.",
        "Четвёртый абзац для того же знаменателя доли.",
    ])
    data = write_canon(tmp_path, [doc])
    canon = data / "canonical"
    nodes = read_nodes(canon, "owner_doc1.md")
    nodes[1]["heading_path"] = []
    rewrite_nodes(canon, "owner_doc1.md", nodes)
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 0
    assert detector(rep, "heading_path_empty") == 1
    assert "heading_path_frac" not in rep["verdict"]["threshold_breaches"]


def test_revision_breakage(tmp_path):
    doc = make_doc("owner/doc1.md", GOOD_PARAS)
    data = write_canon(tmp_path, [doc])
    ddir = data / "canonical" / "owner_doc1.md"
    # обрыв: каталог ревизии с узлами, но без манифеста
    (ddir / "revision-r2").mkdir()
    (ddir / "revision-r2" / "nodes.jsonl").write_text(
        '{"logical_node_id": "x"}\n', encoding="utf-8")
    # обратный обрыв: манифест без узлов (ревизия r3)
    (ddir / "revision-r3").mkdir()
    (ddir / "revision-r3" / "manifest.json").write_text("{}", encoding="utf-8")
    # previous_revision в никуда
    mpath = ddir / "manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    manifest["previous_revision"] = "r9"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rc, rep = run_audit(data, "--no-codes")
    assert rc == 1
    assert detector(rep, "revision_nodes_without_manifest") == 1
    assert detector(rep, "revision_incomplete") == 1
    assert detector(rep, "previous_revision_missing") == 1


def _write_index(data: Path, node_meta: dict) -> None:
    """data/index/meta.json с заданной метой (статусы - как в реальном индексе)."""
    index_dir = data / "index"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text(
        json.dumps({"node_meta": node_meta}), encoding="utf-8")


def test_count_mismatch_index(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    # в индексе 5 узлов документа, в каноне 2 индексируемых (не-root) -> расхождение
    _write_index(data, {
        f"n{i}": {"document_id": "owner/doc1.md", "status": "current"} for i in range(5)
    })
    rc, rep = run_audit(data, "--no-codes", "--index")
    assert rc == 1
    assert detector(rep, "count_mismatch_index") == 1
    assert "канон 2" in examples(rep, "count_mismatch_index")[0]
    assert "индекс 5" in examples(rep, "count_mismatch_index")[0]


def test_count_match_index_green(tmp_path):
    """Позитив counts: ровно 2 не-root узла в индексе и каноне - ЗЕЛЁНЫЙ
    (root-узлы в сверку не входят, superseded не считаются)."""
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    _write_index(data, {
        "n1": {"document_id": "owner/doc1.md", "status": "current"},
        "n2": {"document_id": "owner/doc1.md", "status": "current"},
        "sup": {"document_id": "owner/doc1.md", "status": "superseded"},
    })
    rc, rep = run_audit(data, "--no-codes", "--index")
    assert rc == 0, rep["detectors"]
    assert detector(rep, "count_mismatch_index") == 0
    assert detector(rep, "doc_not_in_index") == 0


def test_doc_not_in_index_warning(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    _write_index(data, {})
    rc, rep = run_audit(data, "--no-codes", "--index")
    assert rc == 0  # warning не роняет exit
    assert detector(rep, "doc_not_in_index") == 1
    assert rep["verdict"]["ok"] is True


def test_count_mismatch_jobs(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    uploads = data / "uploads"
    uploads.mkdir()
    # джоб пишет len(doc.nodes) - с root, т.е. 3; пишем 99 -> расхождение
    (uploads / "ingest-jobs.jsonl").write_text(
        json.dumps({"job_id": "j1", "status": "done",
                    "document_id": "owner/doc1.md", "nodes": 99}) + "\n",
        encoding="utf-8",
    )
    rc, rep = run_audit(data, "--no-codes", "--jobs")
    assert rc == 1
    assert detector(rep, "count_mismatch_jobs") == 1


def test_json_report_and_usage_error(tmp_path):
    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    out = data / "report.json"
    rc = ingest_audit.main([
        "--data", str(data), "--no-codes", "--json", str(out), "--json-documents",
    ])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["documents"], "в JSON-отчёте должны быть построчные записи документов"
    assert rep["documents"][0]["document_id"] == "owner/doc1.md"
    # использование: каталог без canonical* -> exit 2
    with pytest.raises(SystemExit) as ei:
        ingest_audit.main(["--data", str(tmp_path / "nowhere")])
    assert ei.value.code == 2


def test_iter_documents_dedups_duplicate_document_id(tmp_path, capsys):
    """Фикс K13 (canonical/store): два каталога одного document_id (легаси-близнецы
    до фикса имени 28.09) -> iter_documents отдаёт документ ОДИН раз, иначе
    build_index двойно-индексирует контент (прецедент прода: НК РФ 46 355 узлов
    в индексе при ~24 000 в каноне). Без фикса тест падает: docs было бы 2."""
    import shutil

    from veles.canonical.store import iter_documents

    data = write_canon(tmp_path, [make_doc("owner/doc1.md", GOOD_PARAS)])
    canon = data / "canonical"
    shutil.copytree(canon / "owner_doc1.md", canon / "owner_doc1_twin_legacy_name")
    docs = list(iter_documents(root=canon))
    assert len(docs) == 1  # дубль пропущен, документ не задвоен
    assert docs[0].document_id == "owner/doc1.md"
    assert "дубль document_id" in capsys.readouterr().err  # предупреждение в stderr

    # ревизии живого каталога по-прежнему отдаются (дедуп только по document_id)
    rdir = canon / "owner_doc1.md" / "revision-r1"
    rdir.mkdir()
    archive_manifest = json.loads(
        (canon / "owner_doc1.md" / "manifest.json").read_text(encoding="utf-8"))
    archive_manifest["status"] = "superseded"  # как делает _archive_current_revision
    (rdir / "manifest.json").write_text(
        json.dumps(archive_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(canon / "owner_doc1.md" / "nodes.jsonl", rdir / "nodes.jsonl")
    with_rev = list(iter_documents(root=canon, include_revisions=True))
    assert len(with_rev) == 2  # текущая + архивная ревизия живого каталога
    assert sum(1 for d in with_rev if d.status == "superseded") == 1
    capsys.readouterr()


def test_codex_detectors_on_synthetic_code(tmp_path):
    """Специфика кодексов: статьи узнаваемы, «утратил силу» считается,
    разрыв нумерации и перезапуск нумерации находятся."""
    doc = make_doc("docs/Кодекс проверочный.pdf", [
        "Статья 1. Первая статья кодекса с содержательным текстом нормы.",
        "Статья 2. Вторая статья кодекса с содержательным текстом нормы.",
        "Статья 5. Пятая статья: между 2 и 5 разрыв нумерации (3 и 4 нет).",
        "Статья 1. Снова первая статья - новая часть, нумерация началась заново.",
        "Статья 2. После перезапуска нумерация продолжается без разрыва.",
        "Статья 3. Утратила силу.",
    ], title="Кодекс проверочный")
    data = write_canon(tmp_path, [doc])
    rc, rep = run_audit(data)  # кодексы по умолчанию включены
    assert rc == 0
    assert len(rep["codes"]) == 1
    c = rep["codes"][0]
    assert c["article_nodes"] == 6  # все 6 узлов содержат «статья N»
    assert c["article_share"] == 1.0
    assert c["lost_nodes"] == 1 and c["empty_lost_nodes"] == 1
    assert c["numbering_gaps"] == 1  # разрыв 2 -> 5; после перезапуска разрывов нет
    assert "нет ст.3..4" in c["gap_examples"][0]
    assert c["numbering_restarts"] == 1  # 5 -> 1
