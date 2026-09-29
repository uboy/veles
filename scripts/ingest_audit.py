#!/usr/bin/env python3
"""Аудит целостности канона veles (K13): «сохранялись данные в базу правильно».

Read-only обход канона (каталоги <VELES_DATA>/canonical* - те же корни, что
читает фасад; можно передать и один каталог канона). На каждый документ
проверяется:

  узлы всего / пустые тексты;
  дубли logical_node_id (в пределах документа и между документами одного корня);
  дубли node_version_id между документами (это коллизия ключа мета индекса);
  битые heading_path (не список / пустые элементы; пустой путь у не-root -
    warning: у docling легальны абзацы до первого заголовка);
  битые ссылки parent_id (родителя нет среди узлов документа);
  orphan-ассеты в обе стороны (файл без узла - warning, узел без файла - ошибка);
  узлы без node_version_id и с node_version_id, не сходящимся с пересчётом
    sha256(revision_id|logical_node_id|текст) - формула CanonicalNode;
  ревизии: revision-r*/ с узлами, но без manifest.json (обрыв архивации);
    previous_revision, указывающий на несуществующую ревизию;
  расхождение counts: манифест схемы v3.0 не хранит число узлов, поэтому пункт
    сверяется по внешним источникам счётчиков: (а) --index: индексируемые узлы
    документа (не-root, не superseded) в meta.json/overlay индекса против
    canonical-корня - индекс строится из него (первый каталог при каталогах-
    близнецах, как у iter_documents после дедупа d135953; копии в прочих
    canonical*-корнях не суммируются - иначе зеркальные документы давали бы
    ложное расхождение); документ в индексе без каталога в canonical-корне -
    отдельная находка (дрейф канон/индекс);
    (б) --jobs: nodes последнего done-джоба (len(doc.nodes), с root)
    в uploads/ingest-jobs.jsonl против канона;
  два каталога одного корня с одним document_id (legacy-дубли имён до фикса
    28.09) - при пересборке такой документ индексируется дважды;
  document_id в нескольких корнях canonical* (легаси-копии) - информационно.

Специфика кодексов (--codes, по умолчанию вкл): для документов с «кодекс» в
названии - узнаваемость статей (regexp «статья N»), доля «Утратил силу»,
средняя/медианная длина узла, разрывы нумерации статей в документе.

Пороги -> exit-code: 0 - чисто; 1 - есть структурные ошибки или превышены
пороги warnings; 2 - ошибка использования. Скрипт stdlib-only (numpy нужен
только для чтения overlay при --index; его отсутствие - warning, не падение).
Канон не изменяется: только чтение.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# детекторы-ошибки: каждая находка -> exit 1
HARD_DETECTORS = (
    "manifest_unreadable",
    "nodes_unreadable",
    "doc_dir_without_manifest",
    "lnid_duplicate_in_doc",
    "lnid_duplicate_cross_doc",
    "nvid_duplicate_cross_doc",
    "node_version_id_missing",
    "version_id_mismatch",
    "parent_missing",
    "heading_path_broken",
    "asset_file_missing",
    "revision_nodes_without_manifest",
    "previous_revision_missing",
    "duplicate_document_id_dirs",
    "count_mismatch_index",
    "count_mismatch_jobs",
)
# детекторы-предупреждения: exit 1 только при превышении порога
WARN_DETECTORS = (
    "empty_text",
    "heading_path_empty",
    "asset_file_orphan",
    "revision_incomplete",
    "legacy_manifest",
    "doc_not_in_index",
)

_ARTICLE_RE = re.compile(r"статья\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_LOST_RE = re.compile(r"утратил(?:а|и)?\s+силу", re.IGNORECASE)
_CODEX_RE = re.compile(r"кодекс", re.IGNORECASE)


def _version_id(revision_id: str, logical_node_id: str, text: str) -> str:
    """Та же формула, что CanonicalNode.make_version_id (дублируется намеренно:
    аудит stdlib-only и не зависит от версии установленного пакета)."""
    normalized = " ".join(text.split())
    return hashlib.sha256(
        f"{revision_id}|{logical_node_id}|{normalized}".encode("utf-8")
    ).hexdigest()


# ------------------------------------------------------------------ locating

def find_canon_roots(data: Path) -> list[Path]:
    """Каталоги канона: <data>/canonical* (как фасад) либо сам каталог, если
    передан канон напрямую (внутри лежат каталоги документов)."""
    data = Path(data)
    if data.name.startswith("canonical"):
        return [data]
    roots = sorted(p for p in data.glob("canonical*") if p.is_dir())
    if roots:
        return roots
    print(f"не найдено canonical* в {data}", file=sys.stderr)
    raise SystemExit(2)


def iter_doc_dirs(root: Path):
    """Каталоги документов корня: подкаталоги с manifest.json и/или nodes.jsonl
    (есть узлы, но нет манифеста - обрыв сохранения, репортится; пустой каталог
    игнорируется)."""
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        has_manifest = (d / "manifest.json").exists()
        has_nodes = (d / "nodes.jsonl").exists()
        if has_manifest or has_nodes:
            yield d, has_manifest, has_nodes


# ------------------------------------------------------------- index / jobs

def _count_nodes_in_index(data: Path) -> dict[str, int] | None:
    """document_id -> число индексируемых узлов в индексе (meta.json снапшот +
    overlay.npz дельты; superseded/deleted не считаются - как их видит поиск).
    None - индекса нет."""
    index_dir = Path(data) / "index"
    meta_path = index_dir / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    counts: Counter = Counter()
    for m in (meta.get("node_meta") or {}).values():
        if not isinstance(m, dict) or m.get("status", "current") in ("superseded", "deleted"):
            continue
        counts[m.get("document_id") or "(без документа)"] += 1
    try:  # дельта инкрементальных загрузок поверх снапшота
        import numpy as np

        overlay = index_dir / "overlay.npz"
        if overlay.exists():
            with np.load(overlay, allow_pickle=False) as npz:
                ometa = json.loads(npz["meta"].tobytes().decode("utf-8"))
            for m in (ometa.get("nodes") or {}).values():
                if m.get("status", "current") in ("superseded", "deleted"):
                    continue
                counts[m.get("document_id") or "(без документа)"] += 1
    except ImportError:
        print("предупреждение: numpy недоступен - overlay индекса не сверен",
              file=sys.stderr)
    return dict(counts)


def _count_nodes_in_jobs(data: Path) -> dict[str, int] | None:
    """document_id -> nodes последнего done-джоба очереди (K4)."""
    ledger = Path(data) / "uploads" / "ingest-jobs.jsonl"
    if not ledger.exists():
        return None
    last: dict[str, dict] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("status") == "done" and rec.get("document_id"):
            last[rec["document_id"]] = rec
    return {did: int(rec.get("nodes") or 0) for did, rec in last.items()}


# --------------------------------------------------------------- audit core

def _found(state: dict, detector: str, example: str) -> None:
    state[detector] = state.get(detector, 0) + 1
    examples = state.setdefault("_examples", {}).setdefault(detector, [])
    if len(examples) < state["_max_examples"]:
        examples.append(example)


def audit_document(d: Path, root_name: str, *, has_manifest: bool, has_nodes: bool,
                   state: dict) -> dict:
    """Аудит одного каталога документа. Находки дописываются в state
    (счётчики + примеры), сводка по узлам - в state['_docsum']."""
    row = {
        "root": root_name,
        "dir": d.name,
        "document_id": "",
        "revision_id": None,
        "nodes": 0,
        "indexable_nodes": 0,
        "empty_text": 0,
        "dup_lnid_in_doc": 0,
        "bad_heading_path": 0,
        "empty_heading_path": 0,
        "orphan_assets": 0,
    }
    manifest: dict = {}
    if has_manifest:
        try:
            manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            _found(state, "manifest_unreadable", f"{root_name}/{d.name}: {e}")
    else:
        _found(state, "doc_dir_without_manifest", f"{root_name}/{d.name}")
    row["document_id"] = str(manifest.get("document_id") or "")
    row["revision_id"] = manifest.get("revision_id")
    revision_id = str(manifest.get("revision_id") or "r1")
    if manifest and not manifest.get("ingested_at"):
        _found(state, "legacy_manifest", f"{root_name}/{d.name}")

    # ревизии: архивы-обрывки и битые ссылки previous_revision
    archived: set[str] = set()
    for rdir in sorted(d.glob("revision-r*")):
        m = re.fullmatch(r"revision-(r\d+)", rdir.name)
        if not m:
            continue
        archived.add(m.group(1))
        r_manifest = (rdir / "manifest.json").exists()
        r_nodes = (rdir / "nodes.jsonl").exists()
        if r_nodes and not r_manifest:
            _found(state, "revision_nodes_without_manifest",
                   f"{root_name}/{d.name}/{rdir.name}")
        elif r_manifest and not r_nodes:
            _found(state, "revision_incomplete", f"{root_name}/{d.name}/{rdir.name}")
    prev = manifest.get("previous_revision")
    if prev and prev not in archived:
        _found(state, "previous_revision_missing",
               f"{root_name}/{d.name}: previous_revision={prev}")

    # узлы
    if has_nodes:
        try:
            lines = (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError as e:
            _found(state, "nodes_unreadable", f"{root_name}/{d.name}: {e}")
            lines = []
        lnids: set[str] = set()
        lnid_dup: Counter = Counter()
        parents: set[str] = set()
        assets_ref: set[str] = set()
        for line in lines:
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except ValueError as e:
                _found(state, "nodes_unreadable",
                       f"{root_name}/{d.name}: битая JSON-строка: {e}")
                continue
            row["nodes"] += 1
            if node.get("node_type") != "root" and node.get("status") != "superseded":
                row["indexable_nodes"] += 1
            lnid = str(node.get("logical_node_id") or "")
            text = str(node.get("text") or "")
            if not text.strip():
                row["empty_text"] += 1
                _found(state, "empty_text", f"{root_name}/{d.name}#{lnid}")
            if lnid in lnids:
                lnid_dup[lnid] += 1
            lnids.add(lnid)
            if not node.get("node_version_id"):
                _found(state, "node_version_id_missing", f"{root_name}/{d.name}#{lnid}")
            elif node.get("node_version_id") != _version_id(revision_id, lnid, text):
                _found(state, "version_id_mismatch", f"{root_name}/{d.name}#{lnid}")
            hp = node.get("heading_path")
            if node.get("node_type") != "root":
                if not hp:
                    row["empty_heading_path"] += 1
                    _found(state, "heading_path_empty", f"{root_name}/{d.name}#{lnid}")
                elif not isinstance(hp, list) or any(
                    not isinstance(x, str) or not x.strip() for x in hp
                ):
                    row["bad_heading_path"] += 1
                    _found(state, "heading_path_broken",
                           f"{root_name}/{d.name}#{lnid}: {hp!r}")
            if node.get("parent_id"):
                parents.add(str(node["parent_id"]))
            if node.get("asset"):
                assets_ref.add(str(node["asset"]))
        for lnid, n in lnid_dup.items():
            row["dup_lnid_in_doc"] += n
            _found(state, "lnid_duplicate_in_doc", f"{root_name}/{d.name}#{lnid} x{n + 1}")
        # parent_id вторым проходом: родитель может идти позже узла в файле
        for parent in sorted(parents):
            if parent not in lnids:
                _found(state, "parent_missing", f"{root_name}/{d.name}#{parent}")

        # ассеты в обе стороны
        adir = d / "assets"
        on_disk: set[str] = set()
        if adir.is_dir():
            on_disk = {p.name for p in adir.iterdir() if p.is_file()}
        for name in sorted(on_disk - assets_ref):
            row["orphan_assets"] += 1
            _found(state, "asset_file_orphan", f"{root_name}/{d.name}/assets/{name}")
        for name in sorted(assets_ref - on_disk):
            _found(state, "asset_file_missing", f"{root_name}/{d.name}/assets/{name}")

    docsum = state["_docsum"]
    for key in ("nodes", "indexable_nodes", "empty_text", "dup_lnid_in_doc",
                "bad_heading_path", "empty_heading_path", "orphan_assets"):
        docsum[key] += row[key]
    return row


def scan_cross_doc_duplicates(root: Path, root_name: str, state: dict) -> None:
    """Дубли logical_node_id / node_version_id МЕЖДУ документами одного корня
    (второй проход по узлам; в пределах документа ловит audit_document)."""
    seen_l: dict[str, str] = {}
    seen_v: dict[str, str] = {}
    for d, _hm, has_nodes in iter_doc_dirs(root):
        if not has_nodes:
            continue
        for line in (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                node = json.loads(line)
            except ValueError:
                continue
            lnid = str(node.get("logical_node_id") or "")
            nvid = str(node.get("node_version_id") or "")
            if lnid and lnid in seen_l and seen_l[lnid] != d.name:
                _found(state, "lnid_duplicate_cross_doc",
                       f"{lnid}: {seen_l[lnid]} / {d.name}")
            if lnid:
                seen_l.setdefault(lnid, d.name)
            if nvid and nvid in seen_v and seen_v[nvid] != d.name:
                _found(state, "nvid_duplicate_cross_doc",
                       f"{nvid[:16]}…: {seen_v[nvid]} / {d.name}")
            if nvid:
                seen_v.setdefault(nvid, d.name)


# ------------------------------------------------------------------ кодексы

def audit_codex(d: Path) -> dict | None:
    """Специфика кодекса для одного каталога документа; None - не кодекс."""
    try:
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    title = str(manifest.get("title") or "")
    if not _CODEX_RE.search(title):
        return None
    nodes = article_like = lost = empty_lost = 0
    lens: list[int] = []
    prev: float | None = None  # предыдущий номер статьи в текущей последовательности
    restarts = 0
    gaps: list[str] = []
    for line in (d / "nodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            node = json.loads(line)
        except ValueError:  # ревью K13 P2: битая строка не валит весь аудит
            continue
        if node.get("node_type") == "root":
            continue
        text = str(node.get("text") or "")
        nodes += 1
        lens.append(len(text))
        m = _ARTICLE_RE.search(text[:200])
        if m:
            article_like += 1
            num = float(m.group(1))
            if prev is not None and num < prev:
                restarts += 1  # новая часть/раздел: нумерация началась заново
            elif prev is not None and num > prev + 1 and len(gaps) < 50:
                gaps.append(f"после ст.{prev:g} нет ст.{prev + 1:g}..{num - 1:g}")
            prev = num
        if _LOST_RE.search(text):
            lost += 1
            if len(text.strip()) < 40:
                empty_lost += 1
    lens_sorted = sorted(lens)
    avg = sum(lens) / len(lens) if lens else 0
    median = lens_sorted[len(lens) // 2] if lens else 0
    return {
        "root": d.parent.name,
        "document_id": manifest.get("document_id"),
        "title": title,
        "nodes": nodes,
        "article_nodes": article_like,
        "article_share": round(article_like / nodes, 4) if nodes else 0,
        "lost_nodes": lost,
        "lost_share": round(lost / nodes, 4) if nodes else 0,
        "empty_lost_nodes": empty_lost,
        "avg_len": round(avg, 1),
        "median_len": median,
        "numbering_restarts": restarts,
        "numbering_gaps": len(gaps),
        "gap_examples": gaps[:10],
    }


# -------------------------------------------------------------------- main

def run_audit(args: argparse.Namespace) -> dict:
    data = Path(args.data)
    roots = find_canon_roots(data)
    if args.index and len(roots) == 1 and roots[0].name != "canonical":
        # ревью K13 P2: индекс строится из canonical-корня; сверка против
        # одиночного корня с чужим именем даёт массовые ложные «отсутствует»
        print("предупреждение: --index с одиночным корнем без имени canonical - "
              "сверка с индексом даст ложные расхождения", file=sys.stderr)
    state: dict = {"_max_examples": args.max_examples, "_docsum": Counter()}
    index_counts = _count_nodes_in_index(data) if args.index else None
    jobs_counts = _count_nodes_in_jobs(data) if args.jobs else None

    doc_rows: list[dict] = []
    root_stats: list[dict] = []
    # Индекс строится из canonical-корня (veles.config.CANONICAL_DIR), поэтому
    # сверка с индексом идёт против ПЕРВОГО каталога документа в canonical
    # (sorted-порядок - та же семантика, что у iter_documents после дедупа
    # d135953). Копии документа в других корнях (canonical-docling и т.п.) не
    # суммируются: иначе каждый зеркальный документ давал бы ложное
    # расхождение (на проде 29.09 таких ложных - 88).
    canon_canonical: dict[str, int] = {}   # document_id -> не-root узлы canonical-корня (первый каталог)
    canon_rest: dict[str, int] = {}        # document_id -> не-root узлы прочих корней (информация)
    canon_nodes_all: dict[str, int] = {}   # document_id -> все узлы (для джобов)
    canon_roots_of: dict[str, set[str]] = {}

    for root in roots:
        docs = nodes_total = 0
        doc_id_dirs: Counter = Counter()
        for d, has_manifest, has_nodes in iter_doc_dirs(root):
            row = audit_document(
                d, root.name, has_manifest=has_manifest, has_nodes=has_nodes,
                state=state,
            )
            doc_id_dirs[row["document_id"]] += 1
            doc_rows.append(row)
            docs += 1
            nodes_total += row["nodes"]
            did = row["document_id"] or f"(без id:{d.name})"
            if root.name == "canonical":
                # первый каталог в sorted-порядке побеждает (дедуп iter_documents)
                if did not in canon_canonical:
                    canon_canonical[did] = row["indexable_nodes"]
            else:
                canon_rest[did] = canon_rest.get(did, 0) + row["indexable_nodes"]
            canon_nodes_all[did] = canon_nodes_all.get(did, 0) + row["nodes"]
            canon_roots_of.setdefault(did, set()).add(root.name)
        # дубли document_id ВНУТРИ корня (каталоги-близнецы legacy-имён)
        for doc_id, n in doc_id_dirs.items():
            if n > 1 and doc_id:
                _found(state, "duplicate_document_id_dirs",
                       f"{root.name}: {doc_id} x{n} каталогов")
        scan_cross_doc_duplicates(root, root.name, state)
        root_stats.append({"root": root.name, "documents": docs, "nodes": nodes_total})

    cross_root = {
        did: sorted(rs) for did, rs in sorted(canon_roots_of.items())
        if len(rs) > 1 and not did.startswith("(без id")
    }

    # сверка counts с индексом и журналом джобов
    if index_counts is not None:
        for did in sorted(canon_canonical):
            idx_n = index_counts.get(did)
            if idx_n is None:
                _found(state, "doc_not_in_index",
                       f"{did}: в каноне {canon_canonical[did]}, в индексе 0")
            elif idx_n != canon_canonical[did]:
                _found(state, "count_mismatch_index",
                       f"{did}: канон {canon_canonical[did]} (не-root, canonical), индекс {idx_n}")
        for did in sorted(set(index_counts) - set(canon_canonical)):
            note = ""
            if did in canon_rest:
                note = f" (копия только в прочих корнях: {canon_rest[did]} узлов)"
            _found(state, "count_mismatch_index",
                   f"{did}: в индексе {index_counts[did]}, в canonical-корне отсутствует{note}")
    if jobs_counts is not None:
        for did, jnodes in sorted(jobs_counts.items()):
            canon_n = canon_nodes_all.get(did, 0)
            if canon_n and canon_n != jnodes:
                _found(state, "count_mismatch_jobs",
                       f"{did}: канон {canon_n} (с root), джоб {jnodes}")

    # кодексы
    codes: list[dict] = []
    if args.codes:
        for root in roots:
            for d, has_manifest, has_nodes in iter_doc_dirs(root):
                if has_manifest and has_nodes:
                    rep = audit_codex(d)
                    if rep:
                        codes.append(rep)

    totals = {k: int(state.get(k, 0)) for k in HARD_DETECTORS + WARN_DETECTORS}
    docsum = dict(state["_docsum"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": str(data),
        "roots": root_stats,
        "documents_total": sum(r["documents"] for r in root_stats),
        "nodes_total": docsum.get("nodes", 0),
        "document_node_sums": {
            "empty_text": docsum.get("empty_text", 0),
            "duplicate_lnid_in_doc": docsum.get("dup_lnid_in_doc", 0),
            "broken_heading_path": docsum.get("bad_heading_path", 0),
            "empty_heading_path": docsum.get("empty_heading_path", 0),
            "orphan_asset_files": docsum.get("orphan_assets", 0),
        },
        "cross_root_duplicate_documents": cross_root,
        "detectors": totals,
        "examples": state.get("_examples", {}),
        "codes": codes,
        "thresholds": {
            "max_empty_frac": args.max_empty_frac,
            "max_heading_empty_frac": args.max_heading_empty_frac,
            "max_orphan_assets": args.max_orphan_assets,
        },
        "documents": doc_rows if args.json_documents else [],
    }

    # вердикт по порогам
    hard = {k: v for k, v in totals.items() if k in HARD_DETECTORS and v > 0}
    warn = {k: v for k, v in totals.items() if k in WARN_DETECTORS and v > 0}
    breached: dict[str, object] = {}
    total_nodes = docsum.get("nodes", 0)
    if total_nodes:
        empty_frac = docsum.get("empty_text", 0) / total_nodes
        if empty_frac > args.max_empty_frac:
            breached["empty_text_frac"] = round(empty_frac, 4)
        hp_frac = (docsum.get("empty_heading_path", 0)
                   + docsum.get("bad_heading_path", 0)) / total_nodes
        if hp_frac > args.max_heading_empty_frac:
            breached["heading_path_frac"] = round(hp_frac, 4)
    if docsum.get("orphan_assets", 0) > args.max_orphan_assets:
        breached["orphan_assets"] = docsum["orphan_assets"]
    report["verdict"] = {
        "hard": hard,
        "warnings": warn,
        "threshold_breaches": breached,
        "ok": not hard and not breached,
    }
    return report


def print_summary(rep: dict) -> None:
    print(f"канон: {rep['data']}")
    for r in rep["roots"]:
        print(f"  корень {r['root']}: документов {r['documents']}, узлов {r['nodes']}")
    print(f"всего: документов {rep['documents_total']}, узлов {rep['nodes_total']}")
    ds = rep["document_node_sums"]
    print(
        f"пустые тексты: {ds['empty_text']}; дубли logical_node_id в документе: "
        f"{ds['duplicate_lnid_in_doc']}; битых heading_path: {ds['broken_heading_path']}; "
        f"пустых heading_path (не-root): {ds['empty_heading_path']}; "
        f"orphan-файлов ассетов: {ds['orphan_asset_files']}"
    )
    crd = rep["cross_root_duplicate_documents"]
    print(f"document_id в нескольких корнях: {len(crd)}")
    active = {k: v for k, v in rep["detectors"].items() if v}
    if active:
        print("детекторы:")
        for k, v in active.items():
            print(f"  {k}: {v}")
    v = rep["verdict"]
    for k, n in sorted(v["hard"].items()):
        ex = (rep["examples"].get(k) or ["-"])[0]
        print(f"  ОШИБКА {k}: {n}; пример: {ex}")
    for k, n in sorted(v["threshold_breaches"].items()):
        print(f"  ПОРОГ {k}: {n}")
    if v["ok"]:
        print("ВЕРДИКТ: ЗЕЛЁНЫЙ (структурных ошибок нет, пороги не превышены)")
    else:
        print("ВЕРДИКТ: КРАСНЫЙ (см. ошибки/пороги выше)")
    if rep["codes"]:
        print(f"кодексы ({len(rep['codes'])}):")
        for c in rep["codes"]:
            print(
                f"  [{c['root']}] {c['title'][:44]}: узлов {c['nodes']}; статей "
                f"{c['article_nodes']} ({c['article_share']:.0%}); «утратил силу» "
                f"{c['lost_nodes']} ({c['lost_share']:.0%}, пустышек {c['empty_lost_nodes']}); "
                f"ср.длина {c['avg_len']:.0f} / медиана {c['median_len']} зн.; "
                f"разрывов {c['numbering_gaps']}, перезапусков {c['numbering_restarts']}"
            )
        for c in rep["codes"]:
            if c["gap_examples"]:
                print(f"  разрывы [{c['title'][:36]}]: " + "; ".join(c["gap_examples"][:5]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Аудит целостности канона veles (read-only; K13)"
    )
    ap.add_argument("--data", required=True,
                    help="корень VELES_DATA (аудируются все canonical*) или каталог канона")
    ap.add_argument("--json", dest="json_out", default="",
                    help="путь JSON-отчёта (не пишется при пустом)")
    ap.add_argument("--json-documents", action="store_true",
                    help="включить построчные записи документов в JSON-отчёт")
    ap.add_argument("--index", action="store_true",
                    help="сверить число узлов канона с индексом (<data>/index)")
    ap.add_argument("--jobs", action="store_true",
                    help="сверить с nodes последнего done-джоба (uploads/ingest-jobs.jsonl)")
    ap.add_argument("--no-codes", dest="codes", action="store_false",
                    help="не считать специфику кодексов")
    ap.add_argument("--max-examples", type=int, default=5,
                    help="примеров на детектор в отчёте (по умолчанию 5)")
    ap.add_argument("--max-empty-frac", type=float, default=0.02,
                    help="порог доли пустых текстов (0.02)")
    ap.add_argument("--max-heading-empty-frac", type=float, default=0.30,
                    help="порог доли пустых/битых heading_path (0.30)")
    ap.add_argument("--max-orphan-assets", type=int, default=10,
                    help="порог числа orphan-файлов ассетов (10)")
    args = ap.parse_args(argv)

    rep = run_audit(args)
    print_summary(rep)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON-отчёт: {args.json_out}")
    return 0 if rep["verdict"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
