"""CLI veles: scan-corpus | ingest | index | ask | add | delete | status.

Запуск: python -m veles <command> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .canonical.store import ingest_document, iter_documents
from .config import CANONICAL_DIR, CORPUS_ROOT, EMBED_DIM, EMBED_MODEL, INDEX_DIR
from .ingest.allowlist import SPECS, scan_and_copy
from .ingest.txt_md import TxtMdParserAdapter
from .indexing.bm25 import Bm25Index  # noqa: F401 (реэкспорт для типа)
from .indexing.embedder import SentenceTransformersProvider
from .indexing.indexer import build_index, load_index
from .indexing.vector_store import make_store
from .policy.answer import decide
from .policy.identifiers import extract_identifiers
from .retrieval.retriever import hybrid_search


def cmd_scan_corpus(args: argparse.Namespace) -> int:
    from .ingest.allowlist import SPECS

    spec = SPECS[args.spec]
    m = scan_and_copy(spec=spec, copy=not args.dry)
    print(f"манифест: {m.name} @ {m.created_at}")
    print(f"включено: {m.stats['included_files']} файлов ({m.stats['included_bytes']} байт), "
          f"скопировано сейчас: {m.stats['copied_now']}, чёрный список: {m.stats['excluded_by_blacklist']}, "
          f"вне allowlist: {m.stats['outside_allowlist']}")
    for e in m.included[:5]:
        print(f"  + {e.dest_rel} [{e.sha256[:12]}]")
    if len(m.included) > 5:
        print(f"  ... и ещё {len(m.included) - 5}")
    return 0


def _manifest_known_relpaths() -> set[str]:
    """Все dest_rel из манифестов allowlist-конвейера (любые корпусы)."""
    import json

    from .config import MANIFESTS_DIR

    known: set[str] = set()
    if MANIFESTS_DIR.exists():
        for mpath in sorted(MANIFESTS_DIR.glob("*.json")):
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - битый манифест не роняет ingest
                continue
            known.update(e["dest_rel"] for e in m.get("included", []))
    return known


def cmd_ingest(args: argparse.Namespace) -> int:
    parser = TxtMdParserAdapter()
    docs = 0
    nodes = 0
    skipped = 0
    updated = 0
    unchanged = 0
    corpus_dir = CORPUS_ROOT / "05_vyshka_academic" / "lectures"
    if not corpus_dir.is_dir():
        raise SystemExit(
            f"корпус не найден: {corpus_dir}\n"
            "источники в ingest не попадут сами - положите копии файлов в корень корпуса "
            "(env VELES_CORPUS_ROOT, по умолчанию ~/veles-data/corpus) и повторите"
        )
    known = set() if args.no_manifest_check else _manifest_known_relpaths()
    if not known and not args.no_manifest_check:
        raise SystemExit("манифесты allowlist пусты/отсутствуют - ingest запрещён (fail-closed)")
    for path in sorted(corpus_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        rel = path.relative_to(CORPUS_ROOT).as_posix()
        if not known and not args.no_manifest_check:
            raise SystemExit("манифесты allowlist пусты/отсутствуют - ingest запрещён (fail-closed)")
        if known and rel not in known:
            print(f"  ! пропущен (нет в allowlist-манифесте): {rel}")
            skipped += 1
            continue
        doc = parser.parse_file(path, corpus_root=CORPUS_ROOT)
        outcome = ingest_document(doc, by=args.by)
        docs += 1
        nodes += len(doc.nodes)
        if outcome.action == "updated":
            updated += 1
            print(f"  {doc.document_id}: обновлён до {outcome.revision_id}, "
                  f"предыдущая {outcome.previous_revision} сохранена")
        elif outcome.action == "unchanged":
            unchanged += 1
            print(f"  {doc.document_id}: без изменений")
        else:
            print(f"  {doc.document_id}: {len(doc.nodes)} узлов")
    print(f"итого: документов={docs} (новых={docs - updated - unchanged}, "
          f"обновлено={updated}, без изменений={unchanged}), узлов={nodes}, "
          f"пропущено мимо конвейера={skipped} -> {CANONICAL_DIR}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    docs = list(iter_documents())
    if not docs:
        print("канон пуст - сначала `ingest`")
        return 1
    print(f"эмбеддер: {EMBED_MODEL}")
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    store = make_store(prefer=args.backend)
    bundle = build_index(embedder, store)
    bundle.save(INDEX_DIR, embed_meta={"model": EMBED_MODEL, "backend": store.name})
    print(f"проиндексировано узлов: {len(bundle.node_meta)}; backend={store.name}; bm25={len(bundle.bm25.ids)} -> {INDEX_DIR}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not (INDEX_DIR / "meta.json").exists():
        print("индекс не собран - сначала `index`")
        return 1
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)
    qvec = embedder.embed_query(args.query)
    evidence = hybrid_search(bundle, qvec, args.query, top_k=args.top_k)
    decision = decide(
        evidence,
        query_identifiers=extract_identifiers(args.query),
        known_identifiers=bundle.identifiers,
    )
    print(f"решение: {decision.decision.upper()} (доверие {decision.confidence}; {decision.reason})")
    if decision.decision in ("answer", "warn_conflict"):
        top = decision.evidence[0]
        print(f"ответ (текст узла, {len(top.text)} знаков):\n{top.text}\n")
        print("цитаты:")
        for ev in decision.evidence[:3]:
            for c in ev.citations:
                hp = " > ".join(c.heading_path[-3:]) if c.heading_path else ""
                print(f"  - {c.human()} {('| ' + hp) if hp else ''}")
    if decision.conflicts:
        print("возможные противоречащие редакции:")
        for c in decision.conflicts:
            print(f"  - «{c['title']}» {c['document_id']} ({c['revision_id']}): {c['signal']}")
    print("diagnostics:")
    for ev in decision.evidence[:3]:
        print(f"  {ev.node_version_id[:40]} rrf={ev.rrf:.4f} dense={ev.dense_rank} bm25={ev.bm25_rank}")
    return 0


def build_embedder():
    """Фабрика эмбеддера для ask/add (отдельная функция - тесты подменяют стабом)."""
    return SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)


def cmd_add(args: argparse.Namespace) -> int:
    """Добавить один txt/md-файл в базу без пересборки индекса (K15, демо в lite).

    Тот же конвейер, что у загрузки через фасад (K4): parse -> canonical
    (ревизии K3) -> инкрементальный апсерт живого бандла + overlay (документ
    ищется сразу и переживает рестарт без пересборки). L0 кладётся в
    uploads/owner/<имя> - та же физика, что у фасада, поэтому `veles delete`
    снимает такой документ целиком. Синхронно: вернулся - можно спрашивать.
    """
    import os
    import shutil

    from . import config
    from .indexing.overlay import append_overlay
    from .ingest.queue import safe_filename, uploads_root, upsert_document, user_dir

    src = Path(args.file).expanduser()
    if not src.is_file():
        print(f"файл не найден: {src}")
        return 1
    if src.suffix.lower() not in (".txt", ".md"):
        print("поддерживаются форматы txt и md (pdf - через фасад: POST /v1/documents)")
        return 1
    if not (config.INDEX_DIR / "meta.json").exists():
        print("индекс не собран - сначала `veles index` (или install.sh --demo)")
        return 1

    directory = user_dir("")  # static-режим: владелец, как у фасада
    filename = safe_filename(src.name)
    if filename is None:
        print(f"недопустимое имя файла: {src.name}")
        return 1
    l0_dir = uploads_root() / directory
    l0_dir.mkdir(parents=True, exist_ok=True)
    l0_path = l0_dir / filename
    shutil.copyfile(src, l0_path)

    try:
        # corpus_root = корень uploads: document_id = "owner/<имя>", как у фасада
        doc = TxtMdParserAdapter().parse_file(l0_path, corpus_root=uploads_root())
    except Exception as e:  # noqa: BLE001 - битый файл: человеческая причина
        print(f"не удалось разобрать документ: {e}")
        return 1
    if args.title:
        doc.title = args.title
    by = args.by or os.environ.get("VELES_INGESTED_BY", "").strip() or "cli"
    outcome = ingest_document(doc, by=by)
    if outcome.action == "unchanged":
        print(f"{doc.document_id}: без изменений (такая версия уже в базе)")
        return 0

    bundle = load_index(config.INDEX_DIR)
    delta = upsert_document(bundle, doc, outcome, build_embedder())
    if delta is not None:
        append_overlay(
            config.INDEX_DIR, delta.nodes, delta.patches,
            delta.ids, delta.matrix, delta.bm25,
        )
    print(f"{doc.document_id}: {len(doc.nodes)} узлов "
          f"({outcome.action}, {outcome.revision_id}) - документ в поиске")
    print('проверка: veles ask "<вопрос по документу>"')
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    docs = list(iter_documents())
    all_revs = list(iter_documents(include_revisions=True))
    archived = len(all_revs) - len(docs)
    print(f"корпус: {CORPUS_ROOT}")
    print(f"канон: {len(docs)} документов, {sum(len(d.nodes) for d in docs)} узлов"
          + (f", архивных ревизий: {archived}" if archived else "")
          + f" ({CANONICAL_DIR})")
    print(f"индекс: {'есть' if (INDEX_DIR / 'meta.json').exists() else 'нет'} ({INDEX_DIR})")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Удаление документа (K10): канон (все ревизии и assets) -> trash, L0 -
    за ним, узлы - tombstone в индексе (overlay: живой фасад закроет документ
    после рестарта). Инженерный путь - тот же код удаления, что у DELETE
    /v1/documents (store.delete_document + tombstone_document); без --yes
    только показывает, что будет удалено."""
    import os

    from .canonical.store import delete_document, find_document_manifest, trash_root
    from .indexing.overlay import tombstone_document
    from .ingest.queue import discard_l0_file

    manifest = find_document_manifest(args.document_id)
    if manifest is None:
        print(f"документ не найден: {args.document_id}")
        return 1
    if not args.yes:
        print(f"будет удалён документ: {args.document_id}"
              f" ({manifest.get('title') or 'без названия'},"
              f" ревизия {manifest.get('revision_id') or 'r1'})")
        print("канон со всеми ревизиями и assets переедет в trash, L0 будет удалён,")
        print("узлы исчезнут из поиска. Подтвердите: --yes")
        return 1
    info = delete_document(args.document_id, by=os.environ.get("VELES_INGESTED_BY", "").strip() or "cli")
    if info is None:  # манифест видели, канон исчез (гонка)
        print(f"документ не найден: {args.document_id}")
        return 1
    l0_removed = discard_l0_file(args.document_id)
    nodes = tombstone_document(INDEX_DIR, args.document_id)
    print(f"удалён: {args.document_id} (ревизии: {', '.join(info['revisions'])})")
    print(f"канон в корзине: {trash_root() / info['trash'][0]}")
    print(f"L0: {'удалён' if l0_removed else 'не было'}")
    if nodes:
        print(f"узлов выведено из индекса: {nodes} (tombstone в overlay);"
              " работающий фасад закроет документ после рестарта")
    else:
        print("узлов в индексе не найдено (индекс без документа - ок)"
              if (INDEX_DIR / "meta.json").exists() else "индекс не собран - tombstone не нужен")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    from .api.app import create_app

    # fail-closed: ненулевой хост без bearer-токена открывает базу всей сети
    token = os.environ.get("VELES_API_TOKEN", "")
    loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if not loopback and not token:
        raise SystemExit(
            "отказ старта: не-loopback хост без VELES_API_TOKEN молча открывает базу "
            "в сеть (fail-closed). Задайте VELES_API_TOKEN или биндьте 127.0.0.1."
        )
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import run as run_mcp

    run_mcp()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="veles", description="Knowledge Core CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan-corpus", help="allowlist-скан Вышки -> копии в корпус + манифест")
    p.add_argument("--spec", default="module1", choices=sorted(SPECS),
                   help="module1 = 1 курс Модуль_1, um = Управление производством (2 курс)")
    p.add_argument("--dry", action="store_true", help="только манифест, без копирования")
    p.set_defaults(func=cmd_scan_corpus)

    p = sub.add_parser("ingest", help="корпус -> канон (manifest.json + nodes.jsonl)")
    p.add_argument("--by", default="", help="кто загружает (логин/почта); иначе env VELES_INGESTED_BY")
    p.add_argument("--no-manifest-check", action="store_true",
                   help="не сверять файлы с allowlist-манифестом (по умолчанию сверка включена)")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("index", help="канон -> L2 (dense + BM25)")
    p.add_argument("--backend", default="faiss", choices=["faiss", "numpy"])
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("serve", help="HTTP-фасад (/v1/knowledge/query + chat/completions)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8400)
    p.set_defaults(func=cmd_serve)

    sub.add_parser("mcp", help="MCP-сервер (stdio) для кодинг-агентов").set_defaults(func=cmd_mcp)

    p = sub.add_parser("ask", help="вопрос с цитатами")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=8)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("add", help="добавить txt/md-файл в базу без пересборки индекса")
    p.add_argument("file", help="путь к файлу (.txt или .md)")
    p.add_argument("--title", default="", help="название документа (по умолчанию - имя файла)")
    p.add_argument("--by", default="", help="кто добавляет (логин/почта); иначе env VELES_INGESTED_BY")
    p.set_defaults(func=cmd_add)

    sub.add_parser("status", help="состояние корпуса/канона/индекса").set_defaults(func=cmd_status)

    p = sub.add_parser("delete", help="удалить документ: канон -> trash, L0, tombstone узлов")
    p.add_argument("document_id", help="идентификатор документа (как в цитатах/source-ссылках)")
    p.add_argument("--yes", action="store_true", help="удалить без показа плана")
    p.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
