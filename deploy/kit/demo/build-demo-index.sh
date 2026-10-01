#!/usr/bin/env bash
# build-demo-index.sh - пересборка мини-индекса demo/ из demo/documents.
#
# Запускается на машине с готовым окружением veles (extras [st]):
#   VELES_PYTHON=/path/to/venv/bin/python bash build-demo-index.sh
# (VELES_PYTHON не задан - используется python3 из PATH).
#
# Пайплайн тот же, что у пользовательской загрузки документов (K4):
# TxtMdParserAdapter -> ingest_document -> build_index -> bundle.save.
# Данные собираются во временном каталоге (VELES_DATA), оттуда переносятся
# в demo/index и demo/canonical. Сеть нужна один раз - на скачивание
# модели e5-small (~130 МБ, HuggingFace).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# veles импортируется из корня репозитория (demo/ лежит на три уровня глубже: deploy/kit/demo)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
PY="${VELES_PYTHON:-python3}"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/data"

# VELES_DATA - временный корень: канон ляжет в $WORK/data/canonical,
# индекс - в $WORK/data/index
cd "$REPO_ROOT"
VELES_DATA="$WORK/data" "$PY" - "$SCRIPT_DIR" <<'PYEOF'
"""Сборка демо-индекса: docs -> канон -> L2 (dense numpy + BM25)."""
import shutil
import sys
from pathlib import Path

demo = Path(sys.argv[1])
docs_dir = demo / "documents"

from veles.canonical.store import ingest_document
from veles.config import EMBED_DIM, EMBED_MODEL, CANONICAL_DIR, INDEX_DIR
from veles.indexing.embedder import SentenceTransformersProvider
from veles.indexing.indexer import build_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.ingest.txt_md import TxtMdParserAdapter

parser = TxtMdParserAdapter()
files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() == ".md")
if not files:
    raise SystemExit(f"нет .md документов в {docs_dir}")
for path in files:
    doc = parser.parse_file(path, corpus_root=docs_dir)
    # K21: демо-контент - общий (группа all), а не личный по умолчанию
    # (DEFAULT_ACCESS_GROUPS=personal): иначе новый пользователь без группы
    # personal видит в чате «пустую базу» - ловушка демо
    doc.access_groups = ["all"]
    for node in doc.nodes:
        node.access_groups = ["all"]
    outcome = ingest_document(doc, by="demo-build")
    print(f"  {doc.document_id}: {len(doc.nodes)} узлов ({outcome.action}, {outcome.revision_id})")

embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
bundle = build_index(embedder, NumpyFlatStore(EMBED_DIM))
bundle.save(INDEX_DIR, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
print(f"индекс: {len(bundle.node_meta)} узлов, backend=numpy-flat -> {INDEX_DIR}")

# канон и индекс переезжают в кит (demo/index пересобирается с нуля:
# остатки прежнего бандла другой сборки не должны подмешиваться)
target = demo / "canonical"
if target.exists():
    shutil.rmtree(target)
shutil.copytree(CANONICAL_DIR, target)
idx_target = demo / "index"
if idx_target.exists():
    shutil.rmtree(idx_target)
shutil.copytree(INDEX_DIR, idx_target)
print(f"канон -> {target}")
PYEOF

echo "готово:"
du -sh "$SCRIPT_DIR/index" "$SCRIPT_DIR/canonical"
