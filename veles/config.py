"""Пути и константы среза. Корни переопределяются переменными окружения."""

from __future__ import annotations

import os
from pathlib import Path

# Корень корпуса источников (только копии исходников, без производных данных)
CORPUS_ROOT = Path(
    os.environ.get("VELES_CORPUS_ROOT", r"<local-pc>\proj\knowledge-sources")
)

# Рантайм-данные veles (canonical store, индексы) - вне репозитория источников
VELES_DATA = Path(os.environ.get("VELES_DATA", Path(__file__).resolve().parents[1] / "data"))
CANONICAL_DIR = VELES_DATA / "canonical"
INDEX_DIR = VELES_DATA / "index"

# Манифесты allowlist-конвейера живут в репо veles, не в каталоге источников
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "manifests"

# Стек среза: малая мультиязычная модель (валидация на малой модели перед бенчмарком)
EMBED_MODEL = os.environ.get("VELES_EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = 384

# RRF и AnswerPolicy (provisional; калибровка - фаза 2)
RRF_K = 60
ABSTAIN_CONFIDENCE = 0.55
