"""Пути и константы среза. Корни переопределяются переменными окружения."""

from __future__ import annotations

import os
from pathlib import Path

# Корень корпуса источников (только копии исходников, без производных данных).
# Без env - ~/veles-data/corpus (общий корень данных veles); машинных путей в коде нет.
CORPUS_ROOT = Path(
    os.environ.get("VELES_CORPUS_ROOT", str(Path.home() / "veles-data" / "corpus"))
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

DEFAULT_ACCESS_GROUPS = {"personal"}

# Публичный URL фасада для ссылок на источники в цитатах (виден клиентам:
# веб-чат, Telegram с телефона). Задаётся через VELES_PUBLIC_URL.
VELES_PUBLIC_URL = os.environ.get("VELES_PUBLIC_URL", "http://127.0.0.1:8400").rstrip("/")

# Корни с оригиналами L0 для GET /v1/source (разделитель os.pathsep: ':' на linux).
# Пусто - источник отдаётся текстовой сборкой из канона.
VELES_SOURCE_ROOTS = [
    Path(p)
    for p in os.environ.get("VELES_SOURCE_ROOTS", "").split(os.pathsep)
    if p.strip()
]

# RRF и AnswerPolicy (provisional; калибровка - фаза 2)
RRF_K = 60
ABSTAIN_CONFIDENCE = 0.55
# K12 (калибровка retrieval, evals/results/2026-09-29-k12-retrieval-calibration.md):
# дефолты выдачи; поверх них hybrid_search принимает явные kwargs (бенч, тесты).
# Замер 29.09 (прод-индекс, golden 57 answerable + rrncb50): буст 2.0 - единственная
# комбинация, выполняющая все цели (k11-056 в top-5, R@5 golden 1.000, rrncb50 без
# регрессии); дедуп каналов даёт лучшие агрегаты (top1 0.772, MRR 0.877), но на
# rrncb50 R@5 -0.04 (узлы выборки сами раздувались дубль-вкладами дефекта индекса)
# - включить после плановой пересборки с дедупом (d135953) и перегоном бенча.
CODEX_BOOST = 2.0  # буст узлов акта, названного в вопросе («согласно НК РФ»); 0 = выключен
FILTER_JUNK_NODES = False  # пустышки не доходят до top-5 на глубине 25 (замер: без эффекта); knob для depth>
DEDUPE_CHANNELS = False  # снимать дубликаты узла внутри каналов RRF (прод-индекс: 52% узлов x2)
# warn_conflict (K3, реализуемый минимум): топ-2 доказательства считаются
# кандидатами в конфликт только при сопоставимом score - дельта нормализованного
# score (RRF/потолок двух каналов, шкала 0..1) меньше; на сыром RRF порог 0.05
# невыполним (максимум RRF ~ 0.033)
CONFLICT_SCORE_DELTA = 0.05
