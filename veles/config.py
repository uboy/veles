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

# RRF и AnswerPolicy (провизорные; финальная калибровка - K14)
RRF_K = 60
# K14 (замер 29.09 на ЧИСТОМ индексе после пересборки, 78 вопросов: golden 57
# answerable + 19 abstain, scratch/k14gate.py -> evals/results/2026-09-29-k14-...):
# потолок 2/(K+1) рассчитан на скоры дублированного индекса; на чистом одиночный
# ранг-1 одного канала = 0.4, и порог 0.55 давал ложный отказ (k11-026). Порог
# 0.35 -> 0 ложных отказов на всех 78. Различающую силу несёт честный отказ
# синтеза (100% на трёх снапшотах, K11/K12/K14), гейт остаётся tripwire для
# глубокого хвоста; вернуть различимость - cross-encoder reranker (рекомендация
# K12: top1 0.860, 18/57 expected на рангах 2-5).
ABSTAIN_CONFIDENCE = 0.35
# K12 (калибровка retrieval, evals/results/2026-09-29-k12-retrieval-calibration.md):
# дефолты выдачи; поверх них hybrid_search принимает явные kwargs (бенч, тесты).
# Замер 29.09 (прод-индекс, golden 57 answerable + rrncb50): буст 2.0 - единственная
# комбинация, выполняющая все цели (k11-056 в top-5, R@5 golden 1.000, rrncb50 без
# регрессии); дедуп каналов даёт лучшие агрегаты (top1 0.772, MRR 0.877), но на
# rrncb50 R@5 -0.04 (узлы выборки сами раздувались дубль-вкладами дефекта индекса)
# - включить после плановой пересборки с дедупом (d135953) и перегоном бенча.
LEGAL_BOOST = 2.0  # буст узлов акта, названного в вопросе («согласно НК РФ»); 0 = выключен
FILTER_JUNK_NODES = False  # пустышки не доходят до top-5 на глубине 25 (замер: без эффекта); knob для depth>
# K14 (пересборка прода с дедупом, evals/results/2026-09-29-k14-rebuild-dedupe.md):
# дефолт True по плану K12. Замер 29.09 на ЧИСТОМ индексе (127 498 узлов, без
# дублей): True vs False - идентичные метрики golden и rrncb50 до знака (knob
# no-op без дублей в каналах); включение - защита от возврата дублей при будущих
# пересборках, поведение прода не меняет.
DEDUPE_CHANNELS = True  # снимать дубликаты узла внутри каналов RRF (no-op на чистом индексе)
# warn_conflict (K3, реализуемый минимум): топ-2 доказательства считаются
# кандидатами в конфликт только при сопоставимом score - дельта нормализованного
# score (RRF/потолок двух каналов, шкала 0..1) меньше; на сыром RRF порог 0.05
# невыполним (максимум RRF ~ 0.033)
CONFLICT_SCORE_DELTA = 0.05
# K17: cross-encoder reranker - опциональный пост-этап hybrid_search (ПОСЛЕ RRF
# и кодекс-буста, ПЕРЕД усечением top_k). Пусто (дефолт) = выключено, прежнее
# поведение без изменений; имя модели = HuggingFace id (например
# BAAI/bge-reranker-v2-m3). Модель грузится лениво при первом вызове и кэшируется;
# ошибка загрузки/инференса = деградация к обычному порядку выдачи (fail-open,
# лог-warning). Инференс по умолчанию CPU; VELES_RERANKER_DEVICE/MAX_LENGTH/BATCH -
# для GPU-бенчей (батч и обрезка длинных входов; 0 = штатные дефолты модели).
VELES_RERANKER_MODEL = os.environ.get("VELES_RERANKER_MODEL", "")
VELES_RERANKER_DEVICE = os.environ.get("VELES_RERANKER_DEVICE", "cpu")


def _env_int(name: str, default: int) -> int:
    """int из env без роняния импорта: опечатка в .env кита не должна
    валить конфиг целиком (facade ушёл бы в crash-loop) - дефолт + warning."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"config: {name}={raw!r} не целое - беру {default}", flush=True)
        return default


VELES_RERANKER_MAX_LENGTH = _env_int("VELES_RERANKER_MAX_LENGTH", 0)
VELES_RERANKER_BATCH = _env_int("VELES_RERANKER_BATCH", 0)
# Глубина пула кандидатов под rerank (узлов после RRF/буста; бриф K17 - 25).
RERANK_CANDIDATES = 25


def _env_float(name: str, default: float) -> float:
    """float из env без роняния импорта (та же дисциплина, что _env_int)."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"config: {name}={raw!r} не число - беру {default}", flush=True)
        return default


# Таймаут ожидания синтеза LLM, с (K21): при медленном/деградирующем LLM-хосте
# фолбэк на extractive-ответ наступает раньше, чем клиент отвалится по таймауту.
VELES_LLM_TIMEOUT_S = _env_float("VELES_LLM_TIMEOUT_S", 60.0)
