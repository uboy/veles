"""K17: cross-encoder reranker - опциональный пост-этап гибридного поиска.

Гибридный поиск (dense+BM25 -> RRF -> codex-буст) даёт пул кандидатов
(RERANK_CANDIDATES=25 верхних узлов RRF); cross-encoder скорит пары
«вопрос-фрагмент» и пересортировывает пул, в выдачу идут топ-k. Включение -
env VELES_RERANKER_MODEL (пусто/не задано = выключено, прежнее поведение
без изменений: hybrid_search не импортирует этот модуль и не строит пул).

Свойства (бриф K17):
- ленивая загрузка: модель грузится при первом вызове rerank()/preload() и
  кэшируется в процессных переменных; сам импорт модуля модель не тянет;
- fail-open: ошибка загрузки/инференса модели -> лог-warning и обычный порядок
  кандидатов (ответы не ронять); провал загрузки кэшируется негативно - модель
  не дёргает сеть на каждом запросе, сервис продолжает работать без реранка
  до рестарта (снять - reset_cache());
- латентность этапа замеряется внутри и пишется в лог (logger veles.reranker);
- инференс по умолчанию CPU: device берётся из env VELES_RERANKER_DEVICE
  (дефолт "cpu") - GPU стенд заняты 1С, решение о GPU-инференсе - отдельный
  замер владельца.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol, Sequence

log = logging.getLogger("veles.reranker")

# Кэш загруженной модели (процессный): повторные rerank-вызовы не перегружают.
_model = None
_model_name = ""
# Негативный кэш: имена моделей, чья загрузка уже падала (fail-open без
# повторных сетевых походов на каждый запрос). Очищается reset_cache().
_load_failed: set[str] = set()


class Scorer(Protocol):
    """Минимальный интерфейс скорера: predict(pairs) -> скоры (выше = релевантнее)."""

    def predict(self, pairs: Sequence[Sequence[str]]): ...


def _load(model_name: str) -> Scorer:
    """Загрузка CrossEncoder (лениво, кэшируется в процессных переменных).

    Провал загрузки запоминается в негативном кэше (_load_failed): повторные
    вызовы падают мгновенно, не дёргая сеть на каждом запросе (fail-open без
    латентной деградации). Снять - reset_cache() или рестарт процесса."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    if model_name in _load_failed:
        raise RuntimeError(
            f"reranker: модель {model_name} уже падала при загрузке (негативный кэш)"
        )
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    try:
        # Дефолт cpu: инференс CPU-only (см. докстринг модуля). VELES_RERANKER_DEVICE
        # и VELES_RERANKER_MAX_LENGTH - для бенчей/хостов с GPU: устройство и
        # обрезка длинных фрагментов (дефолт 512 - узел индекса помещается почти
        # всегда); батч тоже env (длина 4096+ при дефолтном 32 рвёт память).
        from ..config import (
            VELES_RERANKER_BATCH,
            VELES_RERANKER_DEVICE,
            VELES_RERANKER_MAX_LENGTH,
        )

        model = CrossEncoder(
            model_name,
            device=VELES_RERANKER_DEVICE,
            max_length=VELES_RERANKER_MAX_LENGTH or 512,
        )
    except Exception:
        _load_failed.add(model_name)
        raise
    _model = model
    _model_name = model_name
    log.info(
        "reranker: модель %s загружена за %.1f с (device=%s, max_length=%d)",
        model_name,
        time.perf_counter() - started,
        VELES_RERANKER_DEVICE,
        VELES_RERANKER_MAX_LENGTH or 512,
    )
    return model


def preload() -> None:
    """Прогрев: загрузить модель, если reranker включён в конфиге.

    Вызывается фасадом при старте в фоновом потоке прогрева (порт не
    блокируется - паттерн прогрева эмбеддера). Ошибки не роняют сервис:
    имя попадает в негативный кэш, первый rerank-вызов быстро деградирует
    к обычной выдаче (fail-open)."""
    from ..config import VELES_RERANKER_MODEL

    if not VELES_RERANKER_MODEL:
        return
    try:
        _load(VELES_RERANKER_MODEL)
    except Exception as e:  # noqa: BLE001 - прогрев не должен ронять сервис
        log.warning("reranker: прогрев не удался (fail-open без реранка): %s", e)


def reset_cache() -> None:
    """Сброс кэшей модели и провалов (тесты: подмена скорера между кейсами)."""
    global _model, _model_name
    _model = None
    _model_name = ""
    _load_failed.clear()


def rerank(
    query: str,
    candidates: list,
    top_k: int,
    model_name: str | None = None,
    scorer: Scorer | None = None,
) -> list:
    """Пересортировка кандидатов cross-encoder-скором пары «вопрос-фрагмент».

    candidates - Evidence с заполненным text; возвращает пересортированный
    список (устойчиво к равным скорам: ties сохраняют прежний гибридный
    порядок), усечённый до top_k. Вызывается из hybrid_search ДО финального
    усечения top_k; при выключенном/упавшем реранкере порядок не меняется.
    Органические скоры Evidence (rrf/rrf_base) не трогаются - confidence
    abstain-гейта и warn_conflict считаются по органическим скорам, но нового
    порядка топов: после реранка их значения могут отличаться от выключенного
    режима (планку rrncb50 держит бенч до/после).

    model_name=None -> имя из конфига (VELES_RERANKER_MODEL); scorer=None ->
    реальная модель (ленивая загрузка). Ошибка модели -> лог-warning и входной
    порядок (fail-open). Латентность этапа (скоринг + сортировка) замеряется
    и пишется в лог.
    """
    if not candidates:
        return []
    limit = max(1, int(top_k))
    if len(candidates) <= 1 and limit >= len(candidates):
        # одному кандидату пересортировка ничего не меняет (скор не считаем -
        # ноль лишней латентности)
        return list(candidates)[:limit]

    started = time.perf_counter()
    scorer_label = "подан снаружи"
    try:
        if scorer is None:
            from ..config import VELES_RERANKER_MODEL

            name = VELES_RERANKER_MODEL if model_name is None else model_name
            if not name:
                # выключено (без модели порядок и так входной) - без похода в лог
                return list(candidates)[:limit]
            scorer = _load(name)
            scorer_label = name
        from ..config import VELES_RERANKER_BATCH

        if VELES_RERANKER_BATCH:  # env для длинных входов на GPU; дефолт 0 - штатный predict
            scores = scorer.predict(
                [[query, c.text] for c in candidates],
                batch_size=max(1, int(VELES_RERANKER_BATCH)),
            )
        else:
            scores = scorer.predict([[query, c.text] for c in candidates])
        if len(scores) != len(candidates):
            raise ValueError(
                f"скорер вернул {len(scores)} скоров на {len(candidates)} пар"
            )
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
    except Exception as e:  # noqa: BLE001 - fail-open: не роняем ответ
        log.warning(
            "rerank: ошибка модели (%s) - возвращаю обычный порядок (%d кандидатов)",
            e,
            len(candidates),
        )
        return list(candidates)[:limit]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    out = []
    for i in order[:limit]:
        ev = candidates[i]
        # скор реранкера - в diagnostics-поля Evidence; RRF/rrf_base не трогаем:
        # confidence abstain-гейта и warn_conflict считаются по органическому RRF
        ev.rerank_score = float(scores[i])
        ev.rerank_rank_hybrid = i + 1  # позиция в гибридном порядке до перестановки
        out.append(ev)
    log.info(
        "rerank: %d пар за %.0f мс (скорер %s, топ-%d)",
        len(candidates),
        elapsed_ms,
        scorer_label,
        limit,
    )
    return out
