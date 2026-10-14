"""Регрессия ревью раунда 1 (находка 1): гейт памяти в eval-скрипте
evals/index-format-validate.py обязан мерить метрику критерия (б) плана -
RSS при загрузке (порог >= 3x), а не подменять её пик-аллокациями tracemalloc
(факт подмены: критерий «RSS >= 3x», фактический RSS 2.5x, а PASS брался с
tracemalloc 6.7x)."""

from __future__ import annotations

import ast
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1] / "evals" / "index-format-validate.py"


def _section_source(fn_name: str) -> str:
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return ast.unparse(node)
    raise AssertionError(f"функция {fn_name} не найдена в {EVAL.name}")


def test_memory_gate_3x_is_on_rss_metric():
    """Все ассерты с порогом 3x в замере памяти - на RSS-метрике; tracemalloc
    в гейте отсутствует (он может быть только информационной метрикой)."""
    src = _section_source("section_memory_speed")
    gates = [ln for ln in src.splitlines() if ">= 3.0" in ln or ">=3.0" in ln]
    assert gates, "гейт памяти (порог 3x) пропал из section_memory_speed"
    for ln in gates:
        assert "rss" in ln.lower(), (
            f"гейт с порогом 3x не на RSS-метрике (подмена метрики): {ln.strip()}"
        )


def test_rss_delta_is_measured_and_gated():
    """Подпроцесс замеряет прирост RSS (rss_delta) отдельной загрузкой, а секция
    памяти гейтит именно его и падает, если RSS измерить не удалось."""
    src = _section_source("_child_load")
    assert "rss_delta" in src, "подпроцесс не измеряет прирост RSS (rss_delta)"
    body = _section_source("section_memory_speed")
    # ast.unparse нормализует строковые кавычки к одинарным
    assert body.count("['rss_delta']") >= 2, (
        "гейт памяти не использует rss_delta обоих форматов (старый и новый)"
    )
    assert "['rss_delta'] is not None" in body, (
        "отсутствует проверка, что RSS удалось измерить (критерий не должен "
        "молча пропускаться, когда метрика недоступна)"
    )


def test_tracemalloc_is_informational_only():
    """Пик аллокаций tracemalloc не является гейтом: у ассертов с tracemalloc
    нет порога 3x, а в тексте проверки помечено «не гейт»."""
    src = _section_source("section_memory_speed")
    for ln in src.splitlines():
        low = ln.lower()
        if "peak_alloc" in low and ("assert" in low or "/ new" in low):
            assert ">= 3.0" not in low and ">=3.0" not in low, (
                f"tracemalloc снова в гейте: {ln.strip()}"
            )
