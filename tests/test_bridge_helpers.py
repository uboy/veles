"""Регрессия хелперов моста: fmt_stats/fmt_sources/fmt_toc, apply_footer.

Импорт bridge требует env-стабов (токены читаются на уровне модуля), клиенты
создаются только в main() - сеть не трогается.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:stub")
os.environ.setdefault("OPENWEBUI_API_KEY", "sk-stub")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

import bridge  # noqa: E402


def make_stats() -> dict:
    docs = [
        {"document_id": f"d{i}", "title": f"Док {i:02d}", "nodes": 100 - i, "sections": [f"Раздел {j}" for j in range(i + 1)]}
        for i in range(45)
    ]
    docs.append({"document_id": "dx", "title": "Абзацный источник", "nodes": 3, "sections": []})
    return {"totals": {"documents": len(docs), "nodes": 1234}, "embed_model": "test", "documents": docs}


def test_fmt_stats_summary_and_top10() -> None:
    out = bridge.fmt_stats(make_stats())
    assert "46 документов" in out and "1234 фрагментов" in out
    # 10 крупнейших + строка "… и ещё N"
    assert out.count("- ") == 11
    assert "и ещё 36" in out
    assert "/sources" in out


def test_fmt_sources_numbering_paren() -> None:
    out = bridge.fmt_sources(make_stats())
    # "N." конвертер съедает как markdown-список - только "N)"
    assert "1) Абзацный источник" in out and "2) Док 00" in out
    assert "\n1. " not in out


def test_fmt_toc_number_out_of_range_and_multi_match() -> None:
    stats = make_stats()
    # цифра вне диапазона -> честный отказ, БЕЗ падения в подстрочный поиск
    assert "не найден" in bridge.fmt_toc(stats, "0")
    assert "не найден" in bridge.fmt_toc(stats, "999")
    multi = bridge.fmt_toc(stats, "Док 0")  # матчится 10 документов 00..09
    assert "Уточните" in multi
    single = bridge.fmt_toc(stats, "Абзацный")
    assert "без верхнеуровневых разделов" in single


def test_fmt_toc_sections_cap_at_40() -> None:
    stats = make_stats()
    out = bridge.fmt_toc(stats, "46")  # "Док 44": 45 разделов
    assert "Раздел 39" in out and "и ещё 5" in out
    assert "Раздел 44" not in out


def test_apply_footer_braces_safe() -> None:
    bridge.SOURCE_FOOTER = "\n<i>модель {model}, скобки {oops} и {0} не роняют</i>"
    try:
        out = bridge.apply_footer("<b>ответ</b>", "glm-5.3-flash")
        assert "<b>ответ</b>" in out
        assert "модель glm-5.3-flash" in out
        assert "{oops}" in out and "{0}" in out
    finally:
        bridge.SOURCE_FOOTER = ""


def test_apply_footer_disabled_when_empty() -> None:
    bridge.SOURCE_FOOTER = ""
    try:
        assert bridge.apply_footer("<b>ответ</b>", "x") == "<b>ответ</b>"
    finally:
        bridge.SOURCE_FOOTER = ""
