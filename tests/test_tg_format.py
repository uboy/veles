"""Регрессия форматирования Telegram-вывода моста (перенос пайплайна донора).

Проверки на реальном образце ответа GLM: **bold** -> <b>, заголовки -> <b>,
маркеры списков -> bullet, отсутствие литеральных markdown-символов в HTML,
экранирование HTML, защита code-блоков, чанкинг по абзацам.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

from tg_format import (  # noqa: E402
    format_markdown_to_html,
    split_for_telegram,
    strip_html_tags,
)

SAMPLE = """## Управление производством

**Управление производством** — это деятельность по планированию и контролю.

**Основные задачи:**
- **Планирование производства** — определение объёмов выпуска
- **Организация процессов** — расстановка оборудования

1. Время выполнения заказа (lead time)
2. Загрузка оборудования

Используйте `python script.py` и методологии **Lean**, **Six Sigma**.
"""


def test_sample_converts_bold_headings_bullets() -> None:
    out = format_markdown_to_html(SAMPLE)
    assert "<b>Управление производством</b>" in out
    assert "<b>Планирование производства</b>" in out
    assert "**" not in out
    assert "##" not in out
    assert out.count("• ") >= 2
    assert "<code>python script.py</code>" in out


def test_html_entities_escaped() -> None:
    out = format_markdown_to_html("Сравните a<b и b>c, читайте https://x.ru/?a=1&b=2")
    assert "a&lt;b" in out and "b&gt;c" in out
    assert "&amp;" in out


def test_code_block_preserved_verbatim() -> None:
    out = format_markdown_to_html("Текст:\n```python\nif a < b && c:\n    x = '**not bold**'\n```\nКонец")
    # содержимое кода экранируется для Telegram-HTML, но не теряет символы
    assert "if a &lt; b &amp;&amp; c:" in out
    assert "x = '**not bold**'" in out
    assert "<pre><code>" in out
    assert "<b>not bold</b>" not in out


def test_strip_html_fallback_readable() -> None:
    html_text = format_markdown_to_html(SAMPLE)
    plain = strip_html_tags(html_text)
    assert "<b>" not in plain and "**" not in plain
    assert "Управление производством" in plain
    assert "• " in plain


def test_split_long_text_by_paragraphs() -> None:
    text = "\n\n".join(f"Абзац {i} " + "слово " * 60 for i in range(120))
    chunks = split_for_telegram(text, limit=3900)
    assert len(chunks) > 1
    assert all(len(c) <= 3900 for c in chunks)
    assert "".join(chunks).count("Абзац") == 120


def test_unclosed_code_fence_stashed_as_code() -> None:
    out = format_markdown_to_html("Начало:\n```python\nx = 1")
    assert "<pre><code>" in out
    assert "x = 1" in out
    assert out.count("```") == 0


def test_placeholder_literal_collision_isolated() -> None:
    out = format_markdown_to_html("`pwd` и литерал __INLINE_CODE_0__ в тексте")
    assert out.count("<code>") == 1
    assert "__INLINE_CODE_0__" in out


def test_md_links_become_telegram_anchors() -> None:
    out = format_markdown_to_html("Сноска [1](http://host/v1/source/a?token=x&node=y) и текст")
    assert '<a href="http://host/v1/source/a?token=x&amp;node=y">1</a>' in out
    assert "__MD_LINK" not in out
    assert "[1]" not in out.replace('<a href', '', 1)


def test_short_text_single_chunk() -> None:
    assert split_for_telegram("короткий", limit=3900) == ["короткий"]
