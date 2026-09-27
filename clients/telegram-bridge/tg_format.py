"""Форматирование ответов для Telegram (перенос пайплайна telegram_bot_ai).

Донор: uboy/telegram_bot_ai, shared/utils.py (format_markdown_to_html,
strip_html_tags) и frontend/bot_handlers.py (отправка parse_mode='HTML'
кусками <=3900 с fallback на plain). Порт без async-зависимостей: модуль
только преобразует текст и не знает про сеть и env.

Пайплайн моста: сырой markdown-текст модели -> split_for_telegram() куски
по абзацам -> format_markdown_to_html() каждый кусок -> отправка с
parse_mode='HTML'; ошибка парсинга Telegram -> strip_html_tags() и повтор
plain. Содержимое code-блоков защищается плейсхолдерами от экранирования.
"""

from __future__ import annotations

import html
import re

# Лимит Telegram на сообщение с запасом (жёсткий лимит 4096)
CHUNK_LIMIT = 3900

_CODE_FENCE = re.compile(r"```([a-zA-Z0-9+_-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+?)`")


def _esc(text: str) -> str:
    return html.escape(text, quote=False)


def format_markdown_to_html(text: str) -> str:
    """Конвертировать markdown-подобные конструкции в HTML для Telegram.

    Порядок (как в доноре): спрятать fenced/inline код за плейсхолдерами,
    экранировать HTML, заголовки #/##/### -> <b>, **x** -> <b>x</b>,
    *x* -> <i>x</i> (только с границами слова), списки -> •, восстановить
    код как <pre><code>/<code>.
    """
    if not text:
        return ""

    code_blocks: list[tuple[str, str]] = []

    def _stash_fence(match: re.Match) -> str:
        lang = (match.group(1) or "").strip()
        content = match.group(2)
        if lang and not content.startswith("\n") and "\n" not in content:
            # inline-fenced вида ```bash echo 1``` - "язык" это часть содержимого
            code_blocks.append(("", f"{lang} {content}".strip()))
        else:
            code_blocks.append((lang, content))
        return f"__CODE_BLOCK_{len(code_blocks) - 1}__"

    text = _CODE_FENCE.sub(_stash_fence, text)

    inline_code: list[str] = []

    def _stash_inline(match: re.Match) -> str:
        inline_code.append(match.group(1))
        return f"__INLINE_CODE_{len(inline_code) - 1}__"

    text = _INLINE_CODE.sub(_stash_inline, text)

    text = _esc(text)

    text = re.sub(r"^\s*#{1,6}\s+(.+?)(?=\n|$)", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"(?<!<)\*\*([^*]+?)\*\*(?!>)", r"<b>\1</b>", text)
    text = re.sub(r"(?<!<)(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)(?!>)", r"<i>\1</i>", text)

    text = re.sub(r"^\d+\.\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^[-]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\s+", "• ", text, flags=re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)

    for idx, code in enumerate(inline_code):
        text = text.replace(f"__INLINE_CODE_{idx}__", f"<code>{_esc(code)}</code>")
    for idx, (_lang, code) in enumerate(code_blocks):
        text = text.replace(f"__CODE_BLOCK_{idx}__", f"<pre><code>{_esc(code)}</code></pre>")

    return text.strip()


def strip_html_tags(text: str) -> str:
    """Безопасно убрать HTML-теги, оставив читаемый plain (fallback донора)."""
    if not text:
        return ""
    text = re.sub(r"</(p|div|li|h[1-6])\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_for_telegram(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Разбить plain-текст на куски <=limit по границам абзацев (fallback донора)."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(para) > limit:
            cut = para.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(para[:cut])
            para = para[cut:].lstrip("\n")
        current = para
    if current:
        chunks.append(current)
    return chunks
