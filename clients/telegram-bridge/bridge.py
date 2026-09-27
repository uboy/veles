#!/usr/bin/env python3
"""Telegram-мост dm_chat_ai_helper -> Open WebUI (фаза Ф4, corp-ai-kb).

Telegram-клиент поверх Open WebUI: base-модель - GLM через z.ai (coding plan),
/kb <вопрос> - вопрос в базу знаний (модель veles-knowledge того же WebUI).
Персональный бот: при PAIR_MODE=1 первый /start становится владельцем (state file),
сообщения других чатов игнорируются.

Egress в Telegram: из офисной сети стенд api.telegram.org закрыт DPI - трафик идёт через
HTTPS CONNECT-прокси на <vps-provider> (dumbproxy за HAProxy :443, переменная TG_PROXY,
напр. https://user:pass@<ddns-name>:443). WebUI и z.ai доступны напрямую.

Запуск (см. deploy/стенд/veles-telegram-bridge.service):
  python clients/telegram-bridge/bridge.py
Секреты и конфиг - только через окружение (EnvironmentFile), не в репозитории.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

from tg_format import format_markdown_to_html, split_for_telegram, strip_html_tags

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
WEBUI_BASE = os.environ.get("OPENWEBUI_BASE_URL", "http://127.0.0.1:8300").rstrip("/")
WEBUI_KEY = os.environ.get("OPENWEBUI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENWEBUI_MODEL", "")
KB_MODEL = os.environ.get("OPENWEBUI_KB_MODEL", "")
SYSTEM_PROMPT = os.environ.get(
    "BRIDGE_SYSTEM_PROMPT",
    "Ты - полезный личный ассистент. Отвечай по делу, на русском языке. "
    "Оформляй ответы markdown-разметкой (заголовки, списки, **выделение**).",
)
TG_PROXY = os.environ.get("TG_PROXY") or None
# Прямой доступ к фасаду Veles для /stats /sources /toc (оглавление базы).
VELES_BASE_URL = os.environ.get("VELES_BASE_URL", "http://127.0.0.1:8400").rstrip("/")
VELES_API_TOKEN = os.environ.get("VELES_API_TOKEN", "")
# Подпись provenance на ответах базовой модели: честно показывает, что текст
# НЕ из базы знаний. Пустая переменная выключает подпись. {model} подставляется.
SOURCE_FOOTER = os.environ.get(
    "BRIDGE_SOURCE_FOOTER",
    "\n\n<i>— ответ общей модели {model}, без обращения к базе знаний. "
    "Спросить базу знаний: /kb &lt;вопрос&gt;</i>",
)
PAIR_MODE = os.environ.get("PAIR_MODE", "1") == "1"
STATE_FILE = Path(os.environ.get("BRIDGE_STATE_FILE", "bridge-state.json"))
HISTORY_LIMIT = int(os.environ.get("BRIDGE_HISTORY", "16"))

log = logging.getLogger("tg-bridge")

histories: dict[int, list[dict]] = {}
tg_client: httpx.Client
webui_client: httpx.Client
veles_client: httpx.Client


def load_owner() -> int | None:
    try:
        return int(json.loads(STATE_FILE.read_text(encoding="utf-8"))["owner_chat_id"])
    except Exception:  # noqa: BLE001 - нет файла/битый json = не спарен
        return None


def save_owner(chat_id: int) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"owner_chat_id": chat_id}), encoding="utf-8")
    log.info("owner paired: chat %s -> %s", chat_id, STATE_FILE)


def tg(method: str, **params: object) -> dict:
    r = tg_client.post(f"{TG_API}/{method}", json=params)
    data = r.json()
    if not data.get("ok"):
        log.error("telegram %s: %s", method, str(data)[:300])
    return data


def webui_chat(messages: list[dict], model: str) -> str:
    body: dict = {"messages": messages, "stream": False}
    if model:
        body["model"] = model
    r = webui_client.post(
        f"{WEBUI_BASE}/api/chat/completions",
        headers={"Authorization": f"Bearer {WEBUI_KEY}"},
        json=body,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def send(chat_id: int, text: str) -> None:
    """Отправить markdown-текст: куски по абзацам, каждый как Telegram-HTML
    (перенос пайплайна донора); ok=false от Telegram -> plain fallback куска.
    Лимит куска 3400 по plain-тексту с запасом на разрастание HTML-разметки."""
    text = (text or "").strip() or "(пустой ответ)"
    for chunk in split_for_telegram(text, limit=3400):
        html_text = format_markdown_to_html(chunk)
        sent = tg("sendMessage", chat_id=chat_id, text=html_text, parse_mode="HTML")
        if not sent.get("ok"):
            log.warning(
                "HTML send failed (%s), fallback to plain",
                str(sent.get("description"))[:120],
            )
            tg("sendMessage", chat_id=chat_id, text=strip_html_tags(html_text))


def veles_stats() -> dict:
    """Оглавление базы из фасада: документы, верхнеуровневые разделы, объёмы."""
    headers = {"Authorization": f"Bearer {VELES_API_TOKEN}"} if VELES_API_TOKEN else {}
    r = veles_client.get(f"{VELES_BASE_URL}/v1/knowledge/stats", headers=headers)
    r.raise_for_status()
    return r.json()


def sorted_docs(stats: dict) -> list[dict]:
    return sorted(stats["documents"], key=lambda d: d["title"].lower())


def fmt_stats(stats: dict) -> str:
    docs = stats["documents"]
    n_sections = sum(len(d["sections"]) for d in docs)
    lines = [
        f"База знаний: {stats['totals']['documents']} документов, "
        f"{stats['totals']['nodes']} фрагментов, {n_sections} верхнеуровневых разделов.",
        f"Модель эмбеддингов: {stats['embed_model']}.",
        "",
        "Крупнейшие источники:",
    ]
    for d in docs[:10]:
        lines.append(f"- {d['title']} ({d['nodes']} фр., разделов {len(d['sections'])})")
    if len(docs) > 10:
        lines.append(f"- … и ещё {len(docs) - 10} (полный список: /sources)")
    lines += ["", "Оглавление документа: /toc <номер из /sources>"]
    return "\n".join(lines)


def fmt_sources(stats: dict) -> str:
    lines = [f"Источники базы знаний ({len(stats['documents'])}):", ""]
    for i, d in enumerate(sorted_docs(stats), 1):
        lines.append(f"{i}) {d['title']} ({d['nodes']} фр.)")
    lines += ["", "Оглавление: /toc <номер> или /toc <часть названия>"]
    return "\n".join(lines)


def fmt_toc(stats: dict, arg: str) -> str:
    docs = sorted_docs(stats)
    doc = None
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(docs):
            doc = docs[idx]
    if doc is None:
        needle = arg.lower()
        matches = [d for d in docs if needle in d["title"].lower()]
        if not matches:
            return f"Источник по запросу '{arg}' не найден. Список: /sources"
        if len(matches) > 1:
            lines = [f"Уточните ({len(matches)} совпадений):", ""]
            lines += [f"{docs.index(d) + 1}) {d['title']}" for d in matches[:15]]
            return "\n".join(lines)
        doc = matches[0]
    sections = doc["sections"]
    lines = [f"{doc['title']}", f"({doc['nodes']} фрагментов, разделов {len(sections)})", ""]
    if sections:
        lines += [f"- {s}" for s in sections[:40]]
        if len(sections) > 40:
            lines.append(f"- … и ещё {len(sections) - 40}")
    else:
        lines.append("(документ без верхнеуровневых разделов)")
    return "\n".join(lines)


def handle(update: dict, owner: int | None) -> int | None:
    """Обрабатывает один update; возвращает актуального владельца."""
    msg = update.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    text = (msg.get("text") or "").strip()
    if not chat_id or not text:
        return owner

    if owner is None:
        if PAIR_MODE and text.startswith("/start"):
            save_owner(chat_id)
            send(chat_id, "Бот спарен с этим чатом. Пишите вопрос; /new - сброс контекста; "
                          "/kb <вопрос> - вопрос в базу знаний.")
            return chat_id
        log.warning("drop message from unpaired chat %s", chat_id)
        return owner
    if chat_id != owner:
        log.warning("drop message from non-owner chat %s", chat_id)
        return owner

    if text == "/start":
        send(chat_id, "Готов. Пишите вопрос; /new - сброс контекста; "
                      "/kb <вопрос> - вопрос в базу знаний; "
                      "/stats - сводка базы; /sources - источники; "
                      "/toc <номер> - оглавление документа.")
        return owner
    if text == "/new":
        histories.pop(chat_id, None)
        send(chat_id, "Контекст сброшен.")
        return owner

    if text in ("/stats", "/sources") or text.startswith("/toc"):
        try:
            stats = veles_stats()
        except Exception as e:  # noqa: BLE001
            log.exception("veles stats failed")
            send(chat_id, f"База знаний недоступна: {type(e).__name__}: {str(e)[:150]}")
            return owner
        if text == "/stats":
            send(chat_id, fmt_stats(stats))
        elif text == "/sources":
            send(chat_id, fmt_sources(stats))
        else:
            arg = text[4:].strip()
            send(chat_id, fmt_toc(stats, arg) if arg else
                 "Использование: /toc <номер из /sources> или /toc <часть названия>")
        return owner

    if text.startswith("/kb"):
        question = text[3:].strip()
        if not question:
            send(chat_id, "Использование: /kb <вопрос к базе знаний>")
            return owner
        if not KB_MODEL:
            send(chat_id, "KB-модель не настроена (OPENWEBUI_KB_MODEL).")
            return owner
        try:
            answer = webui_chat([{"role": "user", "content": question}], KB_MODEL)
        except Exception as e:  # noqa: BLE001 - наружу только краткая причина
            log.exception("kb call failed")
            send(chat_id, f"Ошибка запроса к базе знаний: {type(e).__name__}: {str(e)[:200]}")
            return owner
        send(chat_id, answer)
        return owner

    hist = histories.setdefault(chat_id, [])
    hist.append({"role": "user", "content": text})
    try:
        answer = webui_chat(
            [{"role": "system", "content": SYSTEM_PROMPT}] + hist[-HISTORY_LIMIT:],
            DEFAULT_MODEL,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("webui call failed")
        hist.pop()
        send(chat_id, f"Ошибка запроса к модели: {type(e).__name__}: {str(e)[:200]}")
        return owner
    hist.append({"role": "assistant", "content": answer})
    if SOURCE_FOOTER:
        answer += SOURCE_FOOTER.format(model=DEFAULT_MODEL or "по умолчанию")
    send(chat_id, answer)
    return owner


def main() -> int:
    owner = load_owner()
    me = tg("getMe")
    if not me.get("ok"):
        log.error("getMe failed: %s", str(me)[:200])
        return 1
    import re as _re

    proxy_masked = _re.sub(r"//[^@/]+@", "//***@", TG_PROXY) if TG_PROXY else "direct"
    log.info(
        "bot @%s started; webui=%s model=%r kb_model=%r proxy=%s owner=%s",
        me["result"].get("username"), WEBUI_BASE, DEFAULT_MODEL, KB_MODEL,
        proxy_masked, owner,
    )
    offset = 0
    fails = 0
    while True:
        try:
            r = tg_client.get(f"{TG_API}/getUpdates", params={"timeout": 50, "offset": offset})
            r.raise_for_status()
            data = r.json()
            if not data.get("ok"):
                log.error("getUpdates: %s", str(data)[:200])
                time.sleep(3)
                continue
            fails = 0
            for u in data.get("result", []):
                offset = u["update_id"] + 1
                try:
                    owner = handle(u, owner)
                except Exception:  # noqa: BLE001 - один update не роняет цикл
                    log.exception("handle failed")
        except Exception:  # noqa: BLE001
            log.exception("poll failed")
            fails += 1
            if fails >= 10:
                log.error("10 последовательных ошибок poll - выход для рестарта супервизором")
                return 1
            time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx на INFO пишет полный URL с userinfo прокси (пароль в journalctl)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    tg_client = httpx.Client(
        proxy=TG_PROXY,
        timeout=httpx.Timeout(75.0, connect=15.0),
    )
    webui_client = httpx.Client(timeout=httpx.Timeout(300.0, connect=15.0))
    veles_client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))
    sys.exit(main())
