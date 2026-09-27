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

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"
WEBUI_BASE = os.environ.get("OPENWEBUI_BASE_URL", "http://127.0.0.1:8300").rstrip("/")
WEBUI_KEY = os.environ.get("OPENWEBUI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("OPENWEBUI_MODEL", "")
KB_MODEL = os.environ.get("OPENWEBUI_KB_MODEL", "")
SYSTEM_PROMPT = os.environ.get(
    "BRIDGE_SYSTEM_PROMPT",
    "Ты - полезный личный ассистент. Отвечай по делу, на русском языке.",
)
TG_PROXY = os.environ.get("TG_PROXY") or None
PAIR_MODE = os.environ.get("PAIR_MODE", "1") == "1"
STATE_FILE = Path(os.environ.get("BRIDGE_STATE_FILE", "bridge-state.json"))
HISTORY_LIMIT = int(os.environ.get("BRIDGE_HISTORY", "16"))

log = logging.getLogger("tg-bridge")

histories: dict[int, list[dict]] = {}
tg_client: httpx.Client
webui_client: httpx.Client


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
    text = (text or "").strip() or "(пустой ответ)"
    for i in range(0, len(text), 4000):
        tg("sendMessage", chat_id=chat_id, text=text[i : i + 4000])


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
                      "/kb <вопрос> - вопрос в базу знаний.")
        return owner
    if text == "/new":
        histories.pop(chat_id, None)
        send(chat_id, "Контекст сброшен.")
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
    sys.exit(main())
