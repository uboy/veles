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
import re
import sys
import threading
import time
from pathlib import Path

import httpx

from tg_format import format_markdown_to_html, split_for_telegram, strip_html_tags
import tracker

TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TRACKER_ROOT = os.environ.get("TRACKER_ROOT", "")
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
# Явная привязка владельца (chat_id) - закрывает окно спаривания первого /start;
# задан -> state-file и PAIR_MODE игнорируются.
OWNER_CHAT_ID = int(os.environ.get("OWNER_CHAT_ID", "0") or 0)
# Прямой доступ к фасаду Veles для /stats /sources /toc (оглавление базы).
VELES_BASE_URL = os.environ.get("VELES_BASE_URL", "http://127.0.0.1:8400").rstrip("/")
VELES_API_TOKEN = os.environ.get("VELES_API_TOKEN", "")
# Подпись provenance на ответах базовой модели: показывает, что текст
# НЕ из базы знаний. Пустая переменная выключает подпись. {model} подставляется.
SOURCE_FOOTER = os.environ.get(
    "BRIDGE_SOURCE_FOOTER",
    "\n\n<i>— ответ общей модели {model}, без обращения к базе знаний. "
    "Спросить базу знаний: /kb &lt;вопрос&gt;</i>",
)
PAIR_MODE = os.environ.get("PAIR_MODE", "1") == "1"
STATE_FILE = Path(os.environ.get("BRIDGE_STATE_FILE", "bridge-state.json"))
HISTORY_LIMIT = int(os.environ.get("BRIDGE_HISTORY", "16"))
if HISTORY_LIMIT <= 0:
    # hist[-0:] в python - ВЕСЬ список; 0 или меньше означало бы обратное желаемому
    logging.getLogger("tg-bridge").warning("BRIDGE_HISTORY=%s некорректен, использую 16", HISTORY_LIMIT)
    HISTORY_LIMIT = 16
MAX_MESSAGE_CHARS = 8000

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


HELP_TEXT = (
    "Личный ассистент + база знаний.\n"
    "/help - эта справка\n"
    "/new - сброс контекста диалога\n"
    "/kb - вопрос к базе знаний, ответ с источниками\n"
    "/stats - сводка базы: документы, объём\n"
    "/sources - список источников\n"
    "/toc - оглавление документа: /toc и номер из /sources\n"
    "/task текст - создать задачу с дедлайном: due:2026-10-05; длинный текст = транскрипция (GLM извлечёт задачи)\n"
    "/tasks - открытые задачи; /done часть-id - закрыть\n"
    "/note текст - заметка; /notes - последние заметки\n"
    "Обычное сообщение - вопрос к базе знаний."
)
KNOWN_COMMANDS = {"/start", "/new", "/help", "/kb", "/stats", "/sources", "/toc",
                  "/task", "/tasks", "/done", "/note", "/notes"}
TG_COMMANDS = [
    {"command": "new", "description": "Сброс контекста диалога"},
    {"command": "help", "description": "Справка по командам"},
    {"command": "kb", "description": "Вопрос к базе знаний (с источниками)"},
    {"command": "stats", "description": "Сводка базы знаний"},
    {"command": "sources", "description": "Список источников"},
    {"command": "toc", "description": "Оглавление документа: /toc <номер>"},
    {"command": "task", "description": "Создать задачу: /task текст"},
    {"command": "tasks", "description": "Мои открытые задачи"},
    {"command": "done", "description": "Закрыть задачу: /done <часть id>"},
    {"command": "note", "description": "Заметка: /note текст"},
    {"command": "notes", "description": "Последние заметки"},
]


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


def _tg_send(chat_id: int, text: str, parse_html: bool) -> dict | None:
    """sendMessage с одним ретраем: транспортные ошибки и 429 (retry_after)."""
    params: dict = {"chat_id": chat_id, "text": text}
    if parse_html:
        params["parse_mode"] = "HTML"
    for attempt in range(2):
        try:
            sent = tg("sendMessage", **params)
        except httpx.HTTPError as e:
            log.warning("sendMessage transport error (%s), попытка %s", type(e).__name__, attempt + 1)
            time.sleep(3)
            continue
        if sent.get("ok"):
            return sent
        retry_after = (sent.get("parameters") or {}).get("retry_after")
        if retry_after:
            log.warning("rate limited, sleep %ss", retry_after)
            time.sleep(int(retry_after) + 1)
            continue
        return sent
    return None


def apply_footer(answer_html: str, model: str) -> str:
    """HTML-подпись provenance; replace вместо format (env-строка с фигурными
    скобками не должна ронять ответ)."""
    if not SOURCE_FOOTER:
        return answer_html
    return answer_html + SOURCE_FOOTER.replace("{model}", model or "по умолчанию")


IMG_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")


def send_images(chat_id: int, content: str) -> None:
    """Картинки из markdown-ответа -> sendPhoto (до 3, до 9 МБ каждая).
    Файлы скачиваются с фасада обычным HTTP (без прокси Telegram)."""
    for url in IMG_RE.findall(content)[:3]:
        try:
            r = veles_client.get(url)
            r.raise_for_status()
            data = r.content
            if not data or len(data) > 9 * 1024 * 1024:
                continue
        except Exception:  # noqa: BLE001
            log.exception("image download failed")
            continue
        try:
            tg_client.post(
                f"{TG_API}/sendPhoto",
                data={"chat_id": str(chat_id)},
                files={"photo": ("image.png", data, "image/png")},
            )
        except httpx.HTTPError:
            log.exception("sendPhoto failed")


def send(chat_id: int, text: str, footer_model: str = "") -> None:
    """Отправить markdown-текст: куски по абзацам, каждый как Telegram-HTML
    (перенос пайплайна донора); ok=false -> plain fallback куска; транспортные
    ошибки и 429 - ретрай. Лимит 3400 plain с запасом до 4096 Telegram.
    footer_model: HTML-подпись provenance на последнем куске ПОСЛЕ конвертации
    (иначе конвертер двойно-экранирует теги подписи)."""
    text = (text or "").strip() or "(пустой ответ)"
    chunks = split_for_telegram(text, limit=3400)
    for i, chunk in enumerate(chunks):
        html_text = format_markdown_to_html(chunk)
        if footer_model and i == len(chunks) - 1:
            html_text = apply_footer(html_text, footer_model)
        sent = _tg_send(chat_id, html_text, parse_html=True)
        if sent is None or not sent.get("ok"):
            log.warning(
                "HTML send failed (%s), fallback to plain",
                str((sent or {}).get("description"))[:120],
            )
            _tg_send(chat_id, strip_html_tags(html_text), parse_html=False)


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
        else:
            return f"Источник №{arg} не найден (всего {len(docs)}). Список: /sources"
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
    text = (msg.get("text") or "").strip()[:MAX_MESSAGE_CHARS]
    if not chat_id or not text:
        return owner

    if owner is None:
        if PAIR_MODE and text.startswith("/start"):
            save_owner(chat_id)
            send(chat_id, "Бот спарен с этим чатом. " + HELP_TEXT)
            return chat_id
        log.warning("drop message from unpaired chat %s", chat_id)
        return owner
    if chat_id != owner:
        log.warning("drop message from non-owner chat %s", chat_id)
        return owner

    if text in ("/start", "/help"):
        send(chat_id, HELP_TEXT)
        return owner
    if text == "/new":
        histories.pop(chat_id, None)
        send(chat_id, "Контекст сброшен.")
        return owner

    if text in ("/stats", "/sources") or text.startswith("/toc"):
        try:
            stats = veles_stats()
        except Exception as e:  # noqa: BLE001 - наружу только тип (str может нести URL)
            log.exception("veles stats failed")
            send(chat_id, f"База знаний недоступна ({type(e).__name__}); повторите позже.")
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
        except Exception as e:  # noqa: BLE001 - наружу только тип (str может нести URL)
            log.exception("kb call failed")
            send(chat_id, f"Ошибка запроса к базе знаний: {type(e).__name__}")
            return owner
        send(chat_id, answer)
        send_images(chat_id, answer)
        return owner

    if text in ("/task", "/note") or text.startswith(("/task ", "/note ")):
        kind = text[:5].strip().lstrip("/")
        body = text[5:].strip()
        if not body:
            send(chat_id, f"Использование: /{kind} текст")
            return owner
        if not TRACKER_ROOT:
            send(chat_id, "Трекер не настроен (TRACKER_ROOT).")
            return owner
        if len(body) > 1500:
            source = "transcript"
            try:
                tasks = tracker.extract_tasks(
                    lambda prompt: webui_chat(
                        [{"role": "user", "content": prompt}], "veles-knowledge"
                    ),
                    body,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("task extraction failed")
                tasks = []
            if not tasks:
                send(chat_id, "Задач в тексте не нашлось (или LLM недоступен).")
                return owner
            ids = [tracker.create_item(Path(TRACKER_ROOT), chat_id, "task", t,
                                        source=source) for t in tasks]
            send(chat_id, "Извлечённые задачи:\n" +
                 "\n".join(f"• {t} ({i.split('-', 3)[-1]})" for t, i in zip(tasks, ids)))
            return owner
        item_id = tracker.create_item(Path(TRACKER_ROOT), chat_id, kind, body)
        send(chat_id, f"Создано: {item_id}")
        return owner

    if text in ("/tasks", "/notes") or text.startswith("/done "):
        kind = "task" if text == "/tasks" or text.startswith("/done") else "note"
        if not TRACKER_ROOT:
            send(chat_id, "Трекер не настроен (TRACKER_ROOT).")
            return owner
        if text.startswith("/done "):
            send(chat_id, tracker.set_status(Path(TRACKER_ROOT), chat_id, "task",
                                             text[6:].strip(), "done"))
            return owner
        items = tracker.list_items(Path(TRACKER_ROOT), chat_id, kind)
        if not items:
            send(chat_id, "Пока пусто.")
            return owner
        lines = [f"• [{i['status']}] {i['id']} - {i['title']}" for i in items]
        send(chat_id, f"{'Открытые задачи' if kind == 'task' else 'Заметки'}:\n" +
             "\n".join(lines))
        return owner

    if text.startswith("/") and text.split(" ", 1)[0] not in KNOWN_COMMANDS:
        # неизвестная команда не уходит модели как обычный вопрос
        send(chat_id, f"Неизвестная команда {text.split(' ', 1)[0]}. " + HELP_TEXT)
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
        send(chat_id, f"Ошибка запроса к модели: {type(e).__name__}")
        return owner
    hist.append({"role": "assistant", "content": answer})
    del hist[:-HISTORY_LIMIT]
    send(chat_id, answer, footer_model=DEFAULT_MODEL)
    send_images(chat_id, answer)
    return owner


def reminder_loop() -> None:
    """Напоминалки: задача с due сегодня/просроченная и не done -> сообщение
    владельцу раз в сутки (метка reminded: <день> в файле)."""
    if not TRACKER_ROOT:
        return
    root = Path(TRACKER_ROOT)
    while True:
        try:
            today = time.strftime("%Y-%m-%d")
            for user in root.iterdir():
                if not user.is_dir() or user.name.startswith("."):
                    continue
                tdir = user / "tasks"
                if not tdir.exists():
                    continue
                for p in tdir.glob("*.md"):
                    text = p.read_text(encoding="utf-8")
                    m = re.search(r"(?m)^due: (\d{4}-\d{2}-\d{2})$", text)
                    if not m or m.group(1) > today or "status: done" in text:
                        continue
                    if f"reminded: {today}" in text:
                        continue
                    tm = re.search(r'(?m)^title: "?([^"\n]+?)"?$', text)
                    try:
                        send(int(user.name), f"⏰ Задача к сроку {m.group(1)}: "
                             f"{tm.group(1) if tm else p.stem}")
                    except Exception:  # noqa: BLE001 - чат мог отвалиться, не роняем цикл
                        log.exception("reminder send failed")
                        continue
                    p.write_text(text + f"reminded: {today}\n", encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.exception("reminder loop failed")
        time.sleep(300)


def main() -> int:
    owner = OWNER_CHAT_ID or load_owner()
    if owner is None and not PAIR_MODE:
        log.error("PAIR_MODE=0 и владелец не задан (OWNER_CHAT_ID/state-file) - бот был бы глух; отказ старта")
        return 1
    me = tg("getMe")
    if not me.get("ok"):
        log.error("getMe failed: %s", str(me)[:200])
        return 1
    # адрес чата для веб-страницы трекера: мост - источник истины, фасад читает файл
    if TRACKER_ROOT:
        try:
            Path(TRACKER_ROOT).mkdir(parents=True, exist_ok=True)
            (Path(TRACKER_ROOT) / "chat-url.txt").write_text(
                f"https://t.me/{me['result'].get('username', '')}", encoding="utf-8"
            )
        except OSError:
            log.warning("не смог записать chat-url.txt в TRACKER_ROOT")
    threading.Thread(target=reminder_loop, daemon=True).start()
    menu = tg("setMyCommands", commands=TG_COMMANDS)
    if not menu.get("ok"):
        log.warning("setMyCommands failed: %s", str(menu)[:150])
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
