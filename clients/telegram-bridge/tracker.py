"""Персональный трекер veles: задачи и заметки в markdown (стиль ai-tracker).

Хранение: <root>/<chat_id>/tasks/task-ГГГГММДД-<slug>.md и notes/note-...md.
Файл = YAML frontmatter + тело; читается Obsidian'ом как есть; каталог - git-репо
(инициализируется при первой записи, коммит после каждого изменения).

Требования владельца (28.09): персональные задачи и заметки на пользователя,
просмотр через Obsidian, создание в Telegram в т.ч. из транскрипций (длинный
текст /task - GLM извлекает задачи), стык с Jira - следующие фазы.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
MAX_LIST = 20


def _slug(text: str) -> str:
    # только латиница/цифры: кириллические имена файлов ломают запись на
    # некоторых ФС/инструментах; кириллица живёт в frontmatter title
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words[:4])
    if not slug:
        slug = "i-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return slug


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=root, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _commit(root: Path, message: str) -> None:
    if not (root / ".git").exists():
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "veles-tracker")
        _git(root, "config", "user.email", "veles@local")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _user_dir(root: Path, chat_id: int | str, kind: str) -> Path:
    d = Path(root) / str(chat_id) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_item(root: Path, chat_id: int | str, kind: str, text: str,
                status: str = "todo", source: str = "telegram") -> str:
    """Создать md-файл задачи/заметки; вернуть id (имя файла без .md)."""
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    base = _slug(text)
    items = list_items(root, chat_id, kind, include_done=True)
    n = 1
    while any(i["id"].startswith(f"{kind}-{day}-{n:02d}-") for i in items):
        n += 1
    item_id = f"{kind}-{day}-{n:02d}-{base}"
    title = text.strip().splitlines()[0][:120]
    fm = [
        "---",
        f"id: {item_id}",
        f"title: \"{title.replace(chr(34), chr(39))}\"",
        f"status: {status}",
        f"created: {now.isoformat(timespec='seconds')}",
        f"source: {source}",
        "---",
        "",
        text.strip(),
        "",
    ]
    path = _user_dir(root, chat_id, kind) / f"{item_id}.md"
    path.write_text("\n".join(fm), encoding="utf-8")
    _commit(Path(root), f"{kind}: {title[:60]}")
    return item_id


def list_items(root: Path, chat_id: int | str, kind: str,
               include_done: bool = False) -> list[dict]:
    d = _user_dir(root, chat_id, kind)
    out = []
    for p in sorted(d.glob("*.md"), reverse=True)[:MAX_LIST * 3]:
        m = FRONTMATTER_RE.match(p.read_text(encoding="utf-8"))
        if not m:
            continue
        fields = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip()] = v.strip()
        if not include_done and fields.get("status") == "done":
            continue
        body = (m.group(2) or "").strip()
        out.append({
            "id": fields.get("id", p.stem),
            "title": fields.get("title", p.stem).strip("\"'"),
            "status": fields.get("status", "todo"),
            "created": fields.get("created", ""),
            "preview": body[:100],
        })
        if len(out) >= MAX_LIST:
            break
    return out


def set_status(root: Path, chat_id: int | str, kind: str, id_part: str, status: str) -> str:
    d = _user_dir(root, chat_id, kind)
    matches = [p for p in d.glob("*.md") if id_part in p.stem]
    if not matches:
        return f"не найдено: {id_part}"
    p = matches[0]
    text = p.read_text(encoding="utf-8")
    new = re.sub(r"(?m)^status: .*$", f"status: {status}", text, count=1)
    if new == text:
        return f"у {p.stem} уже такой статус"
    p.write_text(new, encoding="utf-8")
    _commit(Path(root), f"{kind}: {p.stem} -> {status}")
    return f"{p.stem} -> {status}"


def extract_tasks(llm_answer_general, text: str) -> list[str]:
    """Длинный текст (транскрипция) -> список задач через LLM. llm_answer_general -
    функция (prompt) -> str; парсим строки-пункты из ответа."""
    prompt = (
        "Из текста ниже извлеки задачи/поручения/сделать-пункты. "
        "Верни ТОЛЬКО нумерованный список коротких формулировок задач, "
        "по одной на строку, без вступлений. Если задач нет - верни ровно: НЕТ\n\n"
        f"Текст:\n{text[:12000]}"
    )
    answer = llm_answer_general(prompt)
    tasks = []
    for line in answer.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.)]\s+(.+)$", line)
        if m and not line.upper().startswith("НЕТ"):
            tasks.append(m.group(1).strip()[:200])
    return tasks[:10]


def short_id(item_id: str) -> str:
    return hashlib.sha1(item_id.encode()).hexdigest()[:6]
