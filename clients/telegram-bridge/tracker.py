"""Персональный трекер veles: задачи и заметки в markdown (стиль ai-tracker).

Хранение: <root>/<chat_id>/tasks/task-ГГГГММДД-<slug>.md и notes/note-...md.
Файл = YAML frontmatter + тело; читается Obsidian'ом как есть; каталог - git-репо
(инициализируется при первой записи, коммит после каждого изменения).

Требования владельца (28.09): персональные задачи и заметки на пользователя,
просмотр через Obsidian, создание в Telegram в т.ч. из транскрипций (длинный
текст /task - GLM извлекает задачи), стык с Jira - следующие фазы.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
DUE_RE = re.compile(r"\b(?:due|дедлайн)\s*[:=]\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE)
MAX_LIST = 20
GIT_LOCK_TIMEOUT = 30.0
# служебные файлы каталога-репозитория, в коммиты не должны попадать
GITIGNORE_ENTRIES = (".git-lock", ".*.tmp-*", "chat-url.txt")

log = logging.getLogger(__name__)


class LockTimeout(RuntimeError):
    """Не дождались flock на .git-lock за GIT_LOCK_TIMEOUT секунд."""


def _atomic_write(path: Path, text: str) -> None:
    """Запись в md через tmp-файл в том же каталоге + os.replace.

    Читатель (бот, фасад, Obsidian) видит файл целиком старым или целиком
    новым; полузаписанного файла и обрывков не бывает. tmp создаётся open
    с O_EXCL и правами 0600: предсказуемое имя + symlink-гонка закрыты
    (O_EXCL поддержан и на Windows), данные выгоняются fsync до os.replace
    (карточка переживает сбой питания); при ошибке tmp удаляется.

    os.replace на Windows падает PermissionError, если целевой файл в этот
    момент ОТКРЫТ читателем (читать-пока-пишут законно, а вот replace поверх
    открытого - нет; на Linux replace всегда проходит). Ретраим с короткой
    паузой: чтение читателя длится миллисекунды, инвариант не нарушается
    (до успешного replace читатель видит целиком старый файл). На POSIX
    PermissionError не возникает - ретраи мертвы.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.002 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink()


@contextlib.contextmanager
def _git_lock(root: Path):
    """Эксклюзивный лок каталога: flock на <root>/.git-lock.

    Два одновременных /task от разных пользователей не должны сталкиваться на
    git index.lock - запись карточки и git-фазу (add+commit) выполняет только
    держатель лока, поэтому каждое изменение = свой коммит (и git add -A не
    захватывает чужую незакоммиченную запись). Таймаут GIT_LOCK_TIMEOUT секунд,
    после - LockTimeout с понятным текстом. На Windows fcntl нет: работаем без
    лока (fallback для одного пользователя; при реальном конфликте git сам
    вернёт ошибку index.lock в _git, check=False её проглотит). Лок обязателен
    там, где git есть (Linux-хосты, стенд).
    """
    lock_path = Path(root) / ".git-lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # Windows-хост: flock недоступен, см. докстринг
        yield
        return
    f = open(lock_path, "a+")
    deadline = time.monotonic() + GIT_LOCK_TIMEOUT
    while True:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if time.monotonic() >= deadline:
                f.close()
                raise LockTimeout(
                    f"не дождались flock на {lock_path} за {GIT_LOCK_TIMEOUT:.0f} с: "
                    "git-операцию держит другой процесс; повторите команду позже"
                )
            time.sleep(0.1)
    try:
        yield
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def parse_due(text: str) -> str:
    """Дедлайн из текста: 'due:2026-10-05' или 'дедлайн: 2026-10-05'."""
    m = DUE_RE.search(text)
    return m.group(1) if m else ""


def _slug(text: str) -> str:
    # только латиница/цифры: кириллические имена файлов ломают запись на
    # некоторых ФС/инструментах; кириллица живёт в frontmatter title
    words = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(words[:4])
    if not slug:
        slug = "i-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return slug


def _git(root: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", *args], cwd=root, check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        # отказ git (index.lock, битый репо) не глушим: хвост stderr в журнал
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        log.warning("git %s: код %s: %s", " ".join(args), proc.returncode, err[:200])


def _git_commit(root: Path, message: str) -> None:
    """git-фаза без взятия лока: вызывать только под _git_lock."""
    if not (root / ".git").exists():
        _git(root, "init", "-q")
        _git(root, "config", "user.name", "veles-tracker")
        _git(root, "config", "user.email", "veles@local")
    gi = root / ".gitignore"
    if not gi.exists():
        # самохил: лечит и существующие установки (напр. стенд) - служебные
        # .git-lock, tmp-файлы записи и chat-url.txt не идут в коммиты
        gi.write_text("\n".join(GITIGNORE_ENTRIES) + "\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def _user_dir(root: Path, chat_id: int | str, kind: str) -> Path:
    d = Path(root) / str(chat_id) / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_item(root: Path, chat_id: int | str, kind: str, text: str,
                status: str = "todo", source: str = "telegram") -> str:
    """Создать md-файл задачи/заметки; вернуть id (имя файла без .md).

    Подбор порядкового номера и запись - целиком под _git_lock: два
    одновременных /task одного пользователя иначе выбирают один item_id,
    и вторая запись затирает первую. Номер перепроверяется по существованию
    пути: файл без валидной шапки list_items не видит, а имя занять может.
    """
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    base = _slug(text)
    title = DUE_RE.sub("", text).strip().splitlines()[0][:120].strip()
    due = parse_due(text)
    root = Path(root)
    with _git_lock(root):
        items = list_items(root, chat_id, kind, include_done=True)
        n = 1
        path = _user_dir(root, chat_id, kind) / f"{kind}-{day}-{n:02d}-{base}.md"
        while any(i["id"].startswith(f"{kind}-{day}-{n:02d}-") for i in items) \
                or path.exists():
            n += 1
            path = path.with_name(f"{kind}-{day}-{n:02d}-{base}.md")
        item_id = path.stem
        fm = [
            "---",
            f"id: {item_id}",
            f"title: \"{title.replace(chr(34), chr(39))}\"",
            f"status: {status}",
            f"created: {now.isoformat(timespec='seconds')}",
            f"source: {source}",
        ]
        if due:
            fm.append(f"due: {due}")
        fm += ["---", "", text.strip(), ""]
        _atomic_write(path, "\n".join(fm))
        _git_commit(root, f"{kind}: {title[:60]}")
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
            "due": fields.get("due", ""),
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
    root = Path(root)
    with _git_lock(root):
        _atomic_write(p, new)
        _git_commit(root, f"{kind}: {p.stem} -> {status}")
    return f"{p.stem} -> {status}"


def _reminder_title(text: str, stem: str, today: str) -> str | None:
    """Условия напоминания по тексту карточки: title, если пора, иначе None.

    Есть due <= today, статус не done и метки reminded: {today} ещё нет
    (метка ищется по всему файлу: она живёт последней строкой файла,
    не в frontmatter).
    """
    m = re.search(r"(?m)^due: (\d{4}-\d{2}-\d{2})$", text)
    if not m or m.group(1) > today or "status: done" in text:
        return None
    if f"reminded: {today}" in text:
        return None
    tm = re.search(r'(?m)^title: "?([^"\n]+?)"?$', text)
    return tm.group(1) if tm else stem


def reminder_due(path: Path, today: str) -> tuple[str, str] | None:
    """Пора ли напоминать по карточке: (due, title), если да, иначе None.

    Одиночное чтение без лока (атомарность чтения дают писатели через
    _atomic_write); решение принимается по снимку файла, поэтому перед
    отправкой напоминания условия перепроверяет mark_reminded. Метка
    reminded: дописывается последней строкой файла, не в frontmatter.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    title = _reminder_title(text, path.stem, today)
    if title is None:
        return None
    due = re.search(r"(?m)^due: (\d{4}-\d{2}-\d{2})$", text).group(1)
    return (due, title)


def mark_reminded(root: Path, path: Path, today: str) -> bool:
    """Поставить метку reminded: <today> под _git_lock; вернуть, поставлена ли.

    Между reminder_due и этой функцией секунды (отправка в Telegram): статус
    могли успеть сменить на done или метку уже поставил другой процесс -
    файл перечитывается и те же условия перепроверяются. Метка reminded:
    дописывается последней строкой файла (с гарантированным переводом строки
    перед ней), не в frontmatter; запись через _atomic_write + _git_commit.
    """
    root = Path(root)
    path = Path(path)
    with _git_lock(root):
        text = path.read_text(encoding="utf-8")
        if _reminder_title(text, path.stem, today) is None:
            return False
        if not text.endswith("\n"):
            text += "\n"
        text += f"reminded: {today}\n"
        _atomic_write(path, text)
        _git_commit(root, f"task: {path.stem} напомянуто {today}")
    return True


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
