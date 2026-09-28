"""Регрессия трекера: md-хранение, статусы, извлечение задач из транскрипции."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

import tracker


def test_create_and_list_task(tmp_path) -> None:
    tid = tracker.create_item(tmp_path, 100, "task", "Подготовить отчёт по базе")
    assert tid.startswith("task-") and "i-" in tid  # кириллица -> хеш-слаг
    items = tracker.list_items(tmp_path, 100, "task")
    assert len(items) == 1 and items[0]["status"] == "todo"
    assert items[0]["title"] == "Подготовить отчёт по базе"
    text = (tmp_path / "100" / "task" / f"{tid}.md").read_text(encoding="utf-8")
    assert "status: todo" in text and "Подготовить отчёт" in text


def test_done_filters_and_set_status(tmp_path) -> None:
    t1 = tracker.create_item(tmp_path, 100, "task", "Задача раз")
    tracker.create_item(tmp_path, 100, "task", "Задача два")
    msg = tracker.set_status(tmp_path, 100, "task", t1[-8:], "done")
    assert "done" in msg
    open_items = tracker.list_items(tmp_path, 100, "task")
    assert [i["title"] for i in open_items] == ["Задача два"]
    all_items = tracker.list_items(tmp_path, 100, "task", include_done=True)
    assert len(all_items) == 2
    assert "не найдено" in tracker.set_status(tmp_path, 100, "task", "nope", "done")


def test_notes_isolated_from_tasks(tmp_path) -> None:
    tracker.create_item(tmp_path, 100, "note", "Мысль про дедлайн")
    assert tracker.list_items(tmp_path, 100, "task") == []
    assert len(tracker.list_items(tmp_path, 100, "note")) == 1


def test_extract_tasks_parses_numbered_list() -> None:
    def fake_llm(prompt: str) -> str:
        return "1. Согласовать смету\n2. Позвонить поставщику\n3. Обновить график"

    assert tracker.extract_tasks(fake_llm, "длинная транскрипция...") == [
        "Согласовать смету", "Позвонить поставщику", "Обновить график"
    ]


def test_extract_tasks_none_case() -> None:
    assert tracker.extract_tasks(lambda p: "НЕТ", "кухонный рецепт") == []


# --- K2: атомарная запись и git-lock ---


def test_atomic_write_reader_sees_whole_file(tmp_path) -> None:
    """Во время записи читатель видит файл целиком старым или целиком новым."""
    target = tmp_path / "task" / "card.md"
    target.parent.mkdir()
    old_text = "status: todo\n" + "x" * 5000
    new_text = "status: done\n" + "y" * 5000
    target.write_text(old_text, encoding="utf-8")
    stop = threading.Event()
    partial: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            data = target.read_text(encoding="utf-8")
            if data not in (old_text, new_text):
                partial.append(data[:80])

    rd = threading.Thread(target=reader)
    rd.start()
    try:
        for _ in range(60):
            tracker._atomic_write(target, new_text)
            tracker._atomic_write(target, old_text)
    finally:
        stop.set()
        rd.join(10)

    assert partial == []
    assert list(target.parent.glob("*tmp-*")) == []  # tmp после записи не остаётся


def test_concurrent_set_status_two_threads(tmp_path) -> None:
    """2 потока по разным карточкам одного каталога: обе записи, git без index.lock."""
    t1 = tracker.create_item(tmp_path, 100, "task", "Задача один")
    t2 = tracker.create_item(tmp_path, 100, "task", "Задача два")
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(item_id: str) -> None:
        try:
            barrier.wait(timeout=10)
            tracker.set_status(tmp_path, 100, "task", item_id[-8:], "done")
        except Exception as exc:  # noqa: BLE001 - собираем исключения обоих потоков
            errors.append(exc)

    th1 = threading.Thread(target=worker, args=(t1,))
    th2 = threading.Thread(target=worker, args=(t2,))
    th1.start()
    th2.start()
    th1.join(60)
    th2.join(60)

    assert errors == []
    items = tracker.list_items(tmp_path, 100, "task", include_done=True)
    assert sorted(i["status"] for i in items) == ["done", "done"]
    log = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "--oneline"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    dirty = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert dirty == ""  # no-loss: оба изменения закоммичены, ничего не висит
    try:
        import fcntl  # noqa: F401
        have_fcntl = True
    except ImportError:
        have_fcntl = False
    if have_fcntl:
        # лок дисциплинирует git: ровно 2 create + 2 set_status,
        # потерянных коммитов из-за index.lock нет
        assert len(log) == 4, f"ожидали 4 коммита, получили: {log}"
    else:
        assert len(log) >= 2  # Windows без flock: не ловим гонку, проверяем запись


def test_git_lock_timeout_negative(tmp_path, monkeypatch) -> None:
    """Занятый .git-lock -> LockTimeout с понятным текстом, не зависание."""
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(tracker, "GIT_LOCK_TIMEOUT", 0.5)
    held = open(tmp_path / ".git-lock", "a+")
    fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        t0 = time.monotonic()
        with pytest.raises(tracker.LockTimeout) as exc_info:
            with tracker._git_lock(tmp_path):
                pass
        elapsed = time.monotonic() - t0
    finally:
        fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        held.close()

    assert ".git-lock" in str(exc_info.value)
    assert 0.3 <= elapsed < 20  # упало по таймауту, а не зависло навсегда
    with tracker._git_lock(tmp_path):  # после снятия лока вход снова возможен
        pass


# --- ревью K2: гонка NN-номера и напоминалка ---


def test_concurrent_create_two_threads(tmp_path) -> None:
    """2 потока создают задачи ОДНОГО пользователя: номера не сталкиваются.

    До фикса подбор номера шёл до лока: оба потока выбирали один item_id,
    второй os.replace затирал карточку первого.
    """
    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    ids: list[str] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            for text in ("Задача раз", "Задача два"):
                ids.append(tracker.create_item(tmp_path, 100, "task", text))
        except Exception as exc:  # noqa: BLE001 - собираем исключения обоих потоков
            errors.append(exc)

    th1 = threading.Thread(target=worker)
    th2 = threading.Thread(target=worker)
    th1.start()
    th2.start()
    th1.join(60)
    th2.join(60)

    assert errors == []
    assert len(ids) == 4
    log = subprocess.run(
        ["git", "-C", str(tmp_path), "log", "--oneline"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    dirty = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--porcelain"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert dirty == ""  # no-loss: все создания закоммичены, ничего не висит
    try:
        import fcntl  # noqa: F401
        have_fcntl = True
    except ImportError:
        have_fcntl = False
    if have_fcntl:
        # лок дисциплинирует нумерацию: id все разные (затирания нет),
        # все 4 карточки на диске, 4 create = 4 коммита, потерянных нет
        assert len(set(ids)) == 4, f"id столкнулись: {ids}"
        files = list((tmp_path / "100" / "task").glob("*.md"))
        assert len(files) == 4
        assert len(log) == 4, f"ожидали 4 коммита, получили: {log}"
    else:
        # Windows без flock: гонку не ловим (лок no-op), проверяем запись
        assert len(set(ids)) >= 2
        assert len(list((tmp_path / "100" / "task").glob("*.md"))) >= 2
        assert len(log) >= 2


def test_create_skips_existing_target(tmp_path) -> None:
    """Занятое имя, невидимое list_items (нет валидной шапки), перескакивается
    по существованию пути - порядковый номер не переиспользуется."""
    d = tmp_path / "100" / "task"
    d.mkdir(parents=True)
    day = datetime.now().strftime("%Y%m%d")
    occupied = d / f"task-{day}-01-test.md"
    occupied.write_text("просто текст без шапки - list_items такой файл не видит\n",
                        encoding="utf-8")
    item_id = tracker.create_item(tmp_path, 100, "task", "Test")
    assert item_id == f"task-{day}-02-test"
    assert occupied.read_text(encoding="utf-8").startswith("просто текст")


def test_reminder_due_cases(tmp_path) -> None:
    """reminder_due: да - due<=today, не done, метки нет; нет - done,
    уже напомянуто, due в будущем, due нет вовсе."""
    today = time.strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    tid = tracker.create_item(tmp_path, 100, "task", f"Будняя due:{future}")
    assert tracker.reminder_due(tmp_path / "100" / "task" / f"{tid}.md", today) is None

    tid = tracker.create_item(tmp_path, 100, "task", "Без срока вовсе")
    assert tracker.reminder_due(tmp_path / "100" / "task" / f"{tid}.md", today) is None

    tid = tracker.create_item(tmp_path, 100, "task", "Закрыта due:2026-01-01")
    tracker.set_status(tmp_path, 100, "task", tid[-8:], "done")
    assert tracker.reminder_due(tmp_path / "100" / "task" / f"{tid}.md", today) is None

    tid = tracker.create_item(tmp_path, 100, "task", "Просрочка due:2026-01-01")
    p = tmp_path / "100" / "task" / f"{tid}.md"
    assert tracker.reminder_due(p, today) == ("2026-01-01", "Просрочка")
    # уже напомянуто сегодня - второй раз не напоминаем
    assert tracker.mark_reminded(tmp_path, p, today) is True
    assert tracker.reminder_due(p, today) is None


def test_mark_reminded_positive(tmp_path) -> None:
    """due в прошлом, todo -> mark_reminded True: метка последней строкой
    файла (не в frontmatter) + коммит."""
    tracker.create_item(tmp_path, 100, "task", "Оплатить счёт due:2026-01-15")
    path = next((tmp_path / "100" / "task").glob("*.md"))
    today = time.strftime("%Y-%m-%d")
    assert tracker.reminder_due(path, today) == ("2026-01-15", "Оплатить счёт")

    def commits() -> int:
        out = subprocess.run(
            ["git", "-C", str(tmp_path), "log", "--oneline"],
            capture_output=True, text=True,
        ).stdout
        return len(out.strip().splitlines())

    n0 = commits()
    assert tracker.mark_reminded(tmp_path, path, today) is True
    text = path.read_text(encoding="utf-8")
    body = text.split("---", 2)[2]  # всё после закрывающей шапки
    assert body.rstrip("\n").endswith(f"reminded: {today}")
    assert commits() == n0 + 1


def test_mark_reminded_negative(tmp_path) -> None:
    """После reminder_due статус сменили на done -> mark_reminded False,
    файл не изменился, метки нет."""
    tid = tracker.create_item(tmp_path, 100, "task", "Отправить отчёт due:2026-01-15")
    path = tmp_path / "100" / "task" / f"{tid}.md"
    today = time.strftime("%Y-%m-%d")
    assert tracker.reminder_due(path, today) is not None
    tracker.set_status(tmp_path, 100, "task", tid[-8:], "done")
    snapshot = path.read_text(encoding="utf-8")
    assert tracker.mark_reminded(tmp_path, path, today) is False
    assert path.read_text(encoding="utf-8") == snapshot
