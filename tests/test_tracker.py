"""Регрессия трекера: md-хранение, статусы, извлечение задач из транскрипции."""

from __future__ import annotations

import sys
from pathlib import Path

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
