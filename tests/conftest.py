"""Общие настройки тестового окружения veles."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_parse_isolation(monkeypatch):
    """Изоляция разбора (отдельный процесс + таймаут) в тестах выключена:
    spawn-дочерний процесс не наследует подстановки конфигурации теста и
    перепивает окружение. Сама изоляция покрыта тестами
    tests/test_audit_fixes.py (там таймаут передаётся аргументом, мимо env)."""
    monkeypatch.setenv("VELES_QUEUE_PARSE_TIMEOUT_S", "0")
