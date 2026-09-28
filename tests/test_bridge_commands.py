"""Регрессия команд моста: /help, неизвестная команда, setMyCommands payload.

handle() тестируется с подменённым tg() (захват вызовов Telegram API).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:stub")
os.environ.setdefault("OPENWEBUI_API_KEY", "sk-stub")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

import bridge  # noqa: E402

CHAT = 100


class FakeTg:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, method: str, **params: object) -> dict:
        self.calls.append((method, dict(params)))
        return {"ok": True, "result": []}

    def sent_texts(self) -> list[str]:
        return [p["text"] for m, p in self.calls if m == "sendMessage"]


def run_handle(text: str, owner: int | None = CHAT) -> tuple[FakeTg, int | None]:
    fake = FakeTg()
    orig = bridge.tg
    bridge.tg = fake
    try:
        new_owner = bridge.handle({"message": {"chat": {"id": CHAT}, "text": text}}, owner)
    finally:
        bridge.tg = orig
    return fake, new_owner


def test_help_command_replies_without_llm() -> None:
    fake, _ = run_handle("/help")
    texts = fake.sent_texts()
    assert len(texts) == 1
    assert "/kb" in texts[0] and "/stats" in texts[0] and "/toc" in texts[0]


def test_start_and_help_share_text() -> None:
    fake_start, _ = run_handle("/start")
    fake_help, _ = run_handle("/help")
    assert fake_start.sent_texts()[0] == fake_help.sent_texts()[0]


def test_unknown_command_gets_help_not_llm() -> None:
    called = {"webui": False}
    fake = FakeTg()
    orig_tg, orig_webui = bridge.tg, bridge.webui_chat
    bridge.tg = fake

    def _fail(*a, **k):
        called["webui"] = True
        raise AssertionError("LLM не должен вызываться на неизвестную команду")

    bridge.webui_chat = _fail
    try:
        bridge.handle({"message": {"chat": {"id": CHAT}, "text": "/stats2 лишнее"}}, CHAT)
    finally:
        bridge.tg, bridge.webui_chat = orig_tg, orig_webui
    assert not called["webui"]
    texts = fake.sent_texts()
    assert len(texts) == 1 and "Неизвестная команда" in texts[0]


def test_known_command_with_args_still_routed() -> None:
    # /kb с аргументом не должен попасть в ветку неизвестных команд
    fake = FakeTg()
    orig_tg, orig_webui, orig_kb = bridge.tg, bridge.webui_chat, bridge.KB_MODEL
    bridge.tg = fake
    bridge.KB_MODEL = "veles-knowledge"
    bridge.webui_chat = lambda messages, model: "KB ответ [1]"
    try:
        bridge.handle({"message": {"chat": {"id": CHAT}, "text": "/kb норма управляемости"}}, CHAT)
    finally:
        bridge.tg, bridge.webui_chat, bridge.KB_MODEL = orig_tg, orig_webui, orig_kb
    assert any("KB ответ [1]" in t for t in fake.sent_texts())


def test_setmycommands_payload_registered() -> None:
    cmds = {c["command"] for c in bridge.TG_COMMANDS}
    assert {"new", "help", "kb", "stats", "sources", "toc"} <= cmds
    assert all(c["description"].strip() for c in bridge.TG_COMMANDS)
