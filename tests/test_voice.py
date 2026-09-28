"""Регрессия голосового потока моста: download/submit/poll, handle_voice ветки."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:stub")
os.environ.setdefault("OPENWEBUI_API_KEY", "sk-stub")
os.environ.setdefault("VOXFUSION_API_TOKEN", "vx-stub")
os.environ.setdefault("VOXFUSION_API_URL", "http://vx-stub:8500")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

import voice  # noqa: E402


class FakeResponse:
    def __init__(self, data=None, status=200, text=""):
        self._data = data if data is not None else {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeVx:
    """Фейковый клиент voxfusion-api: job проходит queued→running→done."""

    def __init__(self, statuses=None, fail=False):
        self.statuses = list(statuses or ["queued", "running", "done"])
        self.fail = fail
        self.text = "Распознанный текст встречи."
        self.submitted = 0

    def get(self, url):
        import time

        time.sleep(0)
        if self.fail:
            return FakeResponse({}, status=500)
        idx = min(self.submitted, len(self.statuses) - 1)
        status = self.statuses[idx]
        if status == "done":
            self.submitted += 1
        if url.endswith("jobs/x1"):
            self.submitted += 0
        return FakeResponse({"status": status, "text": self.text,
                             "error": "boom" if status == "error" else None})

    def post(self, url, files=None, data=None):
        self.submitted += 1
        if self.fail:
            return FakeResponse({}, status=500)
        return FakeResponse({"job_id": "x1"})


def test_submit_and_poll_happy_path(tmp_path) -> None:
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS fake")
    vx = FakeVx(statuses=["queued", "done"])
    job = voice.submit_job(vx, audio, language="ru")
    assert job == "x1"
    # подменяем интервал, чтобы тест не спал
    voice.POLL_INTERVAL_S = 0
    status, text = voice.poll_result(vx, "x1", timeout_s=5)
    assert status == "done" and text.startswith("Распознанный")


def test_poll_error_and_timeout(tmp_path) -> None:
    voice.POLL_INTERVAL_S = 0
    vx = FakeVx(statuses=["error"])
    status, text = voice.poll_result(vx, "x1", timeout_s=5)
    assert status == "error" and "boom" in text
    status, _ = voice.poll_result(FakeVx(statuses=["queued"]), "x1", timeout_s=0.2)
    assert status == "timeout"


def test_feature_disabled_without_token(monkeypatch) -> None:
    monkeypatch.setenv("VOXFUSION_API_TOKEN", "")
    assert not voice.feature_enabled()
    monkeypatch.setenv("VOXFUSION_API_TOKEN", "t")
    assert voice.feature_enabled()


def test_handle_voice_full_flow(monkeypatch, tmp_path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))
    import bridge

    sent: list[str] = []
    monkeypatch.setattr(bridge, "send", lambda chat_id, text, footer_model="": sent.append(text))
    monkeypatch.setattr(bridge, "TRACKER_ROOT", str(tmp_path))
    monkeypatch.setattr(bridge, "VOXFUSION_API_TOKEN", "vx-stub", raising=False)
    monkeypatch.setattr(bridge, "tg_client", FakeVx(), raising=False)
    monkeypatch.setattr(bridge, "veles_client", FakeVx(), raising=False)
    monkeypatch.setattr(bridge.voice, "download_tg_voice", lambda tc, fid, d: d / "v.ogg")
    monkeypatch.setattr(bridge.voice, "submit_job", lambda vc, path, language="ru": "job1")
    monkeypatch.setattr(bridge.voice, "poll_result", lambda vc, jid, timeout_s: ("done", "текст встречи"))
    monkeypatch.setattr(bridge, "webui_chat", lambda messages, model: "1. Согласовать смету")

    (tmp_path).mkdir(exist_ok=True)
    bridge.handle_voice(100, {"file_id": "f1", "duration": 30})
    assert any("Транскрипция:" in t and "текст встречи" in t for t in sent)

    # /vt режим: голосовое -> задачи
    sent.clear()
    monkeypatch.setattr(bridge, "VOICE_PENDING", {100})
    bridge.handle_voice(100, {"file_id": "f2", "duration": 30})
    tasks = bridge.tracker.list_items(Path(tmp_path), 100, "task")
    assert tasks, "задачи из голосового должны создаваться"
    assert any("Согласовать смету" in i["title"] for i in tasks)


def test_handle_voice_no_token(monkeypatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))
    import bridge

    sent: list[str] = []
    monkeypatch.setattr(bridge, "send", lambda chat_id, text, footer_model="": sent.append(text))
    monkeypatch.setattr(bridge, "VOXFUSION_API_TOKEN", "")
    bridge.handle_voice(100, {"file_id": "f1", "duration": 10})
    assert sent and "не настроена" in sent[0]
