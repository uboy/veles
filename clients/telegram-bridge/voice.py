"""Голосовые сообщения -> транскрипция через voxfusion-api (Ф голосовые, v1).

Поток: Telegram voice (ogg/opus) -> getFile -> скачивание во временный файл ->
POST /v1/transcribe (multipart) -> поллинг GET /v1/jobs/{id} -> текст.
Функции принимают клиентов параметрами (тестируемость): tg_client - с прокси
Telegram, vx_client - прямой до voxfusion-api.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

TG_DOWNLOAD_TIMEOUT = 60.0
SUBMIT_TIMEOUT = 60.0
POLL_INTERVAL_S = 10.0


def api_url() -> str:
    return os.environ.get("VOXFUSION_API_URL", "http://127.0.0.1:8500").rstrip("/")


def api_token() -> str:
    return os.environ.get("VOXFUSION_API_TOKEN", "")


def feature_enabled() -> bool:
    return bool(api_token())


def download_tg_voice(tg_client: Any, file_id: str, dest_dir: Path) -> Path:
    """Скачать файл голосового из Telegram во временный каталог."""
    r = tg_client.post(
        f"https://api.telegram.org/bot{__import__('os').environ['TELEGRAM_BOT_TOKEN']}/getFile",
        json={"file_id": file_id},
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"getFile: {str(data)[:120]}")
    file_path = data["result"]["file_path"]
    fname = Path(file_path).name or "voice.ogg"
    dest = dest_dir / fname
    with dest.open("wb") as f:
        with tg_client.stream(
            "GET", f"https://api.telegram.org/file/bot{__import__('os').environ['TELEGRAM_BOT_TOKEN']}/{file_path}"
        ) as resp:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
    return dest


def submit_job(vx_client: Any, audio_path: Path, language: str = "ru") -> str:
    """Поставить задачу транскрибации; вернуть job_id."""
    with audio_path.open("rb") as f:
        r = vx_client.post(
            f"{api_url()}/v1/transcribe",
            files={"file": (audio_path.name, f, "application/octet-stream")},
            data={"language": language},
        )
    r.raise_for_status()
    d = r.json()
    if "job_id" not in d:
        raise RuntimeError(f"transcribe: {str(d)[:120]}")
    return d["job_id"]


def poll_result(vx_client: Any, job_id: str, timeout_s: float) -> tuple[str, str]:
    """Поллинг результата до done/error/таймаута. Возвращает (status, text)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = vx_client.get(f"{api_url()}/v1/jobs/{job_id}")
        r.raise_for_status()
        d = r.json()
        status = d.get("status", "queued")
        if status == "done":
            return "done", d.get("text") or ""
        if status == "error":
            return "error", str(d.get("error") or "ошибка транскрибации")[:200]
        time.sleep(POLL_INTERVAL_S)
    return "timeout", ""
