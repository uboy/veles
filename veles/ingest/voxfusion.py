"""Клиент VoxFusion (транскрибация аудио) для конвейера видео.

Та же пара переменных окружения, что у Telegram-моста:
VOXFUSION_API_URL (по умолчанию http://127.0.0.1:8500) и VOXFUSION_API_TOKEN.
Сегменты с таймкодами запрашиваются полем include_segments=true - тогда в
ответе задачи приходит segments: [{start, end, text[, speaker]}] (контракт
проверен живым пробником 2026-10-11). HTTP-клиент передаётся параметром -
тесты идут без сети (см. tests/test_video_ingest.py).
"""

from __future__ import annotations

import os
import time

POLL_INTERVAL_S = 5.0
DEFAULT_TIMEOUT_S = 1800.0


def api_url() -> str:
    return os.environ.get("VOXFUSION_API_URL", "http://127.0.0.1:8500").rstrip("/")


def api_token() -> str:
    return os.environ.get("VOXFUSION_API_TOKEN", "")


def enabled() -> bool:
    """Мост настроен (есть токен) - иначе видео-конвейер отказывает честно."""
    return bool(api_token())


def submit(client, audio_path, language: str = "ru") -> str:
    """Поставить аудио в транскрибацию; вернуть job_id."""
    with open(audio_path, "rb") as f:
        r = client.post(
            f"{api_url()}/v1/transcribe",
            files={"file": (os.path.basename(audio_path), f, "application/octet-stream")},
            data={"language": language, "include_segments": "true"},
            headers={"Authorization": f"Bearer {api_token()}"},
        )
    r.raise_for_status()
    d = r.json()
    if "job_id" not in d:
        raise RuntimeError(f"voxfusion transcribe: {str(d)[:200]}")
    return d["job_id"]


def poll_segments(client, job_id: str, timeout_s: float = DEFAULT_TIMEOUT_S):
    """Поллинг до done/error/таймаута.

    Возвращает (status, segments, text): status - done|error|timeout;
    segments - список {start, end, text[, speaker]}; text - полный текст
    (или описание ошибки при error/timeout).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(
            f"{api_url()}/v1/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_token()}"},
        )
        r.raise_for_status()
        d = r.json()
        status = d.get("status", "queued")
        if status == "done":
            segs = d.get("segments") or []
            clean = [
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": str(s.get("text") or "").strip(),
                    **({"speaker": s["speaker"]} if s.get("speaker") else {}),
                }
                for s in segs
                if str(s.get("text") or "").strip()
            ]
            return "done", clean, str(d.get("text") or "")
        if status == "error":
            return "error", [], str(d.get("error") or "ошибка транскрибации")[:300]
        time.sleep(POLL_INTERVAL_S)
    return "timeout", [], "транскрибация не уложилась в таймаут"
