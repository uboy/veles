"""Документы в базу знаний через Telegram (K9, corp-ai-kb).

Файл боту -> getFile -> скачивание (механизм переиспользован у voice) ->
POST /v1/documents (multipart) -> поллинг GET /v1/documents/jobs/{id} ->
завершающее сообщение. Форматы первого шага: txt/md/pdf (по расширению);
лимит Telegram на getFile - 20 МБ, крупнее боту не отдаётся (сообщаем честно).
Функции принимают клиентов параметрами (тестируемость, как voice.py):
tg_client - с прокси Telegram, vx_client - прямой до фасада Veles.
Ошибки фасада (>=400) - FacadeError с текстом detail: мост показывает его
пользователю как есть (запрос группы, которой у пользователя нет, и т.п.).
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

# Лимит Telegram Bot API на скачивание файлов через getFile
TG_GETFILE_MAX_BYTES = 20 * 1024 * 1024
# Форматы первого шага загрузки (тот же список, что у фасада: ALLOWED_EXTENSIONS)
DOC_EXTENSIONS = {".txt", ".md", ".pdf"}


def esc(text: Any) -> str:
    """Экранирование Telegram-HTML (паттерн tg_format._esc) для имён файлов
    и текстов ошибок: пользовательский контент не должен ломать разметку."""
    return html.escape(str(text or ""), quote=False)


def base_url() -> str:
    return os.environ.get("VELES_BASE_URL", "http://127.0.0.1:8400").rstrip("/")


def token() -> str:
    return os.environ.get("VELES_API_TOKEN", "")


class FacadeError(RuntimeError):
    """Фасад ответил ошибкой (>=400): detail показывается пользователю как есть."""


def _detail(r: Any) -> str:
    """Человекочитаемая причина отказа фасада: detail из JSON-ошибки FastAPI."""
    try:
        d = r.json()
    except Exception:  # noqa: BLE001 - не-json тело: берём текст как есть
        return str(getattr(r, "text", "") or "")[:300]
    return str(d.get("detail") or d)[:300]


def _auth_headers() -> dict:
    t = token()
    return {"Authorization": f"Bearer {t}"} if t else {}


def submit_document(vx_client: Any, path: Path, title: str = "", group: str = "") -> str:
    """POST /v1/documents (multipart): поставить загрузку в очередь фасада,
    вернуть job_id (ответ 202). group - серверная группа доступа (/docgroup),
    пусто - дефолт фасада (личная user:<uid>)."""
    data: dict = {}
    if title:
        data["title"] = title
    if group:
        data["group"] = group
    with path.open("rb") as f:
        r = vx_client.post(
            f"{base_url()}/v1/documents",
            headers=_auth_headers(),
            files={"file": (path.name, f, "application/octet-stream")},
            data=data,
        )
    if r.status_code >= 400:
        raise FacadeError(_detail(r))
    r.raise_for_status()
    d = r.json()
    if "job_id" not in d:
        raise RuntimeError(f"documents: {str(d)[:120]}")
    return d["job_id"]


def get_job(vx_client: Any, job_id: str) -> dict:
    """Текущий статус джоба: queued/running/done/failed (+error, title, groups)."""
    r = vx_client.get(f"{base_url()}/v1/documents/jobs/{job_id}", headers=_auth_headers())
    if r.status_code >= 400:
        raise FacadeError(_detail(r))
    r.raise_for_status()
    return r.json()


def list_groups(vx_client: Any) -> list[dict]:
    """Список групп для выбора при загрузке: [{"id", "title"}] из GET /v1/groups."""
    r = vx_client.get(f"{base_url()}/v1/groups", headers=_auth_headers())
    if r.status_code >= 400:
        raise FacadeError(_detail(r))
    r.raise_for_status()
    return r.json().get("groups", [])


def fmt_file_size(num_bytes: int) -> str:
    """Размер для статусных сообщений (лимит Telegram - в МБ)."""
    return f"{num_bytes / (1024 * 1024):.1f} МБ"


def load_doc_state(path: Path) -> dict:
    """Прочитать state групп загрузок; нет файла/битый json - пустой словарь."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - нет файла/битый json = группы не выбраны
        return {}
    return data if isinstance(data, dict) else {}


def save_doc_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
