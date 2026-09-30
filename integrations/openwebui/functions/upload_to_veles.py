"""
title: Загрузить в базу знаний
author: Veles
version: 1.0.0
required_open_webui_version: 0.11.4

Pipe-функция Open WebUI (K16): пользователь прикладывает файл(ы) в чат и пишет
название/комментарий - функция берёт извлечённый WebUI текст и загружает его
на фасад Veles (POST /v1/documents) ЛИЧНО от имени пользователя: документ
попадает в его личную группу user:<email> и виден ему в поиске базы знаний.

Установка (админ Open WebUI):
  1. Admin Panel -> Functions -> «+» (New Function) -> вставить весь этот код
     (или Functions -> Import Functions -> из файла). Имя модели при создании:
     «Загрузить в базу знаний». Сохранить и включить тумблер.
  2. Сделать модель видимой обычным пользователям: в 0.11.4 функция без
     записи в Models видна ТОЛЬКО админам. Admin Panel -> Workspace -> Models
     -> «Загрузить в базу знаний» -> видимость Public (или API
     POST /api/v1/models/create с access_grants principal_id "*").
  3. На карточке функции - шестерёнка (Valves), задать:
       veles_url   - базовый URL фасада Veles, видимый ИЗ КОНТЕЙНЕРА WebUI
                     (например http://host.docker.internal:8400);
       veles_token - владельческий токен фасада (VELES_API_TOKEN из
                     veles-facade.env). Токен не попадает в чат: он живёт
                     в Valves и виден только админам.
  4. В чате выбрать модель «Загрузить в базу знаний», прикрепить файл
     (txt/md/pdf - поддержанные фасадом форматы первого шага) и написать
     название или комментарий. Каждый файл становится отдельным документом.

Заголовок X-Veles-On-Behalf-Of (email пользователя) фасад принимает только с
владельческим токеном, поэтому личность документа определяется WebUI-сессией,
а не вводом пользователя. Требования: без внешних зависимостей (только
stdlib: urllib), pydantic есть в окружении Open WebUI.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid

from pydantic import BaseModel, Field


def _http_bytes(
    method: str, url: str, headers: dict, data: bytes | None, timeout: float
) -> tuple[int, bytes]:
    """Один HTTP-вызов stdlib-средствами (зависимостей у функции нет).
    Возвращает (статус, тело); HTTPError - не исключение, а статус с телом.
    timeout - скаляр (urllib не умеет пары connect/read, как socket)."""
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        except Exception:  # noqa: BLE001 - тело ошибки может не читаться
            return e.code, b""
    except Exception as e:  # noqa: BLE001 - сеть/таймаут: текст для человека
        raise RuntimeError(f"фасад недоступен ({e})") from e


def _multipart_body(fields: dict, file_field: str, filename: str, content: bytes):
    """multipart/form-data без внешних библиотек: (тело, content-type)."""
    boundary = "----velesboundary" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            "Content-Type: text/plain\r\n\r\n"
        ).encode("utf-8")
        + content
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _last_user_text(body: dict) -> str:
    """Текст последнего сообщения пользователя - название/комментарий."""
    for msg in reversed(body.get("messages") or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):  # мультимодальное сообщение
            text = " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
            return text.strip()
    return ""


def _attached_files(body: dict, files_param) -> list[dict]:
    """Список приложенных файлов: 0.11.4 передаёт их в __files__
    (metadata['files'] после process_chat_payload); запасные пути - для
    других версий/прямых API-вызовов."""
    if isinstance(files_param, list):
        items = files_param
    else:
        items = body.get("files") or []
        if not items:
            for msg in reversed(body.get("messages") or []):
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("files"):
                    items = msg["files"]
                    break
    return [f for f in items if isinstance(f, dict) and f.get("type") != "image"]


def _file_name(item: dict) -> str:
    file_obj = item.get("file") or {}
    name = item.get("name") or file_obj.get("filename") or file_obj.get("meta", {}).get("name")
    return str(name or "документ.txt")


def _file_text(item: dict) -> str:
    """Извлечённый WebUI текст файла - item.file.data.content (та же структура,
    что читает retrieval-слой Open WebUI для full-context файлов)."""
    file_obj = item.get("file") or {}
    data = file_obj.get("data") or {}
    content = data.get("content") or item.get("content") or ""
    return content.strip() if isinstance(content, str) else ""


def _veles_filename(name: str) -> str:
    """Имя файла для фасада: фасад принимает txt/md/pdf, текст у нас уже
    извлечён - неподдерживаемое расширение (docx и пр.) честно превращается
    в .txt, а не в отказ."""
    name = (name or "").strip() or "Документ"
    lower = name.lower()
    if lower.endswith(".md") or lower.endswith(".markdown"):
        return name.rsplit(".", 1)[0] + ".md"
    if "." in name:
        return name.rsplit(".", 1)[0] + ".txt"
    return name + ".txt"


class Pipe:
    """Модель «Загрузить в базу знаний»: файл(ы) из чата -> личный документ
    пользователя в базе знаний Veles."""

    class Valves(BaseModel):
        veles_url: str = Field(
            default="",
            description="Базовый URL фасада Veles, видимый из контейнера WebUI "
            "(например http://host.docker.internal:8400)",
        )
        veles_token: str = Field(
            default="",
            description="Владельческий токен фасада (VELES_API_TOKEN)",
        )
        wait_seconds: int = Field(
            default=90,
            description="Сколько секунд ждать окончания обработки документа",
        )
        poll_interval_seconds: float = Field(
            default=2.0,
            description="Интервал опроса статуса обработки, секунды",
        )

    def __init__(self):
        self.valves = self.Valves()

    # ------------------------------------------------------------------ отчёт

    def _emit(self, __event_emitter__, description: str, done: bool = False) -> None:
        if __event_emitter__ is None:
            return
        try:
            __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )
        except Exception:  # noqa: BLE001 - статус-событие не должно ломать загрузку
            pass

    def _upload_one(
        self,
        item: dict,
        *,
        email: str,
        title: str,
        on_behalf_headers: dict,
        auth_headers: dict,
        base_url: str,
    ) -> str:
        """Загрузка одного файла: POST /v1/documents -> поллинг джоба -> строка
        отчёта. Ошибки - человеческим текстом, исключения наружу не идут."""
        name = _file_name(item)
        veles_name = _veles_filename(name)
        text = _file_text(item)
        if not text:
            return (
                f"⚠ «{name}»: в файле не нашлось текста после извлечения - "
                "в базу знаний не отправлен (сканы без текстового слоя функция "
                "не распознаёт)."
            )

        upload_title = title or name.rsplit(".", 1)[0]
        body_bytes, content_type = _multipart_body(
            {"title": upload_title}, "file", veles_name, text.encode("utf-8")
        )
        headers = {**auth_headers, **on_behalf_headers, "Content-Type": content_type}
        try:
            status, resp_body = _http_bytes(
                "POST", f"{base_url}/v1/documents", headers, body_bytes, 60.0
            )
        except RuntimeError as e:
            return f"⚠ «{name}»: {e}."
        if status == 401:
            return (
                f"⚠ «{name}»: фасад отклонил токен (401) - проверьте Valve "
                "veles_token функции."
            )
        if status == 413:
            try:
                detail = json.loads(resp_body.decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001 - без detail тоже человекочитаемо
                detail = ""
            return f"⚠ «{name}»: фасад отклонил загрузку по размеру/квоте (413). {detail}".strip()
        if status != 202:
            detail = ""
            try:
                detail = json.loads(resp_body.decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            extra = f" ({detail})" if detail else ""
            return f"⚠ «{name}»: фасад вернул ошибку {status}{extra}."

        try:
            job_id = json.loads(resp_body.decode("utf-8"))["job_id"]
        except Exception:  # noqa: BLE001 - битый ответ фасада
            return f"⚠ «{name}»: фасад не вернул идентификатор задания."

        deadline = time.monotonic() + max(0, int(self.valves.wait_seconds))
        interval = max(0.2, float(self.valves.poll_interval_seconds))
        while True:
            try:
                status, resp_body = _http_bytes(
                    "GET",
                    f"{base_url}/v1/documents/jobs/{job_id}",
                    auth_headers,
                    None,
                    30.0,
                )
            except RuntimeError as e:
                return f"⚠ «{name}»: документ принят, но статус недоступен: {e}."
            record = {}
            if status == 200:
                try:
                    record = json.loads(resp_body.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    record = {}
            state = record.get("status")
            if state == "done":
                action = record.get("action") or "created"
                verb = "обновлён (новая ревизия)" if action == "updated" else "загружен"
                return (
                    f"✅ Документ «{record.get('title') or upload_title}» {verb} "
                    "в базу знаний: доступен в поиске, группа доступа - личная "
                    f"({email})."
                )
            if state == "failed":
                reason = (record.get("error") or "причина не указана").strip()
                return f"⚠ «{name}»: загрузка не удалась - {reason}"
            if time.monotonic() >= deadline:
                return (
                    f"⏳ «{name}»: документ принят фасадом и обрабатывается дольше "
                    f"{int(self.valves.wait_seconds)} с. Он появится в поиске, "
                    "когда обработка закончится (группа - личная)."
                )
            time.sleep(interval)

    # ------------------------------------------------------------------- pipe

    def pipe(self, body: dict, __user__: dict, __files__=None, __event_emitter__=None, __task__=None) -> str:
        # Фоновые задачи WebUI (генерация поисковых запросов для приложенных
        # файлов, теги, follow-up) вызывают ТУ ЖЕ модель: без этого guard'а
        # загрузка происходила дважды, а документ получал названием служебный
        # промпт задачи (живой тест стенд 2026-09-30). Задача - не загрузка.
        if __task__:
            return ""
        base_url = (self.valves.veles_url or "").strip().rstrip("/")
        token = (self.valves.veles_token or "").strip()
        if not base_url or not token:
            return (
                "Функция не настроена: администратору нужно задать Valves - "
                "veles_url (адрес фасада) и veles_token (VELES_API_TOKEN). "
                "Admin Panel -> Functions -> эта функция -> Valves."
            )
        email = str((__user__ or {}).get("email") or "").strip()
        if not email:
            return (
                "Не удалось определить ваш пользователя (нет email в сессии "
                "Open WebUI) - документ мог бы получиться чужим, загрузка "
                "отменена."
            )

        files = _attached_files(body or {}, __files__)
        if not files:
            return (
                "Прикрепите к сообщению файл (txt, md или pdf) - он будет "
                "загружен в базу знаний как ваш личный документ. Название или "
                "комментарий напишите текстом сообщения."
            )

        title = _last_user_text(body or {})
        auth_headers = {"Authorization": f"Bearer {token}"}
        on_behalf_headers = {"X-Veles-On-Behalf-Of": email}
        report: list[str] = []
        for idx, item in enumerate(files):
            self._emit(
                __event_emitter__,
                f"Загрузка в базу знаний: файл {idx + 1} из {len(files)}",
            )
            report.append(
                self._upload_one(
                    item,
                    email=email,
                    title=title if len(files) == 1 else "",
                    on_behalf_headers=on_behalf_headers,
                    auth_headers=auth_headers,
                    base_url=base_url,
                )
            )
        self._emit(__event_emitter__, "Загрузка в базу знаний завершена", done=True)
        header = (
            "Загрузка в базу знаний:"
            if len(files) > 1
            else "Загрузка в базу знаний"
        )
        return header + "\n\n" + "\n\n".join(report)
