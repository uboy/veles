"""
title: База знаний (Veles)
author: Veles
version: 1.0.0
required_open_webui_version: 0.11.4

Pipe-функция Open WebUI (K19): ЕДИНАЯ модель чата «загрузка + вопросы».
Пользователь не выбирает, что делать, - модель сама понимает по сообщению:

  1. Приложен файл (без текста вопроса) - загрузка в ЛИЧНОЕ хранилище
     пользователя: POST /v1/documents с заголовком X-Veles-On-Behalf-Of
     (email из сессии WebUI __user__; владельческий токен фасада живёт
     в Valves - пользователь его не видит и не вводит). Ответ-подтверждение
     по каждому файлу: название, статус обработки, «появится в поиске
     в течение ~минуты». Название документа - имя файла (текст сообщения
     при файлах трактуется как вопрос, см. п. 3).
  2. Только текстовый вопрос - ответ по базе знаний: POST /v1/knowledge/query,
     рендер цитат тот же, что у модели veles-knowledge фасада:
     [название документа](ссылка) в тексте и человекочитаемые сноски.
  3. Файлы + вопрос одним сообщением - сначала загружаются все файлы,
     затем идёт ответ по базе. Если документы ещё обрабатываются, ответ
     даётся по текущей базе, и это говорится честно.

Замена пары «veles-knowledge + Загрузить в базу знаний» (K16): после
переключения в деплое старую функцию можно удалить - она не ломается
этой моделью (совместимость до переключения).

Установка (админ Open WebUI):
  1. Admin Panel -> Functions -> «+» (New Function) -> вставить весь этот
     код (или Import из файла). Имя модели: «База знаний (Veles)».
     Сохранить и включить тумблером.
  2. Сделать модель видимой обычным пользователям: в 0.11.4 функция без
     записи в Models видна ТОЛЬКО админам. Admin Panel -> Workspace ->
     Models -> «База знаний (Veles)» -> видимость Public (или API
     POST /api/v1/models/create с access_grants principal_id "*").
  3. На карточке функции - шестерёнка (Valves):
       veles_url        - базовый URL фасада Veles, видимый ИЗ КОНТЕЙНЕРА
                          WebUI (например http://host.docker.internal:8400);
       veles_token      - владельческий токен фасада (VELES_API_TOKEN из
                          veles-facade.env); в чат не попадает;
       veles_public_url - URL фасада, видимый ИЗ БРАУЗЕРА пользователя
                          (для ссылок на источники); пусто - берётся
                          veles_url.
  4. В чате выбрать модель «База знаний (Veles)»: спросить - получить
     ответ с цитатами; прикрепить файл - загрузить в личную базу;
     файл + вопрос - загрузить и ответить.

Требования: без внешних зависимостей (только stdlib: urllib), pydantic
есть в окружении Open WebUI.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import quote

from pydantic import BaseModel, Field


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Ревью K16 P2: редиректы запрещены - заголовки авторизации не должны
    уезжать другому хосту; 3xx вернётся вызывающему как ошибка."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = None  # инъекционный открыватель - тесты подменяют моком


def _opener():
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(_NoRedirect)
    return _OPENER


def _http_bytes(
    method: str, url: str, headers: dict, data: bytes | None, timeout: float
) -> tuple[int, bytes]:
    """Один HTTP-вызов stdlib-средствами (зависимостей у функции нет).
    Возвращает (статус, тело); HTTPError - не исключение, а статус с телом.
    timeout - скаляр (urllib не умеет пары connect/read, как socket).
    Редиректы НЕ следуются - Authorization и X-Veles-On-Behalf-Of ушли бы
    целевому хосту редиректа; 3xx возвращается как статус."""
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _opener().open(req, timeout=timeout) as resp:
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
    # кавычки/переводы строк ломают quoted-string заголовка - зачищаем
    filename = re.sub(r'[\r\n"]', "", filename or "")
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
    """Текст последнего сообщения пользователя - вопрос/комментарий."""
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


def _source_link(public_url: str, token: str, citation: dict) -> str:
    """Ссылка на источник для цитаты - тот же формат, что render_chat_content
    фасада (static): /v1/source/<document_id>?token=..&node=..#n-<node>."""
    doc = quote(str(citation.get("document_id") or ""), safe="/")
    node = quote(str(citation.get("node_version_id") or ""))
    q = f"token={quote(token)}&node={node}" if token else f"node={node}"
    return f"{public_url}/v1/source/{doc}?{q}#n-{node}"


def _asset_link(public_url: str, token: str, citation: dict) -> str:
    """Ссылка на изображение-цитату (PDF) - формат asset_url фасада:
    /v1/asset/<document_id>/<name>?token=.."""
    doc = quote(str(citation.get("document_id") or ""), safe="/")
    name = quote(str(citation.get("asset") or ""))
    q = f"token={quote(token)}" if token else ""
    return f"{public_url}/v1/asset/{doc}/{name}" + (f"?{q}" if q else "")


def _render_answer(result: dict, public_url: str, token: str) -> str:
    """Рендер ответа /v1/knowledge/query - тот же, что у veles-knowledge:
    сноски [n] в тексте синтеза -> [название документа](ссылка),
    ниже - человекочитаемые сноски-источники; вне корпуса - честный abstain."""
    decision = result.get("decision")
    if decision not in ("answer", "warn_conflict"):
        reason = (result.get("diagnostics") or {}).get("reason") or ""
        return (
            "Ответа на этот вопрос в базе знаний нет."
            + (f"\nПричина: {reason}." if reason else "")
            + "\nУточните вопрос, назовите конкретный документ или попросите "
            "добавить источник в базу (прикрепите файл к сообщению)."
        )

    citations = result.get("citations") or []
    links = {
        c["n"]: _source_link(public_url, token, c)
        for c in citations
        if c.get("document_id") and c.get("node_version_id")
    }
    titles = {
        c["n"]: (c.get("title") or c.get("document_id") or f"источник {c['n']}")[:80]
        for c in citations
    }
    answer = result.get("answer") or ""
    if links:
        answer = re.sub(
            r"\[(\d{1,2})\]",
            lambda m: (
                f"[«{titles[int(m.group(1))]}»]({links[int(m.group(1))]})"
                if int(m.group(1)) in links else m.group(0)
            ),
            answer,
        )
    blocks = [answer, "", "Источники:"]
    for c in citations:
        lines = c.get("lines") or []
        where = f", строки {lines[0]}-{lines[1]}" if len(lines) >= 2 else ""
        url = links.get(c["n"])
        entry = f"  [{c['n']}]({url}) «{c.get('title')}» {c.get('source_id')}{where}" if url else (
            f"  [{c['n']}] «{c.get('title')}» {c.get('source_id')}{where}"
        )
        blocks.append(entry)
    # изображения-цитаты (PDF) - как в render_chat_content фасада
    for c in citations:
        if c.get("asset") and c.get("document_id"):
            blocks.append(f"![изображение]({_asset_link(public_url, token, c)})")
    if decision == "warn_conflict":
        conflicts = (result.get("diagnostics") or {}).get("conflicts") or []
        if conflicts:
            listing = "; ".join(
                f"«{c.get('title') or c.get('document_id')}» "
                f"({c.get('revision_id')}, {c.get('signal')})"
                for c in conflicts
            )
            blocks += [
                "",
                f"⚠ Возможны противоречащие редакции: {listing}.",
                "Уточните, какая редакция вам нужна.",
            ]
    return "\n".join(blocks)


class Pipe:
    """Единая модель «База знаний (Veles)»: файлы -> личная загрузка,
    вопрос -> ответ по базе с цитатами, файлы+вопрос - то и другое."""

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
        veles_public_url: str = Field(
            default="",
            description="URL фасада, видимый из браузера пользователя (для ссылок "
            "на источники в ответах); пусто - используется veles_url",
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
        except Exception:  # noqa: BLE001 - статус-событие не должно ломать ответ
            pass

    def _upload_one(
        self,
        item: dict,
        *,
        email: str,
        on_behalf_headers: dict,
        auth_headers: dict,
        base_url: str,
    ) -> str:
        """Загрузка одного файла: POST /v1/documents -> поллинг джоба -> строка
        отчёта. Ошибки - человеческим текстом, исключения наружу не идут.
        Название документа - имя файла: текст сообщения при файлах - вопрос,
        а не название (K19)."""
        name = _file_name(item)
        veles_name = _veles_filename(name)
        text = _file_text(item)
        if not text:
            return (
                f"⚠ «{name}»: в файле не нашлось текста после извлечения - "
                "в базу знаний не отправлен (сканы без текстового слоя модель "
                "не распознаёт)."
            )

        upload_title = name.rsplit(".", 1)[0]
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
                "veles_token модели (администратору)."
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
                return f"⚠ «{name}»: файл принят, но статус недоступен: {e}."
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
                    "в вашу личную базу знаний: доступен в поиске "
                    f"(группа доступа - личная, {email})."
                )
            if state == "failed":
                reason = (record.get("error") or "причина не указана").strip()
                return f"⚠ «{name}»: загрузка не удалась - {reason}"
            if time.monotonic() >= deadline:
                return (
                    f"⏳ «{name}»: файл принят и обрабатывается - появится "
                    "в поиске в течение ~минуты (группа - личная)."
                )
            time.sleep(interval)

    # ----------------------------------------------------------------- вопрос

    def _ask(self, question: str, *, auth_headers: dict, base_url: str,
             public_url: str, token: str) -> str:
        """Ответ по базе: POST /v1/knowledge/query -> рендер цитат.
        Ошибки - человеческим текстом, исключения наружу не идут."""
        payload = json.dumps(
            {"query": question, "top_k": 5, "use_llm": True}
        ).encode("utf-8")
        headers = {**auth_headers, "Content-Type": "application/json"}
        try:
            status, resp_body = _http_bytes(
                "POST", f"{base_url}/v1/knowledge/query", headers, payload, 120.0
            )
        except RuntimeError as e:
            return f"⚠ Ответить не удалось: {e}."
        if status == 401:
            return (
                "⚠ Фасад отклонил токен (401) - проверьте Valve veles_token "
                "модели (администратору)."
            )
        if status != 200:
            detail = ""
            try:
                detail = json.loads(resp_body.decode("utf-8")).get("detail", "")
            except Exception:  # noqa: BLE001
                pass
            extra = f" ({detail})" if detail else ""
            return f"⚠ Фасад вернул ошибку {status}{extra}."
        try:
            result = json.loads(resp_body.decode("utf-8"))
        except Exception:  # noqa: BLE001 - битый ответ фасада
            return "⚠ Фасад вернул ответ, который не удалось разобрать."
        return _render_answer(result, public_url, token)

    # ------------------------------------------------------------------- pipe

    def pipe(self, body: dict, __user__: dict, __files__=None, __event_emitter__=None, __task__=None) -> str:
        # Фоновые задачи WebUI (генерация поисковых запросов для приложенных
        # файлов, теги, follow-up) вызывают ТУ ЖЕ модель: без этого guard'а
        # файлы загружались бы повторно, а вопросы задач уходили бы в базу
        # как пользовательские (прецедент K16, живой тест стенд). Задача - не чат.
        if __task__:
            return ""
        base_url = (self.valves.veles_url or "").strip().rstrip("/")
        token = (self.valves.veles_token or "").strip()
        if not base_url or not token:
            return (
                "Модель не настроена: администратору нужно задать Valves - "
                "veles_url (адрес фасада) и veles_token (VELES_API_TOKEN). "
                "Admin Panel -> Functions -> эта модель -> Valves."
            )
        email = str((__user__ or {}).get("email") or "").strip()
        if not email:
            return (
                "Не удалось определить ваш пользователя (нет email в сессии "
                "Open WebUI) - документы могли бы получиться чужими, запрос "
                "отменён. Войдите в Open WebUI под своей учётной записью."
            )

        public_url = (self.valves.veles_public_url or "").strip().rstrip("/") or base_url
        question = _last_user_text(body or {})
        files = _attached_files(body or {}, __files__)

        if not files and not question:
            return (
                "Задайте вопрос текстом - отвечу по базе знаний со ссылками "
                "на источники. Или прикрепите файл (txt, md, pdf) - он будет "
                "загружен в вашу личную базу. Можно и то и другое сразу."
            )

        auth_headers = {"Authorization": f"Bearer {token}"}

        report: list[str] = []
        pending = 0  # файлы, обработка которых не успела дойти до конца
        if files:
            on_behalf_headers = {"X-Veles-On-Behalf-Of": email}
            for idx, item in enumerate(files):
                self._emit(
                    __event_emitter__,
                    f"Загрузка в базу знаний: файл {idx + 1} из {len(files)}",
                )
                line = self._upload_one(
                    item,
                    email=email,
                    on_behalf_headers=on_behalf_headers,
                    auth_headers=auth_headers,
                    base_url=base_url,
                )
                if line.startswith("⏳"):
                    pending += 1
                report.append(line)
            self._emit(__event_emitter__, "Загрузка в базу знаний завершена", done=True)

        blocks: list[str] = []
        if report:
            header = (
                "Загрузка в базу знаний:"
                if len(report) > 1
                else "Загрузка в базу знаний"
            )
            blocks.append(header + "\n\n" + "\n\n".join(report))

        if question:
            self._emit(__event_emitter__, "Поиск по базе знаний")
            blocks.append(self._ask(question, auth_headers=auth_headers,
                                    base_url=base_url, public_url=public_url,
                                    token=token))
            if pending:
                blocks.append(
                    "Примечание: только что загруженные файлы ещё обрабатываются, "
                    "ответ дан по текущей базе - задайте вопрос чуть позже, "
                    "и новые документы уже будут в поиске."
                )
        return "\n\n".join(blocks)
