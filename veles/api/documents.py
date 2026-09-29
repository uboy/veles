"""Роутер загрузки документов пользователем (K4): POST /v1/documents + джобы.

POST /v1/documents        - multipart (file + опциональные title, access_groups)
                            или JSON {text, title}; создаёт джоб, ответ 202.
GET /v1/documents/jobs    - список своих джобов (админ - все).
GET /v1/documents/jobs/{id} - статус queued/running/done/failed с причиной.

Личность и группы - только через identity фасада (инвариант 5, тот же шов,
что у query/chat): документ пользователя несёт личную группу user:<uid>,
поэтому чужие загрузки невидимы в поиске и в статусах джобов; админ видит всё.
app.py в этот файл не перекраивался: контекст (identity_fn/pipeline_fn)
приходит через app.state.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..ingest.queue import (
    ALLOWED_EXTENSIONS,
    IngestQueue,
    QueueCapacityExceeded,
    QuotaExceeded,
    check_quota,
    quota_free_bytes,
    quota_limits,
    safe_filename,
    slugify_title,
    upload_max_bytes,
    uploads_root,
    user_dir,
    user_quota_lock,
)

router = APIRouter(prefix="/v1/documents", tags=["documents"])

# одна очередь на процесс фасада; замок закрывает гонку первых двух запросов
_QUEUE_LOCK = threading.Lock()


class TextUpload(BaseModel):
    """JSON-вариант загрузки: текст документа (txt/md-разметка) + название."""

    text: str
    title: str = ""


def _ctx(request: Request) -> dict:
    state = getattr(request.app.state, "veles", None)
    if state is None:  # create_app не достроил контекст - не должно случаться
        raise HTTPException(status_code=500, detail="контекст фасада не инициализирован")
    return state


def _identity(request: Request, authorization: str):
    return _ctx(request)["identity_fn"](request, authorization)


def _queue(request: Request) -> IngestQueue:
    queue = getattr(request.app.state, "veles_ingest", None)
    if queue is not None:
        return queue
    with _QUEUE_LOCK:
        queue = getattr(request.app.state, "veles_ingest", None)
        if queue is None:
            queue = IngestQueue(pipeline_fn=_ctx(request)["pipeline_fn"])
            request.app.state.veles_ingest = queue
    return queue


def _document_groups(ident, groups_param: str, identity_mode: str) -> list[str]:
    """access_groups документа: личная группа пользователя (identity-режимы)
    или группы владельца (static). Явный access_groups - только админу
    (инвариант 5: группы не приходят от клиента)."""
    if groups_param.strip():
        if not ident.admin:
            raise HTTPException(
                status_code=403,
                detail="назначение групп доступа (access_groups) доступно только администратору",
            )
        parsed = sorted({g.strip() for g in groups_param.split(",") if g.strip()})
        if parsed:
            return parsed
    if identity_mode != "static" and ident.user_id:
        return [f"user:{ident.user_id}"]
    groups = sorted(ident.groups - {"*"})
    return groups or sorted({"personal"})


@router.post("")
async def upload_document(
    request: Request, authorization: str = Header(default="")
):
    """Загрузка документа: multipart (file [+ title, access_groups]) или JSON
    ({text, title}). Форматы первого шага: txt/md/pdf. Квоты проверяются ДО
    постановки в очередь (документы - 403, размер L0 - 413).

    P1 ревью K4: Content-Length проверяется ДО разбора тела против жёсткого
    капа VELES_UPLOAD_MAX_MB и против свободного остатка квоты (грубо, без
    имени файла - с запасом на перезапись крупнейшего своего L0) - тело
    заведомо неподъёмного запроса не читается вовсе. Остаточный риск: без
    Content-Length (chunked) и само тело до капа всё ещё буферизуется
    парсером multipart - полный стриминг с капом на лету, отдельная задача.
    Запись L0 - через run_in_threadpool, не блокируя событийный цикл.
    """
    ident = _identity(request, authorization)
    state = _ctx(request)
    queue = _queue(request)

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type and "application/json" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="ожидается multipart/form-data (файл) или application/json (текст)",
        )

    # ранний гейт размера: заголовок есть почти всегда (клиенты ставят его
    # сами); chunked без Content-Length проходит мимо - см. остаточный риск
    declared = (request.headers.get("content-length") or "").strip()
    directory = user_dir(ident.user_id)
    if declared.isdigit():
        length = int(declared)
        max_bytes = upload_max_bytes()
        if max_bytes and length > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"размер запроса {length // (1024 * 1024)} МБ больше "
                    f"жёсткого капа {max_bytes // (1024 * 1024)} МБ "
                    "(VELES_UPLOAD_MAX_MB) - загрузка отклонена до чтения",
            )
        free = quota_free_bytes(directory)
        if free is not None and length > free:
            _, mb_limit = quota_limits()
            raise HTTPException(
                status_code=413,
                detail=f"размер запроса {length // (1024 * 1024)} МБ больше "
                    f"свободного остатка квоты ({mb_limit} МБ, "
                    "VELES_QUOTA_MB) - загрузка отклонена до чтения",
            )

    raw = b""
    filename = ""
    title = ""
    groups_param = ""
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail="файл не передан (поле file)")
        raw = await upload.read()
        filename = upload.filename or ""
        title = str(form.get("title") or "").strip()
        groups_param = str(form.get("access_groups") or "").strip()
    elif "application/json" in content_type:
        try:
            body = TextUpload(**(await request.json()))
        except Exception:
            raise HTTPException(
                status_code=400, detail="ожидается JSON {text, title}"
            )
        raw = body.text.encode("utf-8")
        title = body.title.strip()

    if "multipart/form-data" in content_type:
        if not raw:
            raise HTTPException(status_code=400, detail="пустой файл")
        filename = safe_filename(filename) or ""
        if not filename:
            raise HTTPException(status_code=400, detail="недопустимое имя файла")
        extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail="поддерживаются форматы txt, md, pdf "
                    f"(получено: {extension or 'без расширения'})",
            )
    else:
        if not raw.strip():
            raise HTTPException(status_code=400, detail="пустой документ")
        filename = slugify_title(title) + ".md"

    directory = user_dir(ident.user_id)
    document_id = f"{directory}/{filename}"
    l0_path = uploads_root() / directory / filename

    def _quota_and_write() -> None:
        # P1+P2 ревью K4: check+write в тредпуле (синхронная запись не должна
        # блокировать событийный цикл) и под per-user замком - атомарно для
        # пользователя, два одновременных POST не превышают квоту
        with user_quota_lock(directory):
            check_quota(directory, len(raw), replacing=l0_path.exists())
            l0_path.parent.mkdir(parents=True, exist_ok=True)
            l0_path.write_bytes(raw)

    # K8 ревью P2: ранний (ненадёжный) гейт капа ДО записи L0 - перезапись L0
    # существующего документа необратима, а авторитетный отказ после записи
    # удаляет уже перезаписанный файл. Гонку с отпустившимся капом закрывает
    # авторитетная проверка в submit (ниже, с чисткой L0).
    busy, max_queued = queue.at_capacity()
    if busy:
        raise HTTPException(
            status_code=429,
            detail=f"очередь занята, повторите позже (лимит {max_queued}, "
                   "VELES_INGEST_QUEUE_MAX)",
        )

    try:
        await run_in_threadpool(_quota_and_write)
    except QuotaExceeded as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    try:
        # K8: кап очереди - тот же контракт отказа, что квоты (detail из исключения)
        job = queue.submit(
            user_id=ident.user_id,
            filename=filename,
            title=title,
            document_id=document_id,
            size_bytes=len(raw),
            groups=_document_groups(ident, groups_param, state["identity_mode"]),
        )
    except QueueCapacityExceeded as e:
        # джоба нет и не будет - L0 этой загрузки не оставляем (как _discard_l0
        # у упавшего джоба): отказ очереди не съедает квоту навсегда
        try:
            l0_path.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    return JSONResponse(
        status_code=202,
        content={"job_id": job.job_id, "status": job.status, "document_id": document_id},
    )


@router.get("/jobs")
def list_jobs(request: Request, authorization: str = Header(default="")):
    """Свои джобы (свежие первыми); админ видит все."""
    ident = _identity(request, authorization)
    records = _queue(request).list_jobs()
    if not ident.admin:
        records = [r for r in records if r.get("user_id") == ident.user_id]
    return {"jobs": records}


@router.get("/jobs/{job_id}")
def job_status(request: Request, job_id: str, authorization: str = Header(default="")):
    """Статус джоба. Чужой джоб неотличим от несуществующего (404) - перечень
    джобов не палит чужие загрузки; после рестарта фасада недошедший джоб
    честно failed («сервис перезапущен»), а не вечный running."""
    ident = _identity(request, authorization)
    record = _queue(request).lookup(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="задание не найдено")
    if not ident.admin and record.get("user_id") != ident.user_id:
        raise HTTPException(status_code=404, detail="задание не найдено")
    return record
