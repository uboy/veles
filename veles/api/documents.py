"""Роутер загрузки документов пользователем (K4): POST /v1/documents + джобы.

POST /v1/documents        - multipart (file + опциональные title, group,
                            access_groups) или JSON {text, title, group};
                            создаёт джоб, ответ 202.
GET /v1/documents/jobs    - список своих джобов (админ - все); админка v2:
                            каждая запись несёт exists_in_base (документ уже
                            в каноне) и duplicate_of (id более ранней активной
                            задачи с тем же файлом).
GET /v1/documents/jobs/{id} - статус queued/running/done/failed с причиной.
POST /v1/documents/jobs/{id}/cancel - отмена задачи (админка v2): queued
                            снимается из очереди, running - дочерний процесс
                            разбора; failed «отменена владельцем».
POST /v1/documents/jobs/{id}/retry - повторная постановка сохранённого
                            L0-файла упавшей задачи (новый job_id).
DELETE /v1/documents/jobs/{id} - убрать завершённую задачу из журнала.
POST /v1/documents/{id}/groups - смена групп документа (админка v2): канон
                            (манифест + узлы) и мета живого индекса
                            (overlay-патч), права применяются немедленно;
                            ТОЛЬКО админ (инвариант 5 - группы от клиента
                            принимает только админ).
DELETE /v1/documents/{id} - удаление документа целиком (K10): владелец
                            (user:<uid> в access_groups) или админ, чужой -
                            404 неотличимый; канон (все ревизии и assets) -
                            в trash, L0 удаляется, узлы выводятся из поиска
                            сразу (tombstone).
Операции над задачами - как у DELETE: владелец задачи или админ, чужая - 404
неотличимый.

Личность и группы - только через identity фасада (инвариант 5, тот же шов,
что у query/chat): документ пользователя несёт личную группу user:<uid>,
поэтому чужие загрузки невидимы в поиске и в статусах джобов; админ видит всё.
K7: при загрузке можно выбрать серверную группу (group=<id> из GET /v1/groups)
- документ увидят только члены группы; невалидная группа - 400.
K16: X-Veles-On-Behalf-Of при владельческом bearer (VELES_API_TOKEN) -
«загрузить от имени пользователя»: доверенный путь для Pipe-функции
интеграции Open WebUI (integrations/openwebui/), админ-установленной;
документ личный user:<email>, каталог L0 - uploads/<email>/. Без
владельческого токена заголовок молча игнорируется (обычная семантика).
app.py в этот файл не перекраивался: контекст (identity_fn/pipeline_fn)
приходит через app.state.
"""

from __future__ import annotations

import threading
import hmac
from typing import Optional

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..audit import actor_of, client_ip, get_audit  # журнал аудита действий
from ..ingest.queue import (
    ALLOWED_EXTENSIONS,
    IngestQueue,
    JobOperationRejected,
    QueueCapacityExceeded,
    QuotaExceeded,
    check_quota,
    discard_l0_file,
    quota_free_bytes,
    quota_limits,
    safe_filename,
    slugify_title,
    upload_max_bytes,
    uploads_root,
    user_dir,
    user_quota_lock,
)
from ..indexing.indexer import INDEX_LOCK
from ..indexing.meta_compact import iter_node_fields, patch_node_meta
from ..policy.groups import GROUP_ALL, defined_group_ids, group_exists
from ..policy.identity import RequestIdentity, validated_user_id

router = APIRouter(prefix="/v1/documents", tags=["documents"])

# K16: заголовок «загрузить от имени пользователя» (доверенный путь Pipe-функции)
ACT_AS_HEADER = "X-Veles-On-Behalf-Of"

# одна очередь на процесс фасада; замок закрывает гонку первых двух запросов
_QUEUE_LOCK = threading.Lock()


class TextUpload(BaseModel):
    """JSON-вариант загрузки: текст документа (txt/md-разметка) + название."""

    text: str
    title: str = ""
    # K7: серверная группа для доступа (group=<id> из GET /v1/groups);
    # пусто - личная user:<uid>, безопасный дефолт K4 сохранён
    group: str = ""


class RegroupRequest(BaseModel):
    """Тело смены групп документа (админка v2): полный НОВЫЙ состав групп."""

    groups: list[str]


def _ctx(request: Request) -> dict:
    state = getattr(request.app.state, "veles", None)
    if state is None:  # create_app не достроил контекст - не должно случаться
        raise HTTPException(status_code=500, detail="контекст фасада не инициализирован")
    return state


def _identity(request: Request, authorization: str):
    return _ctx(request)["identity_fn"](request, authorization)


def _acting_identity(request: Request, on_behalf_of: str) -> Optional[RequestIdentity]:
    """K16: личность «от имени пользователя» для доверенного клиента.

    Заголовок ACT_AS_HEADER принимается ТОЛЬКО при точном совпадении bearer с
    владельческим токеном (VELES_API_TOKEN): вызывать будет Pipe-функция,
    админ-установленная в Open WebUI (доверенный путь). В противном случае
    заголовок молча ИГНОРИРУЕТСЯ (None) и документ строится по обычной
    семантике личности: identity-режимы - свой user:<uid>, static - владелец
    или 401. Токен не настроен - доверенного пути нет вовсе.

    Группы act-as: только личная user:<email> (+all) - смысл заголовка именно
    «личный документ пользователя», производные группы владельца не наследуются.
    """
    if not on_behalf_of.strip():
        return None
    state = _ctx(request)
    token = state["token"]
    auth = (request.headers.get("authorization") or "").encode("utf-8")
    # ревью K16 P2: сравнение в константном времени (строка секретная)
    if not token or not hmac.compare_digest(auth, f"Bearer {token}".encode("utf-8")):
        return None
    uid = validated_user_id(on_behalf_of)
    if uid is None:
        raise HTTPException(
            status_code=400,
            detail=f"{ACT_AS_HEADER}: ожидается email пользователя",
        )
    return RequestIdentity(user_id=uid, admin=False, groups={GROUP_ALL, f"user:{uid}"})


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


def _ensure_group_exists(group_param: str) -> None:
    """K7: невалидная группа - 400 СРАЗУ после разбора тела, до квоты и записи
    L0 (поздняя валидация оставляла бы L0-сироту без джоба и съедала квоту).
    Группа обязана существовать в конфиге групп (или all - видно всем)."""
    group_id = group_param.strip()
    if group_id and not group_exists(group_id):
        raise HTTPException(
            status_code=400,
            detail=f"неизвестная группа: {group_id!r}. Доступные: "
                f"{', '.join(sorted(defined_group_ids()))} (список: GET /v1/groups)",
        )


def _document_groups(
    ident, groups_param: str, group_param: str, identity_mode: str
) -> list[str]:
    """access_groups документа.

    K7: явный выбор группы (group=<id>) доступен каждому аутентифицированному;
    валидность группы проверена ранее (_ensure_group_exists, до записи L0).
    access_groups от клиента (инвариант 5: группы не приходят от клиента) -
    по-прежнему только админу. Без всего - безопасный дефолт: личная
    user:<uid> (identity-режимы, а с K16 и act-as: у личности с user_id -
    личная группа); производные группы ident (all и определённые серверные -
    их резолвер добавляет для ЧТЕНИЯ) документу по умолчанию не полагаются,
    иначе каждая личная загрузка стала бы видна всем."""
    if group_param.strip() and groups_param.strip():
        # ревью K7: молчаливый приоритет group= над access_groups= скрывал
        # половину намерения админа - требуем выбрать что-то одно
        raise HTTPException(
            status_code=400,
            detail="укажите что-то одно: group= (выбор группы) или access_groups= (админ)",
        )
    if group_param.strip():
        return [group_param.strip()]
    if groups_param.strip():
        if not ident.admin:
            raise HTTPException(
                status_code=403,
                detail="назначение групп доступа (access_groups) доступно только администратору",
            )
        parsed = sorted({g.strip() for g in groups_param.split(",") if g.strip()})
        if parsed:
            return parsed
    if ident.user_id:
        # личная группа пользователя. Раньше условие было «identity_mode !=
        # static и ident.user_id», но в static у личности user_id всегда пуст
        # (один владелец), поэтому проверка mode была избыточной; с K16
        # act-as создаёт личность с непустым user_id и в static-режиме -
        # документ обязан остаться личным.
        groups = [f"user:{ident.user_id}"]
        if identity_mode == "static":
            # K16, static: владелец - единственный читатель фасада, а wildcard
            # "*" ему давать нельзя (ломает контракт «access_groups= умеет
            # ограничить документ и от самого владельца», K4/K7). Личная группа
            # act-as дополняется дефолтной "personal" - документ находят тем же
            # путём, что и обычные загрузки владельца; в identity-режимах
            # документ остаётся чисто личным (user:<email>), ACL по-настоящему
            # порезан по пользователям.
            groups.append("personal")
        return groups
    derived = defined_group_ids() | {"*"}  # GROUP_ALL входит в defined
    groups = sorted(ident.groups - derived)
    return groups or sorted({"personal"})


@router.post("")
async def upload_document(
    request: Request,
    authorization: str = Header(default=""),
    x_veles_on_behalf_of: str = Header(default="", alias=ACT_AS_HEADER),
):
    """Загрузка документа: multipart (file [+ title, group, access_groups]) или
    JSON ({text, title, group}). Форматы первого шага: txt/md/pdf. Квоты
    проверяются ДО постановки в очередь (документы - 403, размер L0 - 413).
    group=<id> (K7) - серверная группа доступа, невалидная - 400; access_groups -
    только админу (инвариант 5).
    K16: ACT_AS_HEADER + владельческий bearer - документ ЛИЧНО указанного
    пользователя (см. _acting_identity); заголовок без владельческого токена
    молча игнорируется.

    P1 ревью K4: Content-Length проверяется ДО разбора тела против жёсткого
    капа VELES_UPLOAD_MAX_MB и против свободного остатка квоты (грубо, без
    имени файла - с запасом на перезапись крупнейшего своего L0) - тело
    заведомо неподъёмного запроса не читается вовсе. Остаточный риск: без
    Content-Length (chunked) и само тело до капа всё ещё буферизуется
    парсером multipart - полный стриминг с капом на лету, отдельная задача.
    Запись L0 - через run_in_threadpool, не блокируя событийный цикл.
    """
    ident = _identity(request, authorization)
    acting = _acting_identity(request, x_veles_on_behalf_of)
    if acting is not None:
        ident = acting
    state = _ctx(request)
    queue = _queue(request)

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" not in content_type and "application/json" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="ожидается multipart/form-data (файл) или application/json (текст)",
        )

    # ранний гейт размера: заголовок есть почти всегда (клиенты ставят его
    # сами). Аудит 08.10: chunked без Content-Length раньше проходил мимо капа
    # и буферизовался целиком - теперь такой запрос отклоняется до чтения
    # тела (все наши клиенты - curl, веб-интерфейс, мост - ставят Content-Length)
    declared = (request.headers.get("content-length") or "").strip()
    transfer = (request.headers.get("transfer-encoding") or "").lower()
    if not declared.isdigit() and ("chunked" in transfer or not declared):
        raise HTTPException(
            status_code=411,
            detail="требуется заголовок Content-Length: загрузка без указания "
                "размера (chunked) не поддерживается - кап размера и квота "
                "проверяются до чтения тела",
        )
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
    group_param = ""
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail="файл не передан (поле file)")
        raw = await upload.read()
        filename = upload.filename or ""
        title = str(form.get("title") or "").strip()
        group_param = str(form.get("group") or "").strip()
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
        group_param = body.group.strip()

    # K7: невалидная группа - 400 до квоты, записи L0 и очереди
    _ensure_group_exists(group_param)

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
            groups=_document_groups(ident, groups_param, group_param, state["identity_mode"]),
        )
    except QueueCapacityExceeded as e:
        # джоба нет и не будет - L0 этой загрузки не оставляем (как _discard_l0
        # у упавшего джоба): отказ очереди не съедает квоту навсегда
        try:
            l0_path.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    # ответ фиксируется ДО записи аудита: статус джоба mutable и воркер успевает
    # сменить его, пока журнал пишется (регрессия test_upload_json_find_and_
    # citation) - контракт 202 не зависит от латентности аудита
    content = {
        "job_id": job.job_id,
        "status": job.status,
        "document_id": document_id,
    }
    # аудит (план admin-audit): target=имя файла, details=document_id
    get_audit().log(
        "upload", actor=actor_of(ident), admin=ident.admin,
        target=filename, details=document_id, ip=client_ip(request),
    )
    return JSONResponse(
        status_code=202,
        content=content,
    )


def _canon_document_ids() -> set[str]:
    """document_id всех документов канона - один проход по корням на вызов
    (админка v2: exists_in_base для всего списка джобов разом, а не скан на
    каждую запись). Легаси-близнецы не различаются: для подсказки панели
    достаточно факта «документ в каноне есть»."""
    from ..canonical.store import _manifest_document_id, canonical_roots

    ids: set[str] = set()
    for canon in canonical_roots():
        if not canon.is_dir():
            continue
        for child in canon.iterdir():
            did = _manifest_document_id(child)
            if did:
                ids.add(did)
    return ids


def _enrich_jobs(records: list[dict]) -> list[dict]:
    """Поля админ-панели v2 (v2-spec §2) в каждой записи списка задач:
    exists_in_base - document_id уже лежит в каноне (бейдж «уже в базе»);
    duplicate_of - id более ранней АКТИВНОЙ (queued/running) задачи того же
    пользователя с тем же именем файла (бейдж «дубль задачи №N»; завершённые
    дубли не помечаются - их дельта уже применена или честно упала)."""
    in_canon = _canon_document_ids()
    first_active: dict[tuple[str, str], str] = {}
    active = sorted(
        (r for r in records if r.get("status") in ("queued", "running")),
        key=lambda r: (str(r.get("created_at") or ""), str(r.get("job_id") or "")),
    )
    for r in active:
        key = (str(r.get("user_id") or ""), str(r.get("filename") or ""))
        first = first_active.setdefault(key, str(r.get("job_id") or ""))
        r["duplicate_of"] = "" if first == r.get("job_id") else first
    for r in records:
        r.setdefault("duplicate_of", "")
        did = r.get("document_id")
        r["exists_in_base"] = bool(did) and did in in_canon
    return records


@router.get("/jobs")
def list_jobs(request: Request, authorization: str = Header(default="")):
    """Свои джобы (свежие первыми); админ видит все. Каждая запись несёт
    exists_in_base и duplicate_of (админка v2) - для всех, поле безобидное:
    факт «файл уже в базе» виден и из поиска."""
    ident = _identity(request, authorization)
    records = _queue(request).list_jobs()
    if not ident.admin:
        records = [r for r in records if r.get("user_id") == ident.user_id]
    return {"jobs": _enrich_jobs(records)}


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


def _own_job_or_404(request: Request, job_id: str, ident) -> dict:
    """Запись джоба для операций v2; чужой джоб - тот же 404, что и
    несуществующий (права как у DELETE /v1/documents: владелец или админ)."""
    record = _queue(request).lookup(job_id)
    if record is None or (not ident.admin and record.get("user_id") != ident.user_id):
        raise HTTPException(status_code=404, detail="задание не найдено")
    return record


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    request: Request, job_id: str, authorization: str = Header(default="")
) -> dict:
    """Отмена задачи (админка v2): queued снимается из очереди, running -
    дочерний процесс разбора; обе - failed «отменена владельцем». Завершённой
    возвращается запись без изменений (отменять нечего)."""
    ident = _identity(request, authorization)
    record = _own_job_or_404(request, job_id, ident)
    # аудит (план admin-audit): сама попытка отмены - в т.ч. no-op на
    # завершённой задаче; кто и когда дёргал cancel, видно всегда
    get_audit().log(
        "job_cancel", actor=actor_of(ident), admin=ident.admin,
        target=job_id, details=f"статус задачи: {record.get('status', '')}",
        ip=client_ip(request),
    )
    if record.get("status") not in ("queued", "running"):
        return record
    result = _queue(request).cancel(job_id)
    if result is None:  # гонка с рестартом фасада между lookup и cancel
        raise HTTPException(status_code=404, detail="задание не найдено")
    return result


@router.post("/jobs/{job_id}/retry")
def retry_job(
    request: Request, job_id: str, authorization: str = Header(default="")
):
    """Повторная постановка сохранённого L0-файла (админка v2): только для
    упавшей задачи (отмена тоже даёт failed); создаётся НОВАЯ задача, старая
    остаётся в истории. L0 не сохранился (провал разбора удаляет файл, K4) -
    409 «повторите загрузку»."""
    ident = _identity(request, authorization)
    record = _own_job_or_404(request, job_id, ident)
    if record.get("status") != "failed":
        raise HTTPException(
            status_code=409,
            detail="перезапуск возможен только упавшей задачи",
        )
    try:
        job = _queue(request).retry(job_id)
    except JobOperationRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    if job is None:  # гонка с рестартом фасада
        raise HTTPException(status_code=404, detail="задание не найдено")
    # статус новой задачи - в снапшот до записи аудита (тот же контракт 202,
    # что у загрузки: воркер может сменить mutable-статус, пока журнал пишется)
    content = {
        "job_id": job.job_id,
        "status": job.status,
        "document_id": job.document_id,
        "retried_from": job_id,
    }
    # аудит (план admin-audit): повтор задачи - новая задача в details
    get_audit().log(
        "job_retry", actor=actor_of(ident), admin=ident.admin,
        target=job_id, details=f"новая задача: {job.job_id}",
        ip=client_ip(request),
    )
    return JSONResponse(
        status_code=202,
        content=content,
    )


@router.delete("/jobs/{job_id}")
def delete_job(
    request: Request, job_id: str, authorization: str = Header(default="")
) -> dict:
    """Убрать задачу из журнала (админка v2): завершённые (done/failed) -
    из памяти и журнала; активная - 409 «сначала отмените»."""
    ident = _identity(request, authorization)
    record = _own_job_or_404(request, job_id, ident)
    try:
        result = _queue(request).delete_job(job_id)
    except JobOperationRejected as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    if result is None:
        raise HTTPException(status_code=404, detail="задание не найдено")
    # аудит (план admin-audit): файл задачи - в details (что убрали из журнала)
    get_audit().log(
        "job_delete", actor=actor_of(ident), admin=ident.admin,
        target=job_id, details=f"файл: {record.get('filename', '')}",
        ip=client_ip(request),
    )
    return result


def _validate_regroup_groups(groups: list[str]) -> list[str]:
    """Валидация нового состава групп (админка v2): непустой список без
    повторов; серверные группы - из конфига (или all), личные user:<uid> -
    с валидным uid (та же проверка, что у act-as, K16). 400 - при любом
    нарушении, ДО поиска документа."""
    parsed = sorted({g.strip() for g in (groups or []) if isinstance(g, str) and g.strip()})
    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="пустой список групп: документ обязан нести хотя бы одну группу",
        )
    for group in parsed:
        if group.startswith("user:"):
            if validated_user_id(group[len("user:"):]) is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"недопустимая личная группа: {group!r} "
                           "(ожидается user:<email пользователя>)",
                )
        elif not group_exists(group):
            raise HTTPException(
                status_code=400,
                detail=f"неизвестная группа: {group!r}. Доступные: "
                    f"{', '.join(sorted(defined_group_ids()))} (список: GET /v1/groups)",
            )
    return parsed


@router.post("/{document_id:path}/groups")
def regroup_document(
    request: Request, document_id: str, body: RegroupRequest,
    authorization: str = Header(default=""),
) -> dict:
    """Смена групп документа (админка v2, v2-spec §1): ТОЛЬКО админ (403
    прочим - инвариант 5, группы от клиента принимает только админ). Канон:
    манифест + access_group каждого узла (store.set_document_groups); живой
    индекс: мета узлов патчится сразу + overlay-патч для рестарта (паттерн
    tombstone_document). overlay не записался - в ответе overlay_saved=False:
    права работают до рестарта, после - откатятся к снапшоту пересборки
    (канон уже новый), громко в лог."""
    import logging

    from .. import config
    from ..canonical.store import (
        find_document_manifest,
        set_document_groups as store_set_groups,
    )
    from ..indexing.overlay import append_overlay

    ident = _identity(request, authorization)
    if not ident.admin:
        raise HTTPException(
            status_code=403,
            detail="смена групп документа доступна только администратору",
        )
    parsed = _validate_regroup_groups(body.groups)

    manifest = find_document_manifest(document_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="документ не найден")
    canon_id = str(manifest.get("document_id") or document_id)

    info = store_set_groups(canon_id, parsed, by=ident.user_id or "owner")
    if info is None:  # гонка: манифест видели, канон исчез - неотличимый 404
        raise HTTPException(status_code=404, detail="документ не найден")

    # живой индекс: присваивание значений меты (как superseded-патчи
    # upsert_document - размер dict не меняется, читатели без лока безопасны)
    nids: list[str] = []
    pipeline = None
    try:
        pipeline = _ctx(request)["pipeline_fn"]()
    except Exception:  # noqa: BLE001 - нет индекса: канон переписан, патч уйдёт
        pass            # следующей пересборкой
    if pipeline is not None:
        # компактная мета: скан document_id - items() материализовал бы все
        # узлы с распаковкой текстов
        for nid, meta in iter_node_fields(pipeline.bundle.node_meta, "document_id"):
            if meta.get("document_id") == canon_id:
                # правка присваиванием в полученный dict потерялась бы
                # (материализация каждый доступ заново)
                patch_node_meta(pipeline.bundle.node_meta, nid, {"access_groups": parsed})
                nids.append(nid)
    overlay_saved = True
    try:
        append_overlay(
            config.INDEX_DIR,
            {},
            {nid: {"access_groups": parsed} for nid in nids},
            [],
            np.zeros((0, 1), dtype=np.float32),
            [],
        )
    except Exception as e:  # noqa: BLE001 - права уже действуют, но рестарт
        overlay_saved = False
        logging.getLogger("veles.api.documents").error(
            "overlay-патч групп %s не записан (%s): после рестарта откатится "
            "к снапшоту пересборки", canon_id, e,
        )
    # аудит (план admin-audit): target=document_id, details=новые группы
    get_audit().log(
        "document_regroup", actor=actor_of(ident), admin=ident.admin,
        target=canon_id, details=", ".join(parsed), ip=client_ip(request),
    )
    return {
        "status": "regrouped",
        "document_id": canon_id,
        "groups": parsed,
        "nodes": len(nids),
        "overlay_saved": overlay_saved,
    }


@router.delete("/{document_id:path}")
def delete_document(
    request: Request, document_id: str, authorization: str = Header(default=""),
) -> dict:
    """Удаление документа целиком (K10): владелец документа (его user:<uid> в
    access_groups) или админ; чужой и несуществующий - один и тот же 404
    (перечень документов не палится). Член серверной группы с правом ЧТЕНИЯ
    удалять не может - удаление строже чтения, как в брифе.

    Физика: канон (doc_dir со всеми revision-r*/ и assets) переезжает в
    trash/<timestamp>-<id>/ (не безвозвратно, восстановление вручную админом),
    L0 удаляется, узлы выводятся из живого поиска СРАЗУ - tombstone в бандле +
    overlay-патч для рестарта (см. veles/indexing/overlay.py). Событие delete -
    в ingest-audit.jsonl.

    Синхронный def: fs-операции уходят в тредпул, событийный цикл не блокируется.
    Гонка с идущим джобом ингеста того же document_id не закрывается (очередь
    последовательная, удаление - ручная операция инженера; повторный DELETE
    снимет и результат гонки).
    """
    import logging

    from .. import config
    from ..canonical.store import delete_document as store_delete
    from ..canonical.store import find_document_manifest
    from ..indexing.overlay import tombstone_document

    ident = _identity(request, authorization)

    manifest = find_document_manifest(document_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="документ не найден")
    doc_groups = set(manifest.get("access_groups") or [])
    # тот же взгляд на «админа», что у чтения (_visible_by_groups): flag admin
    # или wildcard "*" в группах; static-владелец - admin=True
    if not (ident.admin or "*" in ident.groups or f"user:{ident.user_id}" in doc_groups):
        raise HTTPException(status_code=404, detail="документ не найден")

    info = store_delete(document_id, by=ident.user_id or "owner")
    if info is None:  # гонка: манифест видели, канон исчез - неотличимый 404
        raise HTTPException(status_code=404, detail="документ не найден")

    # ревью K10 P2: L0 ищем по каноническому id из манифеста - sanitizer-
    # эквивалентный id из URL не находит файл, и L0-сирота молча ест квоту
    discard_l0_file(manifest.get("document_id") or document_id)

    pipeline = None
    try:
        pipeline = _ctx(request)["pipeline_fn"]()
    except Exception:  # noqa: BLE001 - нет индекса: tombstone соберётся с диска
        pass
    nodes = tombstone_document(
        config.INDEX_DIR, document_id,
        bundle=pipeline.bundle if pipeline is not None else None,
    )
    if pipeline is not None:
        # decide(): удалённый заголовок больше не «известный документ», если его
        # не несёт ни один живой узел. K21: снапшот под INDEX_LOCK - воркер
        # очереди расширяет node_meta, итерация по живому dict небезопасна
        title = (info.get("title") or "").strip()
        # компактная мета: скан title/status - полный values() материализовал
        # бы все узлы с распаковкой текстов
        with INDEX_LOCK:
            metas = [
                m for _, m in iter_node_fields(pipeline.bundle.node_meta, "title", "status")
            ]
        if title and not any(
            m.get("title") == title and m.get("status") != "deleted"
            for m in metas
        ):
            pipeline.doc_titles.discard(title)

    if nodes == 0 and (config.INDEX_DIR / "meta.json").exists():
        logging.getLogger("veles.api.documents").warning(
            "delete %s: узлов в индексе не найдено (индекс без документа?)",
            document_id,
        )
    # аудит (план admin-audit): target=document_id, сколько узлов выведено
    get_audit().log(
        "document_delete", actor=actor_of(ident), admin=ident.admin,
        target=document_id, details=f"узлов: {nodes}", ip=client_ip(request),
    )
    return {
        "status": "deleted",
        "document_id": document_id,
        "nodes": nodes,
        "revisions": info["revisions"],
        "trash": info["trash"],
    }
