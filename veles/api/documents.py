"""Роутер загрузки документов пользователем (K4): POST /v1/documents + джобы.

POST /v1/documents        - multipart (file + опциональные title, group,
                            access_groups) или JSON {text, title, group};
                            создаёт джоб, ответ 202.
GET /v1/documents/jobs    - список своих джобов (админ - все).
GET /v1/documents/jobs/{id} - статус queued/running/done/failed с причиной.
DELETE /v1/documents/{id} - удаление документа целиком (K10): владелец
                            (user:<uid> в access_groups) или админ, чужой -
                            404 неотличимый; канон (все ревизии и assets) -
                            в trash, L0 удаляется, узлы выводятся из поиска
                            сразу (tombstone).

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
        with INDEX_LOCK:
            metas = list(pipeline.bundle.node_meta.values())
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
    return {
        "status": "deleted",
        "document_id": document_id,
        "nodes": nodes,
        "revisions": info["revisions"],
        "trash": info["trash"],
    }
