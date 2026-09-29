"""GET /v1/groups - список групп для выбора при загрузке (K7).

Группы серверные (veles/policy/groups.py, конфиг <VELES_DATA>/groups.json);
эндпоинт питает выбор группы при загрузке (group=<id> в POST /v1/documents)
и позволяет админу сверить текущий набор. Служебные личные группы user:* не
показываются: их назначает сервер, клиент не выбирает. Авторизация - та же
личность, что у остальных API (инвариант 5), через шов app.state; app.py
больше ничего о группах не знает (как с документами в K4).
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from ..policy.groups import ensure_groups_file, load_groups

router = APIRouter(prefix="/v1/groups", tags=["groups"])


def _identity(request: Request, authorization: str):
    state = getattr(request.app.state, "veles", None)
    if state is None:  # create_app не достроил контекст - не должно случаться
        raise HTTPException(status_code=500, detail="контекст фасада не инициализирован")
    return state["identity_fn"](request, authorization)


@router.get("")
def list_groups(request: Request, authorization: str = Header(default="")):
    """Группы, доступные для выбора при загрузке: id + человекочитаемое
    название. Первый вызов создаёт файл-шаблон <VELES_DATA>/groups.json с
    дефолтными группами, если конфига ещё нет (дальше состав правит админ
    в файле - фасад перечитывает его на каждый запрос)."""
    _identity(request, authorization)
    ensure_groups_file()
    return {
        "groups": [
            {"id": g.id, "title": g.title}
            for g in load_groups()
            if not g.id.startswith("user:")
        ]
    }
