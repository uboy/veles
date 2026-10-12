"""Сквозная валидация LDAP-входа - сценарий ИБ-службы на живом приложении.

Фаза 2 плана docs/plan/ldap-identity/plan.md (эпик ldap-identity, коммит
d150fdc). Живой контур: FastAPI-приложение через TestClient + in-process
LDAP-сервер на ldaptor (Twisted reactor в фоновом потоке), документы загружены
через настоящий POST /v1/documents (очередь -> парсер -> индекс), вопросы -
через настоящий POST /v1/knowledge/query с фильтром по группам.

Сценарий: 2 пользователя каталога (ivanov - отдел + все, petrov - только все),
2 LDAP-группы, маппинг в группы veles [all + otdel]. Проверки:
  (а) вход -> jwt -> вопрос видит документ своей группы и НЕ видит чужой;
  (б) неверный пароль -> 401, неотличимый от неизвестного пользователя,
      пароль не в логах и не в теле ответа;
  (в) отзыв из группы в каталоге + повторный вход -> чужое невидимо
      (старый токен до истечения ещё работает - по дизайну, группы в claims);
  (г) старый jwt (до отзыва) с коротким TTL протухает - 401;
  (д) без VELES_LDAP_URL маршрут отключён (404), текущие режимы не сломаны.

Запуск: python evals/ldap-identity-validate.py   (системный python, из корня репо)
Успех: печатает VALIDATION-PASS, код возврата 0. Любая ошибка - AssertionError,
код возврата 1.

Временные файлы (VELES_DATA: uploads/canonical/groups.json) - во временной
папке ОС, удаляются в конце; захардкоженных путей нет.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------- окружение
# Код валидируемого worktree - первым на sys.path: при запуске
# «python evals/...» sys.path[0] - каталог evals/, и без этого импортируется
# УСТАНОВЛЕННЫЙ в site-packages veles (без коммита ldap-identity) - маршрут
# /v1/auth/ldap отсутствует и сценарий проверяет не тот код.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# VELES_DATA и VELES_GROUPS_FILE ДО импортов veles: config читает env на импорте
ROOT = Path(tempfile.mkdtemp(prefix="veles-ldap-eval-"))
os.environ["VELES_DATA"] = str(ROOT / "data")
# изоляция разбора отдельным процессом выключена (как в tests/conftest.py):
# spawn-дочерний процесс не наследует подстановки окружения
os.environ["VELES_QUEUE_PARSE_TIMEOUT_S"] = "0"
os.environ["VELES_IDENTITY_MODE"] = "jwt"
SECRET = "jwt-secret-eval-ldap"
os.environ["VELES_IDENTITY_JWT_SECRET"] = SECRET
# короткоживущий jwt (проверка (г): старый токен протухает)
os.environ["VELES_LDAP_JWT_TTL_S"] = "10"
os.environ["VELES_LDAP_BIND_DN_TEMPLATE"] = "uid={username},ou=people,dc=corp,dc=ru"
os.environ["VELES_LDAP_BASE_DN"] = "dc=corp,dc=ru"
os.environ["VELES_LDAP_GROUP_BASE_DN"] = "ou=groups,dc=corp,dc=ru"
os.environ["VELES_LDAP_TIMEOUT_S"] = "5"
GROUP_MAP = {
    "cn=corp-all,ou=groups,dc=corp,dc=ru": "all",
    "cn=otdel,ou=groups,dc=corp,dc=ru": "otdel",
}
os.environ["VELES_LDAP_GROUP_MAP"] = json.dumps(GROUP_MAP)

# группы veles: all (видно всем) + otdel (отдел); members пусты - состав
# приходит из каталога через claims jwt, конфиг групп держит только определения
GROUPS_JSON = ROOT / "data" / "groups.json"
GROUPS_JSON.parent.mkdir(parents=True, exist_ok=True)
GROUPS_JSON.write_text(
    json.dumps(
        {
            "groups": [
                {"id": "all", "title": "Видно всем", "members": []},
                {"id": "otdel", "title": "Отдел закупок", "members": []},
            ]
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

import numpy as np  # noqa: E402  (после установки env)
from fastapi.testclient import TestClient  # noqa: E402

from veles.api.app import VelesPipeline, create_app  # noqa: E402
from veles.config import EMBED_DIM  # noqa: E402
from veles.indexing.indexer import build_index  # noqa: E402
from veles.indexing.vector_store import NumpyFlatStore  # noqa: E402
from veles.policy.identity import verify_hs256  # noqa: E402
from veles.policy.ldap_identity import is_configured  # noqa: E402

OWNER_TOKEN = "eval-owner-bearer"
IVANOV, IVANOV_PW = "ivanov", "пароль-иванова"
PETROV, PETROV_PW = "petrov", "пароль-петрова"
BASE_DN = "dc=corp,dc=ru"

OTDEL_DOC_TITLE = "Регламент закупок отдела"
OTDEL_MARK = "350 тысяч рублей"
ALL_DOC_TITLE = "Коучинг и мотивация"
ALL_MARK = "коучинг работает с целями"


class HashingEmbedder:
    """Детерминированный эмбеддер (sublinear tf по хеш-бакетам, как в тестах
    ingest): валидация не грузит модели, фильтр по группам от эмбеддера не
    зависит. dim = EMBED_DIM - как в проде."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in re.findall(r"[а-яёa-z0-9]+", text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _build_ldap_tree():
    """Каталог «корпорация»: ivanov (memberOf otdel + posixGroup corp-all),
    petrov (memberOf corp-all). objectClass задан у каждой записи - фильтр
    (objectClass=*) у ldaptor матчит только записи с заполненным атрибутом."""
    from ldaptor.inmemory import ReadOnlyInMemoryLDAPEntry

    root = ReadOnlyInMemoryLDAPEntry(
        BASE_DN, {"dc": ["corp", "ru"], "objectClass": ["dcObject", "organization"]}
    )
    people = root.addChild(
        "ou=people", {"ou": ["people"], "objectClass": ["organizationalUnit"]}
    )
    people.addChild(
        "uid=ivanov",
        {
            "uid": [IVANOV],
            "cn": ["Иванов И."],
            "objectClass": ["inetOrgPerson"],
            "userPassword": [IVANOV_PW],
            "memberOf": ["cn=otdel,ou=groups,dc=corp,dc=ru"],
        },
    )
    people.addChild(
        "uid=petrov",
        {
            "uid": [PETROV],
            "cn": ["Петров П."],
            "objectClass": ["inetOrgPerson"],
            "userPassword": [PETROV_PW],
            "memberOf": ["cn=corp-all,ou=groups,dc=corp,dc=ru"],
        },
    )
    groups = root.addChild(
        "ou=groups", {"ou": ["groups"], "objectClass": ["organizationalUnit"]}
    )
    # corp-all как posixGroup (memberUid): вторая форма membership, обе проходят
    # маппинг; otdel - memberOf на записи пользователя (форма AD)
    groups.addChild(
        "cn=corp-all",
        {
            "cn": ["corp-all"],
            "objectClass": ["posixGroup"],
            "memberUid": [IVANOV, PETROV],
        },
    )
    groups.addChild(
        "cn=otdel", {"cn": ["otdel"], "objectClass": ["groupOfNames"]}
    )
    return root


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


_results: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Одна проверка сценария: печать OK + накопление для финала."""
    if not cond:
        raise AssertionError(
            f"{name}: условие не выполнено{'; ' + detail if detail else ''}"
        )
    line = f"OK  {name}" + (f" ({detail})" if detail else "")
    print(line, flush=True)
    _results.append(line)


def has_mark(resp, mark: str) -> bool:
    """Маркер документа в выдаче (без учёта регистра - парсер сохраняет регистр
    заголовка и первой буквы предложения)."""
    m = mark.lower()
    return any(m in t.lower() for t in evidence_texts(resp))


def found_texts(resp, mark: str) -> str:
    """Краткая диагностика выдачи для сообщения об ошибке."""
    texts = evidence_texts(resp)
    return (
        f"узлов: {len(texts)}; "
        f"первые: {[t[:60] for t in texts[:2]]}; маркер={mark[:30]!r}"
    )


def evidence_texts(resp) -> list[str]:
    return [ev["text"] for ev in resp.json().get("evidence", [])]


def main() -> int:
    # ------------------------------------------------ in-process LDAP-сервер
    from twisted.internet import reactor

    started = threading.Event()

    def _run_reactor():
        started.set()
        reactor.run(installSignalHandlers=False)

    thread = threading.Thread(target=_run_reactor, daemon=True, name="twisted-reactor")
    thread.start()
    assert started.wait(10), "reactor не поднялся"

    from ldaptor import interfaces as ldap_ifaces
    from ldaptor.protocols.ldap.ldapserver import LDAPServer
    from twisted.internet import protocol as tprotocol
    from twisted.python.components import registerAdapter

    class Factory(tprotocol.ServerFactory):
        protocol = LDAPServer

        def __init__(self, root):
            self.root = root

    registerAdapter(lambda x: x.root, Factory, ldap_ifaces.IConnectedLDAPEntry)

    tree = _build_ldap_tree()
    box: list = []
    done = threading.Event()

    def _listen():
        try:
            box.append(reactor.listenTCP(0, Factory(tree), interface="127.0.0.1"))
        finally:
            done.set()

    reactor.callFromThread(_listen)
    assert done.wait(10), "listenTCP не завершился"
    port = box[0].getHost().port
    os.environ["VELES_LDAP_URL"] = f"ldap://127.0.0.1:{port}"
    print(f"LDAP-сервер: ldap://127.0.0.1:{port} (дерево: 2 пользователя, 2 группы)")

    # -------------------------------------- живое приложение: индекс + загрузки
    assert is_configured(), (
        f"VELES_LDAP_URL={os.environ.get('VELES_LDAP_URL')!r} не виден "
        "ldap_identity.is_configured - маршрут не зарегистрируется"
    )
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=EMBED_DIM))
    pipeline = VelesPipeline(bundle, HashingEmbedder(), llm=None)
    client = TestClient(create_app(pipeline=pipeline, api_token=OWNER_TOKEN))
    owner = {"Authorization": f"Bearer {OWNER_TOKEN}"}

    def upload(title: str, text: str, group: str) -> None:
        """Настоящий цикл загрузки: POST /v1/documents -> очередь -> парсер ->
        индекс (живой апсерт + overlay). Ожидание опустешения очереди."""
        r = client.post(
            "/v1/documents", json={"text": text, "title": title, "group": group},
            headers=owner,
        )
        assert r.status_code == 202, f"загрузка {title!r}: {r.status_code} {r.text}"
        queue = client.app.state.veles_ingest
        assert queue is not None and queue.wait_idle(30), "очередь не опустела"

    upload(
        OTDEL_DOC_TITLE,
        f"# {OTDEL_DOC_TITLE}\n\nПорог единственной закупки составляет {OTDEL_MARK}.\n\n"
        "Закупки свыше порога требуют обязательного тендера.\n",
        "otdel",
    )
    upload(
        ALL_DOC_TITLE,
        f"# {ALL_DOC_TITLE}\n\nКоучинг работает с целями и мотивация сотрудника.\n",
        "all",
    )
    print("Документы загружены: 1 в группу otdel, 1 в группу all")

    def login(username: str, password: str):
        return client.post(
            "/v1/auth/ldap", json={"username": username, "password": password}
        )

    def query(token: str, q: str):
        return client.post(
            "/v1/knowledge/query",
            json={"query": q, "use_llm": False, "top_k": 5},
            headers={"X-OpenWebUI-User-Jwt": token},
        )

    try:
        # -------------------------------------------- (а) вход -> jwt -> вопрос
        r = login(IVANOV, IVANOV_PW)
        check("(а) вход ivanov: 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        token1 = body["access_token"]
        payload1 = verify_hs256(token1, SECRET)
        check(
            "(а) jwt проходит verify_hs256, payload только sub+groups+exp",
            payload1 is not None and set(payload1) == {"sub", "groups", "exp"},
            f"groups={body['groups']}",
        )
        check(
            "(а) ivanov: группы из каталога через маппинг = all + otdel",
            payload1 is not None and payload1["sub"] == IVANOV
            and set(payload1["groups"]) == {"all", "otdel"},
        )
        resp = query(token1, "какой порог единственной закупки и нужен ли тендер")
        check("(а) вопрос ivanov: 200", resp.status_code == 200)
        check(
            "(а) ivanov видит документ своей группы (otdel)",
            has_mark(resp, OTDEL_MARK), found_texts(resp, OTDEL_MARK),
        )
        resp = query(token1, "что такое коучинг и работа с целями сотрудника")
        check(
            "(а) ivanov видит общий документ (all)",
            has_mark(resp, ALL_MARK), found_texts(resp, ALL_MARK),
        )
        r2 = login(PETROV, PETROV_PW)
        check("(а) вход petrov: 200", r2.status_code == 200)
        token2 = r2.json()["access_token"]
        check(
            "(а) petrov: только all (otdel нет в маппинге его групп)",
            set(r2.json()["groups"]) == {"all"},
            f"groups={r2.json()['groups']}",
        )
        resp = query(token2, "какой порог единственной закупки и нужен ли тендер")
        check(
            "(а) petrov НЕ видит чужой документ отдела",
            not has_mark(resp, OTDEL_MARK), found_texts(resp, OTDEL_MARK),
        )
        resp = query(token2, "что такое коучинг и работа с целями сотрудника")
        check(
            "(а) petrov видит общий документ",
            has_mark(resp, ALL_MARK), found_texts(resp, ALL_MARK),
        )

        # ------------------------------- (б) неверный пароль -> 401 без деталей
        cap = _CaptureHandler()
        logging.getLogger().addHandler(cap)
        try:
            wrong = login(IVANOV, "совсем-не-тот-пароль-секрет")
            unknown = login("такого-нет-в-каталоге", "любой-пароль")
        finally:
            logging.getLogger().removeHandler(cap)
        check(
            "(б) неверный пароль -> 401",
            wrong.status_code == 401,
            str(wrong.status_code),
        )
        check(
            "(б) 401 на неверный пароль и неизвестного пользователя неотличимы",
            wrong.json() == unknown.json() == {"detail": "неверные учётные данные"},
        )
        check(
            "(б) пароль не в логах и не в теле ответа",
            "совсем-не-тот-пароль-секрет" not in " ".join(cap.records)
            and "совсем-не-тот-пароль-секрет" not in wrong.text,
            f"записей лога перехвачено: {len(cap.records)}",
        )

        # -------------- (в) отзыв из группы в каталоге + повторный вход
        entry = tree.lookup(f"uid={IVANOV},ou=people,{BASE_DN}").result
        entry.__delitem__("memberOf")  # админ каталога снял членство в otdel
        r3 = login(IVANOV, IVANOV_PW)
        check("(в) повторный вход после отзыва: 200", r3.status_code == 200)
        token3 = r3.json()["access_token"]
        check(
            "(в) новый jwt без отозванной группы",
            r3.json()["groups"] == ["all"],
            f"groups={r3.json()['groups']}",
        )
        resp = query(token3, "какой порог единственной закупки и нужен ли тендер")
        check(
            "(в) после отзыва документ отдела невидим",
            not has_mark(resp, OTDEL_MARK), found_texts(resp, OTDEL_MARK),
        )
        resp = query(token3, "что такое коучинг и работа с целями сотрудника")
        check(
            "(в) общий документ по-прежнему виден",
            has_mark(resp, ALL_MARK), found_texts(resp, ALL_MARK),
        )
        # контроль дизайна: старый живой токен (до отзыва) ещё работает -
        # группы в подписанных claims; это закрывается коротким TTL (проверка г)
        resp = query(token1, "какой порог единственной закупки и нужен ли тендер")
        check(
            "(в) контроль: старый jwt до истечения ещё видит отдел (claims)",
            resp.status_code == 200 and has_mark(resp, OTDEL_MARK),
            found_texts(resp, OTDEL_MARK),
        )

        # -------------------------- (г) старый jwt с коротким TTL протухает
        assert payload1 is not None
        remain = float(payload1["exp"]) - time.time() + 0.5
        if remain > 0:
            print(f"ожидание истечения TTL старого jwt: {remain:.1f}с", flush=True)
            time.sleep(remain)
        check(
            "(г) verify_hs256 отклоняет просроченный jwt",
            verify_hs256(token1, SECRET) is None,
        )
        resp = query(token1, "какой порог единственной закупки и нужен ли тендер")
        check(
            "(г) просроченный jwt на живом приложении -> 401",
            resp.status_code == 401, str(resp.status_code),
        )

        # -------- (д) без VELES_LDAP_URL маршрут отключён, режимы не сломаны
        os.environ.pop("VELES_LDAP_URL", None)
        client_off = TestClient(create_app(pipeline=pipeline, api_token=OWNER_TOKEN))
        r4 = client_off.post(
            "/v1/auth/ldap", json={"username": IVANOV, "password": IVANOV_PW}
        )
        check(
            "(д) без VELES_LDAP_URL вход -> 404 (маршрут не зарегистрирован)",
            r4.status_code == 404, str(r4.status_code),
        )
        r5 = client_off.post(
            "/v1/knowledge/query",
            json={"query": "что такое коучинг и работа с целями сотрудника", "use_llm": False},
            headers=owner,
        )
        check(
            "(д) текущий режим не сломан: вопрос с bearer-токеном -> 200",
            r5.status_code == 200, str(r5.status_code),
        )

        print("\n" + "\n".join(_results))
        print("VALIDATION-PASS", flush=True)
        return 0
    finally:
        reactor.callFromThread(lambda: box[0].stopListening())
        reactor.callFromThread(reactor.stop)
        thread.join(timeout=10)
        shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    try:
        _code = main()
    except AssertionError as e:
        print(f"VALIDATION-FAIL: {e}", file=sys.stderr, flush=True)
        _code = 1
    raise SystemExit(_code)
