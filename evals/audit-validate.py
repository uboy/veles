"""Сквозная валидация журнала аудита на ЖИВОМ uvicorn-сервере (фаза 3 плана
admin-audit, docs/plan/admin-audit/plan.md; коммит реализации a008982).

Не TestClient: FastAPI-приложение поднимается настоящим uvicorn на 127.0.0.1
(свободный порт), клиент - реальные HTTP-запросы httpx. Личность - jwt-режим
identity: токены админа и пользователя минтятся sign_hs256 (известный секрет),
админ - из VELES_ADMIN_EMAILS. Для сценария (е) поднимается in-process
LDAP-каталог (ldaptor): вход по логину/паролю идёт через настоящий
POST /v1/auth/ldap -> ldap3 bind. Эмбеддер - детерминированный hashing
(канон репо: тесты/evals без загрузки моделей), предмет валидации - журнал
аудита, а не модель.

Сценарии плана:
  (а) поиск и вопрос пользователя -> события search/ask с текстом запроса;
  (б) загрузка -> смена групп -> удаление документа админом -> цепочка
      upload/document_regroup/document_delete;
  (в) не-админ зовёт GET /v1/admin/audit -> 403 и событие audit_access_denied;
      админ читает журнал: все события по порядку, фильтры
      actor/action/q/since/until отсекают правильно, limit/offset работают;
  (г) ротация: VELES_AUDIT_ROTATE_MB=1 -> журнал набивается до порога, живой
      запрос рождает запись через ротацию: audit.jsonl.1 создан, старый
      сдвинут, события не теряются;
  (д) латентность поиска с журналом против VELES_AUDIT=0 (50 запросов) -
      деградация в разы недопустима (критерий: среднее и медиана ON <= 2x OFF);
  (е) секреты: jwt из заголовка, владельческий bearer и пароли LDAP (верный и
      неверный) в файлах журнала отсутствуют.

Запуск: python evals/audit-validate.py   (из любого cwd; системный python)
Успех: финальная строка VALIDATION-PASS, код возврата 0; любая ошибка -
AssertionError, VALIDATION-FAIL, код 1.

Временные файлы (VELES_DATA с каноном/индексом/журналом) - во временной папке
ОС, удаляются в конце; захардкоженных машинных путей нет.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import socket
import statistics
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------- окружение
# Код валидируемого worktree - первым на sys.path: при запуске
# «python evals/...» sys.path[0] - каталог evals/, иначе импортируется
# УСТАНОВЛЕННЫЙ в site-packages veles без коммита a008982.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# env ДО импортов veles: config читает VELES_DATA на импорте, AuditLog -
# VELES_AUDIT*/VELES_ADMIN_EMAILS при создании синглтона (первый запрос)
ROOT = Path(tempfile.mkdtemp(prefix="veles-audit-eval-"))
DATA = ROOT / "data"
os.environ["VELES_DATA"] = str(DATA)
os.environ["VELES_IDENTITY_MODE"] = "jwt"
SECRET = "jwt-secret-eval-audit"
os.environ["VELES_IDENTITY_JWT_SECRET"] = SECRET
ADMIN_EMAIL, USER_EMAIL = "admin@corp.ru", "ivanov@corp.ru"
os.environ["VELES_ADMIN_EMAILS"] = ADMIN_EMAIL
os.environ["VELES_RATE_LIMIT_RPM"] = "0"  # лимит запросов - не предмет валидации
# парс в своём процессе (как tests/conftest.py): детерминированно, без spawn
os.environ["VELES_QUEUE_PARSE_TIMEOUT_S"] = "0"
os.environ["VELES_AUDIT_ROTATE_MB"] = "1"  # порог ротации 1 МБ - для сценария (г)
OWNER_BEARER = "veles-eval-bearer-sekret-0987654321"  # для проверки (е) - не в журнал; ascii: заголовки не берут не-ascii
os.environ["VELES_API_TOKEN"] = OWNER_BEARER
os.environ["VELES_LDAP_BIND_DN_TEMPLATE"] = "uid={username},ou=people,dc=corp,dc=ru"
os.environ["VELES_LDAP_BASE_DN"] = "dc=corp,dc=ru"
os.environ["VELES_LDAP_GROUP_BASE_DN"] = "ou=groups,dc=corp,dc=ru"
os.environ["VELES_LDAP_TIMEOUT_S"] = "5"
os.environ["VELES_LDAP_GROUP_MAP"] = json.dumps(
    {"cn=corp-all,ou=groups,dc=corp,dc=ru": "all"}
)

# группы veles: regroup (б) требует существующую группу; membership пуст -
# реальные группы приходят через каталог (для этого eval не нужно)
GROUPS_JSON = DATA / "groups.json"
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

import httpx  # noqa: E402  (после установки env)
import numpy as np  # noqa: E402

from veles.api.app import VelesPipeline, create_app  # noqa: E402
from veles.audit import reset_audit  # noqa: E402  (переключатель (д))
from veles.config import EMBED_DIM, EMBED_MODEL  # noqa: E402
from veles.indexing.indexer import build_index  # noqa: E402
from veles.indexing.vector_store import NumpyFlatStore  # noqa: E402
from veles.policy.identity import sign_hs256  # noqa: E402

SEARCH_QUERY = "какой порог единственной закупки и нужен ли тендер"
ASK_QUESTION = "сколько страниц в регламенте закупок"
DOC_TITLE = "Регламент закупок отдела"
DOC_MARK = "350 тысяч рублей"
ROTATE_QUERY = "запрос после ротации журнала"
PETROV, PETROV_PW = "petrov", "Верный-Пароль-петрова-777"
PETROV_WRONG_PW = "Неверный-пароль-12345"
BASE_DN = "dc=corp,dc=ru"

_results: list[str] = []
_numbers: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    """Одна проверка сценария: печать OK; провал - AssertionError."""
    if not cond:
        raise AssertionError(f"{name}: условие не выполнено{'; ' + detail if detail else ''}")
    line = f"OK  {name}" + (f" ({detail})" if detail else "")
    print(line, flush=True)
    _results.append(line)


def numbers(line: str) -> None:
    """Фактическая цифра для отчёта (число событий/замер)."""
    print(f"ЦИФРА  {line}", flush=True)
    _numbers.append(line)


# ------------------------------------------------------------ эмбеддер/индекс


class HashingEmbedder:
    """Детерминированный эмбеддер (sublinear tf по хеш-бакетам, как в тестах):
    валидация не грузит модели; dim = EMBED_DIM - как в проде."""

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


# --------------------------------------------------------------- LDAP-каталог


def _build_ldap_tree():
    """Каталог «корпорация»: petrov (memberOf corp-all, верный пароль)."""
    from ldaptor.inmemory import ReadOnlyInMemoryLDAPEntry

    root = ReadOnlyInMemoryLDAPEntry(
        BASE_DN, {"dc": ["corp", "ru"], "objectClass": ["dcObject", "organization"]}
    )
    people = root.addChild(
        "ou=people", {"ou": ["people"], "objectClass": ["organizationalUnit"]}
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
    groups.addChild(
        "cn=corp-all",
        {"cn": ["corp-all"], "objectClass": ["posixGroup"], "memberUid": [PETROV]},
    )
    return root


def start_ldap() -> tuple[int, object, threading.Thread]:
    """In-process LDAP-сервер на свободном порту 127.0.0.1 (как
    evals/ldap-identity-validate.py): Twisted reactor в фоновом потоке."""
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
    return box[0].getHost().port, box[0], thread


# ------------------------------------------------------------ журнал с диска


def audit_files() -> list[Path]:
    """Все файлы журнала: текущий + ротированные (audit.jsonl.N)."""
    d = DATA / "audit"
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.name == "audit.jsonl" or p.name.startswith("audit.jsonl.")]
    return sorted(files)


def parse_events(path: Path) -> list[dict]:
    """Валидные JSON-события одного файла (битые строки пропускам - так же
    делает чтение журнала)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def disk_events() -> list[dict]:
    """Все события со всех файлов журнала (без учёта порядка между файлами)."""
    out: list[dict] = []
    for f in audit_files():
        out.extend(parse_events(f))
    return out


def journal_text() -> str:
    """Полный текст журнала - для проверок секретов."""
    return "\n".join(f.read_text(encoding="utf-8") for f in audit_files())


def by_action(events: list[dict]) -> dict[str, int]:
    agg: dict[str, int] = {}
    for e in events:
        agg[e["action"]] = agg.get(e["action"], 0) + 1
    return dict(sorted(agg.items()))


# ------------------------------------------------------------------- сценарии


def main() -> int:
    port, ldap_listener, reactor_thread = start_ldap()
    os.environ["VELES_LDAP_URL"] = f"ldap://127.0.0.1:{port}"
    print(f"LDAP-сервер: ldap://127.0.0.1:{port}", flush=True)

    # живое приложение: настоящий пустой индекс (как kb_env в тестах) -
    # upload -> done -> regroup/delete доходят до канона и overlay
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=EMBED_DIM))
    index_dir = DATA / "index"
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    app = create_app(pipeline=VelesPipeline(bundle, HashingEmbedder(), llm=None))

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    server_thread = threading.Thread(target=server.run, daemon=True, name="uvicorn")
    server_thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        time.sleep(0.2)
        try:
            # trust_env=False: в среде бывает http_proxy - петля 127.0.0.1 через
            # прокси давала бы 502 вместо живого фасада
            with httpx.Client(base_url=base, timeout=2, trust_env=False) as probe:
                r = probe.get("/healthz")
            if r.status_code == 200:
                break
        except Exception:  # noqa: BLE001 - сервер ещё поднимается
            continue
    else:
        raise AssertionError(f"uvicorn не поднялся на {base}")
    print(f"uvicorn жив: {base} (identity=jwt, VELES_DATA={DATA})", flush=True)

    try:
        client = httpx.Client(base_url=base, timeout=30.0, trust_env=False)

        now = int(time.time())
        admin_tok = sign_hs256({"sub": ADMIN_EMAIL, "exp": now + 7200}, SECRET)
        user_tok = sign_hs256({"sub": USER_EMAIL, "exp": now + 7200}, SECRET)
        admin_h = {"X-OpenWebUI-User-Jwt": admin_tok}
        user_h = {"X-OpenWebUI-User-Jwt": user_tok}

        # ------------------------------------------------ (а) search и ask
        r = client.post(
            "/v1/knowledge/query",
            json={"query": SEARCH_QUERY, "use_llm": False, "top_k": 5},
            headers=user_h,
        )
        check("(а) поиск пользователя: 200", r.status_code == 200, str(r.status_code))
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": ASK_QUESTION}]},
            headers=user_h,
        )
        check("(а) вопрос пользователя: 200", r.status_code == 200, str(r.status_code))
        ev = disk_events()
        check(
            "(а) журнал: [search, ask] с текстом запроса и actor пользователя",
            [e["action"] for e in ev] == ["search", "ask"]
            and ev[0]["details"] == SEARCH_QUERY
            and ev[1]["details"] == ASK_QUESTION
            and ev[0]["actor"] == USER_EMAIL
            and ev[0]["admin"] is False
            and set(ev[0]) == {"ts", "actor", "admin", "action", "target", "details", "ip"},
            f"actions={[e['action'] for e in ev]}",
        )
        check("(а) ip клиента записан", ev[0]["ip"] == "127.0.0.1", ev[0]["ip"])
        numbers("(а) событий: 2 (search, ask); actor=ivanov@corp.ru")

        # ------------------------- (б) upload -> regroup -> delete админом
        r = client.post(
            "/v1/documents",
            files={"file": ("регламент-закупок.txt",
                            f"# {DOC_TITLE}\n\nПорог закупки - {DOC_MARK}.\n".encode("utf-8"),
                            "text/plain")},
            data={"title": DOC_TITLE},
            headers=admin_h,
        )
        check("(б) загрузка админом: 202", r.status_code == 202, r.text[:200])
        job_id = r.json()["job_id"]
        document_id = r.json()["document_id"]
        deadline = time.time() + 60
        while time.time() < deadline:
            jr = client.get(f"/v1/documents/jobs/{job_id}", headers=admin_h)
            if jr.status_code == 200 and jr.json().get("status") in ("done", "failed"):
                break
            time.sleep(0.3)
        check(
            "(б) джоб загрузки дошёл до done",
            jr.json().get("status") == "done", jr.text[:200],
        )
        r = client.post(
            f"/v1/documents/{urllib.parse.quote(document_id)}/groups",
            json={"groups": ["otdel"]},
            headers=admin_h,
        )
        check("(б) смена групп: 200 regrouped",
              r.status_code == 200 and r.json().get("status") == "regrouped",
              r.text[:200])
        r = client.delete(f"/v1/documents/{urllib.parse.quote(document_id)}",
                          headers=admin_h)
        check("(б) удаление документа: 200", r.status_code == 200, r.text[:200])
        ev = disk_events()
        chain = [e for e in ev if e["action"] in
                 ("upload", "document_regroup", "document_delete")]
        check(
            "(б) цепочка upload -> document_regroup -> document_delete",
            [e["action"] for e in chain] == ["upload", "document_regroup", "document_delete"],
            f"actions={[e['action'] for e in ev]}",
        )
        check(
            "(б) поля цепочки: actor=admin, target=файл/document_id, детали верные",
            chain[0]["actor"] == ADMIN_EMAIL and chain[0]["admin"] is True
            and chain[0]["target"] == "регламент-закупок.txt"
            and chain[0]["details"] == document_id
            and chain[1]["target"] == document_id and "otdel" in chain[1]["details"]
            and chain[2]["target"] == document_id and "узлов" in chain[2]["details"],
            f"chain={[(e['target'], e['details']) for e in chain]}",
        )
        numbers(f"(б) всего событий: {len(ev)}; по действиям: {by_action(ev)}")

        # ------------- (в) 403 не-админу; админ читает: порядок и фильтры
        r = client.get("/v1/admin/audit", headers=user_h)
        check("(в) не-админ: 403", r.status_code == 403, str(r.status_code))
        ev = disk_events()
        check(
            "(в) отказ записан: audit_access_denied, actor пользователя",
            ev[-1]["action"] == "audit_access_denied"
            and ev[-1]["actor"] == USER_EMAIL and ev[-1]["admin"] is False
            and ev[-1]["target"] == "/v1/admin/audit",
            str(ev[-1]),
        )
        expected = ["search", "ask", "upload", "document_regroup",
                    "document_delete", "audit_access_denied"]
        reads_before = len(ev)
        r = client.get("/v1/admin/audit", params={"limit": 1000}, headers=admin_h)
        check("(в) админ читает журнал: 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        got = [e["action"] for e in body["events"]]
        ts_list = [e["ts"] for e in body["events"]]
        check(
            "(в) все события по порядку, ts хронологичен",
            got == expected and ts_list == sorted(ts_list),
            f"got={got}",
        )
        check("(в) total = числу событий", body["total"] == reads_before,
              f"{body['total']} vs {reads_before}")

        def total_of(**params) -> int:
            rr = client.get("/v1/admin/audit", params=params, headers=admin_h)
            assert rr.status_code == 200, f"{params}: {rr.status_code} {rr.text}"
            return rr.json()["total"]

        check("(в) фильтр actor отсекает", total_of(actor=ADMIN_EMAIL, limit=1000) == 3,
              str(total_of(actor=ADMIN_EMAIL, limit=1000)))
        check("(в) фильтр action отсекает",
              total_of(action="document_delete", limit=1000) == 1)
        q_total = total_of(q="закуп", limit=1000)
        q_case = total_of(q="ЗАКУП", limit=1000)  # без регистра
        # ожидание считаем из файла журнала независимым фильтром (как read()):
        # «закупок» сидит и в target regroup/delete, и в details upload
        expected_q = sum(
            1 for e in ev
            if "закуп" in (str(e.get("target")) + "\n" + str(e.get("details"))).lower()
        )
        check("(в) фильтр q (подстрока target/details, без регистра)",
              q_total == q_case == expected_q,
              f"q={q_total}, Q={q_case}, ожидалось {expected_q}")
        check("(в) q без совпадений -> 0",
              total_of(q="нет-такой-подстроки-xyz", limit=1000) == 0)
        min_ts = min(e["ts"] for e in ev)
        check("(в) since = ts первого события (включительно) -> все",
              total_of(since=min_ts, limit=1000) == reads_before)
        check("(в) since в будущем -> 0",
              total_of(since="2099-01-01T00:00:00+00:00") == 0)
        check("(в) until в будущем -> все",
              total_of(until="2099-01-01T00:00:00+00:00", limit=1000) == reads_before)
        check("(в) until в прошлом -> 0", total_of(until="2000-01-01") == 0)
        bad = client.get("/v1/admin/audit", params={"since": "завтра-утром"},
                         headers=admin_h)
        check("(в) мусорная дата -> 400", bad.status_code == 400, str(bad.status_code))
        check("(в) limit=0 -> 422 (ge=1)",
              client.get("/v1/admin/audit", params={"limit": 0},
                         headers=admin_h).status_code == 422)
        page1 = client.get("/v1/admin/audit", params={"limit": 2, "offset": 0},
                           headers=admin_h).json()
        page2 = client.get("/v1/admin/audit", params={"limit": 2, "offset": 2},
                           headers=admin_h).json()
        page3 = client.get("/v1/admin/audit", params={"limit": 2, "offset": 4},
                           headers=admin_h).json()
        page4 = client.get("/v1/admin/audit", params={"limit": 2, "offset": 6},
                           headers=admin_h).json()
        check(
            "(в) limit/offset: страницы без потерь и дублей, total до среза",
            len(page1["events"]) == 2 and len(page2["events"]) == 2
            and len(page3["events"]) == 2 and page4["events"] == []
            and all(p["total"] == reads_before for p in (page1, page2, page3, page4))
            and [e["action"] for p in (page1, page2, page3) for e in p["events"]]
            == expected,
        )
        # владелец-токен в заголовке тоже был (смешан с jwt) - и для (е)
        rr = client.get("/v1/admin/audit",
                        headers={**admin_h, "Authorization": f"Bearer {OWNER_BEARER}"})
        check("(в) admin + bearer в одном запросе: 200", rr.status_code == 200)
        ev = disk_events()
        check("(в) успешные чтения аудита сами событий не создают",
              len(ev) == reads_before + 0 and ev[-1]["action"] == "audit_access_denied",
              f"{len(ev)} vs {reads_before}")
        numbers(f"(в) событий к моменту: {len(ev)}; 403 -> audit_access_denied записан")

        # --------------------------- (е) LDAP-вход: события и секреты
        wrong = client.post("/v1/auth/ldap",
                            json={"username": PETROV, "password": PETROV_WRONG_PW})
        check("(е) неверный пароль LDAP -> 401", wrong.status_code == 401,
              str(wrong.status_code))
        ok = client.post("/v1/auth/ldap",
                         json={"username": PETROV, "password": PETROV_PW})
        check("(е) верный пароль LDAP -> 200, группа из каталога",
              ok.status_code == 200 and ok.json().get("groups") == ["all"],
              ok.text[:200])
        petrov_tok = ok.json()["access_token"]
        ev = disk_events()
        check(
            "(е) журнал: auth_ldap_fail + auth_ldap_ok, паролей нет",
            ev[-2]["action"] == "auth_ldap_fail" and ev[-2]["actor"] == PETROV
            and ev[-1]["action"] == "auth_ldap_ok" and ev[-1]["actor"] == PETROV
            and "all" in ev[-1]["details"],
            f"actions={[e['action'] for e in ev[-2:]]}",
        )
        text = journal_text()
        for secret_name, secret_value in (
            ("пароль LDAP верный", PETROV_PW),
            ("пароль LDAP неверный", PETROV_WRONG_PW),
            ("jwt админа", admin_tok),
            ("jwt пользователя", user_tok),
            ("jwt от LDAP-входа", petrov_tok),
            ("владельческий bearer", OWNER_BEARER),
        ):
            check(f"(е) секрет не в журнале: {secret_name}",
                  secret_value not in text)
        numbers(f"(е) всего событий: {len(ev)}; секретов в журнале: 0 "
                "(2 пароля, 3 jwt, bearer)")

        # --------------- (д) латентность поиска ON vs VELES_AUDIT=0
        def timed_queries(n: int) -> list[float]:
            for _ in range(5):
                assert client.post("/v1/knowledge/query",
                                   json={"query": SEARCH_QUERY, "use_llm": False},
                                   headers=user_h).status_code == 200
            lat: list[float] = []
            for _ in range(n):
                t0 = time.perf_counter()
                rr = client.post("/v1/knowledge/query",
                                 json={"query": SEARCH_QUERY, "use_llm": False},
                                 headers=user_h)
                lat.append((time.perf_counter() - t0) * 1000.0)
                assert rr.status_code == 200, rr.text[:200]
            return lat

        before_on = len(disk_events())
        on = timed_queries(50)
        after_on = len(disk_events())
        os.environ["VELES_AUDIT"] = "0"
        reset_audit()  # синглтон пересоздастся из env на следующем запросе
        off = timed_queries(50)
        after_off = len(disk_events())
        mean_on, mean_off = statistics.mean(on), statistics.mean(off)
        med_on, med_off = statistics.median(on), statistics.median(off)
        p95_on = sorted(on)[int(0.95 * len(on)) - 1]
        p95_off = sorted(off)[int(0.95 * len(off)) - 1]
        numbers(f"(д) 50 запросов ON: среднее {mean_on:.2f} мс, медиана {med_on:.2f} мс, "
                f"p95 {p95_on:.2f} мс")
        numbers(f"(д) 50 запросов OFF: среднее {mean_off:.2f} мс, медиана {med_off:.2f} мс, "
                f"p95 {p95_off:.2f} мс")
        numbers(f"(д) накладной расход: +{mean_on - mean_off:.2f} мс/запрос "
                f"(x{mean_on / mean_off:.2f} к среднему)")
        check("(д) VELES_AUDIT=0 действительно выключил запись",
              after_on == before_on + 55 and after_off == after_on,
              f"{before_on} -> {after_on} -> {after_off}")
        check(
            "(д) деградации в разы нет (ON <= 2x OFF по среднему и медиане)",
            mean_on <= 2.0 * mean_off and med_on <= 2.0 * med_off,
            f"mean {mean_on:.2f} vs {mean_off:.2f}; median {med_on:.2f} vs {med_off:.2f}",
        )
        del os.environ["VELES_AUDIT"]
        reset_audit()  # журнал снова включён - нужно для (г)

        # ------------------------------- (г) ротация по размеру
        api_before = client.get("/v1/admin/audit", params={"limit": 1000},
                                headers=admin_h).json()["total"]
        pre_events = disk_events()
        check("(г) API до ротации видит всё", api_before == len(pre_events),
              f"api={api_before}, disk={len(pre_events)}")
        # набивка валидными событиями до порога 1 МБ (как набежавший журнал)
        audit_path = DATA / "audit" / "audit.jsonl"
        pad = 0
        with audit_path.open("a", encoding="utf-8") as f:
            while audit_path.stat().st_size < 1024 * 1024 - 4096:
                f.write(json.dumps(
                    {"ts": f"2026-10-12T00:00:{pad % 60:02d}+00:00", "actor": "pad",
                     "admin": False, "action": "search", "target": "",
                     "details": f"набивка-ротации-{pad}", "ip": ""},
                    ensure_ascii=False) + "\n")
                pad += 1
        check("(г) журнал набит до порога",
              audit_path.stat().st_size >= 1024 * 1024 - 4096,
              f"{audit_path.stat().st_size} байт, {pad} строк")
        r = client.post("/v1/knowledge/query",
                        json={"query": ROTATE_QUERY, "use_llm": False},
                        headers=user_h)
        check("(г) живой запрос при переполнении: 200", r.status_code == 200)
        rotated = DATA / "audit" / "audit.jsonl.1"
        check("(г) текущий сдвинут в audit.jsonl.1, второй сдвиг не случился",
              rotated.exists() and not (DATA / "audit" / "audit.jsonl.2").exists(),
              str(sorted(p.name for p in (DATA / "audit").iterdir())))
        rotated_events = parse_events(rotated)
        fresh_events = parse_events(audit_path)
        check(
            "(г) старый файл несёт все события до ротации (не потеряны)",
            len(rotated_events) == len(pre_events) + pad
            and rotated_events[0] == pre_events[0]
            and any(e.get("details") == "набивка-ротации-0" for e in rotated_events)
            and any(e.get("details") == f"набивка-ротации-{pad - 1}"
                    for e in rotated_events),
            f"rotated={len(rotated_events)}, ждали {len(pre_events) + pad}",
        )
        check(
            "(г) свежее событие - в новом файле, всего событий не убавилось",
            len(fresh_events) == 1 and fresh_events[0]["details"] == ROTATE_QUERY
            and fresh_events[0]["action"] == "search"
            and len(disk_events()) == len(pre_events) + pad + 1,
            f"fresh={len(fresh_events)}, total={len(disk_events())}",
        )
        api_after = client.get("/v1/admin/audit", params={"limit": 1000},
                               headers=admin_h).json()
        check(
            "(г) API после ротации читает текущий файл (свежие события)",
            api_after["total"] == 1 and api_after["events"][0]["details"] == ROTATE_QUERY,
            f"api total={api_after['total']}",
        )
        numbers(f"(г) ротация: {len(pre_events)} событий + {pad} набивки -> "
                f"audit.jsonl.1, свежее - в новый файл; всего "
                f"{len(disk_events())} (потерь 0)")

        print("\nИТОГИ ПО ШАГАМ")
        print("\n".join(_numbers))
        print("\n" + "\n".join(_results))
        print("VALIDATION-PASS", flush=True)
        return 0
    finally:
        client.close()
        server.should_exit = True
        server_thread.join(timeout=10)
        from twisted.internet import reactor

        reactor.callFromThread(lambda: ldap_listener.stopListening())
        reactor.callFromThread(reactor.stop)
        reactor_thread.join(timeout=10)
        shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    try:
        _code = main()
    except AssertionError as e:
        print(f"VALIDATION-FAIL: {e}", file=sys.stderr, flush=True)
        _code = 1
    raise SystemExit(_code)
