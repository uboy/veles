"""Сквозная валидация админ-панели v2 (docs/plan/admin-ui/v2-spec.md, раздел
«Проверки» -> «Валидация»). Реализация - коммит 29e0e51 (feature/admin-ui-v2).

Живой контур: FastAPI-приложение через TestClient + настоящий конвейер
(очередь -> парсер txt/pdf -> канон -> индекс -> поиск); эмбеддер -
детерминированный hashing (как в tests/test_delete.py), модели не загружаются.
Приложение поднято с ДВУМЯ серверными группами (legal - петров, fin - иванов)
и засеянным документом, не затронутым сценарием.

Сценарий:
  1. Массовая смена группы: 2 документа (личные user:ivanov@) -> админ меняет
     группу обоих на legal -> /v1/knowledge/stats показывает новый состав
     групп и отсутствие старой личной, поиск подтверждает: петров (legal)
     находит оба, иванов - ни одного (чужая группа не находит); инвариант 5:
     смену групп у не-админа/анонима получить нельзя (403/401), битое тело -
     400. Канон (манифест) переписан - по первичному артефакту.
  2. Массовое удаление выбранных -> поиск пуст у всех, в оглавлении только
     засеянный документ.
  3. Файловая задача с битым файлом (мусорные байты под .pdf, настоящий
     docling-разбор) -> failed «не удалось разобрать документ», L0 удалён
     (битый файл не ест квоту), retry честно 409 «повторите загрузку».
  4. Перезапуск -> отмена -> удаление из истории на задачах с сохранённым L0
     (провал «индекс недоступен»): retry создаёт НОВУЮ задачу (retried_from),
     отмена running ставит failed «отменена владельцем» с сохранением L0,
     активную удалить нельзя (409), завершённая удаляется из журнала (404
     после); дубликат-детекция: на двух активных задачах с тем же файлом
     duplicate_of ровно у одной, exists_in_base у обеих False.

Запуск: python evals/admin-v2-validate.py   (системный python, из корня репо)
Успех: печатает VALIDATION-PASS, код возврата 0. Любая ошибка - AssertionError
и VALIDATION-FAIL, код возврата 1.

Временные файлы (VELES_DATA: канон, индекс, uploads, groups.json) - во
временной папке ОС, удаляются в конце; захардкоженных машинных путей нет.
"""

from __future__ import annotations

import hashlib
import json
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
# УСТАНОВЛЕННЫЙ в site-packages veles - проверялся бы не тот код.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(tempfile.mkdtemp(prefix="veles-admin-v2-eval-"))
# VELES_DATA и VELES_GROUPS_FILE ДО импортов veles: config читает env на импорте
os.environ["VELES_DATA"] = str(ROOT / "data")
# изоляция разбора отдельным процессом выключена (как tests/conftest.py):
# spawn-дочерний процесс не наследует подмену окружения
os.environ["VELES_QUEUE_PARSE_TIMEOUT_S"] = "0"
# identity-профиль как ACL-тесты K4: иванов/петров - пользователи (петров в
# legal, иванов в fin), boss - админ, owner - bearer-владелец
os.environ["VELES_IDENTITY_MODE"] = "headers"
os.environ["VELES_IDENTITY_GROUPS"] = "corp"
os.environ["VELES_ADMIN_EMAILS"] = "boss@corp.ru"
os.environ["VELES_API_TOKEN"] = "owner-token"

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from veles.api.app import VelesPipeline, create_app  # noqa: E402
from veles.canonical.store import _doc_dir, save_document  # noqa: E402
from veles.config import EMBED_DIM, EMBED_MODEL  # noqa: E402
from veles.ingest.queue import IngestQueue, uploads_root, user_dir  # noqa: E402
from veles.ingest.txt_md import TxtMdParserAdapter  # noqa: E402
from veles.indexing.indexer import build_index  # noqa: E402
from veles.indexing.vector_store import NumpyFlatStore  # noqa: E402
from veles.policy.identity import StaticOwnerResolver  # noqa: E402

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""
SEED_ID = "05_vyshka_academic/lectures/коучинг.md"

DOC_V1 = """\
# Регламент закупок

Порог единственной закупки составляет 350 тысяч рублей.

Закупки свыше порога требуют обязательного тендера.
"""
DOC_TITLE = "Регламент закупок"
DOC_QUERY = "какой порог единственной закупки"
DOC_MARK = "350 тысяч"

DOC2_V1 = """\
# Командировки

После поездки сотрудник сдаёт авансовый отчёт в бухгалтерию в течение трёх дней.
"""
DOC2_TITLE = "Командировки"
DOC2_QUERY = "как оформить авансовый отчёт"
DOC2_MARK = "авансовый отчёт"

IVANOV = {"X-OpenWebUI-User-Email": "ivanov@corp.ru"}
PETROV = {"X-OpenWebUI-User-Email": "petrov@corp.ru"}
BOSS = {"X-OpenWebUI-User-Email": "boss@corp.ru"}
OWNER = {"Authorization": "Bearer owner-token"}

_results: list[str] = []


def check(title: str, cond: bool, detail: str = "") -> None:
    """Одна проверка = один assert с человекочитаемым заголовком."""
    if not cond:
        raise AssertionError(f"{title}" + (f": {detail}" if detail else ""))
    _results.append("[ок] " + title + (f" ({detail})" if detail else ""))


class HashingEmbedder:
    """Детерминированный эмбеддер тестов (sublinear tf, md5-бакеты)."""

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
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


def build_env() -> dict:
    """Tmp-окружение: канон с засеянным документом, мини-индекс, конфиг ДВУХ
    серверных групп (legal - петров, fin - иванов). VELES_DATA уже в env."""
    data = ROOT / "data"
    canonical = data / "canonical"
    index_dir = data / "index"
    corpus = ROOT / "corpus"
    seed = corpus / "05_vyshka_academic/lectures/коучинг.md"
    seed.parent.mkdir(parents=True)
    seed.write_text(SEED_DOC, encoding="utf-8", newline="\n")
    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    groups_conf = data / "groups.json"
    groups_conf.write_text(json.dumps({"groups": [
        {"id": "all", "title": "Видно всем", "members": []},
        {"id": "legal", "title": "Юристы", "members": ["petrov@corp.ru"]},
        {"id": "fin", "title": "Финансы", "members": ["ivanov@corp.ru"]},
    ]}, ensure_ascii=False), encoding="utf-8")
    os.environ["VELES_GROUPS_FILE"] = str(groups_conf)
    return {"canonical": canonical, "index_dir": index_dir, "bundle": bundle}


def query_texts(client: TestClient, q: str, headers: dict) -> list[str]:
    d = client.post(
        "/v1/knowledge/query", json={"query": q, "use_llm": False}, headers=headers,
    ).json()
    return [ev["text"] for ev in d["evidence"]]


def wait_status(client: TestClient, job_id: str, want: str,
                timeout_s: float = 30.0, headers: dict | None = None) -> dict:
    """Дождаться статуса джоба (воркер асинхронен; битый pdf - до ~20 с)."""
    rec: dict = {}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        r = client.get(f"/v1/documents/jobs/{job_id}", headers=headers or {})
        assert r.status_code == 200, r.text
        rec = r.json()
        if rec.get("status") == want:
            return rec
        time.sleep(0.05)
    raise AssertionError(f"задача {job_id} не достигла {want!r}: {rec}")


# --------------------------------------------------- акт 1-2: группы и удаление

def act_regroup_and_delete(client: TestClient, env: dict) -> list[str]:
    """Загрузка 2 документов -> массовая смена группы на legal -> stats и поиск
    подтверждают новую группу и потерю старой -> массовое удаление -> поиск
    пуст. Возвращает document_id удалённых (для журнала задач не нужно)."""
    docs: list[tuple[str, str, str]] = []  # (document_id, query, маркер)
    for text, title, q, mark in (
        (DOC_V1, DOC_TITLE, DOC_QUERY, DOC_MARK),
        (DOC2_V1, DOC2_TITLE, DOC2_QUERY, DOC2_MARK),
    ):
        r = client.post("/v1/documents", json={"text": text, "title": title},
                        headers=IVANOV)
        check(f"загрузка «{title}» принята в очередь", r.status_code == 202, r.text)
        body = r.json()
        rec = wait_status(client, body["job_id"], "done", headers=IVANOV)
        check(f"задача «{title}» выполнена", rec["status"] == "done",
              f"error={rec.get('error')!r}")
        docs.append((body["document_id"], q, mark))

    # бейдж «уже в базе» у завершённой задачи
    jobs = {j["job_id"]: j for j in client.get("/v1/documents/jobs", headers=IVANOV).json()["jobs"]}
    done_jobs = [j for j in jobs.values() if j["document_id"] == docs[0][0]]
    check("exists_in_base у выполненной задачи", any(j["exists_in_base"] for j in done_jobs))

    # инвариант 5: группы от клиента принимает только админ
    url0 = f"/v1/documents/{docs[0][0]}/groups"
    body = {"groups": ["legal"]}
    check("смена группы не-админом (петров) - 403",
          client.post(url0, json=body, headers=PETROV).status_code == 403)
    check("смена группы не-админом (иванов) - 403",
          client.post(url0, json=body, headers=IVANOV).status_code == 403)
    check("смена группы анонимом - 401", client.post(url0, json=body).status_code == 401)
    check("пустой список групп - 400",
          client.post(url0, json={"groups": []}, headers=BOSS).status_code == 400)
    check("неизвестная группа - 400",
          client.post(url0, json={"groups": ["нет-такой"]}, headers=BOSS).status_code == 400)
    check("чужой документ - 404",
          client.post("/v1/documents/ivanov@corp.ru/нет-такого.md/groups",
                      json=body, headers=BOSS).status_code == 404)

    # массовая смена группы: та же операция на каждый выбранный документ
    for doc_id, _, _ in docs:
        r = client.post(f"/v1/documents/{doc_id}/groups",
                        json={"groups": ["legal"]}, headers=BOSS)
        check(f"админ сменил группу {doc_id} на legal", r.status_code == 200, r.text)
        answer = r.json()
        check(f"ответ regroup {doc_id}",
              answer["status"] == "regrouped" and answer["groups"] == ["legal"]
              and answer["nodes"] >= 1 and answer["overlay_saved"] is True,
              json.dumps(answer, ensure_ascii=False))
        # канон переписан (первичный артефакт): манифест несёт новый состав
        manifest = json.loads(
            (_doc_dir(doc_id, env["canonical"]) / "manifest.json").read_text(encoding="utf-8"))
        check(f"манифест канона {doc_id} переписан",
              manifest["access_groups"] == ["legal"], str(manifest["access_groups"]))

    # /v1/knowledge/stats: новая группа у обоих, старой личной нет
    stats = client.get("/v1/knowledge/stats", headers=BOSS).json()
    by_id = {d["document_id"]: d for d in stats["documents"]}
    for doc_id, _, _ in docs:
        groups = by_id[doc_id]["groups"]
        check(f"stats: {doc_id} в группе legal", groups == ["legal"], str(groups))
        check(f"stats: {doc_id} потерял личную группу user:ivanov@",
              "user:ivanov@corp.ru" not in groups, str(groups))

    # поиск подтверждает: legal находит, прежняя (личная) - нет
    for doc_id, q, mark in docs:
        check(f"петров (legal) находит «{mark}»",
              any(mark in t for t in query_texts(client, q, PETROV)))
        check(f"иванов (старая группа) НЕ находит «{mark}»",
              not any(mark in t for t in query_texts(client, q, IVANOV)))

    # массовое удаление выбранных
    for doc_id, _, _ in docs:
        r = client.delete(f"/v1/documents/{doc_id}", headers=BOSS)
        check(f"удаление {doc_id}", r.status_code == 200, r.text)

    # поиск пуст у всех, в оглавлении - только незатронутый засеянный документ
    for _, q, mark in docs:
        for name, headers in (("петров", PETROV), ("иванов", IVANOV), ("админ", BOSS)):
            check(f"после удаления «{mark}» не находит {name}",
                  not any(mark in t for t in query_texts(client, q, headers)))
    stats = client.get("/v1/knowledge/stats", headers=BOSS).json()
    check("в оглавлении только засеянный документ",
          {d["document_id"] for d in stats["documents"]} == {SEED_ID},
          str(sorted(d["document_id"] for d in stats["documents"])))


# ------------------------------------------------ акт 3: файловая задача, фол

def act_broken_file(client: TestClient) -> None:
    """Битый файл (мусор под .pdf) -> failed, L0 удалён, retry - честный 409."""
    garbage = b"%PDF-1.7 garbage bytes - not a real pdf \x00\x01\x02"
    r = client.post("/v1/documents",
                    files={"file": ("битый.pdf", garbage)}, headers=IVANOV)
    check("битый файл принят в очередь", r.status_code == 202, r.text)
    job_id = r.json()["job_id"]
    rec = wait_status(client, job_id, "failed", timeout_s=90.0, headers=IVANOV)
    check("битый файл - failed «не удалось разобрать»",
          "не удалось разобрать" in (rec.get("error") or ""), rec.get("error", ""))
    l0 = uploads_root() / user_dir("ivanov@corp.ru") / "битый.pdf"
    check("L0 битого файла удалён (не ест квоту)", not l0.is_file(), str(l0))
    r = client.post(f"/v1/documents/jobs/{job_id}/retry", headers=IVANOV)
    check("retry задачи с удалённым L0 - 409 «повторите загрузку»",
          r.status_code == 409 and "не сохранился" in r.json().get("detail", ""), r.text)


# --------------------------- акт 4: перезапуск -> отмена -> удаление из журнала

def make_blocked_app() -> tuple[TestClient, threading.Event, threading.Event]:
    """Фасад с очередью, чей пайплайн управляется событиями: 1-й вызов висит на
    gate1, 3-й - на gate2, остальные падают сразу («индекс недоступен» - L0
    сохраняется, retry возможен). Bearer не нужен: static-владелец."""
    os.environ.pop("VELES_API_TOKEN", None)  # create_app читает env на вызове
    gate1, gate2 = threading.Event(), threading.Event()
    calls = {"n": 0}

    def pipeline():
        calls["n"] += 1
        if calls["n"] == 1:
            gate1.wait(timeout=30)
        elif calls["n"] == 3:
            gate2.wait(timeout=30)
        raise RuntimeError("пайплайн закрыт")

    queue = IngestQueue(pipeline_fn=pipeline)
    app = create_app(pipeline=None, resolver=StaticOwnerResolver({"personal"}))
    app.state.veles_ingest = queue  # API подхватит её вместо ленивого создания
    app.state.veles["pipeline_fn"] = (
        lambda: (_ for _ in ()).throw(RuntimeError("индекса нет"))
    )
    return TestClient(app), gate1, gate2


def act_retry_cancel_delete(client: TestClient, gate1: threading.Event,
                            gate2: threading.Event) -> None:
    """Файловая задача -> failed -> retry (новая задача) -> cancel (queued и
    running) -> delete из журнала; дубликат-детекция на двух задачах с тем же
    файлом; активную удалить нельзя."""
    def upload(filename: str) -> str:
        r = client.post("/v1/documents",
                        files={"file": (filename, "текст отчёта для разбора".encode("utf-8"))})
        assert r.status_code == 202, r.text
        return r.json()["job_id"]

    job1 = upload("отчёт.txt")
    wait_status(client, job1, "running")  # висит на gate1

    # дубликат: та же задача (тот же файл) ставится второй, пока первая активна
    job1b = upload("отчёт.txt")
    jobs = {j["job_id"]: j for j in client.get("/v1/documents/jobs").json()["jobs"]}
    dups = {job1: jobs[job1]["duplicate_of"], job1b: jobs[job1b]["duplicate_of"]}
    check("дубликат-детекция: ровно одна из пары помечена",
          (dups[job1] == job1b) != (dups[job1b] == job1),
          f"{job1}->{dups[job1] or '-'}, {job1b}->{dups[job1b] or '-'}")
    marked = [jid for jid, d in dups.items() if d]
    check("дубликат-бейдж ссылается на другую задачу пары, не на себя",
          len(marked) == 1 and dups[marked[0]] != marked[0]
          and dups[marked[0]] in (job1, job1b), str(marked))
    check("exists_in_base у обеих активных задач - False",
          all(jobs[j]["exists_in_base"] is False for j in (job1, job1b)))

    # активную (queued) удалить нельзя - сначала отмена
    check("удаление активной задачи - 409",
          client.delete(f"/v1/documents/jobs/{job1b}").status_code == 409)
    rec = client.post(f"/v1/documents/jobs/{job1b}/cancel").json()
    check("отмена queued - failed «отменена владельцем»",
          rec["status"] == "failed" and "отменена" in rec["error"], str(rec["error"]))

    # отмена running: воркер выйдет из заблокированного пайплайна и честно
    # пометит failed; L0 сохранён (разбор не падал)
    check("отмена running принята",
          client.post(f"/v1/documents/jobs/{job1}/cancel").status_code == 200)
    gate1.set()
    rec = wait_status(client, job1, "failed")
    check("running после отмены - failed «отменена владельцем»",
          "отменена" in rec["error"], rec["error"])
    check("L0 отменённой задачи сохранён (перезапуск возможен)",
          (uploads_root() / user_dir("") / "отчёт.txt").is_file())

    # перезапуск упавшей: НОВАЯ задача, старая остаётся в истории
    r = client.post(f"/v1/documents/jobs/{job1}/retry")
    check("retry упавшей - 202", r.status_code == 202, r.text)
    job2 = r.json()["job_id"]
    check("retry создаёт новую задачу с retried_from",
          job2 != job1 and r.json()["retried_from"] == job1)
    wait_status(client, job2, "failed")  # пайплайн по-прежнему недоступен
    r = client.post(f"/v1/documents/jobs/{job2}/retry")
    job3 = r.json()["job_id"]
    check("retry второй степени - 202", r.status_code == 202, r.text)
    wait_status(client, job3, "running")  # третий вызов пайплайна висит на gate2
    check("retry активной (running) - 409 «только упавшей»",
          client.post(f"/v1/documents/jobs/{job3}/retry").status_code == 409)

    # cancel по завершённой - запись без изменений (отменять нечего)
    before = client.get(f"/v1/documents/jobs/{job1}").json()
    check("cancel завершённой задачу не меняет",
          client.post(f"/v1/documents/jobs/{job1}/cancel").json() == before)

    # активную running удалить нельзя; queued третья задача отменяется сразу
    check("удаление running - 409 «сначала отмените»",
          client.delete(f"/v1/documents/jobs/{job3}").status_code == 409)
    job4 = upload("приказ.txt")  # воркер занят job3 - задача остаётся queued
    rec = client.post(f"/v1/documents/jobs/{job4}/cancel").json()
    check("отмена второй queued - failed «отменена владельцем»",
          rec["status"] == "failed" and "отменена" in rec["error"], str(rec["error"]))

    # удаление из истории: завершённая исчезает из журнала и из чтения
    gate2.set()
    wait_status(client, job3, "failed")
    check("delete завершённой из журнала",
          client.delete(f"/v1/documents/jobs/{job3}").json()["status"] == "deleted")
    check("удалённая задача читается как 404",
          client.get(f"/v1/documents/jobs/{job3}").status_code == 404)
    check("повторное delete - 404",
          client.delete(f"/v1/documents/jobs/{job3}").status_code == 404)
    check("удалённой задачи нет в списке",
          job3 not in {j["job_id"] for j in client.get("/v1/documents/jobs").json()["jobs"]})

    # неизвестная задача - 404 на всех трёх операциях
    for method, path in (("post", "/v1/documents/jobs/нет/cancel"),
                         ("post", "/v1/documents/jobs/нет/retry"),
                         ("delete", "/v1/documents/jobs/нет")):
        check(f"неизвестная задача: {method.upper()} {path} - 404",
              getattr(client, method)(path).status_code == 404, path)


def main() -> int:
    env = build_env()
    app = create_app(pipeline=VelesPipeline(env["bundle"], HashingEmbedder(), llm=None))
    client = TestClient(app)

    groups = client.get("/v1/groups", headers=BOSS).json()
    ids = {g["id"] for g in groups["groups"]}
    check("приложение поднято с двумя серверными группами",
          {"legal", "fin"} <= ids, str(sorted(ids)))

    act_regroup_and_delete(client, env)
    print("[акт 1-2] смена группы + массовое удаление - ок", flush=True)
    act_broken_file(client)
    print("[акт 3] файловая задача с битым файлом - ок", flush=True)
    blocked, gate1, gate2 = make_blocked_app()
    act_retry_cancel_delete(blocked, gate1, gate2)
    print("[акт 4] перезапуск -> отмена -> удаление из журнала, дубликаты - ок",
          flush=True)

    print("\n" + "\n".join(_results))
    print(f"\nитог: {len(_results)} проверок пройдено", flush=True)
    print("VALIDATION-PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        _code = main()
    except AssertionError as e:
        print(f"VALIDATION-FAIL: {e}", file=sys.stderr, flush=True)
        _code = 1
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
    raise SystemExit(_code)
