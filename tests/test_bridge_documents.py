"""Регрессия документо-потока моста (K9): owner-гейт, гейты формата/размера,
submit/get_job/list_groups фасада, /docgroup, экранирование, слоты ожидания."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "0:stub")
os.environ.setdefault("OPENWEBUI_API_KEY", "sk-stub")
# ВАЖНО: VELES_API_TOKEN здесь НЕ ставим - это глобальная env процесса pytest,
# фасадные тесты читают её для identity (401 в чужих тестах без всякой причины).

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

import bridge  # noqa: E402
import documents  # noqa: E402

CHAT = 100
GROUPS = [{"id": "eng", "title": "Инженерия"}, {"id": "all", "title": "Видно всем"}]


@pytest.fixture(autouse=True)
def _bridge_runtime(monkeypatch):
    # tg_client/veles_client в проде привязываются в __main__; в тестах - заглушки
    monkeypatch.setattr(bridge, "tg_client", object(), raising=False)
    monkeypatch.setattr(bridge, "veles_client", object(), raising=False)
    bridge._doc_slots = 0
    yield
    bridge._doc_slots = 0


class FakeResponse:
    def __init__(self, data=None, status=200, text=""):
        self._data = data if data is not None else {}
        self.status_code = status
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeVeles:
    """Фейковый клиент фасада: capture POST/GET, настраиваемые ответы."""

    def __init__(self, job=None, post_data=None, post_status=202):
        self.job = job or {"status": "done", "title": "Отчёт", "groups": ["eng"],
                           "error": ""}
        self.post_data = post_data or {"job_id": "j1"}
        self.post_status = post_status
        self.calls: list[tuple] = []

    def post(self, url, headers=None, files=None, data=None):
        self.calls.append(("post", url, dict(data or {})))
        return FakeResponse(self.post_data, status=self.post_status,
                            text=json.dumps(self.post_data, ensure_ascii=False))

    def get(self, url, headers=None):
        self.calls.append(("get", url))
        if url.endswith("/v1/groups"):
            return FakeResponse({"groups": GROUPS})
        return FakeResponse(self.job, text=json.dumps(self.job, ensure_ascii=False))


class FakeThread:
    def __init__(self, registry, target=None, args=(), daemon=None):
        self._registry = registry
        self.target = target
        self.args = args
        self.daemon = daemon

    def start(self):
        self._registry.append((self.target, self.args))


def capture_threads(monkeypatch):
    """threading.Thread в мосту подменяется на запись (target, args) без старта:
    watcher потом можно прогнать синхронно под теми же подменами."""
    spawned: list[tuple] = []
    monkeypatch.setattr(bridge.threading, "Thread",
                        lambda target=None, args=(), daemon=None:
                        FakeThread(spawned, target, args, daemon))
    return spawned


def patch_send(monkeypatch) -> list[str]:
    """Статусные сообщения документо-потока собираем в список: watcher шлёт
    ПОСЛЕ возврата handle(), когда подмена bridge.tg уже снята."""
    sent: list[str] = []
    monkeypatch.setattr(bridge, "send_html", lambda chat_id, text: sent.append(text))
    return sent


def run_handle(update: dict, owner: int | None = CHAT):
    """handle() без реального Telegram: любые sendMessage = провал теста."""
    def _no_tg(method, **params):
        raise AssertionError(f"внезапный вызов Telegram API: {method}")

    orig = bridge.tg
    bridge.tg = _no_tg
    try:
        return bridge.handle(update, owner)
    finally:
        bridge.tg = orig


def doc_update(chat_id: int = CHAT, name="otchet.md", size=100, file_id="f1") -> dict:
    return {"message": {"chat": {"id": chat_id},
                        "document": {"file_name": name, "file_size": size,
                                     "file_id": file_id}}}


def text_command(text: str, chat_id: int = CHAT) -> dict:
    return {"message": {"chat": {"id": chat_id}, "text": text}}


# --- юнит documents.py: клиент фасада ---


def test_submit_document_happy_path(tmp_path) -> None:
    vx = FakeVeles()
    f = tmp_path / "doc.md"
    f.write_bytes(b"# t")
    job_id = documents.submit_document(vx, f, title="t", group="eng")
    assert job_id == "j1"
    method, url, data = vx.calls[0]
    assert method == "post" and url.endswith("/v1/documents")
    assert data["group"] == "eng" and data["title"] == "t"
    assert "job_id" in vx.post_data


def test_facade_error_carries_detail(tmp_path) -> None:
    vx = FakeVeles(post_data={"detail": "очередь занята, повторите позже"},
                   post_status=429)
    f = tmp_path / "doc.md"
    f.write_bytes(b"x")
    with pytest.raises(documents.FacadeError) as e:
        documents.submit_document(vx, f)
    assert "очередь занята" in str(e.value)

    vx2 = FakeVeles()
    vx2.get = lambda url, **k: FakeResponse({"detail": "задание не найдено"},
                                            status=404, text="bad")
    with pytest.raises(documents.FacadeError) as e2:
        documents.get_job(vx2, "nope")
    assert "задание не найдено" in str(e2.value)


def test_get_job_and_list_groups() -> None:
    vx = FakeVeles(job={"status": "running", "title": "", "groups": []})
    assert documents.get_job(vx, "j1")["status"] == "running"
    assert [g["id"] for g in documents.list_groups(vx)] == ["eng", "all"]


def test_auth_header_from_env(monkeypatch) -> None:
    monkeypatch.setenv("VELES_API_TOKEN", "t-secret")
    assert documents._auth_headers() == {"Authorization": "Bearer t-secret"}
    monkeypatch.setenv("VELES_API_TOKEN", "")
    assert documents._auth_headers() == {}


# --- мост: гейты и поток документа ---


def test_document_from_non_owner_ignored(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    patch_send(monkeypatch)

    def _no_download(*a, **k):
        raise AssertionError("чужой файл не должен скачиваться")

    monkeypatch.setattr(bridge.voice, "download_tg_voice", _no_download)
    owner = run_handle(doc_update(chat_id=200))
    assert owner == CHAT


def test_document_wrong_extension_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    run_handle(doc_update(name="notes.docx"))
    assert "не поддерживаю" in sent[0] and ".docx" in sent[0]
    run_handle(doc_update(name="README"))
    assert "без расширения" in sent[1]


def test_document_20mb_gate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    run_handle(doc_update(name="big.pdf", size=21 * 1024 * 1024))
    assert "20 МБ" in sent[0] and "big.pdf" in sent[0]
    assert bridge._doc_slots == 0  # отказ до резервации слота


def test_document_download_fail_message(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("getFile: file is too big")

    monkeypatch.setattr(bridge.voice, "download_tg_voice", _boom)
    run_handle(doc_update())
    assert any("Принял" in t for t in sent)
    assert any("Не смог скачать файл из Telegram: RuntimeError" in t for t in sent)
    assert bridge._doc_slots == 0  # слот освобождён при отказе


def test_document_submit_fail_shows_facade_detail(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda tc, fid, d: d / "doc.md")

    def _fail(*a, **k):
        raise documents.FacadeError("очередь занята, повторите позже")

    monkeypatch.setattr(bridge.documents, "submit_document", _fail)
    run_handle(doc_update())
    assert any("Фасад отклонил загрузку" in t and "очередь занята, повторите позже" in t
               for t in sent)
    assert bridge._doc_slots == 0


def test_document_success_two_messages_and_slot(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda tc, fid, d: d / "otchet.md")
    seen: dict = {}

    def fake_submit(vc, path, title="", group=""):
        seen.update(title=title, group=group)
        return "j1"

    monkeypatch.setattr(bridge.documents, "submit_document", fake_submit)
    monkeypatch.setattr(bridge.documents, "get_job",
                        lambda vc, jid: {"status": "done", "title": "Отчёт",
                                         "groups": ["eng"]})
    spawned = capture_threads(monkeypatch)

    run_handle(doc_update())
    assert any("Принял" in t and "otchet.md" in t for t in sent)
    assert seen["title"] == "otchet"  # title из имени сообщения
    assert seen["group"] == ""  # группа не выбрана - дефолт фасада (личная)
    assert len(spawned) == 1
    assert bridge._doc_slots == 1  # ожидание держит слот

    target, args = spawned[0]
    target(*args)  # doc_watch_loop синхронно
    assert any("✅ Загружено" in t and "Отчёт" in t and "eng" in t for t in sent)
    assert bridge._doc_slots == 0  # завершение освободило слот


def test_document_name_html_escaped(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda tc, fid, d: d / "weird.md")
    monkeypatch.setattr(bridge.documents, "submit_document", lambda vc, p, **k: "j1")
    monkeypatch.setattr(bridge.documents, "get_job",
                        lambda vc, jid: {"status": "done", "title": "a<b>&",
                                         "groups": ["user:u1"]})
    spawned = capture_threads(monkeypatch)
    run_handle(doc_update(name="a<b>&.md"))
    assert any("a&lt;b&gt;&amp;.md" in t for t in sent)
    target, args = spawned[0]
    target(*args)
    final = [t for t in sent if "Загружено" in t][0]
    assert "a&lt;b&gt;&amp;" in final and "личная группа" in final


def test_document_watch_failed_lost_and_timeout(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    monkeypatch.setattr(bridge.documents, "submit_document", lambda vc, p, **k: "j1")
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda tc, fid, d: d / "f.md")

    # failed с причиной
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.documents, "get_job",
                        lambda vc, jid: {"status": "failed", "title": "", "groups": [],
                                         "error": "не удалось разобрать документ"})
    spawned = capture_threads(monkeypatch)
    run_handle(doc_update(name="f.md"))
    spawned[0][0](*spawned[0][1])
    assert any("Не удалось" in t and "не удалось разобрать" in t for t in sent)

    # потеря джоба фасадом (рестарт, 404) - честный отказ
    sent = patch_send(monkeypatch)

    def _gone(vc, jid):
        raise documents.FacadeError("задание не найдено")

    monkeypatch.setattr(bridge.documents, "get_job", _gone)
    spawned = capture_threads(monkeypatch)
    run_handle(doc_update(name="f.md"))
    spawned[0][0](*spawned[0][1])
    assert any("задание не найдено" in t for t in sent)

    # вечный running - таймаут моста с честным сообщением
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.documents, "get_job",
                        lambda vc, jid: {"status": "running", "title": "", "groups": []})
    monkeypatch.setattr(bridge, "DOC_WATCH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(bridge, "DOC_POLL_INTERVAL_S", 0.0)
    spawned = capture_threads(monkeypatch)
    run_handle(doc_update(name="f.md"))
    spawned[0][0](*spawned[0][1])
    assert any("всё ещё обрабатывается" in t for t in sent)
    assert bridge._doc_slots == 0  # таймаут тоже освобождает слот


def test_slot_limit_five_concurrent() -> None:
    # юнит на лимит: 5 резерваций успешны, 6-я - отказ; release возвращает слот
    assert bridge.DOC_MAX_CONCURRENT == 5
    got = [bridge.reserve_doc_slot() for _ in range(6)]
    assert got == [True, True, True, True, True, False]
    bridge.release_doc_slot()
    assert bridge.reserve_doc_slot() is True
    bridge.release_doc_slot()


def test_document_over_limit_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    for _ in range(bridge.DOC_MAX_CONCURRENT):
        assert bridge.reserve_doc_slot()
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    run_handle(doc_update())
    assert "одновременных загрузок" in sent[0]
    assert bridge._doc_slots == bridge.DOC_MAX_CONCURRENT  # слот не съеден отказом


# --- /docgroup: show/set/persist/невалид ---


def test_docgroup_show_and_set_persist(monkeypatch, tmp_path) -> None:
    state_file = tmp_path / "dg.json"
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", state_file)
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.documents, "list_groups", lambda vc: list(GROUPS))

    run_handle(text_command("/docgroup"))
    assert "не выбрана" in sent[0] and "eng" in sent[0] and "all" in sent[0]

    run_handle(text_command("/docgroup eng"))
    assert "eng" in sent[1]
    # persist: файл state записан и читается мостом
    assert json.loads(state_file.read_text(encoding="utf-8")) == {
        str(CHAT): {"doc_group": "eng"}
    }
    assert bridge.load_doc_group(CHAT) == "eng"

    # сохранённая группа уходит в POST следующей загрузки
    monkeypatch.setattr(bridge.voice, "download_tg_voice",
                        lambda tc, fid, d: d / "n.md")
    seen: dict = {}

    def fake_submit(vc, path, title="", group=""):
        seen["group"] = group
        return "j1"

    monkeypatch.setattr(bridge.documents, "submit_document", fake_submit)
    monkeypatch.setattr(bridge.documents, "get_job",
                        lambda vc, jid: {"status": "done", "title": "n",
                                         "groups": ["eng"]})
    spawned = capture_threads(monkeypatch)
    run_handle(doc_update(name="n.md"))
    assert seen["group"] == "eng"
    spawned[0][0](*spawned[0][1])
    assert any("eng" in t for t in sent)


def test_docgroup_invalid_and_facade_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bridge, "DOC_STATE_FILE", tmp_path / "dg.json")
    sent = patch_send(monkeypatch)
    monkeypatch.setattr(bridge.documents, "list_groups", lambda vc: list(GROUPS))

    run_handle(text_command("/docgroup нет-такая"))
    assert "нет в списке" in sent[0] and "нет-такая" in sent[0]
    assert bridge.load_doc_group(CHAT) == ""  # невалид не сохранён

    def _fail(vc):
        raise documents.FacadeError("фасад недоступен")

    monkeypatch.setattr(bridge.documents, "list_groups", _fail)
    run_handle(text_command("/docgroup eng"))
    assert "фасад недоступен" in sent[1]
    run_handle(text_command("/docgroup"))
    assert "фасад недоступен" in sent[2]


def test_docgroup_registered_as_known_command() -> None:
    assert "/docgroup" in bridge.KNOWN_COMMANDS
    assert any(c["command"] == "docgroup" for c in bridge.TG_COMMANDS)
    assert "/docgroup" in bridge.HELP_TEXT


def test_send_html_plain_fallback() -> None:
    """ok=false от Telegram -> тот же текст повторно plain (без parse_mode)."""
    calls: list[tuple[str, dict]] = []

    def fake_tg(method, **params):
        calls.append((method, dict(params)))
        return {"ok": False, "description": "can't parse entities"}

    orig = bridge.tg
    bridge.tg = fake_tg
    try:
        bridge.send_html(CHAT, "❌ <b>имя</b> &lt;тег&gt;")
    finally:
        bridge.tg = orig
    assert len(calls) == 2
    assert calls[0][1]["parse_mode"] == "HTML"
    assert "parse_mode" not in calls[1][1]
    assert calls[1][1]["text"].startswith("❌ имя <тег>")
