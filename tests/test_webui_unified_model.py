"""K19: unit-тесты единой pipe-модели «База знаний (Veles)» (Open WebUI).

Модель живёт в integrations/openwebui/functions/veles_knowledge.py и
исполняется внутри Open WebUI; здесь она загружается из файла как обычный
модуль. HTTP замокан на уровне инъекционного _OPENER (единственная точка
выхода в сеть - _http_bytes), фасадные ответы - те же контракты, что в
tests/test_documents_ingest (202 + джоб, 401) и tests/test_api
(knowledge/query: decision/answer/citations).
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

FUNCTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations" / "openwebui" / "functions" / "veles_knowledge.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("veles_knowledge_test", FUNCTION_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status: int, payload: dict | bytes):
        self.status = status
        self._payload = payload

    def read(self) -> bytes:
        if isinstance(self._payload, bytes):
            return self._payload
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class HttpMock:
    """Замок открывателя: отвечает по паре (метод, подстрока URL);
    элементы - (метод, url_contains, статус, payload). Собирает все вызовы."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.calls: list[dict] = []

    def __call__(self, req, timeout=None):
        call = {
            "method": req.get_method(),
            "url": req.full_url,
            "headers": {k.lower(): v for k, v in req.header_items()},
            "data": req.data,
            "timeout": timeout,
        }
        self.calls.append(call)
        for i, (method, url_contains, status, payload) in enumerate(self.responses):
            if method != call["method"] or url_contains not in call["url"]:
                continue
            return FakeResponse(status, self.responses.pop(i)[3])
        return FakeResponse(200, {})


class HttpRaise:
    """Замок, кидающий исключение (сеть/таймаут; HTTPError - путь редиректа)."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls: list[dict] = []

    def __call__(self, req, timeout=None):
        self.calls.append({"method": req.get_method(), "url": req.full_url})
        raise self.exc


class _FakeOpener:
    def __init__(self, mock):
        self.mock = mock

    def open(self, req, timeout=None):
        return self.mock(req, timeout=timeout)


@pytest.fixture()
def pipe(monkeypatch, mod):
    def _make(http_mock):
        monkeypatch.setattr(mod, "_OPENER", _FakeOpener(http_mock))
        p = mod.Pipe()
        p.valves.veles_url = "http://veles:8400"
        p.valves.veles_token = "owner-token"
        p.valves.wait_seconds = 0
        p.valves.poll_interval_seconds = 0.01
        return p

    return _make


@pytest.fixture()
def direct(monkeypatch, mod):
    """Инъекция открывателя для прямых вызовов _http_bytes (без сети)."""
    def _make(http_mock):
        monkeypatch.setattr(mod, "_OPENER", _FakeOpener(http_mock))
        return mod

    return _make


USER = {"email": "ivanov@corp.ru", "name": "Иванов", "role": "user"}
BODY_QUESTION = {
    "model": "veles_knowledge",
    "messages": [{"role": "user", "content": "Сколько дней ежегодный отпуск?"}],
}
BODY_UPLOAD_ONLY = {"model": "veles_knowledge", "messages": [{"role": "user", "content": ""}]}
FILE_ONE = [{
    "type": "file", "id": "f1", "name": "отпуск.txt",
    "file": {"filename": "отпуск.txt", "data": {"content": "Отпуск 28 дней."}},
}]

QUERY_RESULT = {
    "request_id": "req-1",
    "decision": "answer",
    "answer": "Ежегодный отпуск 28 дней [1].",
    "citations": [{
        "n": 1, "title": "Регламент отпусков", "source_id": "регламент_отпусков.txt",
        "lines": [12, 15], "document_id": "ivanov@corp.ru/регламент_отпусков.txt",
        "node_version_id": "nv-1", "revision_id": "r1",
    }],
    "diagnostics": {"confidence": 0.9, "reason": "", "llm_used": True},
}


def test_no_valves_no_http(mod, pipe) -> None:
    http = HttpMock()
    p = mod.Pipe()  # valves не заполнены
    out = p.pipe(BODY_QUESTION, USER, __files__=FILE_ONE)
    assert "Valves" in out and "veles_url" in out
    assert http.calls == []


def test_no_email_aborts_with_files(pipe) -> None:
    http = HttpMock()
    out = pipe(http).pipe(BODY_UPLOAD_ONLY, {"email": ""}, __files__=FILE_ONE)
    assert "email" in out
    assert http.calls == []


def test_no_email_aborts_with_question(pipe) -> None:
    """Нет email в сессии - отказ и на вопросе тоже (анонимной модели нет)."""
    http = HttpMock()
    out = pipe(http).pipe(BODY_QUESTION, {"email": ""})
    assert "email" in out
    assert http.calls == []


def test_empty_message_instruction(pipe) -> None:
    http = HttpMock()
    out = pipe(http).pipe({"messages": [{"role": "user", "content": "   "}]}, USER)
    assert "Задайте вопрос" in out
    assert http.calls == []


def test_task_guard_no_http(pipe) -> None:
    """Фоновые задачи WebUI (query-gen/теги) вызывают ту же модель с теми же
    файлами: такие вызовы не должны ни загружать, ни спрашивать (прецедент
    K16 - документ получал названием служебный промпт задачи)."""
    http = HttpMock()
    out = pipe(http).pipe(BODY_QUESTION, USER, __files__=FILE_ONE, __task__="search_query")
    assert out == ""
    assert http.calls == []


def test_upload_only_success(pipe) -> None:
    """Только файл, фасад 200 по джобу: личная загрузка, подтверждение."""
    http = HttpMock([
        ("POST", "/v1/documents", 202, {"job_id": "j1", "status": "queued"}),
        ("GET", "/v1/documents/jobs/j1", 200,
         {"status": "done", "action": "created", "title": "отпуск"}),
    ])
    out = pipe(http).pipe(BODY_UPLOAD_ONLY, USER, __files__=FILE_ONE)
    assert "✅" in out and "отпуск" in out
    assert "личная" in out and "ivanov@corp.ru" in out
    # текста-вопроса нет - запрос к базе не уходит
    assert not any("/v1/knowledge/query" in c["url"] for c in http.calls)

    post = [c for c in http.calls if c["method"] == "POST"][0]
    assert post["url"] == "http://veles:8400/v1/documents"
    assert post["headers"]["authorization"] == "Bearer owner-token"
    # личность пользователя - в доверенном заголовке, не во вводе модели
    assert post["headers"]["x-veles-on-behalf-of"] == "ivanov@corp.ru"
    body_text = post["data"].decode("utf-8")
    assert 'name="file"; filename="отпуск.txt"' in body_text
    # название документа - имя файла (текст сообщения при файлах - вопрос)
    assert 'name="title"' in body_text and "отпуск" in body_text
    assert "Отпуск 28 дней." in body_text
    assert http.calls[1]["url"].endswith("/v1/documents/jobs/j1")


def test_upload_facade_500_human_error(pipe) -> None:
    http = HttpMock([("POST", "/v1/documents", 500, {"detail": "internal"})])
    out = pipe(http).pipe(BODY_UPLOAD_ONLY, USER, __files__=FILE_ONE)
    assert "500" in out and "отпуск.txt" in out
    assert "Traceback" not in out and "Exception" not in out


def test_upload_facade_401_token_hint(pipe) -> None:
    http = HttpMock([("POST", "/v1/documents", 401, {"detail": "unauthorized"})])
    out = pipe(http).pipe(BODY_UPLOAD_ONLY, USER, __files__=FILE_ONE)
    assert "401" in out and "veles_token" in out


def test_upload_empty_text_no_http(pipe) -> None:
    http = HttpMock()
    files = [{"type": "file", "name": "скан.pdf", "file": {"data": {"content": "   "}}}]
    out = pipe(http).pipe(BODY_UPLOAD_ONLY, USER, __files__=files)
    assert "скан.pdf" in out and "не нашлось текста" in out
    assert http.calls == []


def test_question_only_answer_with_citations(pipe) -> None:
    """Только вопрос: ответ по базе с цитатами [название](ссылка) и сносками."""
    http = HttpMock([("POST", "/v1/knowledge/query", 200, QUERY_RESULT)])
    out = pipe(http).pipe(BODY_QUESTION, USER)
    assert "28 дней" in out and "Источники:" in out
    # рендер цитат как у veles-knowledge: [название](ссылка) и сноска с файлом
    # (document_id в URL процент-энкодится, как в render_chat_content фасада)
    assert "[«Регламент отпусков»]" in out
    assert "/v1/source/ivanov%40corp.ru" in out
    assert "строки 12-15" in out and "регламент_отпусков.txt" in out
    # сноска [1] в тексте синтеза заменена ссылкой (голых сносок не осталось)
    first_line = out.splitlines()[0]
    assert "[«Регламент отпусков»](" in first_line and "[1]" not in first_line

    call = http.calls[0]
    assert call["url"] == "http://veles:8400/v1/knowledge/query"
    assert call["headers"]["authorization"] == "Bearer owner-token"
    # запрос к базе - НЕ act-as: вопрос не персональный документ
    assert "x-veles-on-behalf-of" not in call["headers"]
    payload = json.loads(call["data"].decode("utf-8"))
    assert payload["query"] == "Сколько дней ежегодный отпуск?"
    assert payload["use_llm"] is True


def test_question_abstain_honest(pipe) -> None:
    http = HttpMock([("POST", "/v1/knowledge/query", 200, {
        "decision": "abstain", "answer": "",
        "citations": [], "diagnostics": {"reason": "низкая уверенность"},
    })])
    out = pipe(http).pipe(BODY_QUESTION, USER)
    assert "Ответа на этот вопрос в базе знаний нет" in out
    assert "низкая уверенность" in out
    assert "прикрепите файл" in out  # путь к решению: добавить источник


def test_question_facade_401_token_hint(pipe) -> None:
    http = HttpMock([("POST", "/v1/knowledge/query", 401, {"detail": "unauthorized"})])
    out = pipe(http).pipe(BODY_QUESTION, USER)
    assert "401" in out and "veles_token" in out


def test_question_facade_unreachable_human_error(pipe) -> None:
    http = HttpRaise(RuntimeError("connection refused"))
    out = pipe(http).pipe(BODY_QUESTION, USER)
    assert "не удалось" in out and "фасад недоступен" in out


def test_files_plus_question_upload_then_answer(pipe) -> None:
    """Файлы + вопрос: сначала загрузка всех файлов, затем ответ по базе."""
    http = HttpMock([
        ("POST", "/v1/documents", 202, {"job_id": "j9"}),
        ("GET", "/v1/documents/jobs/j9", 200,
         {"status": "done", "action": "created", "title": "отпуск"}),
        ("POST", "/v1/knowledge/query", 200, QUERY_RESULT),
    ])
    out = pipe(http).pipe(BODY_QUESTION, USER, __files__=FILE_ONE)
    order = [(c["method"], c["url"]) for c in http.calls]
    assert order == [
        ("POST", "http://veles:8400/v1/documents"),
        ("GET", "http://veles:8400/v1/documents/jobs/j9"),
        ("POST", "http://veles:8400/v1/knowledge/query"),
    ]
    assert "✅" in out and "Источники:" in out
    assert "по текущей базе" not in out  # джоб успел - примечания нет


def test_files_plus_question_pending_honest_note(pipe) -> None:
    """Джобы не успели - ответ по текущей базе, и это сказано честно."""
    http = HttpMock([
        ("POST", "/v1/documents", 202, {"job_id": "j10"}),
        ("GET", "/v1/documents/jobs/j10", 200, {"status": "running"}),
        ("POST", "/v1/knowledge/query", 200, QUERY_RESULT),
    ])
    out = pipe(http).pipe(BODY_QUESTION, USER, __files__=FILE_ONE)
    assert "⏳" in out
    assert "по текущей базе" in out and "чуть позже" in out


def test_public_url_for_links(pipe) -> None:
    """Ссылки ведут на адрес, видимый браузеру пользователя, а не на
    контейнерный veles_url; пустой veles_public_url - fallback на veles_url."""
    http = HttpMock([("POST", "/v1/knowledge/query", 200, QUERY_RESULT)])
    out = pipe(http).pipe(BODY_QUESTION, USER)
    assert "http://veles:8400/v1/source/" in out  # fallback

    http2 = HttpMock([("POST", "/v1/knowledge/query", 200, QUERY_RESULT)])
    p2 = pipe(http2)
    p2.valves.veles_public_url = "http://203.0.113.7:8400"
    out2 = p2.pipe(BODY_QUESTION, USER)
    assert "http://203.0.113.7:8400/v1/source/" in out2
    assert "http://veles:8400/v1/source/" not in out2


def test_redirects_not_followed(direct) -> None:
    """3xx не должен превращаться в следование: Authorization и act-as ушли бы
    целевому хосту. _NoRedirect запрещает redirect_request, HTTPError (статус
    3xx) доходит до _http_bytes и возвращается как статус без исключения."""
    mod = direct(HttpRaise(urllib.error.HTTPError(
        "http://veles:8400/v1/documents", 302, "Found", {},
        io.BytesIO(b"http://evil/moved"),
    )))
    assert mod._NoRedirect().redirect_request(
        None, None, 302, "Found", {}, "http://evil/"
    ) is None
    status, body = mod._http_bytes(
        "POST", "http://veles:8400/v1/documents", {}, b"x", 5.0
    )
    assert status == 302 and b"evil" in body


def test_network_error_human_text(direct) -> None:
    mod = direct(HttpRaise(ConnectionError("timed out")))
    with pytest.raises(RuntimeError, match="фасад недоступен"):
        mod._http_bytes("GET", "http://veles:8400/x", {}, None, 5.0)


def test_multipart_edge_filenames(mod) -> None:
    """Пограничные имена файлов: несколько точек, пробелы, кириллица,
    без расширения - multipart собирается корректно, расширение честное."""
    body, ctype = mod._multipart_body(
        {"title": "отпуск"}, "file", "отчёт 2026.final.txt", "текст".encode("utf-8")
    )
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    text = body.decode("utf-8")
    assert 'filename="отчёт 2026.final.txt"' in text
    assert text.count(f"--{boundary}") == 3  # поле title, файл, финал
    assert text.endswith(f"--{boundary}--\r\n")
    assert "текст" in text

    assert mod._veles_filename("a.b.c.docx") == "a.b.c.txt"
    assert mod._veles_filename("заметки") == "заметки.txt"
    assert mod._veles_filename("  ") == "Документ.txt"
    assert mod._veles_filename("doc.MD") == "doc.md"
    assert mod._veles_filename("doc.markdown") == "doc.md"


def test_multipart_quotes_and_newlines_stripped(mod) -> None:
    """P2 ревью K19: кавычки и переводы строк в имени файла ломали
    quoted-string заголовка - зачищаются до вставки в multipart."""
    body, _ = mod._multipart_body(
        {"title": "t"}, "file", 'отчёт"2026\r\n.docx', b"x"
    )
    text = body.decode("utf-8")
    assert 'filename="отчёт2026.docx"\r\n' in text


def test_render_appends_asset_images(mod) -> None:
    """P2 ревью K19: цитаты с asset (PDF-изображения) рендерятся как
    markdown-картинки тем же форматом, что render_chat_content фасада."""
    result = {
        "decision": "answer",
        "answer": "Ответ [1].",
        "citations": [
            {
                "n": 1,
                "title": "Лекция 04",
                "document_id": "docs/лекция.pdf",
                "source_id": "лекция.pdf",
                "node_version_id": "nv1",
                "asset": "page-3.png",
                "lines": [10, 20],
            }
        ],
    }
    out = mod._render_answer(result, "http://veles:8400", "owner-token")
    # кириллица document_id URL-кодируется, как у фасада (quote)
    assert "![изображение](http://veles:8400/v1/asset/docs/%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D1%8F.pdf/page-3.png?token=owner-token)" in out


def test_images_skipped_question_still_answered(pipe) -> None:
    """Картинка - не документ: загрузки нет, текстовый вопрос работает."""
    http = HttpMock([("POST", "/v1/knowledge/query", 200, QUERY_RESULT)])
    files = [{"type": "image", "url": "http://x/img.png"}]
    out = pipe(http).pipe(BODY_QUESTION, USER, __files__=files)
    assert not any("/v1/documents" in c["url"] for c in http.calls)
    assert "Источники:" in out

    http2 = HttpMock()  # картинка без вопроса - честная инструкция
    out2 = pipe(http2).pipe(BODY_UPLOAD_ONLY, USER, __files__=files)
    assert "Задайте вопрос" in out2 and http2.calls == []


def test_fallback_files_from_body(pipe) -> None:
    """Старый путь: файлы в body['files'] (без __files__) - модель находит."""
    http = HttpMock([
        ("POST", "/v1/documents", 202, {"job_id": "j11"}),
        ("GET", "/v1/documents/jobs/j11", 200,
         {"status": "done", "action": "created", "title": "t"}),
    ])
    body = {**BODY_UPLOAD_ONLY, "files": FILE_ONE}
    out = pipe(http).pipe(body, USER)
    assert "✅" in out


def test_multiple_files_then_question(pipe) -> None:
    files = [
        {"type": "file", "name": "а.txt", "file": {"data": {"content": "Текст А"}}},
        {"type": "file", "name": "б.md", "file": {"data": {"content": "Текст Б"}}},
    ]
    http = HttpMock([
        ("POST", "/v1/documents", 202, {"job_id": "j12"}),
        ("GET", "/v1/documents/jobs/j12", 200,
         {"status": "done", "action": "created", "title": "а"}),
        ("POST", "/v1/documents", 202, {"job_id": "j13"}),
        ("GET", "/v1/documents/jobs/j13", 200,
         {"status": "done", "action": "created", "title": "б"}),
        ("POST", "/v1/knowledge/query", 200, QUERY_RESULT),
    ])
    out = pipe(http).pipe(BODY_QUESTION, USER, __files__=files)
    posts = [c for c in http.calls if c["method"] == "POST" and "documents" in c["url"]]
    assert len(posts) == 2  # каждый файл - отдельный документ
    first = posts[0]["data"].decode("utf-8")
    second = posts[1]["data"].decode("utf-8")
    assert 'filename="а.txt"' in first and "Текст А" in first
    assert 'filename="б.md"' in second and "Текст Б" in second
    # название документа - имя файла (вопрос пользователя не приписывается
    # документам названием)
    assert "Сколько дней" not in first
    assert out.count("✅") == 2 and "Источники:" in out


def test_frontmatter_compat(mod) -> None:
    """Docstring функции обязан парситься фронтматтер-парсером WebUI
    (content начинается с строки '\"\"\"' и несёт title/author/version)."""
    content = FUNCTION_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0].strip() == '"""'
    head = "\n".join(lines[1:10])
    assert "title:" in head and "version:" in head
