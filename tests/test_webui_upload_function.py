"""K16: unit-тесты Pipe-функции «Загрузить в базу знаний» (Open WebUI).

Функция живёт в integrations/openwebui/functions/upload_to_veles.py и
исполняется внутри Open WebUI; здесь она загружается из файла как обычный
модуль и проверяется логика: HTTP замокан на уровне urllib (единственная
точка выхода в сеть - _http_bytes/urlopen), фасадные ответы - те же
контракты, что в tests/test_documents_ingest (202 + джоб, 401, 413).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

FUNCTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations" / "openwebui" / "functions" / "upload_to_veles.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("upload_to_veles_test", FUNCTION_PATH)
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
    """Замок urlopen: собирает вызовы, отвечает по очереди/по методу."""

    def __init__(self, responses):  # responses: list[(method, status, payload)]
        self.responses = responses
        self.calls: list[dict] = []

    def __call__(self, req, timeout=None):
        self.calls.append(
            {
                "method": req.get_method(),
                "url": req.full_url,
                "headers": {k.lower(): v for k, v in req.header_items()},
                "data": req.data,
            }
        )
        for i, (method, status, payload) in enumerate(self.responses):
            if method != self.calls[-1]["method"]:
                continue
            return FakeResponse(status, self.responses.pop(i)[2])
        return FakeResponse(200, {})


@pytest.fixture()
def pipe(monkeypatch, mod):
    def _make(http_mock):
        monkeypatch.setattr(mod.urllib.request, "urlopen", http_mock)
        p = mod.Pipe()
        p.valves.veles_url = "http://veles:8400"
        p.valves.veles_token = "owner-token"
        p.valves.wait_seconds = 0
        p.valves.poll_interval_seconds = 0.01
        return p

    return _make


USER = {"email": "ivanov@corp.ru", "name": "Иванов", "role": "user"}
BODY_ONE = {
    "model": "upload_to_veles",
    "messages": [{"role": "user", "content": "Регламент отпусков 2026"}],
}
FILE_ONE = [{
    "type": "file", "id": "f1", "name": "отпуск.txt",
    "file": {"filename": "отпуск.txt", "data": {"content": "Отпуск 28 дней."}},
}]


def test_no_valves_no_http(mod, pipe) -> None:
    http = HttpMock([])
    p = mod.Pipe()  # valves не заполнены
    out = p.pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "Valves" in out and "veles_url" in out
    assert http.calls == [] and pipe is not None


def test_no_email_aborts(pipe, mod) -> None:
    http = HttpMock([])
    out = pipe(http).pipe(BODY_ONE, {"email": ""}, __files__=FILE_ONE)
    assert "email" in out
    assert http.calls == []
    assert mod is not None  # модуль загружен (структура теста)


def test_no_files_instruction(pipe) -> None:
    http = HttpMock([])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=[])
    assert "Прикрепите" in out
    assert http.calls == []


def test_empty_text_honest_refusal(pipe, mod) -> None:
    http = HttpMock([])
    files = [{"type": "file", "name": "скан.pdf", "file": {"data": {"content": "   "}}}]
    out = pipe(http).pipe(BODY_ONE, USER, __files__=files)
    assert "скан.pdf" in out and "не нашлось текста" in out
    assert http.calls == []
    assert mod._veles_filename("док.docx") == "док.txt"  # неподдерживаемое -> .txt


def test_success_single_file(pipe) -> None:
    http = HttpMock([
        ("POST", 202, {"job_id": "j1", "status": "queued", "document_id": "ivanov@corp.ru/отпуск.txt"}),
        ("GET", 200, {"status": "done", "action": "created", "title": "Регламент отпусков 2026"}),
    ])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "✅" in out and "Регламент отпусков 2026" in out
    assert "личная (ivanov@corp.ru)" in out

    post = http.calls[0]
    assert post["method"] == "POST" and post["url"] == "http://veles:8400/v1/documents"
    assert post["headers"]["authorization"] == "Bearer owner-token"
    # личность пользователя - в доверенном заголовке, не во вводе модели
    assert post["headers"]["x-veles-on-behalf-of"] == "ivanov@corp.ru"
    body_text = post["data"].decode("utf-8")
    assert 'name="file"; filename="отпуск.txt"' in body_text
    assert 'name="title"' in body_text and "Регламент отпусков 2026" in body_text
    assert "Отпуск 28 дней." in body_text
    assert http.calls[1]["url"].endswith("/v1/documents/jobs/j1")


def test_facade_401_token_hint(pipe) -> None:
    http = HttpMock([("POST", 401, {"detail": "unauthorized"})])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "401" in out and "veles_token" in out


def test_facade_quota_413_detail(pipe) -> None:
    http = HttpMock([("POST", 413, {"detail": "квота исчерпана"})])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "413" in out and "квота исчерпана" in out


def test_job_failed_reason_surfaced(pipe) -> None:
    http = HttpMock([
        ("POST", 202, {"job_id": "j2"}),
        ("GET", 200, {"status": "failed", "error": "не удалось разобрать документ"}),
    ])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "не удалась" in out and "не удалось разобрать документ" in out


def test_timeout_running_honest_answer(pipe) -> None:
    http = HttpMock([
        ("POST", 202, {"job_id": "j3"}),
        ("GET", 200, {"status": "running"}),
    ])
    p = pipe(http)
    p.valves.wait_seconds = 0  # сразу исчерпать ожидание
    out = p.pipe(BODY_ONE, USER, __files__=FILE_ONE)
    assert "принят" in out and "обрабатывается" in out


def test_multiple_files_separate_documents(pipe) -> None:
    files = [
        {"type": "file", "name": "а.txt", "file": {"data": {"content": "Текст А"}}},
        {"type": "file", "name": "б.md", "file": {"data": {"content": "Текст Б"}}},
    ]
    http = HttpMock([
        ("POST", 202, {"job_id": "j4"}),
        ("GET", 200, {"status": "done", "action": "created", "title": "а"}),
        ("POST", 202, {"job_id": "j5"}),
        ("GET", 200, {"status": "done", "action": "created", "title": "б"}),
    ])
    out = pipe(http).pipe(BODY_ONE, USER, __files__=files)
    posts = [c for c in http.calls if c["method"] == "POST"]
    assert len(posts) == 2  # каждый файл - отдельный документ
    first = posts[0]["data"].decode("utf-8")
    second = posts[1]["data"].decode("utf-8")
    assert 'filename="а.txt"' in first and "Текст А" in first
    assert 'filename="б.md"' in second and "Текст Б" in second
    # у мульти-загрузки название - имя файла (комментарий одного на всех
    # не приписывается каждому документу)
    assert 'name="title"' in first and "Регламент отпусков" not in first
    assert out.count("✅") == 2


def test_fallback_files_from_body(pipe) -> None:
    """Старый путь: файлы в body['files'] (без __files__) - функция находит."""
    http = HttpMock([
        ("POST", 202, {"job_id": "j6"}),
        ("GET", 200, {"status": "done", "action": "created", "title": "t"}),
    ])
    body = {**BODY_ONE, "files": FILE_ONE}
    out = pipe(http).pipe(body, USER)
    assert "✅" in out


def test_images_skipped_no_false_success(pipe) -> None:
    http = HttpMock([])
    files = [{"type": "image", "url": "http://x/img.png"}]
    out = pipe(http).pipe(BODY_ONE, USER, __files__=files)
    assert http.calls == []
    assert "Прикрепите" in out  # файлов текстового типа нет - честная инструкция


def test_extract_frontmatter_compat(mod) -> None:
    """Docstring функции обязан парситься фронтматтер-парсером WebUI
    (content начинается с строки '\"\"\"' и несёт title/author/version)."""
    content = FUNCTION_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0].strip() == '"""'
    head = "\n".join(lines[1:10])
    assert "title:" in head and "version:" in head
