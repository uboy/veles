#!/usr/bin/env python3
"""Deployed E2E (Уровень 2) стека Veles на стенд: standalone-скрипт с чек-листом PASS/FAIL.

Запуск НА стенд (не pytest):
  cd ~/proj/veles-bench/veles && CUDA_VISIBLE_DEVICES= ~/veles-venv/bin/python \
      e2e/run_deployed_e2e.py --expected-commit <sha-из-локального-main> [--skip-llm] [--only B,C]

Флаги:
  --expected-commit SHA   ожидаемый HEAD main репозитория (сверка с COMMIT.txt деплоя)
  --skip-llm              пропустить пункты, тратящие вызовы z.ai (B7, E2, E3)
  --only A,B7             выполнить только перечисленные пункты
  --known-issue G1        считать пункт известным багом прода (FAIL -> KNOWN-ISSUE в
                          вердикте; чинит владелец, E2E-агент код прода не меняет)

Секреты читаются из ~/secrets/*.env в переменные памяти процесса, НЕ печатаются:
весь вывод прогоняется через mask(). LLM-вызовы считаются счётчиком (полный прогон = 3
вызова glm-5.3-flash: B7, E2, E3; лимит задачи - 6). Telegram API не вызывается.

Exit codes: 0 - PASS/MANUAL/SKIP без FAIL и KNOWN-ISSUE; 2 - есть KNOWN-ISSUE; 1 - есть FAIL.
Stdlib-only (urllib/subprocess): работает системным python3 без внешних зависимостей.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HOME = Path(os.path.expanduser("~"))
REPO_DIR = HOME / "proj" / "veles-bench" / "veles"
VENV_PY = str(HOME / "veles-venv" / "bin" / "python")
FACADE = "http://127.0.0.1:8400"
WEBUI = "http://127.0.0.1:8300"
SECRETS_DIR = HOME / "secrets"
BRIDGE_DIR = REPO_DIR / "clients" / "telegram-bridge"

SECRET_VALUES: list[str] = []

# Паттерны секретов для скана journal (строки НЕ печатаются - только счётчики)
SECRET_PATTERNS = [
    ("tg-bot-token", r"\b\d{8,12}:AA[A-Za-z0-9_-]{20,}"),
    ("ZAI_KEY", r"ZAI_KEY"),
    ("dumbproxy-account", r"стенд-veles2"),
    ("sk-api-key", r"sk-[A-Za-z0-9]{16,}"),
]

results: list[dict] = []
zai_calls = 0


def mask(s: object) -> str:
    """Замаскировать известные секретные значения и userinfo в URL перед печатью."""
    out = str(s)
    for v in SECRET_VALUES:
        if v and len(v) >= 6:
            out = out.replace(v, "***")
    out = re.sub(r"//[^/@\s:]+:[^/@\s]+@", "//***:***@", out)
    out = re.sub(r"\b\d{8,12}:AA[A-Za-z0-9_-]{20,}", "***tg-token***", out)
    return out


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def register_secret(value: str | None) -> None:
    if value and len(value) >= 6:
        SECRET_VALUES.append(value)


def record(item_id: str, name: str, expected: str, actual: str, status: str, note: str = "") -> None:
    results.append({"id": item_id, "name": name, "expected": expected,
                    "actual": mask(actual), "status": status, "note": mask(note)})
    line = f"[{status}] {item_id} {name}: {mask(actual)}"
    if note:
        line += f" | {mask(note)}"
    print(line, flush=True)


def http(method: str, url: str, *, headers: dict | None = None, body: dict | None = None,
         timeout: float = 60) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    hdrs = dict(headers or {})
    if data is not None and "Content-Type" not in hdrs:
        # без Content-Type FastAPI не распарсит JSON и ответит 422 до авторизации
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:  # статус >= 400 - тоже валидный ответ
        return e.code, e.read().decode("utf-8", "replace")


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def run(cmd: list[str], *, timeout: float = 120, env: dict | None = None,
        cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          env=env, cwd=str(cwd) if cwd else None)


def clean_env() -> dict:
    """Окружение для дочерних процессов: без секретов и без переопределений путей veles."""
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("VELES_") or key in ("TELEGRAM_BOT_TOKEN", "OPENWEBUI_API_KEY"):
            del env[key]
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    return env


def journal_text(unit: str, since: str = "1 hour ago") -> str:
    r = run(["sudo", "-n", "journalctl", "-u", unit, "--since", since, "--no-pager", "-o", "cat"],
            timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"journalctl failed: {mask(r.stderr[:200])}")
    return r.stdout


def journal_for_scan(unit: str) -> tuple[str, str]:
    """Журнал для скана паттернов: за час; если за час пусто (idle long-polling
    молчит) - за сутки, чтобы проверка не вырождалась в пустое чтение."""
    text = journal_text(unit, "1 hour ago")
    if text.strip():
        return text, "1 час"
    return journal_text(unit, "24 hours ago"), "1ч пусто (idle), взят 24ч"


# ---------------------------------------------------------------- A. Инфраструктура

def check_services() -> None:
    for unit in ("veles-facade", "veles-telegram-bridge"):
        item = "A1" if "facade" in unit else "A2"
        try:
            r = run(["systemctl", "is-active", unit])
            status = "PASS" if r.stdout.strip() == "active" else "FAIL"
            record(item, f"systemd {unit}", "active", r.stdout.strip() or r.stderr.strip(), status)
        except Exception as e:  # noqa: BLE001
            record(item, f"systemd {unit}", "active", f"error: {e}", "FAIL")


def check_webui_container() -> None:
    fmt = "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}"
    r = run(["docker", "inspect", "-f", fmt, "open-webui"])
    out = r.stdout.strip()
    status = "PASS" if out in ("running|healthy",) else "FAIL"
    if out == "running|no-healthcheck":
        status, out = "FAIL", out + " (healthcheck не задан)"
    record("A3", "docker open-webui", "running|healthy", out or mask(r.stderr[:150]), status)


def check_commit(expected: str | None) -> None:
    commit_file = REPO_DIR / "COMMIT.txt"
    actual = commit_file.read_text(encoding="utf-8").strip() if commit_file.exists() else "(нет файла)"
    if not expected:
        record("A4", "COMMIT.txt == HEAD main", expected or "(не задан)", actual, "SKIP",
               "передайте --expected-commit")
        return
    record("A4", "COMMIT.txt == HEAD main", expected, actual,
           "PASS" if actual == expected else "FAIL")


def check_journal_secrets() -> None:
    try:
        text, window = journal_for_scan("veles-telegram-bridge")
    except Exception as e:  # noqa: BLE001
        record("A5", "секреты в journal моста", "0 совпадений по 4 паттернам", str(e), "FAIL")
        return
    lines = text.splitlines()
    detail = []
    clean = True
    for name, pattern in SECRET_PATTERNS:
        n = sum(1 for ln in lines if re.search(pattern, ln))
        if n:
            clean = False
            detail.append(f"{name}: {n} строк (значения не печатаются)")
    record("A5", "секреты в journal моста", "0 совпадений по 4 паттернам",
           "чисто" if clean else "; ".join(detail), "PASS" if clean else "FAIL",
           f"окно: {window}, строк журнала: {len(lines)}")


# ---------------------------------------------------------------- B. Фасад

def check_healthz() -> None:
    code, body = http("GET", f"{FACADE}/healthz", timeout=30)
    record("B1", "GET /healthz", "200 status=ok", f"{code} {body[:80]}", "PASS" if code == 200 else "FAIL")


def check_401() -> None:
    # POST-тела обязаны нести Content-Type, иначе FastAPI отвечает 422 ДО авторизации
    json_headers = {"Content-Type": "application/json"}
    cases = [
        ("GET", "/v1/models", None, {}),
        ("GET", "/v1/knowledge/stats", None, {}),
        ("POST", "/v1/knowledge/query", {"query": "проба", "use_llm": False}, json_headers),
        ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "проба"}]},
         json_headers),
    ]
    bad = []
    for method, path, body, extra in cases:
        code, _ = http(method, f"{FACADE}{path}", body=body, headers=extra, timeout=30)
        if code != 401:
            bad.append(f"{method} {path} -> {code}")
    record("B2", "401 без bearer на контентных эндпоинтах", "все 401",
           "; ".join(bad) if bad else "все 4 эндпоинта -> 401", "PASS" if not bad else "FAIL")


def check_models() -> None:
    code, body = http("GET", f"{FACADE}/v1/models", headers=bearer(FACADE_TOKEN), timeout=30)
    ids = [m.get("id") for m in json.loads(body).get("data", [])] if code == 200 else []
    record("B3", "GET /v1/models", "200, id veles-knowledge",
           f"{code}, ids={ids}", "PASS" if code == 200 and "veles-knowledge" in ids else "FAIL")


def _doc_has(docs: list[dict], predicate) -> int:
    return sum(1 for d in docs if predicate(d))


def check_stats() -> None:
    code, body = http("GET", f"{FACADE}/v1/knowledge/stats", headers=bearer(FACADE_TOKEN), timeout=60)
    if code != 200:
        record("B4", "GET /v1/knowledge/stats", "200, totals>=120/120000, НК РФ и Вышка",
               f"HTTP {code}: {body[:120]}", "FAIL")
        return
    d = json.loads(body)
    totals = d.get("totals", {})
    docs = d.get("documents", [])
    nk = _doc_has(docs, lambda x: "алоговый кодекс" in x.get("title", "").lower())
    vyshka = _doc_has(docs, lambda x: x.get("document_id", "").startswith("lectures/"))
    ok = totals.get("documents", 0) >= 120 and totals.get("nodes", 0) >= 120000 and nk > 0 and vyshka > 0
    record("B4", "GET /v1/knowledge/stats", "documents>=120, nodes>=120000, НК РФ и Вышка",
           f"documents={totals.get('documents')}, nodes={totals.get('nodes')}, "
           f"НК РФ: {nk} док., lectures/Вышка: {vyshka} док.",
           "PASS" if ok else "FAIL")


def check_query_answer() -> None:
    code, body = http("POST", f"{FACADE}/v1/knowledge/query",
                      headers=bearer(FACADE_TOKEN), timeout=120,
                      body={"query": "транспортный налог согласно НК РФ", "top_k": 5, "use_llm": False})
    if code != 200:
        record("B5", "query: транспортный налог НК РФ", "answer, citations>0, в цитатах НК РФ",
               f"HTTP {code}: {body[:150]}", "FAIL")
        return
    d = json.loads(body)
    cites = d.get("citations", [])
    nk_in = any("алоговый кодекс" in (c.get("title") or "").lower() for c in cites)
    ok = d.get("decision") == "answer" and len(cites) > 0 and nk_in
    record("B5", "query: транспортный налог НК РФ", "answer, citations>0, в цитатах НК РФ",
           f"decision={d.get('decision')}, citations={len(cites)}, "
           f"НК в цитатах: {nk_in}, conf={d.get('diagnostics', {}).get('confidence')}",
           "PASS" if ok else "FAIL")


def check_query_abstain() -> None:
    code, body = http("POST", f"{FACADE}/v1/knowledge/query",
                      headers=bearer(FACADE_TOKEN), timeout=120,
                      body={"query": "как приготовить борщ", "top_k": 5, "use_llm": False})
    if code != 200:
        record("B6", "query: как приготовить борщ", "abstain с причиной",
               f"HTTP {code}: {body[:150]}", "FAIL")
        return
    d = json.loads(body)
    reason = d.get("diagnostics", {}).get("reason") or ""
    ok = d.get("decision") == "abstain" and bool(reason)
    record("B6", "query: как приготовить борщ", "abstain с причиной",
           f"decision={d.get('decision')}, reason={reason[:80]!r}", "PASS" if ok else "FAIL")


def check_chat_synthesis() -> None:
    global zai_calls
    zai_calls += 1
    code, body = http("POST", f"{FACADE}/v1/chat/completions",
                      headers=bearer(FACADE_TOKEN), timeout=300,
                      body={"messages": [{"role": "user", "content": "Норма управляемости руководителя?"}]})
    if code != 200:
        record("B7", "chat/completions синтез (z.ai)", "content с Источники: и [n]",
               f"HTTP {code}: {body[:150]}", "FAIL", f"z.ai вызовов: {zai_calls}")
        return
    d = json.loads(body)
    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    has_n = bool(re.search(r"\[\d\]", content))
    ok = "Источники:" in content and has_n
    record("B7", "chat/completions синтез (z.ai)", "content с Источники: и [n]",
           f"len(content)={len(content)}, Источники: {'Источники:' in content}, "
           f"[n]: {has_n}, decision={d.get('veles', {}).get('decision')}",
           "PASS" if ok else "FAIL", f"z.ai вызовов: {zai_calls}")


# ---------------------------------------------------------------- C. CLI

def check_cli() -> None:
    try:
        r = run([VENV_PY, "-m", "veles", "ask", "норма управляемости"],
                timeout=600, env=clean_env(), cwd=REPO_DIR)
    except subprocess.TimeoutExpired:
        record("C1", "CLI: python -m veles ask", "ANSWER + цитаты", "timeout 600s", "FAIL")
        return
    out = r.stdout
    ok = r.returncode == 0 and "ANSWER" in out and "цитаты:" in out
    first_cite = next((ln for ln in out.splitlines() if ln.strip().startswith("- «")), "")
    record("C1", "CLI: python -m veles ask", "returncode 0, вывод с ANSWER и цитатами",
           f"rc={r.returncode}, ANSWER: {'ANSWER' in out}, цитаты: {'цитаты:' in out}; "
           f"первая цитата: {first_cite.strip()[:100]}",
           "PASS" if ok else "FAIL", mask(r.stderr[-200:]) if r.returncode != 0 else "")


# ---------------------------------------------------------------- D. MCP

class RpcLineReader:
    """Построчный JSON-RPC ридер поверх raw fd (select + os.read): без гонок
    с внутренним буфером TextIOWrapper."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buf = b""

    def read_msg(self, deadline: float):
        """Ждать следующее сообщение с полем id; {} - служебные/некорректные строки."""
        while time.time() < deadline:
            if b"\n" in self.buf:
                line, self.buf = self.buf.split(b"\n", 1)
                try:
                    msg = json.loads(line.decode("utf-8", "replace").strip())
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and "id" in msg:
                    return msg
                continue
            ready, _, _ = select.select([self.fd], [], [], max(0.05, deadline - time.time()))
            if not ready:
                continue
            chunk = os.read(self.fd, 65536)
            if not chunk:
                return {"error": "EOF от процесса"}
            self.buf += chunk
        return {"error": f"timeout {int(deadline - time.time())}s"}


def check_mcp() -> None:
    proc = subprocess.Popen(
        [VENV_PY, "-m", "veles", "mcp"], cwd=str(REPO_DIR), env=clean_env(),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stderr_tail: list[str] = []
    threading.Thread(target=lambda: stderr_tail.extend(
        (proc.stderr.read() or b"").decode("utf-8", "replace").splitlines()[-30:]),
        daemon=True).start()
    reader = RpcLineReader(proc.stdout.fileno())

    def send(obj: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def fail(item: str, name: str, expected: str, actual: str) -> None:
        record(item, name, expected, actual, "FAIL",
               "stderr: " + mask(" | ".join(stderr_tail[-3:])))

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "veles-e2e", "version": "0.1"}}})
        init = reader.read_msg(time.time() + 30)
        if "result" not in init:
            fail("D1", "MCP initialize (smoke)", "result с serverInfo", mask(str(init))[:200])
            record("D2", "MCP tool call knowledge_query", "результат с цитатами",
                   "SKIP: не выполнялся, D1 упал", "SKIP")
            return
        server = init["result"].get("serverInfo", {}).get("name", "?")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tl = reader.read_msg(time.time() + 60)
        tools = [t.get("name") for t in tl.get("result", {}).get("tools", [])] if "result" in tl else []
        if "knowledge_query" not in tools:
            fail("D1", "MCP initialize+tools/list (smoke)", "tools содержит knowledge_query",
                 mask(str(tl))[:200])
            record("D2", "MCP tool call knowledge_query", "результат с цитатами",
                   "SKIP: не выполнялся, D1 упал", "SKIP")
            return
        record("D1", "MCP initialize+tools/list (smoke)", "serverInfo + knowledge_query",
               f"server={server}, tools={tools}", "PASS")

        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "knowledge_query",
                         "arguments": {"query": "норма управляемости", "top_k": 3}}})
        call = reader.read_msg(time.time() + 600)
        if "result" not in call:
            fail("D2", "MCP tool call knowledge_query", "результат с цитатами",
                 mask(str(call))[:200])
            return
        result = call["result"]
        payload = None
        sc = result.get("structuredContent")
        if isinstance(sc, dict):
            payload = sc
        else:
            for c in result.get("content", []):
                if not isinstance(c, dict):
                    continue
                try:
                    payload = json.loads(c.get("text", ""))
                    break
                except (json.JSONDecodeError, TypeError):
                    continue
        if isinstance(payload, dict) and "decision" in payload:
            ncites = len(payload.get("citations", []))
            ok = payload.get("decision") == "answer" and ncites > 0
            record("D2", "MCP tool call knowledge_query", "decision=answer, citations>0",
                   f"decision={payload.get('decision')}, citations={ncites}", "PASS" if ok else "FAIL")
        else:
            fail("D2", "MCP tool call knowledge_query", "результат с цитатами",
                 mask(str(result))[:250])
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()


# ---------------------------------------------------------------- E. Open WebUI

def _webui_ids(body: str) -> list[str]:
    d = json.loads(body)
    ids: list[str] = []
    for m in d.get("data", []) or d.get("models", []) or []:
        ids.append(m.get("id") if isinstance(m, dict) else str(m))
    return ids


def check_webui_models() -> None:
    code, body = http("GET", f"{WEBUI}/api/models", headers=bearer(WEBUI_KEY), timeout=60)
    if code != 200:
        record("E1", "GET /api/models WebUI", "200, glm-5.3-flash и veles-knowledge",
               f"HTTP {code}: {body[:120]}", "FAIL")
        return
    ids = _webui_ids(body)
    ok = "glm-5.3-flash" in ids and "veles-knowledge" in ids
    record("E1", "GET /api/models WebUI", "glm-5.3-flash и veles-knowledge в списке",
           f"моделей: {len(ids)}, glm-5.3-flash: {'glm-5.3-flash' in ids}, "
           f"veles-knowledge: {'veles-knowledge' in ids}", "PASS" if ok else "FAIL")


def check_webui_chat_kb() -> None:
    global zai_calls
    zai_calls += 1
    code, body = http("POST", f"{WEBUI}/api/chat/completions", headers=bearer(WEBUI_KEY),
                      timeout=300,
                      body={"model": "veles-knowledge", "stream": False,
                            "messages": [{"role": "user", "content": "Норма управляемости руководителя?"}]})
    if code != 200:
        record("E2", "WebUI chat model=veles-knowledge (z.ai)", "content с Источники:",
               f"HTTP {code}: {body[:150]}", "FAIL", f"z.ai вызовов: {zai_calls}")
        return
    content = json.loads(body).get("choices", [{}])[0].get("message", {}).get("content", "")
    ok = "Источники:" in content
    record("E2", "WebUI chat model=veles-knowledge (z.ai)", "content с Источники:",
           f"len(content)={len(content)}, Источники: {'Источники:' in content}",
           "PASS" if ok else "FAIL", f"z.ai вызовов: {zai_calls}")


def check_webui_chat_glm() -> None:
    global zai_calls
    zai_calls += 1
    code, body = http("POST", f"{WEBUI}/api/chat/completions", headers=bearer(WEBUI_KEY),
                      timeout=300,
                      body={"model": "glm-5.3-flash", "stream": False,
                            "messages": [{"role": "user", "content": "Ответь одним словом: сколько будет 2+2?"}]})
    if code != 200:
        record("E3", "WebUI chat model=glm-5.3-flash (z.ai)", "200, content непуст",
               f"HTTP {code}: {body[:150]}", "FAIL", f"z.ai вызовов: {zai_calls}")
        return
    d = json.loads(body)
    content = d.get("choices", [{}])[0].get("message", {}).get("content", "")
    model = d.get("model", "?")
    ok = bool(content.strip())
    record("E3", "WebUI chat model=glm-5.3-flash (z.ai)", "200, content непуст",
           f"model={model}, len(content)={len(content)}", "PASS" if ok else "FAIL",
           f"z.ai вызовов: {zai_calls}")


def check_signup_closed() -> None:
    # Валидный по форме payload: если бы signup был открыт, вернулся бы 2xx
    # (заведение одноразовой учётки = сам факт критической находки); закрытый
    # signup отвечает 403 "Signups are disabled".
    code, body = http("POST", f"{WEBUI}/api/v1/auths/signup", timeout=60,
                      body={"name": "e2e-probe", "email": "e2e-probe@example.com",
                            "password": "e2e-probe-not-a-real-password"})
    ok = code >= 400
    record("E4", "signup закрыт (валидная форма)", "отказ (HTTP >= 400)", f"HTTP {code}: {body[:100]}",
           "PASS" if ok else "FAIL")


# ---------------------------------------------------------------- F. Мост

BRIDGE_CHECK_CODE = '''
import json, sys
sys.path.insert(0, {bridge_dir!r})
import bridge, tg_format
src = "## Проверка\\n\\nФормат **жирный** и <тег> & амперсанд."
html = tg_format.format_markdown_to_html(src)
out = bridge.apply_footer(html, "glm-5.3-flash")
res = {{
    "bold_ok": "<b>жирный</b>" in html,
    "tag_escaped_once": "&lt;тег&gt;" in html and "&amp;lt;" not in html,
    "amp_escaped_once": "&amp;" in html and "&amp;amp;" not in html,
    "footer_model_ok": "ответ общей модели glm-5.3-flash" in out,
    "footer_tag_live": "<i>" in out and "&lt;i&gt;" not in out,
    "no_double_escape": "&amp;lt;" not in out and "&amp;amp;" not in out,
}}
print(json.dumps(res))
'''


def check_pair_state() -> None:
    state_path = Path(bridge_env.get("BRIDGE_STATE_FILE", str(SECRETS_DIR / "dm-bridge-state.json")))
    state_path = Path(os.path.expanduser(str(state_path)))
    if not state_path.exists():
        record("F1", "pair-state моста", "файл существует, owner_chat_id внутри",
               f"{state_path}: нет файла", "FAIL")
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        cid = state.get("owner_chat_id")
        ok = isinstance(cid, int) and cid > 0
        record("F1", "pair-state моста", "файл существует, owner_chat_id внутри",
               f"{state_path.name}: owner_chat_id присутствует ({type(cid).__name__}); "
               "значение не печатается", "PASS" if ok else "FAIL")
    except Exception as e:  # noqa: BLE001
        record("F1", "pair-state моста", "файл существует, owner_chat_id внутри",
               f"битый json: {e}", "FAIL")


def check_bridge_render() -> None:
    code_str = BRIDGE_CHECK_CODE.format(bridge_dir=str(BRIDGE_DIR))
    env = clean_env()
    env.update({"TELEGRAM_BOT_TOKEN": "000000000:TEST_STUB", "OPENWEBUI_API_KEY": "stub",
                "PAIR_MODE": "1"})
    try:
        r = run([VENV_PY, "-c", code_str], timeout=60, env=env, cwd=REPO_DIR)
    except subprocess.TimeoutExpired:
        record("F2", "мост: apply_footer + форматер", "рендер без двойного экранирования",
               "timeout 60s", "FAIL")
        return
    if r.returncode != 0:
        record("F2", "мост: apply_footer + форматер", "рендер без двойного экранирования",
               f"rc={r.returncode}: {mask(r.stderr[-200:])}", "FAIL")
        return
    res = json.loads(r.stdout.strip().splitlines()[-1])
    ok = all(res.values())
    failed = [k for k, v in res.items() if not v]
    record("F2", "мост: apply_footer + форматер", "все 6 флагов true (без двойного экранирования)",
           ", ".join(f"{k}={v}" for k, v in res.items()),
           "PASS" if ok else "FAIL", f"падающие флаги: {failed}" if failed else "")


def check_bridge_journal() -> None:
    try:
        text, window = journal_for_scan("veles-telegram-bridge")
    except Exception as e:  # noqa: BLE001
        record("F3", "мост: журнал без traceback", "0 Traceback", str(e), "FAIL")
        return
    n = sum(1 for ln in text.splitlines() if "Traceback" in ln)
    record("F3", "мост: журнал без traceback", "0 Traceback",
           f"Traceback: {n}", "PASS" if n == 0 else "FAIL",
           f"окно: {window}, строк журнала: {len(text.splitlines())}")


# ---------------------------------------------------------------- G. Бэкап

def check_backup() -> None:
    cr = run(["crontab", "-l"], timeout=30)
    cron_lines = [ln for ln in (cr.stdout or "").splitlines() if "veles-backup" in ln]
    script_path = None
    if cron_lines:
        m = re.search(r"(\S*veles-backup\.sh)", cron_lines[0])
        script_path = Path(m.group(1)) if m else None
    script_ok = bool(script_path and script_path.exists() and os.access(script_path, os.X_OK))
    log_exists = (HOME / "veles-backup.log").exists()
    backups_dir = HOME / "backups"
    n_archives = len(list(backups_dir.glob("veles-*.tar.gz"))) if backups_dir.exists() else 0
    ok = script_ok and bool(cron_lines)
    record("G1", "бэкап: скрипт + cron", "скрипт существует+исполняем, cron-строка есть",
           f"cron-строк: {len(cron_lines)}; скрипт {script_path}: "
           f"{'есть, исполняем' if script_ok else 'ОТСУТСТВУЕТ или не исполняем'}; "
           f"log: {'есть' if log_exists else 'нет'}; архивов в ~/backups: {n_archives}",
           "PASS" if ok else "FAIL")


# ---------------------------------------------------------------- main

def main() -> int:
    global FACADE_TOKEN, WEBUI_KEY, bridge_env

    ap = argparse.ArgumentParser(description="Deployed E2E стека Veles на стенд")
    ap.add_argument("--expected-commit", default="")
    ap.add_argument("--skip-llm", action="store_true", help="пропустить B7/E2/E3 (вызовы z.ai)")
    ap.add_argument("--only", default="", help="запятая-список id пунктов")
    ap.add_argument("--known-issue", default="", help="запятая-список id, считающихся KNOWN-ISSUE")
    args = ap.parse_args()

    facade_env = load_env_file(SECRETS_DIR / "veles-facade.env")
    bridge_env = load_env_file(SECRETS_DIR / "dm-bridge.env")
    FACADE_TOKEN = facade_env.get("VELES_API_TOKEN", "")
    WEBUI_KEY = bridge_env.get("OPENWEBUI_API_KEY", "")
    for v in (FACADE_TOKEN, WEBUI_KEY, bridge_env.get("TELEGRAM_BOT_TOKEN"),
              bridge_env.get("TG_PROXY"), bridge_env.get("OPENWEBUI_BASE_URL")):
        register_secret(v)
    if not FACADE_TOKEN:
        print("FAIL: VELES_API_TOKEN не прочитан из ~/secrets/veles-facade.env", flush=True)
        return 1
    if not WEBUI_KEY:
        print("FAIL: OPENWEBUI_API_KEY не прочитан из ~/secrets/dm-bridge.env", flush=True)
        return 1

    llm_ids = {"B7", "E2", "E3"}
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    known = {x.strip().upper() for x in args.known_issue.split(",") if x.strip()}

    checks = [
        ("A1", check_services), ("A2", None), ("A3", check_webui_container),
        ("A4", lambda: check_commit(args.expected_commit or None)),
        ("A5", check_journal_secrets),
        ("B1", check_healthz), ("B2", check_401), ("B3", check_models),
        ("B4", check_stats), ("B5", check_query_answer), ("B6", check_query_abstain),
        ("B7", check_chat_synthesis),
        ("C1", check_cli),
        ("D1", check_mcp),
        ("E1", check_webui_models), ("E2", check_webui_chat_kb),
        ("E3", check_webui_chat_glm), ("E4", check_signup_closed),
        ("F1", check_pair_state), ("F2", check_bridge_render), ("F3", check_bridge_journal),
        ("G1", check_backup),
    ]

    plan = []
    skip_llm = []
    for item_id, fn in checks:
        if only and item_id not in only:
            continue
        if item_id in llm_ids and args.skip_llm:
            skip_llm.append(item_id)
            continue
        plan.append((item_id, fn))
    if skip_llm:
        for item_id in skip_llm:
            record(item_id, "LLM-пункт", "-", "SKIP (--skip-llm)", "SKIP")

    for item_id, fn in plan:
        if fn is None:  # A2 обрабатывается в check_services вместе с A1
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - ошибка проверки не должна ронять прогон
            record(item_id, "неожиданная ошибка проверки", "-", f"{type(e).__name__}: {mask(str(e))[:200]}",
                   "FAIL")

    record("T1", "Telegram-сквозной (реальная отправка)", "сообщение владельца доставлено",
           "MANUAL: not automated (нужен живой пользователь; Telegram API из тестов запрещён)",
           "MANUAL")

    counts: dict[str, int] = {}
    for r in results:
        st = r["status"]
        if st == "FAIL" and r["id"] in known:
            r["status"] = "KNOWN-ISSUE"
            st = "KNOWN-ISSUE"
        counts[st] = counts.get(st, 0) + 1

    print("\n=== ИТОГ deployed E2E ===", flush=True)
    for r in results:
        print(f"[{r['status']:>11}] {r['id']:<3} {r['name']}: {r['actual'][:110]}", flush=True)
    print(f"\nz.ai вызовов: {zai_calls} (лимит задачи: 6)", flush=True)
    print("счёт: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)

    if counts.get("FAIL"):
        print("ВЕРДИКТ: КРАСНЫЙ (есть FAIL)", flush=True)
        return 1
    if counts.get("KNOWN-ISSUE"):
        print("ВЕРДИКТ: ЗЕЛЁНЫЙ-С-ОГОВОРКАМИ (есть KNOWN-ISSUE)", flush=True)
        return 2
    print("ВЕРДИКТ: ЗЕЛЁНЫЙ", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
