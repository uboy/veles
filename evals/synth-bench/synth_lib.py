"""Общая библиотека synth-bench (K11): чистые функции без сети и без импорта veles.

Здесь живёт то, что тестируется pytest-ом без сети и без стенд:
- сборка сообщений синтеза (зеркало промпта фасада, см. veles/api/llm.py SYNTHESIS_SYSTEM);
- извлечение сносок [n] из ответа;
- механическая детекция отказа (abstain-кейсы: отказ = 1, выдумка = 0);
- разбор ответа судьи (строгий JSON) и агрегация метрик;
- рендер таблицы [MODEL, FAITH, CITE, COV, ABST, RU, LAT].

HTTP-клиент (chat_completion) - stdlib urllib, стиль e2e/run_deployed_e2e.py.
"""

from __future__ import annotations

import json
import re
import statistics
import urllib.error
import urllib.request
from pathlib import Path

# Порядок и подписи колонок таблицы - по брифу K11.
METRIC_KEYS = ("faithfulness", "citation_accuracy", "coverage", "russian_quality")
METRIC_RU = {
    "faithfulness": "FAITH",
    "citation_accuracy": "CITE",
    "coverage": "COV",
    "russian_quality": "RU",
}

# Маркеры отказа (строчные подстроки): механика abstain-кейсов.
REFUSAL_MARKERS = (
    "ответа нет",
    "ответ отсутствует",
    "в базе нет",
    "нет в базе",
    "не содержат ответа",
    "не содержит ответа",
    "не содержится",
    "нет информации",
    "нет данных",
    "не найдено",
    "не удалось найти",
    "не могу ответить",
    "не могу дать ответ",
    "не позволяет ответить",
    "затрудняюсь",
    "нет сведений",
    "фрагменты не содержат",
    "нет ответа",
)

FOOTNOTE_RE = re.compile(r"\[(\d{1,2})\]")


def load_jsonl(path: str | Path) -> list[dict]:
    """Чтение golden.jsonl: по валидному JSON-объекту на строку, пустые строки - пропуск."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def build_synthesis_messages(question: str, fragments: list[str]) -> list[dict]:
    """Сообщения синтеза - точное зеркало veles.api.llm.OpenAICompatibleProvider.synthesize.

    System-промпт фасада подставляется вызывающим кодом (run_synth_bench импортирует
    SYNTHESIS_SYSTEM из veles.api.llm, чтобы промпт не мог разойтись с продом);
    здесь - пользовательское сообщение с нумерованными фрагментами.
    """
    numbered = "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(fragments))
    return [
        {
            "role": "user",
            "content": f"Вопрос: {question}\n\nФрагменты базы знаний:\n{numbered}",
        }
    ]


def extract_cited_nums(answer: str) -> list[int]:
    """Номера сносок [n] в порядке появления (без дублей)."""
    seen: list[int] = []
    for m in FOOTNOTE_RE.finditer(answer):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def is_refusal(answer: str) -> bool:
    """Механическая детекция отказа для abstain-кейсов.

    Отказ = в ответе нет содержательных фактов (бриф K11):
    1) явный маркер отказа в тексте - отказ;
    2) маркера нет, но ответ короткий (<= 40 слов), без цифр и без сносок - тоже
       отказ (уклончивая отговорка без содержания);
    3) иначе ответ считается содержательным (выдумка на abstain-вопросе).
    """
    low = answer.lower()
    if any(marker in low for marker in REFUSAL_MARKERS):
        return True
    words = low.split()
    has_digits = any(ch.isdigit() for ch in low)
    has_footnote = bool(FOOTNOTE_RE.search(answer))
    if len(words) <= 40 and not has_digits and not has_footnote:
        return True
    return False


def abstain_score(answer: str) -> int:
    """Механический скор abstain-кейса по брифу: отказ = 1, выдумка = 0."""
    return 1 if is_refusal(answer) else 0


def parse_judge_json(text: str) -> dict | None:
    """Разбор ответа судьи: первый '{'..последний '}' как строгий JSON.

    Возвращает dict только если все четыре метрики присутствуют и лежат в 0..2;
    иначе None (звонок считается неразобранным, вызывающий код ставит parse_error).
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    for key in METRIC_KEYS:
        val = data.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or not 0 <= val <= 2:
            return None
    return data


def aggregate(results: list[dict]) -> dict:
    """Агрегация прогона одной модели -> строка метрик.

    Ожидает записи вида:
      {"qid", "abstain", "answer", "latency_s", "prompt_tokens", "completion_tokens",
       метрики ключами верхнего уровня: "faithfulness"/"citation_accuracy"/
       "coverage"/"russian_quality" = 0..2 | None (только не-abstain),
       "abstain_ok": bool}
    FAITH/CITE/COV/RU - среднее 0..2 по отвечаемым кейсам с валидным скором;
    ABST - процент корректных отказов на abstain-кейсах; LAT - средняя задержка,
    сек; TOK - среднее полное число токенов (prompt+completion) на ответ.
    """
    answerable = [r for r in results if not r.get("abstain")]
    abstains = [r for r in results if r.get("abstain")]
    row: dict = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in answerable if isinstance(r.get(key), int)]
        row[key] = round(statistics.mean(vals), 2) if vals else None
    row["abst_pct"] = (
        round(100.0 * sum(1 for r in abstains if r.get("abstain_ok")) / len(abstains), 1)
        if abstains
        else None
    )
    lats = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    row["lat_s"] = round(statistics.mean(lats), 1) if lats else None
    toks = [
        (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in results
    ]
    row["tok_mean"] = round(statistics.mean(toks), 0) if any(toks) else None
    row["n_total"] = len(results)
    row["n_answerable"] = len(answerable)
    row["n_abstain"] = len(abstains)
    row["n_parse_errors"] = sum(
        1 for r in answerable if not isinstance(r.get("faithfulness"), int)
    )
    return row


def render_table(rows: list[dict]) -> str:
    """Markdown-таблица [MODEL, FAITH, CITE, COV, ABST, RU, LAT] (+N, TOK).

    rows - результат aggregate() с добавленным "model"; ABST в процентах.
    """
    header = (
        "| MODEL | FAITH | CITE | COV | ABST | RU | LAT, c | N | TOK/отв |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in rows:
        fmt = lambda v: "-" if v is None else v  # noqa: E731
        lines.append(
            f"| {r['model']} | {fmt(r.get('faithfulness'))} | {fmt(r.get('citation_accuracy'))} "
            f"| {fmt(r.get('coverage'))} | {fmt(r.get('abst_pct'))}% | {fmt(r.get('russian_quality'))} "
            f"| {fmt(r.get('lat_s'))} | {r.get('n_total')} | {fmt(r.get('tok_mean'))} |\n"
        )
    return "".join(lines)


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    api_key: str = "",
    max_tokens: int = 3072,
    temperature: float = 0.2,
    timeout: float = 180.0,
) -> dict:
    """Один OpenAI-совместимый вызов chat/completions (stdlib, ретраев нет - на верху).

    Возврат: {"content", "reasoning_present", "finish_reason", "prompt_tokens",
    "completion_tokens", "model"}. Ответом считается ТОЛЬКО message.content;
    reasoning_content (thinking-канал GLM/gemma) в ответ не попадает.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {}) or {}
    usage = data.get("usage", {}) or {}
    return {
        "content": msg.get("content") or "",
        "reasoning_present": bool(msg.get("reasoning_content")),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "model": data.get("model", model),
    }


def load_env_file(path: str | Path) -> dict[str, str]:
    """Чтение KEY=VALUE файла (стенд ~/secrets/*.env); кавычки снимаются."""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
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
