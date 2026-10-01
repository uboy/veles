#!/usr/bin/env python3
"""K11 judge-bias: валидация судьи перед доверием его цифрам.

Три оси (все на готовых артефактах):
1. ШУМ - подмножество уже суденных ответов пересуживается тем же судьёй:
   доля смен вердикта и средняя дельта по метрикам = полоса шума, которую
   надо приписывать разницам таблиц (разница меньше шума ничего не значит).
2. РУБРИКА - контролируемые кейсы на реальных фрагментах: заведомая выдумка
   (faithfulness 0), корректный ответ (2), честный отказ, факт без сноски.
   Судья не обязан угадать точное число, но направление обязано быть верным.
3. СТИЛЬ - реальные ответы в оригинале и «плоской» версии (markdown снят,
   факты и сноски те же): систематическая дельта = смещение к оформлению.

Запуск (стенд):
  PYTHONUNBUFFERED=1 ~/veles-venv/bin/python judge_bias.py \
      --bench-dir <каталог с judged.json, golden.jsonl, retrieval-cache.json> \
      --secrets-file ~/secrets/veles-facade.env --out report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_lib import (  # noqa: E402
    METRIC_KEYS,
    chat_completion,
    extract_cited_nums,
    load_env_file,
    parse_judge_json,
)
from judge import build_judge_messages  # noqa: E402

REPEAT_N = 24      # повторных суждений для оси шума
STYLE_N = 3        # пар оригинал/плоский для оси стиля
MAX_TOKENS = 3072  # как у judge.py: thinking-канал GLM ест бюджет, 1024 обрезали content


def ask_judge(secrets: dict, question: str, fragments: list, answer: str, expected: list) -> dict | None:
    cited = extract_cited_nums(answer)
    messages = build_judge_messages(question, fragments, answer, cited, expected)
    for attempt in range(3):
        try:
            resp = chat_completion(
                secrets["VELES_LLM_BASE_URL"], secrets["VELES_LLM_MODEL"], messages,
                api_key=secrets.get("VELES_LLM_API_KEY", ""), max_tokens=MAX_TOKENS,
            )
            verdict = parse_judge_json(resp["content"])
            if verdict is None:
                print(f"  parse fail, content[:120]: {resp['content'][:120]!r}", flush=True)
                continue
            # судья нестабилен в типе вывода (число/строка) - нормализуем,
            # иначе sum/delta падают на смешанных int/str
            try:
                return {k: int(round(float(verdict[k]))) for k in METRIC_KEYS}
            except (KeyError, ValueError, TypeError) as e:
                print(f"  normalize fail ({e}), verdict: {verdict}", flush=True)
                continue
        except Exception as e:  # noqa: BLE001 - сеть мерцает: повтор
            print(f"  judge call retry {attempt + 1}: {e}", flush=True)
        time.sleep(2.0)
    return None


def load_all(bench: Path):
    judged = json.loads((bench / "judged.json").read_text(encoding="utf-8"))
    rows: list = []
    for e in judged.get("files", []):
        rows.extend(e.get("judged", []))
    golden_by_qid = {}
    for g in (bench / "golden.jsonl").read_text(encoding="utf-8").splitlines():
        if g.strip():
            item = json.loads(g)
            golden_by_qid[item["id"]] = item
    cache = json.loads((bench / "retrieval-cache.json").read_text(encoding="utf-8"))
    return rows, golden_by_qid, cache.get("questions", {})


def frag_by_qid(questions: dict) -> dict:
    return {qid: entry.get("fragments", []) for qid, entry in questions.items()}


def expected_of(golden_by_qid: dict, qid: str) -> list:
    g = golden_by_qid.get(qid, {})
    return g.get("expected_nodes") or g.get("expected") or []


def flat_version(answer: str) -> str:
    """«Плоская» версия: markdown-оформление снято, факты и сноски на месте."""
    s = re.sub(r"^#{1,6}\s*", "", answer, flags=re.MULTILINE)   # заголовки
    s = s.replace("**", "").replace("__", "")                    # жирный
    s = re.sub(r"^\|.*\|\s*$", "", s, flags=re.MULTILINE)        # таблицы -> прочь
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", required=True)
    ap.add_argument("--secrets-file", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    bench = Path(args.bench_dir)
    secrets = load_env_file(Path(args.secrets_file))
    rows, golden_by_qid, cache = load_all(bench)
    frags = frag_by_qid(cache)
    lines = ["# judge-bias проверка судьи", ""]

    # --- ось 1: шум повторных суждений ---
    cand = [r for r in rows if all(r.get(k) is not None for k in METRIC_KEYS)]
    step = max(1, len(cand) // REPEAT_N)
    sample = cand[::step][:REPEAT_N]
    print(f"ось шума: {len(sample)} повторных суждений", flush=True)
    deltas, flips = [], 0
    for r in sample:
        v1 = {k: r[k] for k in METRIC_KEYS}
        v2 = ask_judge(secrets, r["question"], frags.get(r["qid"], []), r["answer"], expected_of(golden_by_qid, r["qid"]))
        if v2 is None:
            continue
        d = {k: v2[k] - v1[k] for k in METRIC_KEYS}
        deltas.append(d)
        if any(v != 0 for v in d.values()):
            flips += 1
        time.sleep(0.3)
    parsed = len(deltas)
    if parsed == 0:
        lines += [
            "## Ось 1: шум (повторное суждение того же ответа тем же судьёй)",
            f"- записей пересужено: {len(sample)}, РАЗОБРАНО 0 - судья не дал валидных вердиктов, проверка НЕ состоялась",
            "",
        ]
    else:
        mean_d = {k: round(sum(d[k] for d in deltas) / parsed, 3) for k in METRIC_KEYS}
        mean_abs = {k: round(sum(abs(d[k]) for d in deltas) / parsed, 3) for k in METRIC_KEYS}
        lines += [
            "## Ось 1: шум (повторное суждение того же ответа тем же судьёй)",
            f"- записей пересужено: {len(sample)}, разобрано: {parsed}, смен вердикта хотя бы по одной метрике: {flips} ({round(100 * flips / parsed)}%)",
            f"- средняя дельта по метрикам: {mean_d}",
            f"- средняя |дельта| (полоса шума): {mean_abs}",
            "",
        ]

    # --- ось 2: контролируемые кейсы рубрики ---
    print("ось рубрики: 4 контролируемых кейса", flush=True)
    qid0 = next(q for q in frags if frags[q])
    q0 = golden_by_qid[qid0]["question"]
    f0 = frags[qid0]
    f1 = f0[0]["text"]
    cases = [
        ("выдумка с ложной сноской",
         "Согласно базе, ставка налога составляет 77 процентов от кадастровой стоимости недвижимости и земельных участков [1].",
         {"faithfulness": 0}),
        ("корректный ответ со сноской",
         f"Из фрагмента базы: {f1[:300]} [1]",
         {"faithfulness": 2, "citation_accuracy": 2}),
        ("честный отказ",
         "В предоставленных фрагментах информация по вашему вопросу отсутствует. Не могу ответить.",
         {"faithfulness": 2}),
        ("факт без сноски",
         f1[:300],
         {"citation_accuracy": 0}),
    ]
    lines += ["## Ось 2: контролируемые кейсы рубрики", ""]
    for name, answer, want in cases:
        v = ask_judge(secrets, q0, f0, answer, expected_of(golden_by_qid, qid0))
        got = {k: v[k] for k in want} if v else None
        if v is None:
            lines.append(f"  (кейс '{name}': судья не разобран за 3 попытки, см. retry-лог)")
        ok = v is not None and all(got[k] == w for k, w in want.items())
        lines.append(f"- {name}: {got} (ожидалось {want}) - {'OK' if ok else 'РАСХОЖДЕНИЕ'}")
        time.sleep(0.3)

    # --- ось 3: стилевое смещение ---
    print(f"ось стиля: {STYLE_N} пар оригинал/плоский", flush=True)
    styled = [r for r in cand if "**" in r["answer"] or "\n#" in r["answer"]][:STYLE_N]
    lines += ["", "## Ось 3: оформление (оригинал vs плоская версия)", ""]
    style_deltas = []
    for r in styled:
        b = flat_version(r["answer"])
        s1 = ask_judge(secrets, r["question"], frags.get(r["qid"], []), r["answer"], expected_of(golden_by_qid, r["qid"]))
        s2 = ask_judge(secrets, r["question"], frags.get(r["qid"], []), b, expected_of(golden_by_qid, r["qid"]))
        if s1 and s2:
            d = {k: s2[k] - s1[k] for k in METRIC_KEYS}
            style_deltas.append(d)
            lines.append(f"- {r['qid']}: сумма {sum(s1.values())} -> {sum(s2.values())} (плоский), дельта {d}")
        time.sleep(0.3)
    if style_deltas:
        m = max(1, len(style_deltas))
        mean_style = {k: round(sum(d[k] for d in style_deltas) / m, 3) for k in METRIC_KEYS}
        lines.append(f"- средняя дельта (плоский минус оригинал): {mean_style}")

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
