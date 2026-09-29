#!/usr/bin/env python3
"""K11 judge: LLM-судья GLM с рубрикой 0-2 + механика abstain + таблица результатов.

Судья = GLM (тот же z.ai env, что у фасада). Судье даются ФРАГМЕНТЫ из
замороженного retrieval + ответ модели + извлечённые сноски; expected_nodes
передаются как контекст ожидаемых узлов, но рубрика предписывает оценивать
СТРОГО по фрагментам (эталон - для будущей ручной валидации владельцем).

Abstain-кейсы судье не отправляются - механика (synth_lib.abstain_score):
отказ = 1, выдумка = 0.

Запуск (стенд, из каталога с артефактами):
  PYTHONUNBUFFERED=1 ~/veles-venv/bin/python judge.py \
      --answers answers-glm.json --answers answers-qwen.json ... \
      --golden golden.jsonl --retrieval-cache retrieval-cache.json \
      --out judged.json --md table.md --secrets-file ~/secrets/veles-facade.env
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_lib import (  # noqa: E402
    METRIC_KEYS,
    abstain_score,
    aggregate,
    chat_completion,
    extract_cited_nums,
    load_env_file,
    load_jsonl,
    parse_judge_json,
    render_table,
)

JUDGE_SYSTEM = (
    "Ты - строгий судья качества ответов RAG-ассистента. Тебе дают вопрос, "
    "фрагменты базы знаний и ответ модели. Оценивай ответ СТРОГО по фрагментам: "
    "что не подтверждается фрагментами - выдумка. Шкала по каждой метрике 0-2:\n"
    "faithfulness: 2 - каждое фактическое утверждение подтверждается фрагментами; "
    "1 - ядро подтверждено, есть мелкие неподтверждённые формулировки; "
    "0 - есть утверждения, которых нет во фрагментах или противоречащие им.\n"
    "citation_accuracy: 2 - сноски [n] есть и каждая ведёт на фрагмент с утверждаемым "
    "фактом; 1 - сноски есть, но часть неверна или пропущена на важных утверждениях; "
    "0 - сносок нет совсем или они указывают не на тот фрагмент. Если ответ - отказ "
    "от ответа (фактов нет), оценивай citation_accuracy: 2, если отказ честный, "
    "0 - если ответ выдаёт себя за содержательный без сносок.\n"
    "coverage: 2 - ответ покрывает все существенные для вопроса факты из фрагментов; "
    "1 - покрыта только часть; 0 - вопрос фактически не отвечен.\n"
    "russian_quality: 2 - грамотный связный русский; 1 - заметные шероховатости; "
    "0 - несвязный или нерусский текст.\n"
    "Ответь ТОЛЬКО JSON без markdown: "
    '{"faithfulness": 0, "citation_accuracy": 0, "coverage": 0, "russian_quality": 0, '
    '"comment": "кратко"}'
)


def build_judge_messages(
    question: str,
    fragments: list[dict],
    answer: str,
    cited: list[int],
    expected_nodes: list[str],
) -> list[dict]:
    frag_block = "\n\n".join(
        f"[{f['n']}] ({Path(f.get('source_id', '') or f.get('document_id', '')).name}) {f['text']}"
        for f in fragments
    )
    expected = ", ".join(expected_nodes) if expected_nodes else "(не заданы)"
    user = (
        f"Вопрос: {question}\n\n"
        f"Фрагменты базы знаний:\n{frag_block}\n\n"
        f"Ответ модели:\n{answer}\n\n"
        f"Сноски, указанные в ответе: {cited if cited else 'нет'}\n"
        f"Ожидаемые узлы (контекст автора вопроса; оценивай строго по фрагментам): {expected}\n\n"
        "Выставь оценки по рубрике и ответь только JSON."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def judge_answers_file(
    path: Path,
    golden_by_qid: dict,
    frag_by_qid: dict,
    base_url: str,
    model: str,
    api_key: str,
    max_tokens: int,
    limit: int = 0,
) -> tuple[dict, list[dict]]:
    """Судить один answers-файл; возврат (row агрегата, judged-записи)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload["meta"]
    results = payload["results"]
    if limit:
        results = results[:limit]
    judged = []
    zai_calls = 0
    for i, r in enumerate(results, 1):
        entry = golden_by_qid.get(r["qid"], {})
        frags = frag_by_qid.get(r["qid"], {}).get("fragments", [])
        out = dict(r)
        out["model"] = meta["model"]
        if r.get("error") and not r.get("answer"):
            out.update({k: None for k in METRIC_KEYS})
            out["judge_comment"] = f"ошибка прогона: {r['error'][:120]}"
        elif r.get("abstain"):
            score = abstain_score(r.get("answer", ""))
            out["abstain_ok"] = score == 1
            out["judge_comment"] = "механика abstain (без вызова судьи)"
            out["abstain_score"] = score
        else:
            cited = extract_cited_nums(r.get("answer", ""))
            messages = build_judge_messages(
                r["question"], frags, r.get("answer", ""), cited,
                entry.get("expected_nodes", []),
            )
            zai_calls += 1
            t0 = time.time()
            try:
                resp = chat_completion(
                    base_url, model, messages, api_key=api_key, max_tokens=max_tokens
                )
                scores = parse_judge_json(resp["content"])
            except Exception as e:  # noqa: BLE001
                scores = None
                out["judge_comment"] = f"judge error: {type(e).__name__}: {e}"[:200]
            out["judge_latency_s"] = round(time.time() - t0, 2)
            if scores:
                for k in METRIC_KEYS:
                    out[k] = scores[k]
                out["judge_comment"] = str(scores.get("comment", ""))[:300]
            else:
                for k in METRIC_KEYS:
                    out.setdefault(k, None)
                out["judge_comment"] = out.get("judge_comment") or "parse_error судьи"
        judged.append(out)
        if i % 10 == 0 or i == len(results):
            print(f"  {path.name}: {i}/{len(results)} (судья вызовов: {zai_calls})", flush=True)
    row = aggregate(judged)
    row["model"] = meta["model"]
    row["answers_file"] = path.name
    return row, judged


def main() -> int:
    ap = argparse.ArgumentParser(description="K11 judge + таблица")
    ap.add_argument("--answers", required=True, nargs="+", help="answers-JSON файлы прогонов")
    ap.add_argument("--golden", default="golden.jsonl")
    ap.add_argument("--retrieval-cache", default="retrieval-cache.json")
    ap.add_argument("--out", default="judged.json")
    ap.add_argument("--md", default="", help="куда положить markdown-таблицу")
    ap.add_argument("--secrets-file", default=str(Path.home() / "secrets" / "veles-facade.env"))
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=0, help="судить только первые N ответов каждого файла")
    args = ap.parse_args()

    secrets = load_env_file(args.secrets_file)
    base_url = secrets.get("VELES_LLM_BASE_URL", "")
    model = secrets.get("VELES_LLM_MODEL", "")
    api_key = secrets.get("VELES_LLM_API_KEY", "")
    if not (base_url and model and api_key):
        print("FAIL: судья требует VELES_LLM_* из secrets-file", file=sys.stderr)
        return 1

    golden_by_qid = {e["id"]: e for e in load_jsonl(args.golden)}
    cache = json.loads(Path(args.retrieval_cache).read_text(encoding="utf-8"))
    frag_by_qid = cache["questions"]

    rows, all_judged = [], []
    for p in args.answers:
        print(f"сужу {p}...", flush=True)
        row, judged = judge_answers_file(
            Path(p), golden_by_qid, frag_by_qid, base_url, model, api_key,
            args.max_tokens, args.limit,
        )
        rows.append(row)
        all_judged.append({"answers_file": p.name, "row": row, "judged": judged})

    table = render_table(rows)
    out = Path(args.out)
    out.write_text(
        json.dumps({"meta": {"judge_model": model}, "files": all_judged}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(table, flush=True)
    if args.md:
        Path(args.md).write_text(table, encoding="utf-8")
    print(f"судьи вердикты: {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
