#!/usr/bin/env python3
"""K11 synth-bench runner: синтез ответов по замороженному retrieval (бриф K11).

Прогоняет одну модель по golden-набору НА ОДНИХ И ТЕХ ЖЕ фрагментах
(retrieval-cache.json из freeze_retrieval.py) и пишет сырые ответы + latency/tokens.
Промпт синтеза - В ТОЧНОСТИ фасадский: SYNTHESIS_SYSTEM импортируется из
veles.api.llm (не копируется), пользовательское сообщение собирается как в
OpenAICompatibleProvider.synthesize. Ответ модели = ТОЛЬКО message.content;
thinking-канал (reasoning_content у GLM/gemma) ответом не считается.

Провайдеры:
  --provider glm   - env фасада из --secrets-file (стенд ~/secrets/veles-facade.env):
                     base_url/модель/ключ берутся те же, что у прода;
  --provider local - OpenAI-совместимый URL без ключа: --base-url http://...:8471/v1
                     --model qwen3.8-27b (llama.cpp на V100).

Запуск (стенд, из каталога с кэшем):
  VELES_REPO=$HOME/proj/veles-bench/veles PYTHONUNBUFFERED=1 ~/veles-venv/bin/python \
  e2e/run_synth_bench.py --provider glm --golden golden.jsonl \
      --retrieval-cache retrieval-cache.json --out answers-glm.json [--limit 5]

Gemma: держите --max-tokens >= 2048 (thinking-канал съедает бюджет).
Exit codes: 0 - ок; 1 - ошибка конфигурации.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

VELES_REPO = Path(
    os.environ.get("VELES_REPO", str(Path.home() / "proj" / "veles-bench" / "veles"))
)
sys.path.insert(0, str(VELES_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals" / "synth-bench"))

from synth_lib import build_synthesis_messages, chat_completion, load_env_file, load_jsonl  # noqa: E402

# Промпт синтеза - единственный источник истины фасад (прод-чекаут с тем же main).
from veles.api.llm import SYNTHESIS_SYSTEM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="K11 synth-bench runner")
    ap.add_argument("--provider", required=True, choices=["glm", "local"])
    ap.add_argument("--base-url", default="", help="URL /v1 для --provider local")
    ap.add_argument("--model", default="", help="имя модели для --provider local")
    ap.add_argument("--secrets-file", default=str(Path.home() / "secrets" / "veles-facade.env"))
    ap.add_argument("--golden", default="golden.jsonl")
    ap.add_argument("--retrieval-cache", default="retrieval-cache.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="только первые N вопросов (0 = все)")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.2, help="как у фасада")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    if args.provider == "glm":
        secrets = load_env_file(args.secrets_file)
        base_url = secrets.get("VELES_LLM_BASE_URL", "")
        model = secrets.get("VELES_LLM_MODEL", "")
        api_key = secrets.get("VELES_LLM_API_KEY", "")
        if not (base_url and model and api_key):
            print("FAIL: в secrets-file нет VELES_LLM_BASE_URL/MODEL/API_KEY", file=sys.stderr)
            return 1
    else:
        base_url, model, api_key = args.base_url, args.model, ""
        if not (base_url and model):
            print("FAIL: для --provider local нужны --base-url и --model", file=sys.stderr)
            return 1

    golden = load_jsonl(args.golden)
    cache = json.loads(Path(args.retrieval_cache).read_text(encoding="utf-8"))
    frag_by_qid = cache["questions"]
    if args.limit:
        golden = golden[: args.limit]

    print(
        f"провайдер: {base_url} model={model}; вопросов: {len(golden)}; "
        f"SYNTHESIS_SYSTEM: {len(SYNTHESIS_SYSTEM)} знаков (импортирован из veles.api.llm)",
        flush=True,
    )
    results = []
    for i, entry in enumerate(golden, 1):
        qid = entry["id"]
        fr = frag_by_qid.get(qid)
        if not fr:
            print(f"[{i}/{len(golden)}] {qid}: НЕТ в кэше retrieval - пропуск", flush=True)
            continue
        fragments = [f["text"] for f in fr["fragments"]]
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            *build_synthesis_messages(entry["question"], fragments),
        ]
        err = ""
        t0 = time.time()
        try:
            resp = chat_completion(
                base_url,
                model,
                messages,
                api_key=api_key,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
        except Exception as e:  # noqa: BLE001 - ошибка одного вопроса не роняет прогон
            resp = {"content": "", "reasoning_present": False, "finish_reason": "error",
                    "prompt_tokens": None, "completion_tokens": None, "model": model}
            err = f"{type(e).__name__}: {e}"[:300]
        latency = round(time.time() - t0, 2)
        results.append(
            {
                "qid": qid,
                "question": entry["question"],
                "abstain": bool(entry.get("abstain")),
                "provenance": entry.get("provenance", ""),
                "answer": resp["content"],
                "reasoning_present": resp["reasoning_present"],
                "finish_reason": resp["finish_reason"],
                "latency_s": latency,
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "error": err,
            }
        )
        flag = " ERR " if err else "  ok "
        print(
            f"[{i}/{len(golden)}]{flag}{qid} abst={entry.get('abstain', False)} "
            f"lat={latency:.1f}s len={len(resp['content'])} "
            f"finish={resp['finish_reason']}" + (f" {err}" if err else ""),
            flush=True,
        )

    payload = {
        "meta": {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": args.provider,
            "base_url": base_url,
            "model": model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "golden": str(args.golden),
            "retrieval_cache": str(args.retrieval_cache),
            "system_prompt_chars": len(SYNTHESIS_SYSTEM),
            "n_questions": len(results),
            "n_errors": sum(1 for r in results if r["error"]),
            "api_key": "(из secrets-file, не печатается)" if api_key else "(не нужен)",
        },
        "results": results,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ответы: {out}; ошибок: {payload['meta']['n_errors']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
