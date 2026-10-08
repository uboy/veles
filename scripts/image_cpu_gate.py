"""Гейт развертывания картинного канала: кодировщик запросов на CPU.

Меряет для модели «текст-картинка»: время загрузки, пиковую память процесса,
задержку кодирования одного запроса. Критерий плана: модель обязана работать
на хосте сервиса в доступной памяти (там свободно порядка 2 ГБ).
Запуск: python scripts/image_cpu_gate.py --model jinaai/jina-clip-v2
"""

from __future__ import annotations

import argparse
import json
import resource
import time

QUERIES = [
    "схема балансировщика нагрузки",
    "микросервисная архитектура диаграмма",
    "хеш-таблица схема",
    "жизненный цикл проекта",
    "шардирование базы данных",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jinaai/jina-clip-v2")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoModel

    t0 = time.time()
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()
    load_s = time.time() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    t1 = time.time()
    n = 0
    for _ in range(args.repeats):
        for q in QUERIES:
            model.encode_text([q])
            n += 1
    lat_ms = (time.time() - t1) / n * 1000
    print(
        json.dumps(
            {
                "model": args.model,
                "device": "cpu",
                "load_seconds": round(load_s, 1),
                "peak_rss_mb": round(rss_mb),
                "query_latency_ms": round(lat_ms),
                "queries": n,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
