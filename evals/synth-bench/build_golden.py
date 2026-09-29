#!/usr/bin/env python3
"""K11: сборка golden-набора synth-bench (запуск НА стенд, venv python).

Источники вопросов (бриф docs/plan/briefs/2026-09-29-k11-synth-bench.md):
  1. RRNCB QA (csv на стенд) - самопроверка: нормализованный ответ имеет общий
     кусок >= 40 знаков с одним из top-5 фрагментов (по духу LCS-gold из bench/rrncb.py);
  2. мини-QA Вышка (bench/miniqa-vyshka-module1-v1.json) - самопроверка: в top-5
     есть узел документа expect_doc_hint;
  3. GLM-генерация по узлам прод-канона - самопроверка: узел-источник попал в
     top-5 retrieval (иначе вопрос выбраковывается);
  4. живые вопросы владельца (provenance=manual) - НЕ выбраковываются автоматом,
     expected_nodes ищутся подстрокой в индексе, результат проверки фиксируется;
  5. abstain-вопросы (узкие темы других доменов, ответа в корпусе нет) - manual,
     retrieval-проверки не проходят (они и должны промахиваться).

Индекс грузится in-process read-only (load_index, НИКАКИХ save) против прод-данных:
VELES_DATA берётся из конфига veles-чекаута (env VELES_REPO, по умолчанию - prod
~/proj/veles-bench/veles на стенд). CUDA скрыт: эмбеддинг запроса на CPU.

Запуск (стенд):
  VELES_REPO=$HOME/proj/veles-bench/vels CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
  ~/veles-venv/bin/python build_golden.py --secrets-file ~/secrets/veles-facade.env \
      --rrncb-csv ~/proj/veles-bench/rrncb/rrncb_public_dataset.csv \
      --miniqa bench/miniqa-vyshka-module1-v1.json --out golden.jsonl
Флаги --skip-glm: только детерминированные источники + abstain (без вызовов z.ai).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # synth_lib рядом
from synth_lib import chat_completion, load_env_file  # noqa: E402

# --- bootstrap импорта veles из чекаута (env VELES_REPO или prod-путь стенд) ---
VELES_REPO = Path(
    os.environ.get("VELES_REPO", str(Path.home() / "proj" / "veles-bench" / "veles"))
)
sys.path.insert(0, str(VELES_REPO))

from veles.config import EMBED_DIM, EMBED_MODEL, INDEX_DIR  # noqa: E402
from veles.indexing.embedder import SentenceTransformersProvider  # noqa: E402
from veles.indexing.indexer import load_index  # noqa: E402
from veles.retrieval.retriever import hybrid_search  # noqa: E402

TOP_K = 5
LCS_MIN_CHARS = 40
GLM_MAX_TOKENS = 3072

# Живые вопросы владельца (заказ K11): соосность блоков, транспортный налог,
# управление производством. expected_nodes ищем подстрокой в индексе; у вопросов
# без находки expected_nodes=[] и честная пометка в note.
MANUAL_QUESTIONS = [
    {
        "question": "Как проверяют соосность рядов укладываемых блоков фундамента?",
        "probe": "Соосность рядов укладываемых блоков",
    },
    {
        "question": "Кто является плательщиком транспортного налога?",
        "probe": "транспортный налог",
    },
    {
        "question": "Что такое норма управляемости и как она влияет на структуру организации?",
        "probe": "Норма управляемости",
    },
]

# Abstain-кейсы: узкие темы ДРУГИХ доменов, ответа в прод-корпусе нет
# (стиль формулировок - деловой, чтобы retrieval давал правдоподобные, но
# нерелевантные фрагменты; это и есть проверка честного отказа).
ABSTAIN_QUESTIONS = [
    "Как приготовить суши-рис для роллов?",
    "Какие упражнения рекомендуют при остеохондрозе поясничного отдела?",
    "Каковы правила розыгрыша в го и как считается территория?",
    "Как часто нужно пересаживать орхидею фаленопсис и какой субстрат выбрать?",
    "В каком году пал Константинополь и какая империя его контролировала после этого?",
    "Как настроить Ingress-контроллер в Kubernetes для внешнего трафика?",
    "Как реализовать конкурентные запросы через asyncio в Python?",
    "Сколько стоит проездной на месяц в московском метро?",
    "Чем кормить гуппи и как обустроить аквариум для их разведения?",
    "Как исправить испорченную кредитную историю перед подачей заявки на ипотеку?",
    "Нужна ли виза в Японию для россиян и какие документы подавать?",
    "Какие маршруты походов по Алтаю подходят новичкам в сентябре?",
    "Как вырастить авокадо из косточки в домашних условиях?",
    "Какой сварочный аппарат выбрать для дачных работ и какие меры безопасности соблюдать?",
    "Как приготовить борщ по классическому рецепту?",
]

# Промпт генерации вопроса по узлу канона.
GEN_SYSTEM = (
    "Ты готовишь golden-набор для оценки RAG-ассистента по базе знаний. "
    "Придумай ОДИН конкретный фактический вопрос по приведённому фрагменту документа. "
    "Вопрос должен однозначно отвечаться содержимым фрагмента, быть на русском языке, "
    "не ссылаться на 'фрагмент'/'документ' и не быть да/нет-вопросом. "
    "Ответь строго JSON: {\"question\": \"...\", \"facts\": [\"ключевой факт 1\", ...]}"
)


def norm_text(s: str) -> str:
    """Нормализация для LCS-самопроверки (в духе bench/rrncb.py): lower, Ё->Е, пробелы."""
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", s).strip()


def lcs_hit(answer: str, texts: list[str]) -> tuple[bool, int]:
    """Общий кусок answer и любого из texts >= LCS_MIN_CHARS; возврат (hit, idx)."""
    na = norm_text(answer)
    if len(na) < LCS_MIN_CHARS:
        return False, -1
    for i, t in enumerate(texts):
        nt = norm_text(t)
        m = SequenceMatcher(None, na, nt, autojunk=True).find_longest_match(
            0, len(na), 0, len(nt)
        )
        if m.size >= LCS_MIN_CHARS:
            return True, i
    return False, -1


def main() -> int:
    ap = argparse.ArgumentParser(description="K11 golden-set builder (стенд)")
    ap.add_argument("--secrets-file", default=str(Path.home() / "secrets" / "veles-facade.env"))
    ap.add_argument("--rrncb-csv", default=str(Path.home() / "proj" / "veles-bench" / "rrncb" / "rrncb_public_dataset.csv"))
    ap.add_argument("--miniqa", default="")
    ap.add_argument("--out", default="golden.jsonl")
    ap.add_argument("--rrncb-sample", type=int, default=30, help="кандидатов RRNCB к проверке")
    ap.add_argument("--rrncb-keep", type=int, default=15)
    ap.add_argument("--miniqa-keep", type=int, default=10)
    ap.add_argument("--glm-docs", type=int, default=22, help="документов для GLM-генерации")
    ap.add_argument("--glm-per-doc", type=int, default=2, help="узлов на документ")
    ap.add_argument("--glm-keep", type=int, default=37, help="максимум принятых GLM-вопросов")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--skip-glm", action="store_true", help="без вызовов z.ai (только генерация раскладки)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    secrets = load_env_file(args.secrets_file)
    base_url = secrets.get("VELES_LLM_BASE_URL", "")
    model = secrets.get("VELES_LLM_MODEL", "")
    api_key = secrets.get("VELES_LLM_API_KEY", "")
    if not args.skip_glm and not (base_url and model and api_key):
        print("FAIL: в secrets-file нет VELES_LLM_* (base_url/model/key)")
        return 1

    print(f"загрузка прод-индекса: {INDEX_DIR} (read-only)", flush=True)
    embedder = SentenceTransformersProvider(EMBED_MODEL, dim=EMBED_DIM)
    bundle = load_index(INDEX_DIR)  # ТОЛЬКО чтение; save() не вызывается никогда
    print(f"индекс: {len(bundle.node_meta)} узлов", flush=True)

    # document_id -> title; группировка узлов по документам
    docs: dict[str, dict] = {}
    for nid, meta in bundle.node_meta.items():
        d = meta["document_id"]
        docs.setdefault(d, {"title": meta["title"], "nodes": []})
        docs[d]["nodes"].append((nid, meta))

    def retrieve(question: str) -> list[dict]:
        vec = embedder.embed_query(question)
        ev = hybrid_search(bundle, vec, question, top_k=TOP_K)
        return [
            {
                "node_version_id": e.node_version_id,
                "document_id": e.citations[0].document_id,
                "title": e.citations[0].title,
                "text": e.text,
            }
            for e in ev
        ]

    golden: list[dict] = []
    qid_n = 0

    def add(entry: dict) -> None:
        nonlocal qid_n
        qid_n += 1
        entry = {"id": f"k11-{qid_n:03d}", "abstain": False, **entry}
        golden.append(entry)

    # ---- 1. RRNCB QA ------------------------------------------------------
    rrncb_rows = []
    with open(args.rrncb_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("answer", "").strip() and row["answer"].strip() != "-":
                rrncb_rows.append(row)
    # фиксированный корпус: только документы, присутствующие в прод-индексе
    idx_docs = set(docs)
    rrncb_rows = [r for r in rrncb_rows if r["document"] in idx_docs]
    sample = rng.sample(rrncb_rows, min(args.rrncb_sample, len(rrncb_rows)))
    kept_rrncb = 0
    rrncb_stat = {"cand": len(sample)}
    for row in sample:
        frags = retrieve(row["question"])
        hit, idx = lcs_hit(row["answer"], [fr["text"] for fr in frags])
        rrncb_stat.setdefault("selfcheck_hit", 0)
        if not hit or kept_rrncb >= args.rrncb_keep:
            continue
        rrncb_stat["selfcheck_hit"] += 1
        kept_rrncb += 1
        add(
            {
                "question": row["question"],
                "expected_nodes": [frags[idx]["node_version_id"]],
                "source_doc": row["document"],
                "provenance": "rrncb",
                "note": "RRNCB QA, самопроверка LCS>=40 против top-5",
            }
        )
    print(f"RRNCB: {rrncb_stat}, принято {kept_rrncb}", flush=True)

    # ---- 2. мини-QA Вышка -------------------------------------------------
    if args.miniqa and Path(args.miniqa).exists():
        items = json.loads(Path(args.miniqa).read_text(encoding="utf-8"))["items"]
        kept_mq = 0
        for it in items:
            if kept_mq >= args.miniqa_keep:
                break
            frags = retrieve(it["question"])
            hit = any(it.get("expect_doc_hint", "") in fr["document_id"] for fr in frags)
            if not hit:
                print(f"  miniqa отбракован {it['id']}: hint не в top-5", flush=True)
                continue
            kept_mq += 1
            add(
                {
                    "question": it["question"],
                    "expected_nodes": [fr["node_version_id"] for fr in frags
                                       if it.get("expect_doc_hint", "") in fr["document_id"]],
                    "source_doc": next(
                        (fr["document_id"] for fr in frags if it.get("expect_doc_hint", "") in fr["document_id"]), ""
                    ),
                    "provenance": "miniqa",
                    "note": f"мини-QA Вышка {it['id']}, самопроверка doc-hint в top-5",
                }
            )
        print(f"мини-QA: принято {kept_mq}/{len(items)}", flush=True)

    # ---- 3. GLM-генерация по узлам канона ---------------------------------
    if not args.skip_glm:
        # срез документов: НК РФ, стройка (docs/Data1_*), лекции-пересказы (без транскриптов)
        nk = [d for d in docs if "Налоговый кодекс" in docs[d]["title"]]
        lecture = [d for d in docs if d.startswith("lectures/") and "Пересказ" in docs[d]["title"]]
        constr = [
            d for d in docs
            if d.startswith("docs/") and "Налоговый кодекс" not in docs[d]["title"]
        ]
        picks = (
            nk
            + rng.sample(lecture, min(8, len(lecture)))
            + rng.sample(constr, min(max(args.glm_docs - len(nk) - 8, 0), len(constr)))
        )[: args.glm_docs]
        print(f"GLM-генерация: документов {len(picks)} (НК {len(nk)}, лекции {len(lecture)}, стройка {len(constr)})", flush=True)
        gen_stat = {"cand": 0, "json_bad": 0, "selfcheck_miss": 0, "kept": 0}
        for d in picks:
            if gen_stat["kept"] >= args.glm_keep:
                break
            # узлы-кандидаты: содержательные листья 300..1500 знаков
            cand = [
                (nid, meta)
                for nid, meta in docs[d]["nodes"]
                if 300 <= len(meta["text"]) <= 1500
                and meta["text"].lstrip()[:1].isalnum()
            ]
            if not cand:
                continue
            for nid, meta in rng.sample(cand, min(args.glm_per_doc, len(cand))):
                if gen_stat["kept"] >= args.glm_keep:
                    break
                gen_stat["cand"] += 1
                try:
                    resp = chat_completion(
                        base_url,
                        model,
                        [
                            {"role": "system", "content": GEN_SYSTEM},
                            {
                                "role": "user",
                                "content": f"Документ: {meta['title']}\n\nФрагмент:\n{meta['text'][:1800]}",
                            },
                        ],
                        api_key=api_key,
                        max_tokens=GLM_MAX_TOKENS,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  GLM error: {e}", flush=True)
                    continue
                text = resp["content"]
                m_start, m_end = text.find("{"), text.rfind("}")
                if m_start < 0 or m_end <= m_start:
                    gen_stat["json_bad"] += 1
                    continue
                try:
                    gen = json.loads(text[m_start : m_end + 1])
                    q = str(gen["question"]).strip()
                    facts = [str(x) for x in gen.get("facts", [])][:5]
                except Exception:  # noqa: BLE001 - любой брак генерации = выбраковка
                    gen_stat["json_bad"] += 1
                    continue
                if not (25 <= len(q) <= 220) or not q.endswith("?"):
                    gen_stat["json_bad"] += 1
                    continue
                frags = retrieve(q)
                if not any(fr["node_version_id"] == nid for fr in frags):
                    gen_stat["selfcheck_miss"] += 1
                    continue
                gen_stat["kept"] += 1
                add(
                    {
                        "question": q,
                        "expected_nodes": [nid],
                        "expected_facts": facts,
                        "source_doc": d,
                        "provenance": "auto",
                        "note": "GLM-генерация по узлу канона, самопроверка: узел в top-5",
                    }
                )
            print(f"  {Path(d).name[:40]}: всего принято {gen_stat['kept']}", flush=True)
        print(f"GLM-генерация: {gen_stat}", flush=True)

    # ---- 4. живые вопросы владельца (manual, не выбраковываются) ----------
    for mq in MANUAL_QUESTIONS:
        frags = retrieve(mq["question"])
        hit_nodes = [
            fr["node_version_id"] for fr in frags if mq["probe"].lower() in fr["text"].lower()
        ]
        add(
            {
                "question": mq["question"],
                "expected_nodes": hit_nodes,
                "source_doc": frags[0]["document_id"] if frags else "",
                "provenance": "manual",
                "note": "живой вопрос владельца; expected_nodes=подстрока в top-5 ("
                + (f"найдено {len(hit_nodes)}" if hit_nodes else "НЕ найдено - валидация владельцем")
                + ")",
            }
        )

    # ---- 5. abstain-кейсы --------------------------------------------------
    for q in ABSTAIN_QUESTIONS:
        add(
            {
                "question": q,
                "expected_nodes": [],
                "source_doc": "",
                "provenance": "manual",
                "note": "abstain: темы других доменов, ответа в прод-корпусе нет",
                "abstain": True,
            }
        )

    out = Path(args.out)
    out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in golden), encoding="utf-8"
    )
    n_abst = sum(1 for e in golden if e["abstain"])
    print(
        f"итог: {len(golden)} вопросов (answerable {len(golden) - n_abst}, abstain {n_abst}"
        f" = {100.0 * n_abst / len(golden):.0f}%) -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
