#!/usr/bin/env python
"""K14: ремап устаревших expected-узлов golden на новое поколение узлов.

Методика K12 («пересобрать golden-expected по индексу»): expected узел,
отсутствующий в новом индексе, заменяется на узел ТОГО ЖЕ документа с тем же
нормализованным текстом (легаси-поколение двойной индексации -> текущее).
Без совпадения expected остаётся как есть (честный промах) и фиксируется."""
import json
import unicodedata
from pathlib import Path

DATA = Path("<server-home>/proj/veles-bench/veles/data")
SB = Path("<server-home>/proj/veles-bench/veles/evals/synth-bench")
NEW = json.load(open(DATA / "index" / "meta.json"))["node_meta"]
OLD = json.load(open(DATA / "index-backup-20260929-2137" / "meta.json"))["node_meta"]

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower().replace("ё", "е")
    return " ".join(s.split())

# индекс нового индекса: (document_id, normtext) -> [version_ids]
by_doc_text = {}
for vid, m in NEW.items():
    by_doc_text.setdefault((m.get("document_id"), norm(m.get("text") or "")), []).append(vid)

golden = [json.loads(l) for l in open(SB / "golden.jsonl", encoding="utf-8")]
out, remapped, unmatched = [], 0, []
for e in golden:
    exp, notes = [], []
    for vid in e["expected_nodes"]:
        if vid in NEW:
            exp.append(vid)
            continue
        old_m = OLD.get(vid) or {}
        key = (old_m.get("document_id"), norm(old_m.get("text") or ""))
        cands = by_doc_text.get(key, [])
        if cands:
            exp.extend(sorted(cands))
            remapped += 1
            notes.append(f"remap:{vid[:8]}->{'|'.join(v[:8] for v in sorted(cands))}")
        else:
            exp.append(vid)  # остаётся - честный промах
            unmatched.append((e["id"], vid[:12]))
    e2 = dict(e)
    e2["expected_nodes"] = sorted(set(exp))
    if notes:
        e2["note"] = (e.get("note") or "") + " | K14 ремап: " + "; ".join(notes)
    out.append(e2)

with open(SB / "golden-k14.jsonl", "w", encoding="utf-8") as f:
    for e in out:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f"golden: {len(out)} записей; ремапнуто expected-узлов: {remapped}; "
      f"не найдено наследников: {len(unmatched)} {unmatched}")
print("артефакт:", SB / "golden-k14.jsonl")
