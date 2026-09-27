"""RRNCB-бенчмарк Ф2: flat baseline vs canonical, Recall@K/MRR, FAR-калибровка.

Датасет: FractalGPT/RRNCBPublic (200 QA: question/answer/document) + PDF-документы
(Google Drive RRNCB). Документы в knowledge-sources/06_benchmarks/rrncb/docs.

Ограничения честности (по ревью Ф1):
- gold-узел = локализация answer-строки в юните (извлекаемая выборка, покрытие
  считается и сообщается; метрики - на общем локализуемом подмножестве обеих конфигов);
- метрики retrieval (H2/H4 - гранулярность чанков и heading-префикс); эффект
  small-to-big расширения (H3) на полноту ответа требует LLM-суждения и в Ф2 не
  измеряется - фиксированный прогон expansion не влияет на ранжирование листьев.
"""

from __future__ import annotations

import csv
import io
import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from veles.canonical.model import CanonicalDocument
from veles.ingest.pdf_text import PyMuPdfParserAdapter

import os

RRNCB_DIR = Path(os.environ.get("VELES_RRNCB_DIR", r"<local-pc>\proj\knowledge-sources\06_benchmarks\rrncb"))
QA_CSV = RRNCB_DIR / "rrncb_public_dataset.csv"
DOCS_DIR = RRNCB_DIR / "docs"

_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.lower()).strip()


def load_qa() -> list[dict]:
    rows = list(csv.DictReader(io.open(QA_CSV, encoding="utf-8")))
    return [r for r in rows if r["answer"].strip() and r["answer"].strip() != "-"]


_IDENT_RE = re.compile(
    r"(ГОСТ\s*Р?\s*[\d.]+(?:-\d{2,4})?|СП\s+[\d.]+|СНиП\s*[\d.]+(?:-\d{2,4})?|МДС\s+[\d.-]+|РД\s+[\d.-]+)",
    re.IGNORECASE,
)


def extract_identifiers(text: str) -> list[str]:
    """Номера нормативных документов в тексте вопроса ('ГОСТ 17.2.3.01-86' и т.п.)."""
    out = []
    for m in _IDENT_RE.finditer(text.upper().replace("Ё", "Е")):
        out.append(re.sub(r"\s+", " ", m.group(1)))
    return out


def corpus_identifier_set(docs: dict) -> set[str]:
    """Идентификаторы, присутствующие в корпусе: из названий и заголовков узлов."""
    found = set()
    for name, doc in docs.items():
        found |= {i.upper() for i in extract_identifiers(name)}
        for n in doc.nodes[:400]:
            found |= {i.upper() for i in extract_identifiers(n.text)}
    return found


@dataclass
class Unit:
    uid: str
    text: str
    logical: Optional[str] = None  # logical_node_id для small-to-big расширения


@dataclass
class ConfigIndex:
    name: str
    units: list[Unit] = field(default_factory=list)
    bundle: Optional[object] = None


def flat_units(doc: CanonicalDocument, size: int = 800, overlap: int = 100) -> list[Unit]:
    """Базлайн: фиксированные окна по полному тексту документа, без иерархии."""
    full = "\n".join(n.text for n in doc.nodes if n.node_type != "root")
    units = []
    step = size - overlap
    for i, start in enumerate(range(0, len(full), step)):
        chunk = full[start : start + size].strip()
        if len(chunk) < 40:
            break
        units.append(Unit(uid=f"flat:{doc.document_id}:{i:05d}", text=chunk))
    return units


def canonical_units(doc: CanonicalDocument) -> list[Unit]:
    return [
        Unit(uid=n.node_version_id, text=n.text, logical=n.logical_node_id)
        for n in doc.nodes
        if n.node_type == "paragraph"
    ]


def lcs_gold(
    units: list[Unit], answer: str, norm_cache: list[str], tok_index: dict[str, set[int]],
    min_chars: int = 40,
) -> Optional[set[str]]:
    """Вербатим-LCS gold (ревью-2 P0 fix): юниты с непрерывным общим отрезком
    >= min_chars с ответом. Префильтр кандидатов по редким словам, затем
    SequenceMatcher. Gold-сеты крошечные (~1-3 юнита) и одинаковой ширины между
    конфигурациями - база случайного попадания исчезает (эксперимент 27.09)."""
    na = norm_text(answer)
    fw = rare_tokens(answer)
    rare = sorted(fw, key=lambda w: len(tok_index.get(w, set())))[:6]
    cand: set[int] = set()
    for w in rare:
        cand |= tok_index.get(w, set())
    gold: set[str] = set()
    for i in cand:
        sm = SequenceMatcher(None, na, norm_cache[i], autojunk=True)
        m = sm.find_longest_match(0, len(na), 0, len(norm_cache[i]))
        if m.size >= min_chars:
            gold.add(units[i].uid)
    return gold or None


def rare_tokens(text: str) -> set[str]:
    return {w for w in norm_text(text).split() if len(w) > 4}


def norm_text(t: str) -> str:
    t = re.sub(r"[^а-яёa-z0-9 ]", " ", t.lower().replace("ё", "е"))
    return re.sub(r"\s+", " ", t).strip()


def docling_structural_units(doc: CanonicalDocument, cap: int = 1800) -> list[Unit]:
    """Секционные юниты Ф3: секция целиком (заголовок + абзацы/таблицы) - подобие
    статьи/пункта нормативного документа. Длинные секции режутся по cap с суффиксом."""

    def split_len(text: str) -> list[str]:
        if len(text) <= cap:
            return [text]
        parts, buf = [], ""
        for sent in re.split(r"(?<=[.!?…])\s+", text):
            if buf and len(buf) + len(sent) + 1 > cap:
                parts.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            parts.append(buf)
        return parts

    by_parent: dict[str, list] = {}
    for n in doc.nodes:
        if n.node_type == "paragraph":
            by_parent.setdefault(n.parent_id, []).append(n)

    units: list[Unit] = []
    for n in doc.nodes:
        if n.node_type != "section":
            continue
        children = by_parent.get(n.logical_node_id, [])
        body = "\n".join([n.text] + [c.text for c in children]).strip()
        if len(body) < 60:
            continue
        for k, part in enumerate(split_len(body)):
            suffix = f":part{k + 1}" if k else ""
            units.append(Unit(uid=f"sec:{n.logical_node_id}{suffix}", text=part, logical=n.logical_node_id))
    return units


def _grams_tokens(text: str) -> tuple[set[str], set[str]]:
    t = normalize(text).replace("ё", "е")
    t = re.sub(r"[^a-zа-я0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    words = [w for w in t.split() if len(w) > 4]
    grams = {t[i : i + 3] for i in range(max(len(t) - 2, 0))}
    return grams, set(words)


def localize_gold_best(
    units: list[Unit], answer: str, gram_cache: Optional[list] = None
) -> tuple[Optional[str], Optional[str], float]:
    """(gold_uid_set, best_fragment, best_score): все узлы с покрытием >=0.55 хотя бы
    одним фрагментом + лучший фрагмент (для проверки покрытия evidence-pack'ом)."""
    ans = " ".join(answer.split())
    words = ans.split(" ")
    fragments = []
    step, size = 6, 14
    for i in range(0, max(len(words) - size, 1), step):
        fragments.append(" ".join(words[i : i + size]))
    if not fragments:
        fragments = [ans]

    if gram_cache is None:
        gram_cache = [_grams_tokens(u.text) for u in units]

    unit_scores: dict[str, float] = {}
    best_frag, best_frag_score = None, 0.0
    for frag in fragments:
        fg, fw = _grams_tokens(frag)
        if len(fg) < 10:
            continue
        rare = sorted(fw, key=lambda w: sum(1 for g, wset in gram_cache if w in wset))[:6]
        cand = set()
        for w in rare:
            cand |= {i for i, (g, wset) in enumerate(gram_cache) if w in wset}
        for i in cand:
            ug = gram_cache[i][0]
            score = len(fg & ug) / max(len(fg), 1)
            uid = units[i].uid
            if score > unit_scores.get(uid, 0.0):
                unit_scores[uid] = score
            if score > best_frag_score:
                best_frag_score, best_frag = score, frag
    gold_set = {u for u, s in unit_scores.items() if s >= 0.55}
    return (gold_set or None), best_frag, best_frag_score


def localize_gold(
    units: list[Unit], answer: str, gram_cache: Optional[list] = None
) -> Optional[str]:
    gold_set, _, _ = localize_gold_best(units, answer, gram_cache)
    if not gold_set:
        return None
    return sorted(gold_set)[0]
