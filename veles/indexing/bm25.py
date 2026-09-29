"""BM25 (Okapi, k1/b по канону) поверх простой русской токенизации."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Строчные токены; для среза без стемминга (калибруется в Ф2)."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class Bm25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_lens: list[int] = []
        self.tfs: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0

    def add(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        for nid, text in zip(ids, texts):
            tokens = tokenize(text)
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            for term in tf:
                self.df[term] = self.df.get(term, 0) + 1
            self.doc_lens.append(len(tokens))
            self.tfs.append(tf)
            # ids дописывается ПОСЛЕДНИМ в итерации (K4): search в другом потоке
            # берёт n_docs = len(ids) и индексирует tfs/doc_lens по i < n_docs -
            # безопасно читать индекс, пока воркер очереди загрузок его дополняет
            self.ids.append(nid)
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q_terms = tokenize(query)
        n_docs = len(self.ids)
        scores: list[tuple[str, float]] = []
        for i in range(n_docs):
            score = 0.0
            for term in q_terms:
                tf = self.tfs[i].get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_lens[i] / (self.avgdl or 1))
                score += idf * tf * (self.k1 + 1) / denom
            if score > 0:
                scores.append((self.ids[i], score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "k1": self.k1,
            "b": self.b,
            "ids": self.ids,
            "doc_lens": self.doc_lens,
            "tfs": self.tfs,
            "df": self.df,
            "avgdl": self.avgdl,
        }
        (dir_path / "bm25.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
