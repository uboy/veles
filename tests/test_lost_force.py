"""K18: статусы «утратил силу» - метка узла при индексации + плашка в цитатах.

Бриф docs/plan/briefs/2026-09-30-k18-lost-force-status.md: детектор по тексту
узла пишется в node_meta (build_index И инкрементальный upsert_document);
флаг не меняет ранжирование и не скрывает узел; цитаты JSON несут lost_force;
рендеры обоих клиентов ставят плашку «⚠ Норма утратила силу»; старый индекс
без поля в мете читается без ошибок (lost_force=False, без плашки).
Негативы падают по своей причине: отсутствие плашки у обычного узла и
работоспособность старой меты проверяются отдельными assert'ами.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app, render_chat_content
from veles.canonical.store import save_document
from veles.config import EMBED_DIM
from veles.ingest.queue import upsert_document
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import IndexBundle, build_index, is_lost_force
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import (
    LOST_FORCE_PREFIX,
    hybrid_search,
    lost_force_prefix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clients" / "telegram-bridge"))

from tg_format import (  # noqa: E402
    LOST_FORCE_PLATE,
    format_markdown_to_html,
    lost_force_prefix as tg_lost_force_prefix,
)

# dim = EMBED_DIM: как в test_documents_ingest (валидация модели при load_index)
HASH_DIM = EMBED_DIM
_TOKEN_RE = __import__("re").compile(r"[а-яёa-z0-9]+")


class HashingEmbedder:
    """Детерминированный стаб эмбеддера (канон: тесты без загрузки моделей)."""

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        import hashlib

        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


# --------------------------------------------------------------- детектор форм


@pytest.mark.parametrize(
    "text",
    [
        "Часть вторая утратила силу с 2 июля 2021 г. - ФЗ от 01.07.2021 N 294-ФЗ.",
        "Положение утратило силу.",
        "Норма УТРАТИЛА СИЛУ с 1 января 2020 года",
        "Пункты 3-5 УТРАТИЛИ СИЛУ",
        "Признан утратившим силу приказом от 12.03.2023 N 15.",
        "Закон, утратившие силу в связи с изданием настоящего акта",
        "Абзац не подлежит применению в этой части",
        "НЕ ПОДЛЕЖИТ ПРИМЕНЕНИЮ",
    ],
)
def test_detector_positive_forms(text: str) -> None:
    assert is_lost_force(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Порог единственной закупки составляет 350 тысяч рублей.",
        "Норма действует в редакции от 2024 года.",
        "Положение не утратило силу и действует в текущей редакции.",
        "Норма не утратил силу - продолжает применяться.",
        "Список норм, не утративших силу, приведён в приложении.",
        "",
    ],
)
def test_detector_negative_forms(text: str) -> None:
    assert is_lost_force(text) is False


def test_detector_is_content_level_not_junk_gate() -> None:
    """K18 не путать с K12: контентный узел с фразой матчится, пустышка-заглушка
    тоже метчится детектором, но из выдачи её режет отдельный is_junk_node."""
    from veles.retrieval.retriever import is_junk_node

    content = "Часть вторая утратила силу с 2 июля 2021 г."
    assert is_lost_force(content) and not is_junk_node(content)
    stub = "Утратил силу."
    assert is_lost_force(stub) and is_junk_node(stub)


# -------------------------------------------- индексация: build_index и upsert


LOST_DOC = """\
# Положение о командировках

## Суточные

Суточные составляют 700 рублей за каждый день командировки.

## Прекращение действия

Раздел 2 утратил силу с 1 января 2020 г. - приказ от 20.12.2019 N 55.
"""
NORMAL_DOC = """\
# Регламент закупок

Порог единственной закупки составляет 350 тысяч рублей.
"""


@pytest.fixture()
def mini_canon(tmp_path, monkeypatch):
    """tmp-канон из двух md: один с «утратил силу», один обычный."""
    canonical = tmp_path / "canonical"
    corpus = tmp_path / "corpus"
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    for name, text in (
        ("командировки.md", LOST_DOC),
        ("закупки.md", NORMAL_DOC),
    ):
        path = corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        save_document(TxtMdParserAdapter().parse_file(path, corpus_root=corpus), root=canonical)
    return canonical


def test_build_index_marks_lost_force_nodes(mini_canon) -> None:
    """Позитив: документ с «утратил силу» получает lost_force в node_meta при
    индексации, обычный документ - False."""
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    flagged = {m["title"]: m["lost_force"] for m in bundle.node_meta.values()}
    командировки = [v for k, v in flagged.items() if "командиров" in k.lower()]
    закупки = [v for k, v in flagged.items() if "закуп" in k.lower()]
    assert командировки and any(командировки), flagged  # метка стоит
    assert закупки and not any(закупки), flagged  # обычный - без метки
    # у суточных-узла метки нет: детектор по тексту узла, не документа
    суточные = [
        m["lost_force"] for m in bundle.node_meta.values()
        if "Суточные" in m["text"]
    ]
    assert суточные == [False]


def test_upsert_document_marks_lost_force(tmp_path) -> None:
    """Инкрементальная загрузка (путь K4) ставит ту же метку без пересборки."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    normal = corpus / "закупки.md"
    normal.write_text(NORMAL_DOC, encoding="utf-8", newline="\n")
    doc = TxtMdParserAdapter().parse_file(normal, corpus_root=corpus)
    bundle = IndexBundle(vector=NumpyFlatStore(dim=HASH_DIM), bm25=Bm25Index())
    from veles.canonical.store import IngestOutcome

    upsert_document(bundle, doc, IngestOutcome("created", "r1"), HashingEmbedder())
    assert all(m["lost_force"] is False for m in bundle.node_meta.values())

    lost = corpus / "положение_утратил.md"
    lost.write_text(LOST_DOC, encoding="utf-8", newline="\n")
    doc2 = TxtMdParserAdapter().parse_file(lost, corpus_root=corpus)
    upsert_document(bundle, doc2, IngestOutcome("created", "r1"), HashingEmbedder())
    new_meta = [
        m for m in bundle.node_meta.values() if m["document_id"] == doc2.document_id
    ]
    assert new_meta
    assert any(m["lost_force"] for m in new_meta)  # метка ставится сразу


# ----------------------------------------- retriever: флаг и нейтральность RRF


def make_meta(lost_force: bool | None) -> dict:
    """Мета двух узлов в духе test_api; lost_force=None = старый индекс без поля."""
    def node(text: str) -> dict:
        m = {
            "text": text,
            "access_groups": ["personal"],
            "title": "Пересказ Общий менеджмент",
            "document_id": "lectures/Общий менеджмент/лекция2/пересказ.txt",
            "heading_path": ["Общий менеджмент"],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "пересказ.txt", "lines": [82, 82]},
        }
        return m

    meta = {
        "n1": node("Норма управляемости руководителя - от 5 до 9 пунктов."),
        "n2": node("Коучинг работает с целями и мотивацией."),
    }
    if lost_force is not None:
        meta["n1"]["lost_force"] = lost_force
    return meta


class FakeVector:
    def __init__(self) -> None:
        self.ids = [("n1", 0.95), ("n2", 0.3)]

    def search(self, q, top_k=10):
        return self.ids


class FakeBm25:
    def search(self, q, top_k=10):
        return [("n1", 5.0), ("n2", 0.4)]


def run_search(meta: dict):
    bundle = IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=meta)
    qvec = np.zeros(4, dtype=np.float32)
    return hybrid_search(bundle, qvec, "норма управляемости", top_k=5)


def test_hybrid_search_flag_without_rank_change() -> None:
    """Метка lost_force НЕ меняет ни порядок, ни скоры RRF - меняется только
    флаг цитаты (research: показывать с меткой, не прятать)."""
    with_flag = run_search(make_meta(True))
    without_flag = run_search(make_meta(False))
    assert [e.node_version_id for e in with_flag] == ["n1", "n2"]
    assert [(e.rrf, e.rrf_base, e.dense_rank, e.bm25_rank) for e in with_flag] == [
        (e.rrf, e.rrf_base, e.dense_rank, e.bm25_rank) for e in without_flag
    ]
    assert with_flag[0].citations[0].lost_force is True
    assert without_flag[0].citations[0].lost_force is False


def test_hybrid_search_old_index_meta_without_key() -> None:
    """Негатив-совместимость: мета старого индекса без поля lost_force не падает
    и даёт lost_force=False (узлы считаются действующими)."""
    evidence = run_search(make_meta(None))
    assert evidence, "поиск на старой мете обязан работать"
    assert all(e.citations[0].lost_force is False for e in evidence)


def test_lost_force_prefix_helper() -> None:
    assert lost_force_prefix(True) == f"{LOST_FORCE_PREFIX} - "
    assert lost_force_prefix(False) == ""


# ------------------------------------------------------------------ API/рендер


def make_pipeline(lost_force: bool | None) -> VelesPipeline:
    bundle = IndexBundle(vector=FakeVector(), bm25=FakeBm25(), node_meta=make_meta(lost_force))
    return VelesPipeline(bundle, _ZeroEmbedder(), llm=None)


class _ZeroEmbedder:
    dim = 4

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


def test_api_citations_carry_lost_force() -> None:
    """JSON цитат несёт lost_force: у меченого узла True, у обычного False;
    старые клиенты поле просто игнорируют (новое поле в существующем списке)."""
    app = create_app(pipeline=make_pipeline(True))
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "какая норма управляемости", "top_k": 3})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "answer"
    assert d["citations"][0]["lost_force"] is True

    app2 = create_app(pipeline=make_pipeline(False))
    d2 = TestClient(app2).post(
        "/v1/knowledge/query", json={"query": "какая норма управляемости", "top_k": 3}
    ).json()
    assert d2["citations"][0]["lost_force"] is False


def _rendered_content(lost_force: bool | None) -> str:
    app = create_app(pipeline=make_pipeline(lost_force))
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "veles-knowledge",
        "messages": [{"role": "user", "content": "какая норма управляемости"}],
    })
    assert r.status_code == 200
    return r.json()["choices"][0]["message"]["content"]


def test_chat_render_plate_before_flagged_citation() -> None:
    """Позитив рендера: перед цитатой lost_force-узла стоит плашка."""
    content = _rendered_content(True)
    sources = content.split("Источники:", 1)[1]
    lines = [ln for ln in sources.splitlines() if "[1]" in ln]
    assert lines, content
    assert lines[0].lstrip().startswith(f"{LOST_FORCE_PREFIX} - "), content


def test_chat_render_no_plate_for_normal_node() -> None:
    """Негатив рендера: у обычного узла плашки нет (падает по своей причине,
    если плашка ставится безусловна)."""
    content = _rendered_content(False)
    assert LOST_FORCE_PREFIX not in content
    assert "[1]" in content.split("Источники:", 1)[1]


def test_chat_render_old_index_meta_no_plate_no_crash() -> None:
    """Старый индекс (мета без поля): рендер работает, плашки нет."""
    content = _rendered_content(None)
    assert LOST_FORCE_PREFIX not in content
    assert "Источники:" in content


def test_render_chat_content_direct_without_links() -> None:
    """Рендер без ссылок (token=, нет document_id/node) - плашка на месте."""
    result = {
        "decision": "answer",
        "answer": "От 5 до 9 пунктов.",
        "citations": [
            {
                "n": 1,
                "title": "Пересказ",
                "source_id": "пересказ.txt",
                "lines": [82, 82],
                "heading_path": [],
                "document_id": "",
                "node_version_id": "",
                "lost_force": True,
            }
        ],
        "diagnostics": {},
    }
    content = render_chat_content(result)
    assert f"{LOST_FORCE_PREFIX} - [1]" in content


# -------------------------------------------------------- tg_format (мост)


def test_lost_force_plate_sync() -> None:
    """Единая плашка двух клиентов: контейнеры не делят код, от расползания
    держит этот тест - тексты обязаны совпадать байт в байт."""
    assert LOST_FORCE_PLATE == LOST_FORCE_PREFIX
    assert tg_lost_force_prefix(True) == lost_force_prefix(True)
    assert tg_lost_force_prefix(False) == ""


def test_tg_format_passes_plate_through_to_html() -> None:
    """Мост получает готовый markdown из render_chat_content: конвертер обязан
    донести плашку до Telegram-HTML без искажений (сноска [1] становится ссылкой)."""
    content = _rendered_content(True)
    html = format_markdown_to_html(content)
    assert LOST_FORCE_PLATE in html
    assert html.count(LOST_FORCE_PLATE) == content.count(LOST_FORCE_PLATE)
    # markdown-разбор прошёл (сноска стала ссылкой), плашка осталась обычным текстом
    assert '<a href="' in html
    assert f'>{LOST_FORCE_PLATE}<' not in html
