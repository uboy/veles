"""In-process E2E полного пайплайна (Уровень 1, без сети и без загрузки моделей).

Отличие от unit-тестов Ф4 (ручной IndexBundle): здесь ЧЕСТНЫЙ сквозной цикл на
временной мини-корпусе - L0-файлы (.txt/.md с реальной структурой) -> TxtMdParserAdapter
-> canonical store (tmp) -> build_index (numpy-бэкенд, настоящий индексатор) ->
facade create_app -> TestClient. Ретривер, AnswerPolicy, ACL-фильтр и фасад работают
как в проде; единственная подмена - EmbeddingProvider.

Эмбеддер - детерминированный hashing (канон veles/AGENTS.md: "тесты без загрузки
моделей"): токен -> хеш-бакет bag-of-words с нормализацией. Это честный компонент E2E,
не фейковый retrieval: релевантность возникает из реального пересечения слов запроса
и фрагмента, и тесты падают, если цитаты указывают не на тот файл/строки.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.api.app import VelesPipeline, create_app, render_chat_content
from veles.api.llm import LLMConfig, OpenAICompatibleProvider
from veles.canonical.store import save_document
from veles.config import EMBED_MODEL
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.policy.identity import StaticOwnerResolver

HASH_DIM = 512
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

IN_CORPUS_QUERY = "какая норма управляемости руководителя"
OUT_CORPUS_QUERY = "как приготовить борщ"
GOST_QUERY = "требования ГОСТ 99999 к освещённости"
SECRET_QUERY = "зарплатная ведомость оклады"

# L0-мини-корпус: rel-путь -> (текст, restricted). Реальная структура: заголовки,
# нумерованные пункты. Слова из контрольных запросов не встречаются в других
# документах (иначе смысл теста размывается), служебные слова запросов ("как",
# "сколько") в корпусе отсутствуют вовсе.
ORG_MD = """\
# Организационная структура

## Норма управляемости

Норма управляемости руководителя составляет от 5 до 9 прямых подчинённых.
Превышение нормы управляемости делает команду неуправляемой.

## Принципы делегирования

Делегирование передаёт задачу вместе с полномочиями и ответственностью.
"""

SPRINT_TXT = """\
Регламент работы команды.

1. Спринт занимает ровно две недели.

2. Ежедневный дейли длится не более 15 минут.
"""

SALARY_MD = """\
# Зарплатная ведомость

Оклады сотрудников: значения сумм конфиденциальны.

## Порядок пересмотра окладов

Пересмотр окладов проводится ежегодно в апреле.
"""

COACHING_MD = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.

Модель GROW помогает структурировать беседу.
"""

CORPUS_FILES: dict[str, dict] = {
    "05_vyshka_academic/lectures/Модуль_1/Общий менеджмент/пересказ_оргструктура.md": {
        "text": ORG_MD,
    },
    "05_vyshka_academic/lectures/Модуль_1/Управление проектами/регламент_спринтов.txt": {
        "text": SPRINT_TXT,
    },
    "05_vyshka_academic/lectures/Модуль_1/HR/зарплатная_ведомость.md": {
        "text": SALARY_MD,
        "restricted": True,
    },
    "05_vyshka_academic/lectures/Модуль_1/Лидерство/пересказ_коучинг.md": {
        "text": COACHING_MD,
    },
}

ORG_DOC_REL = "05_vyshka_academic/lectures/Модуль_1/Общий менеджмент/пересказ_оргструктура.md"
SPRINT_DOC_REL = "05_vyshka_academic/lectures/Модуль_1/Управление проектами/регламент_спринтов.txt"
SALARY_DOC_REL = "05_vyshka_academic/lectures/Модуль_1/HR/зарплатная_ведомость.md"


class HashingEmbedder:
    """Детерминированный EmbeddingProvider-стаб: sublinear tf по хеш-бакетам слов.

    Тот же интерфейс, что SentenceTransformersProvider (dim + embed_passages +
    embed_query), нормализация как у e5 - cosine через inner product.
    """

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            # sublinear tf: повторные вхождения усиливают, но с насыщением
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture()
def mini_corpus(tmp_path, monkeypatch):
    """L0 -> L1 -> L2 на tmp: парсинг, канон, сборка индекса настоящим пайплайном."""
    corpus_root = tmp_path / "corpus"
    canonical = tmp_path / "canonical"
    parser = TxtMdParserAdapter()
    for rel, spec in CORPUS_FILES.items():
        path = corpus_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec["text"], encoding="utf-8", newline="\n")
        doc = parser.parse_file(path, corpus_root=corpus_root)
        if spec.get("restricted"):
            doc.access_groups = ["restricted"]
            for node in doc.nodes:
                node.access_groups = ["restricted"]
        save_document(doc, root=canonical)
    # build_index читает канон через CANONICAL_DIR - подменяем на tmp
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    return bundle, corpus_root


def make_client(bundle, *, token: str | None = None, groups: set[str] | None = None,
                llm: OpenAICompatibleProvider | None = None) -> TestClient:
    pipeline = VelesPipeline(bundle, HashingEmbedder(), llm)
    app = create_app(
        pipeline=pipeline,
        api_token=token,
        resolver=StaticOwnerResolver(groups if groups is not None else {"personal"}),
    )
    return TestClient(app)


def test_answer_in_corpus_citations_point_to_right_file(mini_corpus) -> None:
    """Сценарий 1: вопрос из корпуса -> answer, цитаты указывают правильный
    файл/строки/heading_path, chat-рендер содержит блок Источники."""
    bundle, _ = mini_corpus
    client = make_client(bundle)
    r = client.post("/v1/knowledge/query",
                    json={"query": IN_CORPUS_QUERY, "use_llm": False})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "answer"
    assert "от 5 до 9" in d["answer"]
    assert d["citations"], "у answer обязаны быть цитаты"
    top = d["citations"][0]
    assert top["source_id"] == ORG_DOC_REL
    assert top["lines"] == [4, 7]
    assert top["heading_path"][-2:] == ["Организационная структура", "Норма управляемости"]
    assert top["title"] == "пересказ_оргструктура"

    content = render_chat_content(d)
    assert "Источники:" in content
    assert "[1]" in content
    assert ORG_DOC_REL in content
    # [n] в синтезированном/extractive ответе не обязателен, но блок источников
    # обязан показывать тот же документ, что citations[0]


def test_answer_second_document_txt_lines(mini_corpus) -> None:
    """Второй in-corpus кейс на .txt: нумерованный пункт с точными строками."""
    bundle, _ = mini_corpus
    client = make_client(bundle)
    r = client.post("/v1/knowledge/query",
                    json={"query": "сколько минут длится дейли", "use_llm": False})
    d = r.json()
    assert d["decision"] == "answer"
    assert d["citations"][0]["source_id"] == SPRINT_DOC_REL
    assert d["citations"][0]["lines"] == [5, 5]
    assert "15 минут" in d["answer"]


def test_abstain_outside_corpus_with_reason_and_hint(mini_corpus) -> None:
    """Сценарий 2: вопрос вне корпуса -> abstain с причиной и подсказкой."""
    bundle, _ = mini_corpus
    client = make_client(bundle)
    r = client.post("/v1/knowledge/query", json={"query": OUT_CORPUS_QUERY})
    d = r.json()
    assert d["decision"] == "abstain"
    assert d["diagnostics"]["reason"]
    assert not d["citations"] or d["decision"] == "abstain"
    content = render_chat_content(d)
    assert "Ответа на этот вопрос в базе знаний нет" in content
    assert "Причина:" in content
    assert "Уточните вопрос" in content
    # abstain и через chat-фасад не превращается в выдуманный ответ
    rc = client.post("/v1/chat/completions",
                     json={"messages": [{"role": "user", "content": OUT_CORPUS_QUERY}]})
    assert rc.status_code == 200
    assert "Ответа на этот вопрос в базе знаний нет" in rc.json()["choices"][0]["message"]["content"]


def test_abstain_on_unknown_identifier_gost(mini_corpus) -> None:
    """Сценарий 3: identifier-grounding - номер ГОСТ из вопроса отсутствует в
    корпусе -> abstain независимо от скоров."""
    bundle, _ = mini_corpus
    client = make_client(bundle)
    r = client.post("/v1/knowledge/query", json={"query": GOST_QUERY})
    d = r.json()
    assert d["decision"] == "abstain"
    assert "отсутствующие в корпусе" in d["diagnostics"]["reason"]


def test_acl_restricted_excluded_without_rights_via_resolver(mini_corpus) -> None:
    """Сценарий 4 (ACL, инвариант 5): restricted-документ не попадает в evidence
    при группах владельца без права; попадает при наличии права (через resolver)."""
    bundle, _ = mini_corpus
    secret_text = "Оклады сотрудников: значения сумм конфиденциальны."

    denied = make_client(bundle, groups={"personal"})
    d = denied.post("/v1/knowledge/query",
                    json={"query": SECRET_QUERY, "use_llm": False}).json()
    texts = [ev["text"] for ev in d["evidence"]]
    cited_titles = [c["title"] for c in d["citations"]]
    assert secret_text not in texts
    assert "зарплатная_ведомость" not in cited_titles
    assert SALARY_DOC_REL not in {c["source_id"] for c in d["citations"]}

    granted = make_client(bundle, groups={"personal", "restricted"})
    d2 = granted.post("/v1/knowledge/query",
                      json={"query": SECRET_QUERY, "use_llm": False}).json()
    texts2 = [ev["text"] for ev in d2["evidence"]]
    assert d2["decision"] == "answer"
    assert secret_text in texts2
    assert SALARY_DOC_REL in {c["source_id"] for c in d2["citations"]}


def test_llm_unreachable_degrades_to_extractive(mini_corpus) -> None:
    """Сценарий 5: LLM-провайдер с недоступным base_url -> extractive-ответ,
    llm_used=false, фасад не падает (200)."""
    bundle, _ = mini_corpus
    dead_llm = OpenAICompatibleProvider(
        LLMConfig(base_url="http://127.0.0.1:1/v1", model="stub", timeout_s=0.5)
    )
    client = make_client(bundle, llm=dead_llm)
    r = client.post("/v1/knowledge/query", json={"query": IN_CORPUS_QUERY, "use_llm": True})
    assert r.status_code == 200
    d = r.json()
    assert d["decision"] == "answer"
    assert "от 5 до 9" in d["answer"]
    assert d["diagnostics"]["llm_used"] is False
    assert d["diagnostics"]["llm_error"]


def test_401_on_all_content_endpoints_without_bearer(mini_corpus) -> None:
    """Сценарий 6: при заданном токене все контентные эндпоинты отвечают 401 без
    bearer и 200 с ним. /healthz намеренно открыт (liveness-probe) - фиксируем."""
    bundle, _ = mini_corpus
    client = make_client(bundle, token="e2e-bearer-token")
    protected = [
        ("GET", "/v1/models", None),
        ("GET", "/v1/knowledge/stats", None),
        ("POST", "/v1/knowledge/query", {"query": IN_CORPUS_QUERY, "use_llm": False}),
        ("POST", "/v1/chat/completions",
         {"messages": [{"role": "user", "content": IN_CORPUS_QUERY}]}),
    ]
    for method, url, body in protected:
        kwargs = {"json": body} if body is not None else {}
        resp = client.request(method, url, **kwargs)
        assert resp.status_code == 401, f"{method} {url} без bearer обязан быть 401"
        resp_wrong = client.request(method, url,
                                    headers={"Authorization": "Bearer wrong-token"}, **kwargs)
        assert resp_wrong.status_code == 401, f"{method} {url} с чужим токеном обязан быть 401"
        resp_ok = client.request(method, url,
                                 headers={"Authorization": "Bearer e2e-bearer-token"}, **kwargs)
        assert resp_ok.status_code == 200, f"{method} {url} с верным токеном обязан быть 200"
    # liveness-проба намеренно без авторизации (systemd/мониторинг)
    assert client.get("/healthz").status_code == 200


def test_index_roundtrip_rebuildable_from_canonical(mini_corpus, tmp_path) -> None:
    """L2 перестраивается из L1 (инвариант 1): save -> load_index -> тот же ответ."""
    bundle, _ = mini_corpus
    idx_dir = tmp_path / "index"
    bundle.save(idx_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    loaded = load_index(idx_dir, store=NumpyFlatStore(dim=HASH_DIM))
    client = make_client(loaded)
    d = client.post("/v1/knowledge/query",
                    json={"query": IN_CORPUS_QUERY, "use_llm": False}).json()
    assert d["decision"] == "answer"
    assert d["citations"][0]["source_id"] == ORG_DOC_REL
