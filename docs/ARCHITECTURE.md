# Veles — архитектурный стандарт Knowledge Core

> **Версия:** 3.0 (2026-09-26)
> **Статус:** принят
> **Концептуальное имя:** Knowledge Core (реализация — репозиторий `veles`)
> **Предшественник:** [ARCHITECTURE v2.2 (archive)](archive/ARCHITECTURE-v2.2.md) — сокращён
>   по итогам ревью 2026-09-25: инварианты отделены от реализационных решений.
> **База кода:** самостоятельный репозиторий; зрелые части retrieval-ядра переносятся из
>   `uboy/telegram_bot_ai` (master, коммит 2026-08-31). Telegram остаётся клиентом, а не
>   тёзкой проекта.

---

## 1. Что это и для чего

Veles — база знаний с RAG и верифицируемыми цитатами: единый канонический слой (L0/L1)
независим от поисковых индексов (L2) и провайдеров LLM; ответы обязаны ссылаться на
первоисточник (файл, страница/слайд, пункт), при отсутствии фактов — ответ об их отсутствии.

Первичная цель — **личный профиль**: единый поиск по книгам, лекциям, документации и
курируемым знаниям офлайн. Корпоративный профиль — задел на старте (интерфейсные швы),
но не MVP: build-out (SSO, очереди, серверные развёртывания) отложен до подтверждённой
ценности личного контура.

## 2. Инварианты (не зависят от выбора библиотек и моделей)

1. **Слои L0/L1/L2.** L0 — неизменяемые оригиналы. L1a — канонические документы в схеме
   проекта; L1b — курируемые знания со статусами. L2 — производные поисковые индексы:
   перестраиваются 100% из L1 (полностью offline; L0→L1 — offline при наличии локально
   установленных артефактов парсера/моделей). Версии canonical schema, парсера и
   embedding-модели фиксируются в данных (meta индекса); фиксация reranker-модели -
   цель стандарта, в реализации среза meta несёт только embed-модель.
2. **CanonicalDocument принадлежит проекту.** Парсер — адаптер (Docling, PyMuPDF,
   Markdown, MinerU…). Зависимость от парсера не должна превращаться в lock-in так же,
   как зависимость от векторной БД.
3. **Стабильные идентичности.** `document_id`, `revision_id`, `logical_node_id`
   (устойчивая логическая единица, например `clause:12.4.2`), `node_version_id` =
   hash(revision_id + logical_node_id + normalized_text). Сравнение редакций одного
   пункта обязано работать.
4. **Точный provenance.** Locator зависит от формата: PDF — страница+bbox, PPTX —
   слайд, XLSX — лист+диапазон, Markdown — heading_path+строки, DOCX — абзац/закладка.
   Транскрипции/пересказы/claims не притворяются первоисточником: у них `derived_from`.
   `source_id` + относительный locator; абсолютные машинные пути в каноне запрещены
   (корень задаётся deployment-конфигом).
5. **Pre-retrieval authorization.** Отсечение по правам — до формирования контекста.
   Группы доступа выводятся сервером из проверенной identity (корп. профиль); группы из
   клиентского запроса не принимаются. Личный профиль: один principal, фильтр тривиален,
   но интерфейс фильтра присутствует с первого дня (шов).
6. **Гибридный поиск** (dense + lexical + fusion, rerank опционален) — один
   retrieval-движок для всех клиентов.
7. **Policy-driven evidence expansion.** Подтягивание контекста (родитель, соседи,
   таблицы) — стратегия по типу документа, не один универсальный алгоритм (см. §6).
8. **Version/conflict semantics.** При поиске фильтруются статусы: в реализации -
   `current`/`superseded` (канон) плюс tombstone `deleted` удалённых документов;
   `approved`/`obsolete` - целевые статусы стандарта, в срезе не заведены.
   Противоречие двух действующих редакций - явное
   `warn_conflict`, а не молчаливый выбор и не «ответа нет».
9. **Один канонический API.** `POST /v1/knowledge/query` — native API (answer, evidence,
   citations, answerability, request_id). `POST /v1/chat/completions` — compatibility-фасад
   для OpenAI-совместимых клиентов (Open WebUI и др.). Core stateless относительно
   UI-сессий: историю хранит клиентский слой.
10. **Профили развёртывания одной кодовой базы:** `PERSONAL_CORE` (один процесс),
    `PERSONAL_UI` (core + UI-клиент), `CORPORATE` (сервер). Профиль — конфигурация,
    не форк.

## 3. Runtime-схема

```
                 CLIENTS: Telegram / Open WebUI (optional) / CLI / агенты (MCP)
                                      │
                                      ▼
                                 QUERY API
                    (/v1/knowledge/query; chat/completions — фасад)
                                      │
                                      ▼
                        REQUEST CONTEXT (principal, scope,
                          ACL, version/as_of, retrieval hints)
                                      │
                                      ▼
                              RETRIEVAL
                 dense + lexical + fusion + optional rerank
                                      │
                                      ▼
                            EVIDENCE BUILDER
                    policy-driven expansion + provenance
                                      │
                                      ▼
                             ANSWER POLICY
                       answer / abstain / warn_conflict
                                      │
                                      ▼
                              LLM PROVIDER

WRITE PATH:  SOURCES → INGESTION → CANONICAL STORE → INDEXER → RETRIEVAL INDEX
```

Отдельных архитектурных сущностей «Universal LLM Gateway», «Deterministic Router»,
«Answerability Gate» нет: LLM-эндпоинты — конфигурация провайдера; запрос «пункт 12.4» —
retrieval hint в RequestContext; отказ — исход AnswerPolicy. EVIDENCE BUILDER
в реализации среза - provenance без расширения: hybrid_search отдаёт flat top-k
(policy-driven expansion - цель, см. §5).

## 4. Модель данных

Физически документ хранится плоско (потоковая обработка, diff, миграции, частичная
переиндексация):

```
canonical/<source_id>/revision-<n>/
├── manifest.json      # метаданные документа, sha256, статусы, parser_profile
├── nodes.jsonl        # узлы дерева, по строке на узел
├── tables/            # таблицы отдельными артефактами
└── assets/            # изображения и прочее
```

Узел (компактно):

```yaml
document_id: contract-123
revision_id: r7
logical_node_id: clause:12.4.2
node_version_id: sha256(...)
node_type: clause              # section|paragraph|table_row|slide_block|...
parent_id: clause:12.4
heading_path: ["12. Расторжение договора", "12.4 Односторонний отказ"]
text: "..."
provenance: { pdf: {page: 327, bbox: [...] } }   # locator по формату, см. инвариант 4
access_groups: [legal]         # фильтруется pre-retrieval
```

L1b (курируемое) несёт `content_kind: claim|note|synthesis`, `authority: curated`,
`derived_from: [<logical_node_id>…]`, `status: current|superseded` — чтобы тезис и его
первоисточник не считались двумя независимыми подтверждениями и не поднимали confidence
искусственно.

## 5. Retrieval и ответ

Кандидаты — по листовым узлам; в поисковый текст детерминированно добавляется путь
заголовков. После fusion/rerank — расширение контекста по политике типа документа
(цель стандарта; в реализации среза - flat top-k без расширения):

| Тип | Hit | Расширение |
|---|---|---|
| Договор/регламент | пункт | интро родителя, ±1 сосед при необходимости, явные перекрёстные ссылки |
| Книга | параграф | заголовок, оконный контекст |
| Презентация | блок | весь слайд + notes |
| Таблица | строка | заголовки, caption, строка |

**AnswerPolicy** вместо бинарного гейта. Сигналы (калибруются на бенчмарке, не
фиксируются априори): top rerank score, margin top1-top2, согласие dense/BM25, точное
совпадение идентификатора (номер пункта/артикул), число evidence-единиц, конфликт версий.
Исходы: `answer`, `abstain` (отказ без домысливания), `warn_conflict`. `ask_clarification`
отложен (не MVP).

Retrieval-scores пользователю не показываются (шкалы режимов несопоставимы) — только
человекочитаемая ссылка «Документ, ред. N, §12.4.2, стр. 327»; полные score-трейсы — в
diagnostics.

## 6. Провайдеры и профили

> Состав этого раздела - **целевой стандарт**, не описание текущего кода. Реализация
> среза: faiss/numpy + BM25 (свой Okapi), канон JSONL (`nodes.jsonl`), LLM по
> OpenAI-compatible env (SQLite, Qdrant Local и llama-cpp из профиля ниже в коде нет).

```
VectorStore      local | server (один интерфейс; granularity подбирается по документу)
EmbeddingProvider   FastEmbed (ONNX) | SentenceTransformers | remote
RerankerProvider    FastEmbedReranker | TransformersCrossEncoder | remote
LLMProvider         OpenAICompatible (Ollama/vLLM/облако — просто endpoints) | InProcessLlamaCpp
```

Ограничения Qdrant Local mode учтены проектом: small-scale по заявлению вендора,
предупреждение вендора от ~20 000 точек, single-process lock, точный поиск без HNSW,
payload indexes без эффекта. Отсюда: интерфейс `local|server` с первого дня; гранулярность
индексации (не каждый table row механически) — ответственность SearchNodeBuilder.

Профили:

- **PERSONAL_CORE** — один процесс: SQLite, Qdrant Local, embeddings, опционально
  llama-cpp. Установка `pip/uv` на Windows/macOS/Linux без обязательных внешних БД,
  контейнеров и системных сервисов.
- **PERSONAL_UI** — core + UI-клиент (Open WebUI опционален, хранит только свои сессии;
  знания/чанки/эмбеддинги ему не отдаются, тяжёлые функции ограничиваются permissions).
- **CORPORATE** — Qdrant server, IdentityResolver (JWT/OIDC/SSO) → группы → pre-retrieval
  filter, Docker Compose. Для MVP строятся только интерфейсы: VectorStore, ACL-фильтр,
  IdentityResolver.

Конкретные модели (embedding/reranker/LLM) — не часть стандарта: они выбираются
бенчмарком и живут в конфиге. Стартовый стек наследуется от ядра `telegram_bot_ai`
(sentence-transformers + CrossEncoder) — замена только по данным замера.

## 7. Корпус и приватность

Каталог `~/proj/knowledge-sources/` (01_books_pm … 06_benchmarks). Вышка попадает в
корпус только через allowlist-конвейер:

```
OneDrive → SourceScanner → strict allowlist → staging manifest → knowledge-sources → ingestion
```

Прямое индексирование исходного OneDrive-каталога запрещено. Запрещено независимо от
конвейера: ДЗ, оценки, ведомости, `00_Админ`/`01_Оргинфо`, приказы, договоры,
`05_Диссертация`. Художественная литература (04) — стресс-корпус вне gating-метрик, чтобы
не маскировать деградацию на нормативных документах.

`06_benchmarks/rrncb` (65 нормативных PDF + ~900 верифицированных QA) — первичный
калибровочный бенчмарк; `synthetic_traps` — старые/новые редакции, сканы, противоречия
(для AnswerPolicy и version semantics).

## 8. Качество и SLO

Замер по стадиям × профилям (PERSONAL_CPU / WORKSTATION / SERVER): candidate retrieval,
rerank, expansion, search-only response, first-token latency. P50/P95 фиксируются после
замеров; целевые числа v2.2 (3 с, 500 мс, RAM) аннулированы как неизмеренные.

Качество (на общем корпусе, flat-chunk baseline против canonical small-to-big):
Recall@K / MRR, Answerability (FAR < 5% на ловушках), citation accuracy (100% ответов
с проверяемой ссылкой), rebuildability L2 (полное удаление индекса → 100% восстановление
из L1).

## 9. Non-goals MVP

GraphRAG/LightRAG; Visual RAG (ColPali); fine-tuning до исчерпания парсинга и реранкинга;
Kubernetes; корпоративный build-out кроме швов; `ask_clarification`; LLM-маршрутизация
каждого запроса.
