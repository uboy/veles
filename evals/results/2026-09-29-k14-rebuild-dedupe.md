# Eval K14: плановая пересборка прода с дедупликацией (чистка близнецов, свап, бенчи, DEDUPE_CHANNELS)

Дата: 2026-09-29. Эпик: corp-ai-kb. Ветка `feature/corp-ai-kb-rebuild-dedupe`
(база main e37e6cc + коммит 8872c2c), worktree `veles-wt-rebuild`. Бриф:
`docs/plan/briefs/2026-09-29-k14-rebuild-dedupe.md`. Мандат владельца 29.09
(«ну делай»): плановая prod-операция, обратимо.

## Что сделано (хронология, стенд)

1. Baseline «до» (read-only): индекс 158 393 узла / 137 document_id в meta;
   canonical 143 каталога (УК РФ возвращён владельцем), docling 65, vyshka 62;
   детектор `duplicate_document_id_dirs` - ровно 5 пар близнецов.
2. Чистка: 5 не-первых каталогов (sorted-семантика iter_documents, d135953)
   перенесены `mv` в `data/trash/20260929-2114-twin-01..05-*` (обратимо).
   Детектор после чистки: 0 пар; canonical 138 каталогов.
3. Сборка in-process `build_index` (iter_documents с дедупом) в
   `data/index-rebuild-20260929-2115`, OMP_NUM_THREADS=8, CPU, numpy-flat
   (как у прода), модель intfloat/multilingual-e5-small. Время ~21 мин.
4. Свап (deploy/стенд/k14-swap-20260929.sh): mv data/index ->
   data/index-backup-20260929-2137; mv rebuild -> data/index; рестарт
   veles-facade (1 из 2 разрешённых; второй не понадобился). healthz/readyz ok.
5. Проверки прода, бенчи, synth-bench, DEDUPE_CHANNELS, аудит (ниже).

## Индекс до/после

| Метрика | ДО (dup-индекс 29.09 00:04) | ПОСЛЕ (пересборка) |
|---|---|---|
| узлов (node_meta) | 158 393 | **127 498** (-19.5%) |
| записей вектор-стора | 241 429 (52% дублей) | **127 498 (= meta, дублей нет)** |
| записей BM25 | 241 429 | 127 498 |
| документов (фасад totals) | 138 | **138** |
| НК РФ, узлов | 46 355 (двойная индексация) | **23 960** (ровно канон) |
| УК РФ | присутствовал | присутствует |
| близнецы (детектор) | 5 пар | **0** |
| lnid/nvid duplicate_cross_doc | 842 | **0** |

No-loss: множество document_id до/после совпало 1:1 (138=138, потерянных 0,
лишних 0); у недублированных документов счётчики узлов идентичны до узла
(ГК 11 557, ТК ЕАЭС 8 872, КоАП 8 209 - без изменений). Аудит: canonical
138 док / 127 636 узлов с root (127 498 + 138 root) - арифметика сходится.

## Retrieval-бенчи на новом снапшоте (live, дефолты config: boost=2.0, depth=25)

Особенность наборов: expected-узлы golden/rrncb50 построены в K11/K12 на
dup-индексе; часть из них - узлы легаси-поколений двойной индексации
(другой revision_id -> другой node_version_id), которых в чистом индексе нет
по построению. Проверка: у 6 golden-промахов as-measured 5 expected отсутствуют
в новом индексе и есть в бэкапе (k11-013 - настоящий ранговый промах);
из 14 rrncb-промахов 9 отсутствуют в новом. K12 это прямо планировал
(«пересобрать golden-expected по индексу»), поэтому основные числа - на
обновлённых expected.

| Набор | R@5 | top1 | MRR | P@5 | комментарий |
|---|---|---|---|---|---|
| golden as-measured (устаревший expected) | 0.895 | 0.772 | 0.828 | 0.635 | 5 из 6 промахов - отсутствующие узлы |
| **golden-k14 (ремап 14 expected)** | **0.983** | **0.860** | **0.915** | 0.635 | цель R@5>=0.98 ВЫПОЛНЕНА |
| rrncb50 старый (смещён на dup) | 0.720 | 0.480 | 0.566 | 0.792 | 9 из 14 промахов - отсутствующие узлы |
| **rrncb50-k14 (регенерация seed 12)** | **1.000** | 0.600 | 0.724 | 0.788 | покрытие 0.45 (было 0.407) |

Референс K12 (dup-индекс, та же калибровка): golden 1.000/0.684/0.819/0.635;
rrncb50 1.000/0.560/0.713/0.776. На честных expected: R@5 -0.017 (k11-013),
top1 +0.176, MRR +0.096; rrncb top1 +0.040, MRR +0.011, P@5 +0.012,
покрытие выборки выросло. Единственный golden-промах k11-013 (expected d=79 -
глубокий dense-ранг, известный кейс K12, кандидат reranker-фазы).
k11-056 «транспортный налог»: expected ст. 357 - ТОП-1 (в K12 был ранг 4-5).

Ремап expected: `evals/synth-bench/remap_golden_expected.py` - expected узел,
отсутствующий в новом индексе, заменяется узлом того же документа с тем же
нормализованным текстом (все 14 нашлись, 0 исключений); артефакт
`golden-k14.jsonl`. Регенерация rrncb50: `rrncb_sample.py --seed 12` ->
`rrncb50-k14.jsonl`. DEDUPE True vs False на чистом индексе: метрики
идентичны до знака на обоих наборах (knob no-op).

## Synth-bench GLM (72 вопроса, судья GLM, новый retrieval-cache)

| MODEL | FAITH | CITE | COV | ABST | RU | LAT, c | N | TOK/отв |
|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash, K14 (новый кэш) | 1.91 | 1.98 | **2.00** | 93.3%* | 1.96 | 9.7 | 72 | 1164 |
| glm-5.3-flash, K12 (dup-кэш) | 1.95 | 2.00 | 1.96 | 100% | 1.96 | 12.5 | 72 | 1068 |
| glm-5.3-flash, K11 (старый кэш) | 1.98 | 2.00 | 1.98 | 100% | 2.00 | 10.6 | 72 | 1060 |

*ABST 93.3% (14/15) - **не выдумка**: единственный «промах» k11-059 -
семантически корректный отказ («информация ... **отсутствует**» + перечисление,
что есть во фрагментах), не распознанный механикой `is_refusal` (фраза не в
маркерах, ответ длинный со сносками - правила 2-3 не сработали). Вариативность
формулировки отказа GLM (прецедент K12: «отказ вариативен»). Детектор в этой
фазе сознательно не правился (менять измеритель в середине фазы запрещено).
Дополнительно n_parse_errors=1 у судьи (точечная вариативность, как в K12).
COV 2.00 - лучший результат на трёх снапшотах; FAITH -0.04, CITE -0.02 -
в пределах разброса одного судьи.

## codex-boost-check (2 answerable + 4 abstain, синтез по замороженным фрагментам)

6/6 верно: cbc-001 «транспортный налог» - ответ со ст. 357 (цитаты верхнего
ранга); cbc-002 «соосность блоков» - «отвес» [4],[5]; cbc-003..006 (суши-рис,
галстук, борщ, ПДД) - все 4 отказа (mechanical_refusal=1).

## НАХОДКА: abstain-гейт фасада недокалиброван под чистый индекс (решение владельца)

Живой фасад (`/v1/knowledge/query`, use_llm): на вопрос cbc-001 decision=abstain
при ст. 357 в цитате N1. Замер `gate_abstain_report.py` (decide() с продовыми
дефолтами, оба индекса, read-only):

| Индекс | false-abstain на 57 answerable | codex-boost-check гейт |
|---|---|---|
| dup (бэкап) | 1/57 (1.8%) - k11-026 | 3/6 (ошибки: 2 false-answer) |
| чистый (прод) | **2/57 (3.5%) - k11-026, k11-056** | 3/6 (ошибки: 2 false-abstain) |

Причина: `_confidence` (veles/policy/answer.py) считается по rrf_base с
нормировкой ceiling=2/(RRF_K+1), рассчитанной на скоры dup-индекса, где
двойные вклады RRF давали conf~1.0; на чистом индексе честный топ-1 ОДНОГО
канала = 1/61, factor 0.8 -> conf=0.4 < ABSTAIN_CONFIDENCE 0.55 -> abstain.
Дополнительный фактор: фасад до рестарта крутил код c1410bb (до K12), после -
972b131; наблюдаемое поведение = новый код + чистый индекс (раньше сочетание
не существовало). codex_boost не помогает confidence по построению (rrf_base до
буста, фикс ревью K12 P1). Заметим: гейт оба индекса пропускает cbc-006 (ПДД,
conf 1.0) - FAR-кейс не нов.

Варианты (владельцу): (а) перекалибровать порог/нормировку под чистый индекс
(например single-channel потолок 1/(K+1) и/или порог 0.35: тогда single-channel
rank-1 conf 0.408 проходит, rank>10 отсекается) - отдельная фаза K12-типа с
перегоном golden+rrncb50 и FAR-проверками; (б) принять как есть до
reranker-фазы (синтез-качество не затронуто: synth-bench и cbc идут по
замороженным фрагментам мимо гейта; 2/57 answerable абстейнсятся в живом
фасаде); (в) откат индекса (не чинит гейт, возвращает дубль-инфляцию скоров).
Выбор владельца; никакой из вариантов не исполнялся в K14.

## Повторный аудит (scripts/ingest_audit.py --index, после пересборки)

| детектор | K13 (до) | K14 (после) | ожидание K13 |
|---|---|---|---|
| count_mismatch_index | 39 | **0** | ->0 ВЫПОЛНЕНО |
| duplicate_document_id_dirs | 5 | **0** | ->0 ВЫПОЛНЕНО |
| lnid/nvid_duplicate_cross_doc | 842/842 | **0/0** | уходят ВЫПОЛНЕНО |
| doc_not_in_index | 1 (УК) | **0** | УК в canonical ВЫПОЛНЕНО |
| parent_missing | 83 | 84 | остаток: 83 легаси + 1 в возвращённой УК-копии; только реингест, на поиск не влияет (retrieval по узлам, не иерархии) |
| heading_path_empty (warning) | 298 | 302 | +4 УК, порог 0.30 не превышен |
| legacy_manifest (warning) | 204 | 200 | класс тот же |

Состав: canonical 138 док / 127 636 узлов (с root), docling 65 / 117 077,
vyshka 62 / 9 308.

## DEDUPE_CHANNELS: принято True (коммит 8872c2c)

Замер True vs False на чистом индексе: golden и rrncb50 (оба набора) -
метрики идентичны до знака. На чистом индексе дедуп каналов - no-op; включение
выполняет план K12 («включить после пересборки») и защищает от возврата
дублей при будущих пересборках, поведение прода не меняя. Правки: config
дефолт True + комментарий, синхронизирован docstring retriever, юнит-тест
no-op дефолта (`test_dedupe_channels_default_true_noop_on_clean_index`).
Полный pytest ветки на стенд (снапшот HEAD в /tmp/veles-k14):
**260 passed, 1 skipped** (K13-референс 249+1; рост от тестов K12/K13/K14).

## Откат-план (проверен: каталог бэкапа на месте)

```bash
# состояние после K14: data/index = новый, data/index-backup-20260929-2137 = старый
ssh стенд "ls -d <server-home>/proj/veles-bench/veles/data/index-backup-20260929-2137"
# откат:
ssh стенд "sudo systemctl stop veles-facade && \
  mv <server-home>/proj/veles-bench/veles/data/index <server-home>/proj/veles-bench/veles/data/index-k14-rolledback && \
  mv <server-home>/proj/veles-bench/veles/data/index-backup-20260929-2137 <server-home>/proj/veles-bench/veles/data/index && \
  sudo systemctl start veles-facade && sleep 15 && curl -sf http://127.0.0.1:8400/healthz"
# возврат близнецов (если нужен полный откат канона):
# mv <server-home>/proj/veles-bench/veles/data/trash/20260929-2114-twin-*/ <каталоги> в data/canonical/
```

Бэкап НЕ удалять до решения владельца по гейту.

## Команды (воспроизведение, стенд)

```bash
# бенчи (live, read-only):
cd ~/proj/veles-bench/veles/evals/synth-bench
VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
  ~/veles-venv/bin/python retrieval_bench.py --golden golden-k14.jsonl --live --dedupe
VELES_REPO=... ~/veles-venv/bin/python retrieval_bench.py --golden rrncb50-k14.jsonl --live
# перезаморозка кэша (выполнена, committed):
... freeze_retrieval.py --golden golden.jsonl --out retrieval-cache.json
# synth: e2e/run_synth_bench.py --provider glm --secrets-file ~/secrets/veles-facade.env \
#   --golden golden.jsonl --retrieval-cache retrieval-cache.json --out answers-glm-k14.json
# judge: judge.py --answers answers-glm-k14.json --golden golden.jsonl \
#   --retrieval-cache retrieval-cache.json --out judged-k14.json --md table-k14.md
# гейт: ~/veles-venv/bin/python gate_abstain_report.py  (оба индекса, read-only)
# аудит: ~/veles-venv/bin/python scripts/ingest_audit.py --data data --index --jobs \
#   --json-documents --max-examples 1000 --json /tmp/k14/audit-after.json
```

Артефакты: evals/synth-bench/{golden-k14.jsonl, rrncb50-k14.jsonl,
retrieval-cache.json (567 КБ, перезаморожен), retrieval-cache-cbc.json,
codex-boost-check-id.jsonl, answers-glm-k14.json, answers-cbc-k14.json,
judged-k14.json, table-k14.md, remap_golden_expected.py,
gate_abstain_report.py}; стенд: /tmp/k14/{bench-*.json, gate-old-new.json,
audit-after.json, build.log}, лог операции ~/veles-rebuild-dedupe.log.

## Что НЕ проверено / вне объёма

- Живой фасад остался на main 972b131: ветка K14 (DEDUPE_CHANNELS=True) не
  задеплоена - на чистом индексе поведение идентично (замер), деплой - координатор.
- Гейт AnswerPolicy не перекалибровался и не откатывался - решение владельца
  (варианты выше), второй рестарт фасада не потрачен.
- Telegram-мост и клиенты на новом индексе не проверялись (кроме
  healthz/readyz/stats/query фасада).
- Reranker не внедрялся (k11-013, top1 0.860 - его зона, рекомендация K12).
- k11-013: единственный ранговый промах golden; depth/буст под него не
  подбирались (K12 отклонил depth50, ограничение сохранено).
- rrncb50-k14 - новая выборка на новом индексе; прямое сравнение с K12-набором
  некорректно (другие вопросы в покрытой части), даны оба.
- Локальный (Windows) pytest не гонялся - только стенд.

## Критерии готовности брифа

- Свап с бэкапом: ВЫПОЛНЕНО (index-backup-20260929-2137 на месте).
- Проверки зелёные: healthz/readyz/stats ВЫПОЛНЕНО; состав 138=138 ВЫПОЛНЕНО;
  golden R@5 0.983 >= 0.98 ВЫПОЛНЕНО (на ремапнутом expected; as-measured
  0.895 объяснён отсутствием легаси-узлов); synth ABST механически 93.3%
  (k11-059 - вариативность формулировки отказа, семантически 15/15) -
  ФОРМАЛЬНО НЕ ВЫПОЛНЕНО, разобрано; гейт фасада - НАХОДКА владельцу.
- Eval с числами до/после и откат-планом: этот файл.
- стенд pytest полный на ветке с новым дефолтом: 260 passed, 1 skipped.
