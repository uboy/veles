# Eval K11: synth-bench — харнес качества синтеза

Дата: 2026-09-29. Эпик: corp-ai-kb, фаза K11. Бриф:
[`docs/plan/briefs/2026-09-29-k11-synth-bench.md`](../plan/briefs/2026-09-29-k11-synth-bench.md).
Ветка `feature/corp-ai-kb-synthbench`, worktree `veles-wt-synthbench`.

## Что сделано

Полный харнес качества СИНТЕЗА: golden-набор 72 вопроса по прод-корпусу,
замороженный retrieval (один прогон, JSON-кэш), раннер, LLM-судья GLM с рубрикой,
бейзлайны трёх моделей (GLM + обе локальные с V100 — обе достижимы).

## Инфраструктура и решение по сети

- стенд: ssh BatchMode OK; прод-данные `~/proj/veles-bench/veles/data` (137 доков,
  158 394 узла, numpy-индекс); венд `~/veles-venv` (torch 2.14.0+cu130,
  sentence-transformers). Ключ синтеза — `~/secrets/veles-facade.env`,
  `VELES_LLM_BASE_URL=https://api.z.ai/api/coding/paas/v4`, модель `glm-5.3-flash`
  (тот же env, что у прода).
- **V100: `http://<internal-llm-host>:8471/v1` ДОСТИЖИМА с стенд напрямую**
  (llama.cpp, OpenAI-совместимый). Ssh-туннель и перенос бенча не понадобились.
  Прогон qwen3.8-27b на работающем контейнере `synthbench-qwen38`; для
  gemma-4-31b-it контейнер свапнут командой из
  `/data<server-home>/proj/synthbench-stack-20260929.md` (на V100-хосте), после
  прогона стенд восстановлен (`synthbench-qwen38` Up, health: healthy).
  Сервисы 1С (GPU1/GPU2) и чужие процессы не тронуты.
- Прод read-only: индекс грузился in-process `load_index` БЕЗ save; фасад не
  менялся, сквозные пользовательские запросы не отправлялись (запросы к LLM —
  напрямую к провайдерам, минуя прод).
- Скрипты на стенд жили в `/tmp/k11-synthbench` (scratch), импортируя пакет veles
  из прод-чекаута (`VELES_REPO`); после прогона каталог удалён.

## Команды (стенд)

```bash
# 1. golden-набор: RRNCB QA + мини-QA + GLM-генерация с самопроверкой + abstain
VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
~/veles-venv/bin/python evals/synth-bench/build_golden.py \
  --secrets-file ~/secrets/veles-facade.env \
  --rrncb-csv ~/proj/veles-bench/rrncb/rrncb_public_dataset.csv \
  --miniqa bench/miniqa-vyshka-module1-v1.json --out golden.jsonl
# -> 72 вопроса (57 answerable + 15 abstain = 21%); GLM-генерация:
#    кандидатов 39, json-брак 2, самопроверка не прошла 6, принято 31

# 2. заморозка retrieval: top_k=5 на вопрос, один прогон, read-only
VELES_REPO=$HOME/proj/veles-bench/veles CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 \
~/veles-venv/bin/python evals/synth-bench/freeze_retrieval.py \
  --golden golden.jsonl --out retrieval-cache.json
# -> кэш 549 КБ; expected-node в top-5: 56/57 (промах - ручной вопрос про
#    транспортный налог, см. Находки)

# 3. синтез (по одному прогону на модель)
... e2e/run_synth_bench.py --provider glm --out answers-glm.json
... e2e/run_synth_bench.py --provider local \
    --base-url http://<internal-llm-host>:8471/v1 \
    --model qwen3.8-27b --out answers-qwen.json
... e2e/run_synth_bench.py --provider local \
    --base-url http://<internal-llm-host>:8471/v1 \
    --model gemma-4-31b-it --max-tokens 4096 --out answers-gemma.json

# 4. судья GLM (рубрика 0-2 по faithfulness/citation_accuracy/coverage/
#    russian_quality; abstain - механика, без судьи)
... evals/synth-bench/judge.py --answers answers-glm.json answers-qwen.json \
    answers-gemma.json --golden golden.jsonl --retrieval-cache retrieval-cache.json \
    --out judged.json --md table.md
```

Промпт синтеза — в точности фасадский: `SYNTHESIS_SYSTEM` импортируется из
`veles.api.llm` прод-чекаута (не копируется), сообщение собирается как в
`OpenAICompatibleProvider.synthesize`. Ответ модели = только `message.content`;
thinking-канал (`reasoning_content` у GLM и gemma) ответом не считается.

## Таблица результатов (судья GLM, 57 answerable + 15 abstain на каждый)

| MODEL | FAITH | CITE | COV | ABST | RU | LAT, c | N | TOK/отв |
|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | 1.98 | 2.00 | 1.98 | 100% | 2.00 | 10.6 | 72 | 1060 |
| qwen3.8-27b | 1.96 | 1.95 | 1.89 | 100% | 1.96 | 19.4 | 72 | 1242 |
| gemma-4-31b-it | 2.00 | 2.00 | 1.91 | 100% | 2.00 | 17.7 | 72 | 1057 |

- Метрики 0-2 (средние по answerable); ABST — % честных отказов на 15
  abstain-кейсах (механика: отказ=1, выдумка=0); LAT — средняя задержка
  chat-вызова из стенд (GLM — сеть+reasoning, локальные — инференс V100);
  TOK — prompt+completion на ответ.
- Разброс COV по источникам вопросов: manual (живые вопросы владельца) —
  GLM 1.67 / qwen 1.33 / gemma 1.00; auto (GLM-генерация) — 2.00 у всех;
  rrncb — 2.00/1.85/1.85; miniqa — 2.00/1.80/2.00.
- Все прогоны: 72/72 ответов, 0 сетевых ошибок; finish_reason=stop у GLM и
  gemma на всех 72, у qwen 71 stop + 1 length (ответ обрезан бюджетом 3072).
- Судья: 171 вызов на answerable-ответы, 1 parse_error (qwen k11-050, его скоры
  исключены из среднего qwen); abstain-кейсы судье не отправляются.

## Интерпретация (осторожная — судья lenient, см. «Не проверено»)

1. **Все три синтезатора дисциплинированны**: 100% честных отказов на
   abstain-темах, faithfulness ≥1.96, цитирование почти безупречное. Промпт
   фасада держит любые из трёх моделей.
2. **GLM — самый быстрый (10.6 c) и лучший по coverage (1.98)**; локальные
   проигрывают в полноте на «трудных» (manual) вопросах — в основном там, где
   retrieval не принёс нужный фрагмент (см. находку 1), и честный отказ —
   корректное поведение.
3. **qwen3.8-27b** (base-GGUF со спек-декодингом) чуть слабее по COV/CITE и
   самый медленный; один ответ упёрся в max_tokens.
4. Выбор синтезатора по этому замеру не навязывается: разрывы малы и судья-GLM
   склонен завышать (само-оценка своему семейству). Рекомендация — ручная
   выборочная проверка (ниже) и, при необходимости, второй судья другой семьи.

## Находки

1. **Retrieval-промах на живом вопросе владельца**: «Кто является плательщиком
   транспортного налога согласно НК РФ?» — в прод-индексе НК РФ содержит 7 узлов
   гл. 28 («Транспортный налог»), но ни один не попал в top-5 (flexible-поиск
   вытянул ГК РФ и старые отраслевые документы). Все три модели честно
   отказались. Это пробел ПОИСКА (отдельный трек: retrieval-калибровка), не
   синтеза; e2e-проверка B5 с запросом «транспортный налог согласно НК РФ» при
   деплое проходила — поведение зависит от формулировки.
2. Механика abstain изначально занижала GLM (80% вместо 100%): три честных
   отказа с разнесёнными формулировками («ответ … отсутствует», «в базе знаний
   нет прямого ответа») не матчились подстроками. Исправлено регэкспами
   (`synth_lib.REFUSAL_RES`), сохранённые вердикты пересчитаны (скоринг judge по
   answerable не менялся), добавлены регресс-тесты.
3. RRNCB-вопросы — самые строгие по формулировке: часть из них требует точного
   подпункта статьи, которого в top-5 нет, и корректное поведение модели —
   отказ (судья это оценивает высоко). Для future: golden-набор можно
   дополнять вопросами, гарантированно отвечаемыми из top-5.

## Честность и бюджет

- Все числа — из фактических прогонов 2026-09-29 (файлы в
  `evals/synth-bench/`: `golden.jsonl`, `retrieval-cache.json`, `answers-*.json`,
  `judged.json`); LAT/TOK — из записей раннера, метрики — из `judged.json`.
- Расход z.ai (glm-5.3-flash, coding-endpoint): генерация golden 39 + smoke
  раннера 5 + smoke судьи 15 (два пере-запуска после фиксов контракта) + полный
  прогон GLM 72 + судья 171 = **302 вызова**. Локальные модели — без токен-бюджета.
- Кэш retrieval 549 КБ < 1 МБ — закоммичен в `evals/synth-bench/` (обоснование:
  фиксирует замороженные фрагменты, на которых воспроизводимы все ответы; снапшот
  в отчёте не нужен, т.к. сам файл в git). Секреты в репозиторий не попадали
  (ключи читались из `~/secrets/veles-facade.env` на стенд).

## Не проверено (MANUAL — для владельца)

1. **Ручная валидация golden-набора**: 72 вопроса составлены агентом (GLM-генерация
   + самопроверка retrieval); соответствие вопросов ожидаемым фактам владельцем не
   утверждалось. Файл: `evals/synth-bench/golden.jsonl` (provenance у каждой записи).
2. **Выборочная ручная проверка ≥10 ответов** (бриф, п.4): сверка вердиктов судьи
   с человеческими по `evals/synth-bench/judged.json` (комментарии судьи приложены
   к каждой записи).
3. Смещение судьи: судья — GLM, т.е. семейство одного из сравниваемых
   синтезаторов; систематического завышения GLM-ответов не исключено и данным
   замером не контролировалось.
4. Latency локальных моделей на прод-нагрузке (n_slots=4, конкурентность) не
   измерялась — замер одиночный, последовательный.

## Юнит-тесты (без сети)

`tests/test_synth_bench.py`: 16 тестов на чистые функции — сборка промпта
синтеза, извлечение сносок, механика отказа (позитив + негатив-регрессии реальных
ответов), разбор JSON судьи (валидный + 5 форм мусора), судейский промпт,
агрегация и таблица, env/jsonl-ридеры. Полный pytest-набор репозитория не задет:
на HEAD 993ae67 — **199 collected, 198 passed, 1 skipped** (эталонный прогон
ревью K11; цифра «186» в первой редакции отчёта была прогоном без tests/e2e на
промежуточном коммите — исправлено ревью).

## Оговорки к таблице (ревью K11)

- Сравнение моделей — на ОДНИХ замороженных фрагментах (hybrid_search top_k=5),
  промпт и temperature те же, что у фасада, но полный прод-пайплайн шире:
  фасад прогоняет decide() и не фиксирует max_tokens (бенч: 3072/4096). Для
  A/B синтезаторов это некритично (фрагменты идентичны).
- `judged.json`, агрегат `row` GLM: `abst_pct=80.0` — значение ДО фикса механики
  отказа; канон — таблица выше (100%, по записям). Остальные агрегаты row
  соответствуют отчёту.

## Артефакты

- `evals/synth-bench/build_golden.py`, `freeze_retrieval.py`, `synth_lib.py`,
  `judge.py`, `e2e/run_synth_bench.py` — харнес.
- `evals/synth-bench/golden.jsonl` — 72 вопроса (57 answerable / 15 abstain).
- `evals/synth-bench/retrieval-cache.json` — замороженный retrieval (549 КБ).
- `evals/synth-bench/answers-{glm,qwen,gemma}.json` — сырые ответы + latency/tokens.
- `evals/synth-bench/judged.json` — вердикты судьи по каждой записи.
