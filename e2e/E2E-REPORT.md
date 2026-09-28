# E2E-отчёт стека Veles (Knowledge Core + Open WebUI + Telegram-мост)

> **Дата:** 2026-09-28. **Ветка:** `feature/corp-ai-kb-e2e` (от main `23f8f28`), коммиты
> `91dc1bf` (Уровень 1), `6c1aa58` (Уровень 2), + отчёт. **Владелец решает merge** (не мержилось).
> **Вердикт: ЗЕЛЁНЫЙ-С-ОГОВОРКАМИ** — 2 KNOWN-ISSUE прода (MCP-сервер, ночной бэкап),
> автопроверки 20/22 PASS, ручных по существу 0 (1 MANUAL по определению).

## 1. Что построено

| Уровень | Файл | Что делает |
|---|---|---|
| 1. In-process E2E | `tests/e2e/test_pipeline_e2e.py` (pytest, 8 кейсов) | Полный цикл на tmp-мини-корпусе: L0 (.txt/.md) → `TxtMdParserAdapter` → canonical store (tmp) → `build_index` (numpy) → `create_app` → TestClient. Эмбеддер — детерминированный hashing-стаб (канон `veles/AGENTS.md`: тесты без загрузки моделей); retrieval/AnswerPolicy/ACL/фасад — настоящий код |
| 2. Deployed E2E | `e2e/run_deployed_e2e.py` (standalone, stdlib-only, не pytest) | 22 автопункта A–G против живого стека на стенд + T1 MANUAL. Секреты из `~/secrets/*.env` — только в память процесса, вывод через `mask()`; счётчик z.ai-вызовов; флаги `--skip-llm`, `--only`, `--known-issue`; exit 0/1/2 (0 зелёный, 2 есть KNOWN-ISSUE) |

### Команды запуска (точные)

```bash
# Уровень 1 - локально (Windows) или на стенд (из копии ветки, прод-дерево не трогается):
cd <корень veles> && python -m pytest tests/e2e/test_pipeline_e2e.py -q

# Уровень 2 - НА стенд (скрипт лежит в ~/proj/veles-bench/veles/e2e/, один новый файл):
cd ~/proj/veles-bench/veles && CUDA_VISIBLE_DEVICES= ~/veles-venv/bin/python \
  e2e/run_deployed_e2e.py \
  --expected-commit 23f8f283889636aa1d4377045a20d415e947dd63 \
  --known-issue D1,G1
# дешёвый режим без вызовов z.ai: добавить --skip-llm

# Полный тест-набор ветки на стенд (прод-копия не менялась, прогон из ~/veles-e2e-check):
cd ~/veles-e2e-check && CUDA_VISIBLE_DEVICES= ~/veles-venv/bin/python -m pytest tests/ -q
```

## 2. Уровень 1 — in-process E2E (8/8 PASS)

Прогон: локально `8 passed in 7.72s`; на стенд (копия ветки, venv) — в составе полного набора
`50 passed` (42 базовых + 8 новых). Сценарии и доказательства:

| Кейс | Ожидание | Факт |
|---|---|---|
| answer in-corpus (.md) | decision=answer; цитата = правильный файл/строки/heading_path; «Источники:» в render_chat_content | `source_id=…/пересказ_оргструктура.md`, `lines=[4,7]` (сверено с ручной разметкой файла), `heading_path[-2:]=["Организационная структура","Норма управляемости"]` |
| answer in-corpus (.txt) | цитата в нумерованный пункт с точными строками | `lines=[5,5]`, «15 минут» в ответе |
| abstain вне корпуса | abstain с причиной и подсказкой | «Ответа на этот вопрос в базе знаний нет», «Причина:», «Уточните вопрос» — и в query, и в chat/completions |
| identifier-grounding | «ГОСТ 99999» из вопроса, отсутствующего в корпусе → abstain | reason «вопрос ссылается на документ(ы), отсутствующие в корпусе» |
| ACL (инвариант 5) | restricted-документ невидим без права, виден с правом, через resolver на HTTP | без права: текста нет ни в evidence, ни в citations; с правом: decision=answer, документ в evidence и цитатах |
| деградация LLM | недоступный base_url → extractive-ответ, llm_used=false, фасад 200 | llm_used=False, llm_error непуст, «от 5 до 9» в ответе |
| 401 без bearer | 4 контентных эндпоинта 401 без/с чужим токеном, 200 с верным | все 4 → 401/401/200; /healthz намеренно открыт (liveness) |
| roundtrip L2 из L1 | save → load_index → тот же ответ (инвариант 1) | decision=answer, та же цитата |

## 3. Уровень 2 — deployed E2E на стенд (финальный полный прогон)

Факты прогона 2026-09-28 (бм1-время ≈14:26 MSK): **PASS=20, KNOWN-ISSUE=2, SKIP=1, MANUAL=1,
z.ai-вызовов 3 из 6** (B7, E2, E3 — короткие, по одному разу; повторов не было).

| ID | Пункт | Статус | Факт |
|---|---|---|---|
| A1 | systemd veles-facade | PASS | active |
| A2 | systemd veles-telegram-bridge | PASS | active |
| A3 | docker open-webui | PASS | running + healthcheck healthy |
| A4 | COMMIT.txt == HEAD main | PASS | `23f8f283889636aa1d4377045a20d415e947dd63` — совпадает |
| A5 | секреты в journal моста | PASS | паттерны tg-token/ZAI_KEY/dumbproxy/sk-key: 0 строк; за час журнал пуст (idle long-polling) — скан выполнен за 24ч (10 строк), чисто |
| B1 | GET /healthz | PASS | 200 `{"status":"ok"}` |
| B2 | 401 без bearer | PASS | /v1/models, /v1/knowledge/stats, /v1/knowledge/query, /v1/chat/completions → все 401 |
| B3 | GET /v1/models фасада | PASS | `ids=['veles-knowledge']` |
| B4 | GET /v1/knowledge/stats | PASS | documents=127 (>=120), nodes=126258 (>=120000); НК РФ присутствует; документов Вышки (document_id `lectures/…`) — 62 |
| B5 | query «транспортный налог согласно НК РФ» (use_llm=false) | PASS | decision=answer, citations=5, «Налоговый кодекс…» в цитатах, confidence=1.0 |
| B6 | query «как приготовить борщ» | PASS | decision=abstain, reason=«нет достаточных подтверждений в корпусе» |
| B7 | chat/completions синтез (z.ai, вызов 1/3) | PASS | decision=answer, content 2018 зн., «Источники:» + [n] присутствуют |
| C1 | CLI `python -m veles ask "норма управляемости"` | PASS | rc=0, «ANSWER» + «цитаты:»; первая цитата: «Пересказ_Запись_МУОП_1_курс_ДЕСЯТКИ» lectures/Общий менеджмент/… |
| D1 | MCP initialize+tools/list | **KNOWN-ISSUE** | `python -m veles mcp` падает на импорте — см. §5 |
| D2 | MCP tool call knowledge_query | SKIP | не выполнялся: D1 упал на импорте (детерминированно) |
| E1 | GET /api/models WebUI | PASS | 13 моделей; glm-5.3-flash и veles-knowledge присутствуют |
| E2 | WebUI chat model=veles-knowledge (z.ai, вызов 2/3) | PASS | content 2014 зн., «Источники:» присутствуют (сквозной путь WebUI → фасад → синтез) |
| E3 | WebUI chat model=glm-5.3-flash (z.ai, вызов 3/3) | PASS | 200, content 6 зн., в ответе model=glm-5.3-flash |
| E4 | signup закрыт | PASS | POST /api/v1/auths/signup с валидной по форме формой → HTTP 403 (учётка не создаётся) |
| F1 | pair-state моста | PASS | `~/secrets/dm-bridge-state.json` существует, `owner_chat_id` int (значение не печатается) |
| F2 | мост: apply_footer + форматер | PASS | 6/6 флагов: bold, одинарное экранирование `<тег>` и `&`, футер с моделью, живой `<i>` футера, двойного экранирования нет |
| F3 | журнал моста без traceback | PASS | Traceback: 0 (окно 24ч, 10 строк) |
| G1 | бэкап: скрипт + cron | **KNOWN-ISSUE** | cron-строка есть; скрипта по пути из cron нет — см. §5 |
| T1 | Telegram-сквозной (реальная отправка) | MANUAL | not automated: нужен живой пользователь в чате; обращения к Telegram API из тестов запрещены ограничением задачи |

## 4. Найденные и исправленные баги (в тестах/скрипте, не в проде)

1. **B2/E4 ловили 422 вместо 401/403** (тест-баг): POST-тела без заголовка
   `Content-Type: application/json` FastAPI отклоняет 422 до авторизации. Исправлено:
   JSON-заголовок добавлен в `http()` по умолчанию; B2 теперь честно доказывает 401,
   E4 — 403 на валидную форму signup.
2. **E4 был слабым** (тест-баг): пустая форма ловила 422 валидации и «проходила» независимо
   от ENABLE_SIGNUP. Усилен до валидной по форме заявки: 403 = signup закрыт по существу.
3. **MCP-клиент скрипта** переведён с select-поверх-buffered-TextIO (риск подвисания на
   недоконченной строке) на raw-fd построчный ридер.
4. Существующие тесты репо не правились — E2E их слабостей не вскрыл.

## 5. Найденные баги прода (KNOWN-ISSUE, НЕ исправлялись — правки прода за владельцем)

### D1: `python -m veles mcp` сломан — дрейф пакета mcp в venv
- **Факт:** `veles/mcp_server.py:31` делает `from mcp.server.fastmcp import FastMCP`; в
  `~/veles-venv` установлен **mcp 2.2.0**, где FastMCP переименован (`ModuleNotFoundError:
  No module named 'mcp.server.fastmcp' … pin 'mcp<2'`). Процесс умирает до JSON-RPC.
- **Воспроизведение:** `cd ~/proj/veles-bench/veles && ~/veles-venv/bin/python -m veles mcp`
  (stderr: ModuleNotFoundError, см. лог прогона D1).
- **Варианты починки владельцу:** (а) `~/veles-venv/bin/pip install "mcp<2"` — быстрый откат;
  (б) миграция `veles/mcp_server.py` на `from mcp.server.mcpserver import MCPServer`
  (класс проверен — импортируется в venv). После починки прогнать D1/D2 тем же скриптом
  (`--only D1,D2`; z.ai не тратится, use_llm=false).
- **Примечание:** Ф4-1 (27.09) MCP работал — значит, mcp 2.x попал в venv позже; это дрейф
  окружения, не регрессия кода Ф4.

### G1: ночной бэкап не работает — скрипт не установлен
- **Факт:** cron-строка есть (`30 3 * * * <server-home>/bin/veles-backup.sh >> …`), но
  `<server-home>/bin/veles-backup.sh` **отсутствует** (в `~/bin/` его нет), `~/veles-backup.log`
  нет, каталога `~/backups` нет — ни одного бэкапа не создано. Скрипт в репо есть:
  `deploy/стенд/veles-backup.sh`.
- **Починка владельцу:** установить скрипт по пути из cron
  (`install -m 755 deploy/стенд/veles-backup.sh <server-home>/bin/veles-backup.sh`) и проверить
  ручным прогоном; после этого пункт G1 зеленеет тем же скриптом (`--only G1`).

## 6. Регрессия полного тест-набора

- Прод-копия стенд (`~/proj/veles-bench/veles`, канонная команда рунбука): `42 passed` — не сломан.
- Копия ветки (стенд, `~/veles-e2e-check`, после прогона удалена): **50 passed** (42 + 8 новых).
- Существующие тесты не менялись; правок кода `veles/` в ветке нет.

## 7. Ограничения и гигиена прогона

- **z.ai-квота:** 3 вызова glm-5.3-flash на весь E2E (B7, E2, E3; каждый по одному разу,
  max_tokens не ограничивался синтезом фасада). Лимит 6 не превышен. Остальные проверки
  шли без LLM (use_llm=false / --skip-llm).
- **стенд:** GPU не занимался (все процессы с `CUDA_VISIBLE_DEVICES=`); сервисы не
  рестартовались; процессы не убивались; пакеты в venv не ставились; файлы прода не
  менялись. Единственное добавление в прод-дерево — новый файл
  `~/proj/veles-bench/veles/e2e/run_deployed_e2e.py` (сам раннер, требование задачи).
  Временная копия `~/veles-e2e-check` после прогона удалена.
- **Секреты:** читались из `~/secrets/*.env` в память процесса раннера; в выводе, отчёте и
  коммитах не печатались (маскировщик значений + userinfo-URL; строки журнала с паттернами
  секретов не печатаются вовсе — только счётчики).
- **Telegram API:** не вызывался (мост проверен импортом с env-стабами и форматером;
  реальная отправка — T1 MANUAL).
- **Параллельная сессия:** в общем чекауте `<local-pc>\proj\veles` есть чужой
  незакоммиченный WIP (`M clients/telegram-bridge/bridge.py`, `?? tests/test_bridge_commands.py`) —
  не тронут и не закоммичен; коммиты ветки сделаны пофайлово и содержат только файлы E2E.
- **Особенность локального Windows-хоста** (не репо): в site-packages Python лежит
  посторонний пакет `tests`, затеняющий namespace-пакет репозитория — коллекция полного
  набора локально падает на `from tests.test_api import …` и без моих изменений. Канонный
  прогон полного набора — на стенд (там чисто).

## 8. Что осталось ручным

| Пункт | Почему |
|---|---|
| T1 Telegram-сквозной | требуется живой владелец в спаренном чате (отправка реального сообщения); по условию задачи обращения к Telegram API из автоматики запрещены |
| применение фиксов D1/G1 | правки прода (venv/файлы стенд) по ограничению задачи выполняет владелец после ревью; скрипт раннера уже умеет пере-прогонять только эти пункты |
