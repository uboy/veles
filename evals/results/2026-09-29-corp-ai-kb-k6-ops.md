# Eval K6: эксплуатационный минимум (ops hardening) — 29.09.2026

Фаза K6 эпика corp-ai-kb. Ветка `feature/corp-ai-kb-ops`, коммиты 92127f9
(rate-limit + security-заголовки + readiness + тесты), 9e26ea9 (TLS-профиль кита),
18acfda (HEAD на /tracker). Бриф: `docs/plan/briefs/2026-09-29-k6-ops-hardening.md`.

Параметры прогонов: полный `pytest tests -q` на стенд из `git archive HEAD | tar`
в /tmp (venv <server-home>/veles-venv, `CUDA_VISIBLE_DEVICES=`); живой негатив —
тест-инстанс на стенд, порт 18420, tmp VELES_DATA/TRACKER_ROOT, VELES_RATE_LIMIT_RPM=3.
Прод стенд не тронут. Docker на Windows-хосте отсутствует — compose-проверка на стенд.

## 1. Rate-limit фасада

Модуль `veles/api/ratelimit.py`: скользящее окно в памяти (deque меток времени,
threading.Lock), per-identity (headers/jwt — ключ id:<user_id>) / per-token
(static — bearer или token=, ключ sha256-префикс, токены в открытом виде не
хранятся) / подпись ссылки (sig=) / IP как последний фолбэк. Подключение в
`create_app` — одна строка `install_rate_limit(app, state)`. Превышение — 429 +
Retry-After (секунды до ухода самого старого попадания из окна). VELES_RATE_LIMIT_RPM:
пусто = 240 (щедрый дефолт), 0 и мусор = выкл. /healthz, /readyz вне лимита —
проба мониторинга, отвалившаяся по 429, читается как сбой сервиса.

Тесты (`tests/test_ops.py`, 8 шт.): окно скользит (в лимите 200-эквивалент, 4-й —
отказ с retry_after=58 по точному фейковому времени, после окна — снова пропуск),
ключи независимы; 0/мусор = выключено; HTTP-негатив: 3-й запрос (RPM=2) → 429 с
целым Retry-After ≥ 1; per-identity: лимит user A не блокирует user B; пробы
/healthz//readyz не лимитируются.

## 2. Security-заголовки на /tracker

На HTML-ответ `/tracker` добавлены: `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
`Content-Security-Policy: default-src 'none'; frame-ancestors 'none';
base-uri 'none'; form-action 'none'`. JSON-эндпоинты и ошибки не тронуты
(заголовки ставятся только на HTMLResponse страницы — тест «нет заголовков на
/healthz и /v1/models» зелёный).

### Решение по CSP

Бриф допускал unsafe-inline из-за инлайн-стилей/скриптов на странице трекера.
Проверка фактического рендера (veles/api/app.py, tracker_page): страница строится
только из текстовых узлов и ссылок — инлайн-стилей и скриптов НЕТ (emoji в ссылках
— это контент, не стиль). Честный строгий CSP `default-src 'none'` проходит без
unsafe-inline: переходы по `<a href>` CSP fetch-директивами не ограничиваются, а
встроенного кода, который надо было бы выносить в статику, не существует. Вынесение
инлайнов не потребовалось (оценка объёма: 0). Итог: CSP строгий, unsafe-inline
не нужен, альтернатива не понадобилась. Это решение зафиксировано комментарием в
app.py рядом с заголовками.

Вне фазы: HTML-страница `/v1/source` (текстовая сборка) содержит инлайн-скрипт
скролла к анкеру и осталась без заголовков — scope брифа (/tracker), отдельное
решение, если она пойдёт наружу без доверенного прокси.

`/tracker/file` отдаёт `text/plain` (не HTML) — заголовки по брифу не нужны.

### HEAD

`curl -I /tracker` на живом инстансе упёрся в 405: FastAPI не добавляет HEAD к
GET-маршрутам автоматически (в отличие от чистого Starlette). Исправлено минимально:
`@app.api_route("/tracker", methods=["GET", "HEAD"])` (18acfda); заголовки на HEAD
те же, тест в test_ops.

## 3. TLS-профиль кита

`deploy/kit/docker-compose.yml`: сервис `caddy` (image caddy:2) под профилем `tls`,
наружу только 80/443 caddy; фасад (expose 8400) и WebUI остаются в compose-сети.
WebUI публикуется через `${VELES_WEBUI_BIND:-0.0.0.0}:${VELES_WEBUI_PORT:-8300}:8080`:
дефолт — прежнее поведение, при TLS в .env ставится `VELES_WEBUI_BIND=127.0.0.1`
(наружу хоста смотрит только caddy; verify-install.sh с localhost продолжает
работать — проверено ниже по config-выводу host_ip). Шаблон `deploy/kit/caddy/
Caddyfile.example` (переменная VELES_TLS_DOMAIN; /v1/* и /tracker* → veles-facade,
остальное → open-webui; websockets проксируются caddy автоматически). README:
раздел «TLS (corporate)» + строка в «Требования к контуру» + правка устаревшей
фразы «тема K6» в «Безопасности corporate» + строка в «Что где лежит».
.env.example: VELES_RATE_LIMIT_RPM, VELES_TLS_DOMAIN, VELES_WEBUI_BIND.
Caddyfile.example зафиксирован LF в .gitattributes (иначе CRLF ломает Caddy на
Linux; make-kit чистит \r только у .sh/.py/.service/.md).

Проверки (стенд, docker есть там):

```
$ cd /tmp/veles-k6-test/deploy/kit && cp .env.example .env
$ docker compose --profile tls config >/dev/null && echo OK        → CONFIG_OK
$ docker compose --profile tls config --services                   → veles-facade open-webui caddy
$ docker compose config --services                                 → veles-facade open-webui (caddy не входит)
$ VELES_WEBUI_BIND=127.0.0.1 docker compose --profile tls config   → ports webui: host_ip: 127.0.0.1, published: "8300"
```

`bash -n install.sh verify-install.sh` — ок (скрипты не менялись). `docker compose
up` не делался (по брифу — на тест-сервере владельца).

## 4. Readiness

`_pipeline()` создаёт VelesPipeline только после успешной `load_index()` — это и
есть честный факт готовности (`state["pipeline"] is not None`, под тем же локом).
`/healthz` остаётся живостью (200 всегда), контракт расширен обратно-совместимо
полем `index_loaded` (старый тест e2e «/healthz 200» зелёный); добавлен `GET
/readyz`: 200 `{"status":"ready","index_loaded":true}` с индексом, 503
`{"status":"starting","index_loaded":false}` без. Мост и systemd-юниты не тронуты,
деплой-скрипты не тронуты (compose healthcheck ходит по /healthz — не изменился).

## Прогон стенд (полный сет)

```
$ git archive HEAD | ssh стенд "rm -rf /tmp/veles-k6-test && mkdir -p /tmp/veles-k6-test && tar -x -C /tmp/veles-k6-test"
$ cd /tmp/veles-k6-test && CUDA_VISIBLE_DEVICES= <server-home>/veles-venv/bin/python -m pytest tests -q
→ 134 passed, 1 warning in 7.08s   (из них 10 новых — tests/test_ops.py, остальной сет без изменений)
```

Локальный Windows-хост: `python -m pytest tests -q` → 133 passed, 1 skipped
(skipped — платформенный тест, не из K6; 134 vs 133 = разница окружений, не кода).

## Живой негатив на тест-инстансе (стенд, :18420, RPM=3)

```
$ VELES_DATA=/tmp/veles-k6-live TRACKER_ROOT=/tmp/veles-k6-live/tracker \
  VELES_RATE_LIMIT_RPM=3 python -m veles serve --host 127.0.0.1 --port 18420
$ curl -s -w '%{http_code}' http://127.0.0.1:18420/readyz
  → {"status":"starting","index_loaded":false} 503          # индекс в tmp VELES_DATA не строился
$ curl -sI http://127.0.0.1:18420/tracker
  → HTTP/1.1 200 OK
    x-content-type-options: nosniff
    x-frame-options: DENY
    referrer-policy: no-referrer
    content-security-policy: default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'
    content-type: text/html; charset=utf-8
# бюджет 3: /tracker (HEAD) + /tracker (GET) + /v1/models
$ curl /v1/models                                            → 429
$ curl -i /v1/models
  → HTTP/1.1 429 Too Many Requests
    retry-after: 60
    {"detail":"слишком много запросов, повторите позже"}
$ curl /healthz (дважды после 429)                           → 200, 200 (проба вне лимита)
```

После проверок: процесс убит, /tmp/veles-k6-test, /tmp/veles-k6-live и дамп
config-вывода удалены (`ls /tmp/veles-k6-*` — пусто, leftover-процессов нет).
Прод стенд не трогался.

## Не проверено

- **Реальный `docker compose --profile tls up`** — по брифу только на тест-сервере
  владельца: фактический выпуск сертификата ACME, маршрутизация под живым caddy и
  поведение `VELES_WEBUI_BIND=127.0.0.1` при реальном up не проверялись (проверен
  только `config` и интерполяция переменных).
- **Продление/перевыпуск сертификатов caddy** во времени.
- **Rate-limit под реальным multi-user-трафом**: per-identity проверен тестом и на
  static-инстансе; живая проверка в headers/jwt-режиме с двумя реальными
  пользователями не проводилась.
- **429 при переполнении памяти процесса**: окно в памяти — рестарт фасада
  сбрасывает счётчики (осознанно, задокументировано в докстринге модуля); поведение
  при долгой жизни и многих identity (рост dict) не нагружалось — лимитеров столько,
  сколько пользователей, записи чистятся по окну, но только при обращении к ключу.
- **Retry-After точность в бою**: значение 60 c на живом инстансе согласуется с
  расчётом, но истечение окна в реальном времени не ждали (проверено тестом на
  фейковом времени).

## Открытые вопросы к владельцу

1. Дефолт VELES_RATE_LIMIT_RPM=240/мин на личность — ок для пилота или ужесточить
   (решение влияет только на корпоративный контур, personal по умолчанию не лимитируем
   щедрее — тем же значением)?
2. `/v1/source` (HTML с инлайн-скриптом) остался без security-заголовков — закрывать
   в рамках внешнего пилота (за caddy) или завести отдельную задачу на вынос скролла
   в статику?
