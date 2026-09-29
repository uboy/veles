# Veles deploy-kit

Один архив - рабочая установка. Распаковали, запустили `install.sh`, ответили на
вопросы, проверили `verify-install.sh`. Удаляет всё `uninstall.sh`.

## Демо за 5 минут (install --demo)

`--demo` разворачивает базу, уже наполненную демо-контентом: 5 документов
фикшн-компании ООО «Ромашка» (отпуска, командировки, ИТ-доступы, закупки,
FAQ поддержки) с готовым мини-индексом. Реальных данных в демо нет.

```bash
tar xzf veles-deploy-kit-<версия>.tar.gz
cd veles-deploy-kit-<версия>/deploy/kit
./install.sh --demo --yes          # профиль lite (CLI); сеть: pip (CPU-torch, ~200 МБ)
                                   # + модель e5-small с HuggingFace (~130 МБ)
./.venv/bin/veles ask "Сколько дней отпуска положено сотруднику?"
```

В конце установки печатаются 5 проверенных на этом индексе демо-вопросов.
Демо-шаг 2 - свой документ: `./.venv/bin/veles add файл.txt --title "Название"`,
после чего документ ищется сразу (`veles ask`).

Профили с `--demo`: по умолчанию `lite`; `--profile personal --demo` поднимет
фасад с уже наполненной базой; `--profile corporate --demo` скопирует демо-индекс
в `data/` (контейнер подхватит через bind) - сам docker compose запускается
как обычно, отдельной командой владельца на машине с docker.

Важно знать: индекс несёт вектора документов, но для эмбеддинга ВОПРОСА модель
`intfloat/multilingual-e5-small` нужна и на машине демо - при установке она
скачивается с HuggingFace (~130 МБ, кеш `~/.cache/huggingface`); без сети
на huggingface.co демо-вопрос не заработает. Torch ставится CPU-сборкой
(download.pytorch.org); если тот недоступен, pip уйдёт на PyPI и поставит
CUDA-вариант (крупнее на ~5 ГБ). Диск для lite-демо: от 4 ГБ.

Удаление демо целиком: `./uninstall.sh --purge-data` (снесёт данные, venv и .env;
спросит дважды без `--yes`).

## Быстрый старт

```bash
tar xzf veles-deploy-kit-<версия>.tar.gz
cd veles-deploy-kit-<версия>/deploy/kit
./install.sh                 # спросит профиль и минимум вопросов
./verify-install.sh          # таблица OK/FAIL
```

## Три профиля

| Профиль | Что ставит | Кому |
|---|---|---|
| `corporate` | docker compose: фасад + Open WebUI; `--with-bridge` добавит Telegram-мост | сервер компании |
| `personal` | venv + фасад (systemd user-юнит, если есть), WebUI по `--with-webui` | свой Linux/macOS |
| `lite` | только CLI в venv, без демонов и портов | тесты, «как ai-tracker» |

Общие флаги: `--profile corporate|personal|lite`, `--dir PATH` (каталог venv и данных
для lite/personal; corporate ставит всё в каталог кита), `--dry-run`, `--yes`,
`--with-webui`, `--with-bridge`, `--with-voxfusion`, `--demo` (см. «Демо за 5 минут»).
Corporate-only: `--identity-headers` - режим личности через plain-заголовки вместо
дефолтного jwt (см. «Безопасность corporate»).

### Lite: каталог трекера как рабочий каталог агента

Каталог трекера (TRACKER_ROOT, по умолчанию `~/veles-tracker`) - обычный
каталог markdown с git; он же может быть рабочим каталогом кодинг-агента.
Скопируйте туда `templates/AGENTS.md` и подкаталог `templates/skills/`
(в корне каталога трекера, без слова templates): ZCode, Claude Code, Codex
подхватывают AGENTS.md и skills/*/SKILL.md автоматически, и агент получает
правила работы с карточками (формат, статусы, «готово = проверено») и два
скилла - `veles-ask` (вопрос к базе с цитатами) и `tracker-task` (завести
задачу). Тот же приём, что в ai-tracker.

### Бэкапы (corporate - обязательно)

Каталог трекера TRACKER_ROOT (env `TRACKER_ROOT`, по умолчанию
`~/veles-tracker`) - карточки задач и заметок всех пользователей, они
существуют только там. Индекс базы пересобирается из корпуса, а вот трекер
не восстанавливается ни из чего другого - corporate-админ обязан включить
его в регулярный бэкап, например ночным cron (снимок каталога целиком
+ выкидывание копий старше 7 дней):

```bash
TRACKER_ROOT="${TRACKER_ROOT:-$HOME/veles-tracker}"
tar czf "$HOME/veles-tracker-backup-$(date +%F).tar.gz" \
    -C "$(dirname "$TRACKER_ROOT")" "$(basename "$TRACKER_ROOT")"
find "$HOME" -maxdepth 1 -name 'veles-tracker-backup-*.tar.gz' -mtime +7 -delete
```

## Требования к контуру (одна страница)

**corporate**

- Linux с docker и compose v2 (`docker compose version` отвечает).
- Свободный порт 8300 (WebUI). Фасад наружу не публикуется (порт только внутри
  compose-сети).
- TLS (опционально): профиль `--profile tls` - reverse-proxy caddy с авто-HTTPS;
  нужны открытые 80/443 и домен, указывающий на сервер (см. «TLS (corporate)»).
- Диск ≥5 ГБ (минимум preflight); образ фасада с embedding-моделью тяжёлый -
  закладывайте 10+ ГБ. RAM ≥4 ГБ.
- Сеть: docker-реестры (ghcr.io) при первой установке; HuggingFace при первом
  построении индекса; ключ LLM - по адресу провайдера (по умолчанию api.z.ai).
- `openssl` (секреты; есть везде).

**personal**

- Linux/macOS, python 3.11+, bash 4+.
- Порт 8400 свободен (+8300 при `--with-webui`).
- Диск ≥5 ГБ (torch в venv), RAM ≥4 ГБ, `curl`, `openssl`.
- При первом `veles index` - доступ к HuggingFace.

**lite**

- Linux/macOS (на Windows - `windows/install.ps1`), python 3.11+, bash 4+.
- Диск ≥1 ГБ (демо: ≥4 ГБ - CPU-torch и sentence-transformers в venv),
  портов не занимает.
- Сеть: pip при установке (демо ставит CPU-torch с download.pytorch.org
  и модель e5-small ~130 МБ с HuggingFace).

Все профили: ключ LLM спрашивается при установке и пишется только в локальный `.env`
(chmod 600). В архив ключи не встраиваются никогда.

## Безопасность corporate

- **Сеть.** Фасад не публикуется наружу: он доступен только внутри compose-сети
  (`http://veles-facade:8400`), наружу открыт только Open WebUI. Пользователи в базу
  попадают через WebUI; прямых ходов к фасаду из сети нет. Кликабельные ссылки на
  источники (`/v1/source/...`) открываются браузером через reverse-proxy caddy -
  включается TLS-профилем (см. «TLS (corporate)»).
- **Личность.** По умолчанию `jwt`: Open WebUI подписывает пользователя
  (`VELES_IDENTITY_JWT_SECRET`, тот же секрет отдаётся WebUI), фасад проверяет подпись.
  Режим `headers` включается только явно (`--identity-headers`) - заголовкам можно
  подделать личность при прямом доступе к фасаду в обход Open WebUI, поэтому такой
  режим требует сетевого ACL; инсталлятор об этом предупреждает.
- **Fail-closed.** Инсталлятор не поднимет corporate без аутентификации:
  `VELES_IDENTITY_MODE` обязан быть `jwt`/`headers` (static запрещён), bearer-токен
  генерится всегда. Дубль проверки в самом фасаде (старт не-loopback без токена
  отклоняется кодом).
- **Секреты.** `VELES_API_TOKEN`, `VELES_LINK_SECRET`, `WEBUI_SECRET_KEY`,
  `VELES_IDENTITY_JWT_SECRET` генерируются установщиком (`openssl rand -hex 32`) и
  хранятся только в `.env` (chmod 600). `VELES_LINK_SECRET` - подписанные ссылки на
  источники, генерится всегда.

## Группы доступа (K7)

При загрузке документа (`POST /v1/documents`, поле `group`) пользователь выбирает
**группу доступа** - документ увидят только те, у кого эта группа есть. Список
групп для UI: `GET /v1/groups`.

- **Удаление (K10).** `DELETE /v1/documents/{document_id}` (или CLI
  `veles delete <document_id> --yes`): доступно владельцу документа
  (`user:<uid>` в access_groups) или админу; чужому - 404. Канон со всеми
  ревизиями переезжает в `<VELES_DATA>/trash/<timestamp>-<id>/` (не безвозвратно,
  восстановление вручную админом), L0 удаляется, документ исчезает из поиска
  сразу и не всплывает после рестарта/пересборки.
- Дефолтно созданы три группы: `all` («видно всем» - неявно есть у каждого
  аутентифицированного), `on-request` («видно по запросу» - выдача доступа вне
  системы), `test-dept` («тестовый отдел»).
- **Состав** не-all групп правит админ в файле `<VELES_DATA>/groups.json`
  (файл-шаблон создаётся при первом `GET /v1/groups`; фасад перечитывает его на
  каждый запрос - рестарт не нужен):

  ```json
  {
    "groups": [
      { "id": "all", "title": "Видно всем", "members": [] },
      { "id": "on-request", "title": "Видно по запросу", "members": [] },
      { "id": "test-dept", "title": "Тестовый отдел",
        "members": ["ivanov@corp.ru", "petrova@corp.ru"] }
    ]
  }
  ```

  `id` - ascii-слаг `[a-z0-9][a-z0-9_-]*`, `members` - identity (email) сотрудников.
  Битый файл не перезаписывается и не роняет фасад: действуют дефолтные группы
  (membership пуст - документы групп становятся невидимы, т.е. fail-closed).
  Ещё два fail-closed случая: группа удалена из конфига - её документы видит
  только админ (группа живёт в манифесте документа); в static-режиме группы из
  `VELES_ALLOWED_GROUPS`, определённые в `groups.json`, владельцу НЕ
  добавляются (он и так админ, `*`).
- Без `group` документ загружается в **личную** группу `user:<email>` - виден
  только автору (и админу). Неизвестная группа - `400`; одновременно указать
  `group` и `access_groups` - тоже `400`.
- Open WebUI 0.11.4 **не передаёт** группы пользователя в заголовках/JWT
  (`ENABLE_FORWARD_USER_INFO_HEADERS` пересылает только name/id/email/role/JWT),
  поэтому состав групп - только серверный конфиг выше. Группы WebUI можно
  подмешать кастомными заголовками с плейсхолдерами `{{USER_GROUP_IDS}}`, но
  это отдельная настройка и отдельное решение.
- Админ (`VELES_ADMIN_EMAILS`) видит все документы; bearer-владелец - все
  определённые группы.

## TLS (corporate)

Корпоративный контур снаружи ходит по HTTPS. Для этого в ките есть опциональный
reverse-proxy caddy (авто-выпуск и продление сертификатов, Let's Encrypt) отдельным
compose-профилем: `--profile tls`. Наружу открываются только 80/443 самого caddy;
фасад и WebUI остаются внутри compose-сети.

```bash
cd deploy/kit
cp caddy/Caddyfile.example caddy/Caddyfile
# в .env: VELES_TLS_DOMAIN=veles.example.com   (домен указывает на этот сервер)
#         VELES_WEBUI_BIND=127.0.0.1           (наружу хоста - только caddy)
docker compose --profile tls up -d caddy
```

- Маршрутизация: `/v1/*` и `/tracker*` (API и HTML-страницы фасада - по ним браузер
  открывает источники из цитат) идут на veles-facade, остальное - в Open WebUI.
- Сертификат выпускается автоматически (ACME HTTP-01 через порт 80), хранится в
  `data/caddy` и переживает перезапуск. Домен обязан реально указывать на сервер
  и иметь открытый 80-й порт.
- `VELES_PUBLIC_URL` в `.env` поставьте `https://<домен>` - подписанные ссылки в
  цитатах будут вести на домен, а не на внутренний адрес.
- Без TLS-профиля установка работает как раньше: WebUI на 8300, caddy не запускается.
- Проверка конфигурации без запуска: `docker compose --profile tls config`.

## После установки

- Адреса и команды печатает `install.sh`; повторная проверка - `verify-install.sh`.
- `.env` рядом со скриптами: порты, секреты, ключи. Правьте и перезапускайте.

### Наполнение базы

Индекс строится там, где есть корпус: `veles ingest && veles index`
(корпус задаёт `VELES_CORPUS_ROOT`, данные - `VELES_DATA`). Готовый индекс -
каталог `data/index`: corporate-профиль подхватит его из `<kit>/data/index`
(bind в контейнер), personal/lite читают из `VELES_DATA`. Без индекса фасад
здоров (`/healthz`, `/v1/models`), но запросы отвечают, что базы нет.

Для corporate индекс собирается ЗАРАНЕЕ (на машине с корпусом - lite-профилем
или на своей машине) и копируется на сервер: `<kit>/data/index` до `install.sh`.
Внутри corporate-контейнера ингеста нет - в compose не монтируется корпус
нарочно: контур заказчика получает готовый индекс, а не сырьё.

`veles ingest` ждёт корпус определённой структуры: `<CORPUS_ROOT>/<спек>/файлы`,
где спеки перечислены в allowlist (`veles/ingest/allowlist.py`: `module1`, `um`
и др. - каталоги файлов лекций/нормативки). Ошибка «корпус не найден» печатает
подсказку; свою структуру смотрите там же.

### Загрузка документов

Точечное наполнение базы без пересборки индекса - через фасад (K4/K7) или
Telegram-бота (K9). Загруженный документ индексируется очередью фасада и
сразу виден в поиске тем, кому доступен (по умолчанию - только загрузчику).

API (multipart, форматы txt/md/pdf):

```bash
# личный документ (виден только загрузчику)
curl -s -H "Authorization: Bearer $VELES_API_TOKEN" \
  -F "file=@отчёт.pdf" -F "title=Отчёт за сентябрь" \
  http://127.0.0.1:8400/v1/documents          # -> 202 {"job_id": ...}

# документ серверной группы (id из GET /v1/groups) - видно только её членам
curl -s -H "Authorization: Bearer $VELES_API_TOKEN" \
  -F "file=@регламент.md" -F "group=test-dept" \
  http://127.0.0.1:8400/v1/documents

# статус джоба: queued/running/done/failed (+ причина отказа в error)
curl -s -H "Authorization: Bearer $VELES_API_TOKEN" \
  http://127.0.0.1:8400/v1/documents/jobs/<job_id>
```

Telegram-сценарий (мост): перетащить файл (txt/md/pdf, до 20 МБ - лимит
Telegram на скачивание ботом) в чат с ботом -> бот отвечает «Принял ...,
загружаю...», по завершении присылает «✅ Загружено: <title> (группа ...)»
или причину отказа. Группа для загрузок этого чата: `/docgroup` (показать
текущую и список), `/docgroup <id>` (выбрать, persist в state моста). Без
выбора документ уходит в личную группу загрузчика.

### Транскрипция (голосовые)

Отдельный сервис VoxFusion (свой репозиторий, HTTP `POST /v1/transcribe`).
kit его не ставит. После установки впишите в `.env` моста `VOXFUSION_API_URL`
и `VOXFUSION_API_TOKEN` - голосовые сообщения бота пойдут в транскрибацию.

## Удаление

```bash
./uninstall.sh                          # остановить, данные и .env сохранить
./uninstall.sh --purge-data             # + удалить данные; спросит дважды
./uninstall.sh --purge-data --yes       # без вопросов
```

Без `--purge-data` сохраняются: `data/` (база, трекер, WebUI), `.env`. С ним -
удаляются и проверяется, что портов и файлов не осталось.

## Обновление

Данные и `.env` переживают обновление: распакуйте новый архив рядом (или поверх),
запустите `install.sh` с тем же профилем - существующий `.env` сохраняется,
недостающие ключи дополняются. Откат - распаковать архив предыдущей версии тем же
способом. Какой коммит в архиве - `COMMIT.txt` в корне.

## Брендинг

WebUI по умолчанию называется **Veles** (`WEBUI_NAME=Veles`). Лицензия Open WebUI
разрешает ребрендинг при использовании до 50 пользователей - для наших контуров
(personal и корпоративный до 50) это допустимо. Позиция зафиксирована владельцем
28.09.

## Windows (personal/lite)

`windows/install.ps1` - venv + CLI + фасад (run-facade.ps1); `--WithWebUI` ставит
open-webui в venv. Параметры: `Get-Help .\install.ps1 -Full`; снятие -
`windows/uninstall.ps1` (`-PurgeData` удаляет venv, `.env` и данные).

Отличия от bash-версии (осознанные): связка WebUI с фасадом через OPENAI_API_BASE_URLS
не прописывается автоматически (настраивается в UI после `-WithWebUI`); RAM-префлайт
и проверка порта отсутствуют. Ввод ключей без эха в bash; в PowerShell ключ виден
при вводе - ставьте на машине, которой доверяете экран.

## Что где лежит

```
deploy/kit/
  install.sh uninstall.sh verify-install.sh   # установка, снятие, самопроверка
  make-kit.sh                                 # сборка архива из git HEAD
  demo/                                       # демо в один клик (--demo)
    documents/                                #   5 фикшн-документов (ООО «Ромашка»)
    index/ canonical/                         #   готовый мини-индекс + канон
    questions.txt                             #   5 проверенных демо-вопросов
    build-demo-index.sh                       #   пересборка мини-индекса
  docker-compose.yml Dockerfile Dockerfile.bridge
  caddy/Caddyfile.example                     # шаблон reverse-proxy TLS (профиль tls)
  .env.example                                # все переменные с комментариями
  lib/common.sh                               # общие функции
  windows/install.ps1 windows/uninstall.ps1   # Windows-профиль
  templates/AGENTS.md templates/skills/       # правила + скиллы (veles-ask, tracker-task)
                                              # в каталог трекера - см. «Lite: каталог трекера…»
```

Транскрипция голосовых (voxfusion-api) - внешний сервис, в кит не входит: поднимите
его отдельно и укажите `VOXFUSION_API_URL`/`VOXFUSION_API_TOKEN` в `.env` - мост
начнёт принимать голосовые сам.
