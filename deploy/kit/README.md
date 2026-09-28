# Veles deploy-kit

Один архив - рабочая установка. Распаковали, запустили `install.sh`, ответили на
вопросы, проверили `verify-install.sh`. Удаляет всё `uninstall.sh`.

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
`--with-webui`, `--with-bridge`, `--with-voxfusion`.
Corporate-only: `--identity-headers` - режим личности через plain-заголовки вместо
дефолтного jwt (см. «Безопасность corporate»).

## Требования к контуру (одна страница)

**corporate**

- Linux с docker и compose v2 (`docker compose version` отвечает).
- Свободный порт 8300 (WebUI). Фасад наружу не публикуется (порт только внутри
  compose-сети).
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
- Диск ≥1 ГБ, портов не занимает, сети не требует (кроме pip при установке).

Все профили: ключ LLM спрашивается при установке и пишется только в локальный `.env`
(chmod 600). В архив ключи не встраиваются никогда.

## Безопасность corporate

- **Сеть.** Фасад не публикуется наружу: он доступен только внутри compose-сети
  (`http://veles-facade:8400`), наружу открыт только Open WebUI. Пользователи в базу
  попадают через WebUI; прямых ходов к фасаду из сети нет. Кликабельные ссылки на
  источники (`/v1/source/...`) требуют reverse-proxy перед фасадом - тема K6.
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
  docker-compose.yml Dockerfile Dockerfile.bridge
  .env.example                                # все переменные с комментариями
  lib/common.sh                               # общие функции
  windows/install.ps1 windows/uninstall.ps1   # Windows-профиль
```

Транскрипция голосовых (voxfusion-api) - внешний сервис, в кит не входит: поднимите
его отдельно и укажите `VOXFUSION_API_URL`/`VOXFUSION_API_TOKEN` в `.env` - мост
начнёт принимать голосовые сам.
