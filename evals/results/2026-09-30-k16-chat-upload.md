# Eval: K16 - загрузка документов в базу знаний из чата Open WebUI (Pipe Function)

Дата: 2026-09-30 · Ветка: `feature/corp-ai-kb-chat-upload` (от main 778701c) ·
Worktree: `~/proj/veles-k16-chat-upload` (локальный Windows-чекапут трекера veles)

Запрос владельца 30.09: «можно ли перенаправить загрузку из openwebui в нашу
систему? чтобы автоматически, через ту же авторизацию, и документ был личный».

## 1. Факты API функций Open WebUI 0.11.4 (проверено по исходникам контейнера)

Версия на стенд: `GET :8300/api/version` → `{"version":"0.11.4"}`. Контейнер
`open-webui` (ghcr.io/open-webui/open-webui:main), исходники в
`/app/backend/open_webui/`.

- **Pipe-функция**: модуль с классом `Pipe`; `utils/plugin.py::load_function_module_by_id`
  делает `exec(content)` и инстанцирует `module.Pipe()`; вызывается
  `pipe(**params)`, где params = `{'body': form_data}` плюс ТОЛЬКО те
  доп-параметры, что объявлены в сигнатуре `pipe`
  (`functions.py::generate_function_chat_completion` →
  `get_function_params`, фильтр по `inspect.signature`).
- **Файлы**: `metadata = form_data.pop('metadata', {})`,
  `files = metadata.get('files', [])`, передаются параметром `__files__`.
  В `body` файлов НЕТ (в `process_chat_payload` верхнеуровневые `files`
  переезжают в `metadata['files']`). Структура элемента
  (`retrieval/utils.py::get_rag_context` читает так же):
  `{"type": "file", "id": ..., "name": ..., "file": {"data": {"content":
  "<извлечённый WebUI текст>"}}}`.
- **Пользователь**: `__user__ = user.model_dump()` (UserModel): `email`,
  `name`, `role`, `id`.
- **Модель в списке**: каждая активная функция типа `pipe` автоматически
  попадает в `/api/models` (`functions.py::get_function_models`), имя - из
  записи Functions. НО: без записи в таблице Models модель видна ТОЛЬКО
  админам (`utils/models.py::get_filtered_models`: «No DB entry means no
  access control configured yet; only admins can see unconfigured models»).
  Обычному пользователю нужна запись Models с грантом read
  (публичная/группа) - обнаружено живым тестом, см. §3.
- **Фоновые задачи**: WebUI вызывает ТУ ЖЕ модель для генерации
  поисковых запросов по приложенным файлам (task-промпт «Analyze the chat
  history to determine the necessity of generating search queries...»);
  такие вызовы отличаются `metadata['task']` → параметр `__task__`.
- **Admin API**: `POST /api/v1/functions/create` `{id, name, content, meta}`
  (id - python-identifier, создаётся `is_active=false`) →
  `POST /api/v1/functions/id/{id}/toggle` →
  `POST /api/v1/functions/id/{id}/valves/update`;
  запись модели: `POST /api/v1/models/create` c
  `access_grants: [{"principal_type": "user", "principal_id": "*",
  "permission": "read"}]`.
- **Valves**: класс `Valves(BaseModel)` на уровне модуля; админ правит в UI
  (шестерёнка на карточке функции).

## 2. Что сделано (коммиты ветки, 11)

| Коммит | Что |
|---|---|
| 6cc014a | фасад: `X-Veles-On-Behalf-Of` при владельческом bearer → документ личный `user:<email>`; `validated_user_id` в identity; 8 тестов act-as |
| b5401f0 | фикс теста: GET джоба в static требует bearer |
| 284f80d, 2e3db3b, 50f7dd5 | отказ от wildcard у владельца: попытка дать владельцу `*` сломала 5 негативов K4/K7 (access_groups= умеет ограничить документ и от владельца); вместо этого static-act-as документ несёт `[user:<email>, personal]` - владелец находит дефолтными группами, identity-режимы - чисто `user:<email>` |
| fb207a7 | порядок групп в тесте - отсортированный |
| bbff926, 3cd60a1 | Pipe-функция `integrations/openwebui/functions/upload_to_veles.py` + 13 unit-тестов |
| 981c148 | README integrations/openwebui (установка на другом инстансе) |
| 7441c88 | urllib timeout скаляром (живой тест поймал 'tuple object cannot be interpreted as an integer') |
| 36d8515 | guard `__task__`: фоновые задачи WebUI больше не загружают файлы; в инструкции шаг «модель видна только админам без записи в Models» |

Семантика фасада: заголовок принимается ТОЛЬКО при точном
`Authorization: Bearer VELES_API_TOKEN` (доверенный путь: Pipe-функция,
админ-установленная). Без владельческого токена заголовок молча игнорируется
(обычная семантика личности); владельческий токен + мусорный email → 400.
Работает и в static-режиме (стенд живёт в static), и в identity-режимах.
Квоты, каталог L0 (`uploads/<email>/`), джобы и аудит - по email.

## 3. Живой тест на стенд

Прод-фасад `:8400` **не тронут** (старый код, рестарт - в цикле мержа по
согласованию координатора). Для честной проверки семантики K16 поднят
временный фасад ВЕТКИ на `:8401` (`git archive` снапшот ветки в /tmp/k16-pytest,
`VELES_DATA=/tmp/k16-veles-data`, сид-документ + индекс
numpy-flat, токен тестовый). WebUI-функция установлена и оставлена (обратимо,
это и есть задача).

1. Установка: `POST /api/v1/functions/create` (id `upload_to_veles`, name
   «Загрузить в базу знаний») → `toggle` → `is_active: true` → valves
   (veles_url/veles_token/таймауты) → `200`.
2. Модель появилась в `/api/models` админа; у обычного пользователя список
   был ПУСТ (п.1 фактов) → создана запись Models `upload_to_veles` с
   публичным read-грантом → модель видна пользователю.
3. Тестовый пользователь `ivanov@veles.local` (создан через
   `POST /api/v1/auths/add`, удалён после теста).
4. Живые completion-вызовы `POST /api/chat/completions` от имени ivanov с
   файлами (файл - `{type:"file", file:{filename, data:{content}}}`):
   - 1 файл + «Регламент отпусков» → ответ функции за ~2 c:
     «✅ Документ «Регламент отпусков» загружен в базу знаний: доступен в
     поиске, группа доступа - личная (ivanov@veles.local).»
   - 2 файла за раз → два отдельных документа, каждый со своим отчётом;
   - файл без текста (`data.content="   "`) → честный отказ, HTTP к фасаду
     НЕТ («в файле не нашлось текста после извлечения - в базу не
     отправлен»).
5. Проверка на тестовом фасаде:
   - манифесты: `ivanov@veles.local/регламент_отпусков.txt`,
     `.../памятка_командировки.md`, `.../расходы.txt` - все
     `access_groups: [user:ivanov@veles.local, personal]` (static-режим),
     `ingested_by: ivanov@veles.local`, названия чистые;
   - поиск «сколько дней ежегодный отпуск» → `decision: answer`, цитата на
     личный документ ivanov; «какие суточные в командировке» → answer.
6. Найдено и починено живым тестом (до фикса): WebUI вызвал функцию ДВАЖДЫ -
   сначала фоновая задача query-gen (документ получил названием служебный
   промпт, реальная загрузка стала `unchanged`); после guard `__task__` в
   журнале джобов ровно один `created` на документ.

Финальная настройка функции на стенд: valves переключены на ПРОД фасад
(`http://host.docker.internal:8400` + VELES_API_TOKEN из
`~/secrets/veles-facade.env`); тестовый пользователь удалён; тестовые
документы существовали только в `VELES_DATA=/tmp/k16-veles-data` тестового
фасада (каталог и процесс :8401 удаляются после eval, прод не затронут).
Здоровье прода после всех работ: `GET :8400/healthz` → `{"status":"ok",
"index_loaded":true}`.

## 4. стенд: полный pytest ветки

```
cd /tmp/k16-pytest (git archive HEAD 36d8515) && <server-home>/veles-venv/bin/python -m pytest tests -q
291 passed, 1 skipped, 1 warning in 14.96s
```

Новые тесты: `tests/test_documents_act_as.py` (8: personal-doc static, поиск
владельцем, multipart, игнор без токена/без настроенного токена, 401 при
чужом bearer, 400 на мусорный email, identity-режим ACL ivanov/petrov,
манифест), `tests/test_webui_upload_function.py` (14: valves не заданы, нет
email, нет файлов, пустой текст, success c проверкой заголовков
Authorization/X-Veles-On-Behalf-Of и multipart-тела, 401, 413, failed-джоб,
таймаут, мульти-файл, fallback body['files'], картинки, task-guard,
фронтматтер). Регрессии K1-K15 зелёные (включая негативы ACL/ревизий,
которые ловили неудачную wildcard-попытку).

## 5. Что НЕ проверено (MANUAL / границы)

- **Живой чат-тест ИЗ БРАУЗЕРА** владельцем (прикрепить файл в UI,
  модель «Загрузить в базу знаний») - MANUAL: через API файлы переданы
  реалистично (`files` верхнего уровня, как шлёт фронтенд), но клики в UI
  не воспроизводились. Шаги - integrations/openwebui/README.md.
- **Личный документ на ПРОД фасаде стенд** - после рестарта фасада веткой в
  цикле мержа (прод сейчас без K16-кода: заголовок проигнорирует, документ
  ляжет владельцу - это поведение до мержа, не поломка).
- **corporate-режим (headers/jwt) вживую** на стенд - фасад живёт в static;
  ACL identity-режимов (petrov не видит документ ivanov) покрыт юнит-тестами
  на headers-режиме.
- **PDF через чат** - функция берёт извлечённый WebUI текст (для PDF его
  делает WebUI-лоадер); в live-тесте PDF не подкладывался (только txt/md
  с инлайн-контентом). Форматная проверка PDF на фасаде - K4.
- OCR/сканы - вне скоупа фазы (честный отказ функции).

## 6. Откат

Удалить функцию: Admin Panel → Functions → `upload_to_veles` → Delete
(или `DELETE /api/v1/functions/id/upload_to_veles/delete`); запись модели
`upload_to_veles` в Workspace → Models удалить отдельно (если создавалась).
Код фасада - ветка, не смержена.
