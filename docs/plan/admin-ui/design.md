# Дизайн: админ-панель veles (/admin)

Дата: 2026-10-11 · Ветка: feature/admin-ui (worktree C:/Users/devl/proj/veles-admin-ui)
Основа: чтение veles/api/app.py, veles/api/documents.py, veles/api/groups.py,
veles/api/static/upload.js, upload_page.html, upload.css, veles/ingest/queue.py,
tests/test_upload_ui.py. Принцип: простота — плотные таблицы, минимум цвета,
vanilla HTML+JS+CSS, один файл страницы, без сборочных инструментов.

## 1. Размещение и доступ

- Маршрут GET /admin в veles/api/app.py по образцу upload_page (app.py:1022):
  veles/api/admin_page.html читается с диска, отдаётся через HTMLResponse +
  html_security_headers(allow_self_scripts=True) (app.py:94).
- Статика: veles/api/static/admin.js и admin.css по маршрутам /static/admin.js,
  /static/admin.css через существующий _static_page_asset (app.py:1044) —
  та же проверка имени (traversal отсечён) и CSP.
- Страница — пустой shell без секретов, отдаётся без токена. Токен админа
  (VELES_API_TOKEN) вводится в поле (type=password, кнопка «Сохранить»),
  хранится в localStorage; каждый запрос несёт Authorization: Bearer <токен>.
  Токен в URL не используется (отличие от /upload) — не попадает в логи/историю.
- CSP уже подходит: connect-src 'self' (app.py:107); инлайн-скриптов и стилей
  нет; любое текстовое поле экранируется esc() как в upload.js:10.

## 2. Разделы панели (вкладки, каждая — рабочая)

Шапка: заголовок, поле токена, строка состояния API, навигация по вкладкам
(показ/скрытие секций в admin.js, без перезагрузки страницы).

### Обзор
- GET /healthz → status, index_loaded (app.py:743); GET /readyz → 200/503
  (app.py:749). Версия фасада: GET /openapi.json → info.version (same-origin,
  connect-src 'self' разрешает).
- Модели: GET /v1/models → data[].id (app.py:756) — «модель синтеза» панели.
- Итоги: GET /v1/knowledge/stats → totals.documents / nodes /
  superseded_nodes, embed_model (app.py:828-837).
- Вид: одна плотная таблица «ключ — значение», статусы healthz/readyz бейджами.

### Документы
- Данные: GET /v1/knowledge/stats → documents[] (document_id, title, nodes,
  groups — см. п. 4). Поиск по названию — клиентский фильтр по подстроке,
  без обращений к API.
- Таблица: Название | Группы | Узлов | id.
- Удаление: кнопка в строке → confirm(«Удалить „<title>“? Узлы выйдут из
  поиска сразу, канон переедет в trash.») → DELETE /v1/documents/{document_id}
  (documents.py:393). Результат (nodes, revisions, trash, documents.py:466)
  — строкой под таблицей; затем обновить список.

### Задачи
- GET /v1/documents/jobs (documents.py:369): админ видит все, свежие первыми,
  лимит 200 (queue.py:544).
- Колонки из записи джоба (queue.py:343): created_at, filename/title, user_id,
  status (queued/running/done/failed), action, revision_id, nodes, error.
- Кнопка «Обновить»; автопов раз в 5 с, пока есть queued/running.
- Клик по строке → GET /v1/documents/jobs/{id} (documents.py:379) → блок
  деталей: document_id, size_bytes, groups, started_at/finished_at, error.

### Группы
- GET /v1/groups → id + title (groups.py:27). Документов в группе — клиентский
  подсчёт по documents[].groups из stats. Таблица: id | Название | Документов.
- Плашка в интерфейсе: состав групп правится файлом <VELES_DATA>/groups.json,
  API создания/правки групп нет (groups.py:32).

### Вопрос
- Поля: вопрос, top_k (по умолчанию 5), опционально «от имени пользователя» —
  заголовок X-Veles-On-Behalf-Of (принимается только с владельческим bearer,
  app.py:515; без него молча игнорируется).
- POST /v1/knowledge/query (app.py:839). Вид: бейдж вердикта decision
  (answer / warn_conflict / нет ответа + diagnostics.reason), текст answer,
  список цитат: [n] title, source_id, строки, ссылка source_url из данных
  цитат (attach_citation_links, app.py:249); если ссылки нет — без ссылки.
- Ниже — diagnostics: confidence, llm_used, conflicts (app.py:226).

### Загрузка
- Встроенная форма: файл (.txt/.md/.pdf), название, группа (select из
  GET /v1/groups, пусто = личная user:<uid>, documents.py:144) →
  POST /v1/documents multipart (documents.py:197) → задача появляется во
  вкладке «Задачи».
- Ссылка на /upload не даётся: та страница ожидает ?token= в URL, что
  противоречит п. 1 (токен не в URL).

## 3. Чего панель НЕ делает (честно показывается текстом)

- Создание/переименование/удаление групп — в API нет; правка вручную в
  <VELES_DATA>/groups.json (плашка во вкладке «Группы»).
- Управление пользователями — в API нет: личности приходят из заголовков
  identity-режимов (app.py:9); в панели виден только user_id из джобов
  (плашка в футере «Обзора»).

## 4. Структура бэкенда

Новые файлы:
- veles/api/admin_page.html — разметка вкладок (по образцу upload_page.html).
- veles/api/static/admin.js — токен/localStorage, вкладки, fetch, esc().
- veles/api/static/admin.css — по образцу upload.css: плотные таблицы,
  минимум цвета; prefers-color-scheme достаётся бесплатно (upload.css:12).

Правки существующих:
- app.py: маршрут GET /admin + два маршрута /static/admin.* через
  _static_page_asset.
- knowledge_stats (app.py:781): в каждый элемент documents[] добавить
  groups — объединение access_groups живых узлов документа из того же
  снапшота node_meta под INDEX_LOCK (app.py:802). Это ЕДИНСТВЕННОЕ изменение
  данных API: оно кормит и «Документы» (колонку Группы), и «Группы»
  (агрегат документов по группам считается клиентом). Новых
  JSON-эндпоинтов не нужно.

Переиспользуются без правок: /healthz, /readyz, /v1/models, /v1/groups,
POST и DELETE /v1/documents, /v1/documents/jobs[/{id}],
/v1/knowledge/query, /openapi.json.

## 5. Ошибки

- 401 от любого запроса (app.py:502): сообщение «Токен недействителен или
  истёк — введите заново», фокус в поле токена; сохранённое значение не
  стирается до успешного запроса.
- Сбой сети / 5xx: баннер сверху «API недоступен», вкладки показывают
  последние данные с пометкой времени загрузки.
- 503 /readyz: в «Обзоре» статус «индекс загружается», остальные вкладки
  работают (healthz при этом жив, app.py:743).
- 429 (rate limit): «Слишком много запросов — повторите позже».
- detail из ответа API показывается как есть — тексты бэкенда русские.

## 6. Тесты (критерий готовности фазы)

- tests/test_admin_ui.py по образцу test_upload_ui.py: /admin отдаёт 200,
  CSP содержит script-src 'self' без unsafe-inline; /static/admin.js и
  /static/admin.css — 200 с верными content-type; /static/..%2F..%2Fapp.py
  отсечён (400/404).
- stats: у документа в ответе есть groups (админ видит группы документа).
- Негатив: при настроенном VELES_API_TOKEN запрос всех API-вкладок без
  bearer даёт 401 (app.py:502-504) — панель показывает просьбу ввести токен.
