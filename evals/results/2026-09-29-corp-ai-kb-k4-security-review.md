# Ревью безопасности K4 (user-ingest: POST /v1/documents, очередь, upsert) — 29.09.2026

Объект: ветка `feature/corp-ai-kb-user-ingest`, коммит `c13092b` (база `d290963`) —
загрузка документов пользователем (multipart/JSON), фоновая очередь с append-журналом
и boot-id, инкрементальный upsert + overlay, квоты, ACL из identity (личная группа
`user:<uid>`). Проверялись `veles/api/app.py`, `veles/api/documents.py`,
`veles/ingest/queue.py`, `pyproject.toml`, eval фазы.

## Вердикт: FAIL (P0 — приватность на путях чтения)

Фиксы — в этом же eval, ниже (раздел «Фиксы (тот же заход)»).

## Находки

- **[P0] ACL enforced только в `hybrid_search`; три пути чтения о нём не знают.**
  `veles/api/app.py`: фильтр групп стоит в ретривере (`retriever.py:61`), поэтому
  чужой документ невидим в поиске, но:
  - `source_view` (~430-503): `GET /v1/source/<document_id>` отдаёт оригинал из
    VELES_SOURCE_ROOTS и HTML-сборку из канона любому аутентифицированному
    пользователю, зная document_id (утекает через список джобов админа, аудит,
    перебор слагов) — манифест с `access_groups` читается, но по группам не
    проверяется;
  - `asset_file` (~405-428): `GET /v1/asset/<document_id>/<name>` — то же, актив
    документа отдаётся без сверки с манифестом;
  - `knowledge_stats` (~534-570): `/v1/knowledge/stats` строит оглавление из всей
    `node_meta` — чужой пользователь видит titles/document_id/объёмы всех загрузок
    (включая личные `user:<uid>` других пользователей).
- **[P1] Размер запроса не ограничен до чтения; тело целиком в RAM до check_quota;
  синхронная запись на событийном цикле.** `veles/api/documents.py` ~105-156:
  Content-Length не проверяется — `await upload.read()` / `await request.json()`
  поднимают тело целиком в память ДО квоты (нет капа — один запрос может съесть
  произвольный объём RAM); далее `l0_path.write_bytes(raw)` — блокирующая запись
  на событийном цикле (все запросы ждут диск).
- **[P2] TOCTOU квоты в загрузке.** `check_quota` (queue.py:103) и запись L0
  (documents.py:156) не атомарны: два одновременных POST одного пользователя оба
  проходят проверку до записи — квота превышается (признано в eval K4,
  «Не проверено»).
- **[P2] `list_jobs` (queue.py:344-352) не применяет boot-id-семантику lookup:**
  джоб чужого boot_id в queued/running показывается в списке как живой — вечный
  running в списке, хотя точечный lookup честно вернул бы failed «сервис
  перезапущен».
- **[P2] Упавший джоб оставляет L0-файл (queue.py ~449-453):** ошибка парса ->
  failed, но файл в `uploads/<user>/` остаётся — квота пользователя съедена
  навсегда битым файлом.
- **[P2] `pyproject.toml`: `python-multipart>=0.0.9`** — нижняя граница старше
  DoS-фиксов (0.0.18); на стенд стоит 0.0.32, но свежая установка может вытянуть
  уязвимую.
- **[P2] Арифметика бейслайна в eval фазы** (`2026-09-29-corp-ai-kb-k4-user-ingest.md`
  ~84-92): «139 vs 116 (+15)» — 116 унаследована от K3-мерджа, базой K4 является
  `d290963` = 124 passed; правильно 124+15=139.

## Признано корректным (обходов не найдено)

- Поисковые пути: `hybrid_search` отсекает чужие группы до формирования evidence
  (включая `revision=all`); `*` только у админа; личная группа `user:<uid>`
  делает чужие загрузки невидимыми (тест `test_acl_other_user_invisible_everywhere`).
- Джобы: чужой job_id неотличим от несуществующего (404), список джобов фильтруется
  по user_id; boot-id-семантика lookup корректна; журнал append-only, битые строки
  пропускаются.
- `safe_filename` закрывает обход каталогов и скрытые файлы; document_id в путь
  попадает только через `user_dir`+`safe_filename`/`slugify_title`.
- Подписанные ссылки: HMAC от document_id|name_or_node|exp, compare_digest,
  срок через exp — sig скоупит конкретный документ+узел, поэтому групповая
  проверка на sig-пути не применяется (ссылка выдаётся из цитат, уже
  отфильтрованных ACL).
- access_groups назначает только сервер (инвариант 5): клиентский параметр —
  только админ, остальным 403.

## Фиксы (тот же заход)

Ветка `feature/corp-ai-kb-user-ingest`, коммиты по находкам; публичные контракты
эндпоинтов не менялись (кроме добавленных 413 на неподъёмный размер). Прогон —
стенд (Ubuntu, `<server-home>/veles-venv`), код `git archive HEAD | ssh tar -x -C
/tmp/veles-k4-fix`; прод-стек стенд не затронут, каталог и процессы вычищены.

| Находка | Коммит | Чем доказано |
|---|---|---|
| P0 source/asset/stats без ACL | `927ade6` | `source_view`/`asset_file` сверяют группы манифеста с личностью (`_visible_by_groups`): чужому 404, неотличимый от несуществующего; sig-ссылка (ident None) групповой проверке не подлежит — sig скоупит документ+узел; админ/`*`/bearer-владелец видят всё как раньше (static не сломан). `knowledge_stats` фильтрует узлы тем же фильтром, totals.nodes — по видимым. Тест `test_acl_other_user_invisible_everywhere` расширен на source (petrov 404), asset (petrov 404, живой актив в doc_dir), stats (у petrov документа нет, totals.nodes=0; у ivanov/boss/owner есть), sig-ссылка 200 / битая 401; все прежние тесты source (test_source_links, test_identity) зелёные |
| P1 размер запроса до чтения | `2a6cc1d` | Ранний гейт Content-Length ДО разбора тела: жёсткий кап `VELES_UPLOAD_MAX_MB` (env, дефолт 50 МБ, 0 = снят) и грубая проверка свободного остатка квоты (с запасом на перезапись крупнейшего своего L0 — ложных 413 нет, точная проверка в check_quota осталась) → 413 до чтения; запись L0 через `run_in_threadpool`. Остаточный риск задокументирован в докстринге (тело до капа буферизуется парсером, chunked без CL проходит мимо гейта; полный стриминг — отдельная задача). Тесты: `test_upload_content_length_cap_413_before_read` (413, L0 и джобов нет; файл в пределах капа проходит), `test_upload_content_length_fake_header_413` (фейковый CL 10 МБ при крошечном теле → 413 до чтения), `test_upload_content_length_quota_remainder_413` (ранний отказ по остатку; впритык — прежняя семантика 413 от check_quota) |
| P2a TOCTOU квоты | `8fa2fc6` | `user_quota_lock` — per-user `threading.Lock`; check_quota+запись L0 в documents.py под одним замком (в тредпуле, событийный цикл не блокируется). Тест `test_quota_check_and_write_atomic_per_user`: два потока через барьер, квота 1 документ — итог `[ok, quota]`, файл один |
| P2b list_jobs boot-id | `8fa2fc6` | `list_jobs` применяет семантику lookup: чужой boot_id в queued/running показывается failed «сервис перезапущен» и персистентно пишется в журнал; свой живой queued не трогается. Тест `test_list_jobs_marks_stale_boot_failed` (три поколения очереди) |
| P2c L0 упавшего джоба | `8fa2fc6` | `_discard_l0`: ошибка парса → L0 удалён (квота не съедается), неудача удаления → WARN в лог без падения воркера. Тесты: `test_pdf_job_fails_clean_without_docling` расширен (после failed файла «документ.pdf» в uploads нет), `test_discard_l0_warns_on_unlink_failure` (unlink поднял OSError → только WARN-запись в лог) |
| P2d python-multipart | `da57b78` | `pyproject.toml`: `>=0.0.9` → `>=0.0.18` (DoS-фиксы; в venv стенд стоит 0.0.32 — совместимо) |
| P2e арифметика бейслайна | `da57b78` | В eval фазы «vs 116» заменено на базу d290963 = 124 passed, 124+15=139; проверено прогоном архива базового коммита на стенд: `pytest tests -q` → `124 passed, 1 warning in 6.94s` |

Вывод полного сета (стенд, `/tmp/veles-k4-fix`, `git archive HEAD`):

```
$ cd /tmp/veles-k4-fix && CUDA_VISIBLE_DEVICES= <server-home>/veles-venv/bin/python -m pytest tests -q
145 passed, 1 warning in 7.70s        (было 139, +6 новых тестов)
```

Локально (Windows, Python 3.11): `144 passed, 1 skipped, 1 warning in 41.78s`
(skip и warning — унаследованные, не из фиксов: PermissionError-флап K2-теста
трекера, задокументирован в ревью K2).

Мини-E2E ACL на тест-инстансе стенд (`127.0.0.1:18410`, `VELES_IDENTITY_MODE=headers`,
`VELES_DATA=/tmp/veles-k4-fix-e2e/data`, real-модель e5-small, сид 1 документ +
индекс numpy-flat; живой фасад, не TestClient):

```
healthz: 200 ok
upload: {"job_id": "2b525ecc…", "status": "queued", "document_id": "ivanov@corp.ru/Парольная_политика.md"}
job: {"status": "done", "action": "created", "revision_id": "r1", "nodes": 2, …}
petrov: source 404 asset 404 stats documents: [] totals: {'documents': 0, 'nodes': 0, 'superseded_nodes': 0}
ivanov: source 200 asset 200 stats ['ivanov@corp.ru/Парольная_политика.md']
boss:   source 200 asset 200 stats ['seed/politika.txt', 'ivanov@corp.ru/Парольная_политика.md']
owner:  source 200 asset 200 stats ['seed/politika.txt', 'ivanov@corp.ru/Парольная_политика.md']
sig-link (без заголовков): source 200
sig-link битая: 401
E2E ACL: OK
```

Чистка стенд: тест-инстанс убит (порт 18410 свободен), `/tmp/veles-k4-fix` и
`/tmp/veles-k4-fix-e2e` удалены, `pgrep -af 'veles serve'` находит только прод
112960 (`--port 8400`, работал до прогона и не тронут).

## Замечания вне фиксов

- Гейт квоты в раннем отказе намеренно грубый (без имени файла остаток взят с
  запасом на перезапись крупнейшего своего L0): ложных отказов нет, точную
  семантику держит check_quota после разбора — покрыто тестом «впритык».
- `totals.superseded_nodes` в stats остался глобальным счётчиком индекса
  (число, не контент); ACL-фильтр применён к документам/узлам и totals.nodes.

## Отложено (за рамками этого ревью, решение владельца)

- Кап глубины очереди + ротация `ingest-jobs.jsonl` (append-журнал растёт
  неограниченно, очередь не ограничена по глубине).
- Гонка `build_index` vs overlay: загрузки, легшие в канон ПОСЛЕ чтения канона
  сборкой, стираются полной пересборкой (снапшот их не содержит, оверлей
  затирается). Warning'ом документировать не стали — реальный фикс отдельной
  задачей (сборка должна дочитывать оверлей или строиться с блокировкой очереди).
- fsync оверлея и чистка сироты `overlay.npz.tmp` при сбое записи.
- Windows-зарезервированные имена (`CON`, `NUL`, ...) в `safe_filename` —
  на Windows-хосте такое имя L0-файла может уйти мимо файла.
- `\n` в uid из JWT (заголовки/логи; в пути `user_dir` попадает уже
  нормализованным, но перевод строки в uid не запрещён явно).
