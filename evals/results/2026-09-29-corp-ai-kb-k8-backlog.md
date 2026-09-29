# Eval K8: бэклог-тройка (HEAD /v1/source, скролл на #fragment, кап очереди) — 29.09.2026

Фаза K8 эпика corp-ai-kb. Ветка `feature/corp-ai-kb-ops2`, коммиты d839946
(HEAD + CSP/фрагмент), a28c16f (кап очереди + ротация журнала + fsync overlay).
Бриф: `docs/plan/briefs/2026-09-29-k8-backlog-trio.md`.

Параметры прогонов: полный `pytest tests -q` на стенд из
`git archive HEAD | ssh стенд "tar -x -C /tmp/veles-k8"` (venv стенд,
`CUDA_VISIBLE_DEVICES=`); живая проверка — тест-инстанс на стенд, порт 18440
(только 127.0.0.1), tmp `VELES_DATA=/tmp/veles-k8-live/data`, VELES_API_TOKEN.
Прод стенд (порт 8400, PID 348744) не тронут — healthz 200 после проверок;
/tmp/veles-k8 и /tmp/veles-k8-live вычищены, процесс тест-инстанса остановлен.

Базлайн: 164 passed (K7-мердж). Итог стенд: **171 passed** за 8.01s
(164 + 7 новых тестов K8). Локальный Windows-прогон: 169 passed, 1 skipped,
1 упавший — `test_tracker.py::test_atomic_write_reader_sees_whole_file`
(PermissionError на replace); воспроизводится на чистом HEAD без правок K8
(git stash → тот же отказ) — предсуществующий Windows-специфичный кейс
атомарной записи, на Linux (стенд) зелёный.

## 1. HEAD на /v1/source

Маршрут был только GET → `curl -I` давал 405. Переведён на
`@app.api_route("/v1/source/{document_id:path}", methods=["GET", "HEAD"])` —
как давно сделано на /tracker (18acfda). Тест `test_source_head_supported`:
HEAD → 200 с security-заголовками, несуществующий документ под HEAD — тот же
404, что под GET.

## 2. Скролл на #fragment, чистый CSP

Места сборки source-ссылок — найдены ВСЕ (grep по `/v1/source`, `source_url`,
`node=` по veles/, clients/, tests/): ровно две функции в `veles/api/app.py` —
`source_url` (token=) и `signed_source_url` (exp+sig), обе вызываются только из
`render_chat_content._src_link`. `policy/answer.py` и `retriever` ссылок не
строят (цитаты — только данные); `tg_format` и telegram-bridge переформатируют
готовые markdown-ссылки (регэксп `https?://[^)\s]+` пропускает `#` без правок);
documents.py ссылок не строит. Обе функции дописывают `#n-<node_version_id>`
к существующему query; sig в query не меняется — фрагмент сервер не видит и в
HMAC (document_id|node|exp) не входит (тест `test_signed_citation_link_keeps_sig_and_adds_fragment`).

Из `source_view` убран инлайн-скрипт скролла совсем (`<script>` в странице
нет — тест), анкер `id='n-<node_version_id>'` в рендере уже был, формат
совпадает с фрагментом (node_version_id — sha256 hex; тест
`test_citation_fragment_matches_render_anchor` сверяет фрагмент собранной
ссылки с якорем живого рендера). CSP `/v1/source` теперь
`default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'`
— без `script-src 'unsafe-inline'` (компромисс ревью K6 снят).

Заголовки сведены в одну функцию `html_security_headers()` (app.py) — общая
для /v1/source и /tracker, dict больше не дублируется.

Валидация `node` (`[\w.-]+` → 400) сохранена: node входит в сообщение HMAC и
в формат фрагмента/анкера (тест `test_source_rejects_malformed_node` зелёный).

## 3. Кап очереди + ротация журнала (+ бонус overlay)

`veles/ingest/queue.py`: лимит queued-джобов `VELES_INGEST_QUEUE_MAX`
(дефолт 50, 0 = без лимита, мусор = дефолт; читается на вызове, как квоты).
Проверка и постановка под `threading.Lock` — два одновременных submit не
переполняют; превышение → `QueueCapacityExceeded(429, «очередь занята,
повторите позже …(VELES_INGEST_QUEUE_MAX)»)`; documents.py переводит его в
HTTP 429 тем же контрактом, что QuotaExceeded, и удаляет L0 этой загрузки
(джоба нет — файл не съедает квоту; как `_discard_l0` у упавшего джоба).
Кап именно на ждущих (`Queue.qsize()`): джоб, взятый воркером, не считается.

Ротация `ingest-jobs.jsonl`: при достижении `_LEDGER_ROTATE_BYTES` (5 МБ) перед
append активный файл переименовывается в `.1` (прежний `.1` заменяется),
держатся 2 файла; история до ротации в lookup/list_jobs больше не читается
(операционный журнал, персона данных не теряет — финальные статусы в каноне/аудите).

Бонус: в `append_overlay` fsync tmp-файла перед `os.replace` (иначе сбой питания
между savez и replace оставляет пустой/битый overlay — загрузки доезжают только
с пересборкой; дескриптор открыт `r+b` — на Windows fsync требует право записи) +
чистка tmp-сирот `overlay.tmp*.npz` в начале записи (обрыв прошлого write).
Тест `test_overlay_write_cleans_tmp_orphans`: сирота удалена, overlay читается.
Тест на сам вызов fsync не пишется (по брифу) — поведение записи не изменилось.

Тесты K8 (7 новых): кап 429 на submit (лимит 2 → третий отказ, 0 = снят),
кап через эндпоинт (429 + текст + L0 удалён + джобов нет), ротация (порог 1 Б
подменой константы: первая запись не ротирует пустой, вторая → `.1`, третья
заменяет `.1`), HEAD 200, фрагмент подписанной ссылки, фрагмент=якорь рендера,
CSP без unsafe-inline и без `<script>` в странице. Обновлены 2 теста K6
(ожидания ссылок с фрагментом; ожидания CSP).

## Проверки

- стенд полный сет: `171 passed in 8.01s` — команда и вывод в логе карточки.
- Живой инстанс 18440 (tmp VELES_DATA, канон-фикстура lectures/x.txt):
  - `curl -I /v1/source/lectures/x.txt?token=…` → **200 OK** (не 405) с
    `x-content-type-options: nosniff`, `x-frame-options: DENY`,
    `referrer-policy: no-referrer`, CSP `default-src 'none'; frame-ancestors
    'none'; base-uri 'none'; form-action 'none'` — **unsafe-inline нет**;
  - GET того же URL с `#n-nva1b2c3` → 200 (сервер фрагмент не видит, страница
    та же; скролл делает браузер);
  - якорь `id='n-nva1b2c3'` в странице есть, `<script>` — 0 вхождений;
  - без токена → 401 (авторизация не потеряна).
- Прод стенд: `/healthz` на 8400 → 200 до и после; процесс не перезапускался.

## Не проверено

- Скролл в реальном браузере (переход по ссылке с фрагментом визуально) —
  проверен только серверный контракт: фрагмент в ссылке, якорь в HTML, формат
  совпадает; поведение нативное браузерное, кода на странице нет.
- 429 капа очереди на живом инстансе — покрыт тестами с подменой лимита и
  очереди без воркера; на живом фасаде очередь с реальным воркером разбирает
  джобы быстрее, чем кап достижим.
- Ротация журнала на живом инстансе — покрыта юнит-тестом с подменой порога;
  ротация на 5 МБ вживую не дождана.
- fsync под реальным сбоем питания — проверена только эквивалентность
  поведения записи (overlay читается после записи, сироты чистятся).
- Производительность ротации/капа под конкурентной нагрузкой не измерялась
  (замеры вне объёма фазы).
