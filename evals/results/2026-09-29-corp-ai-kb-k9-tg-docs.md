# Eval K9: документы через Telegram (document-message -> POST /v1/documents, /docgroup, статусные сообщения)

Дата: 2026-09-29. Ветка `feature/corp-ai-kb-tgdocs`, worktree `veles-wt-tgdocs`
(от main 80a63e1), коммиты 3dbc318 (код + тесты) + 7700dfe (документация) +
f5624bf (экранирование названий групп в /docgroup) + этот (eval). Бриф:
`docs/plan/briefs/2026-09-29-k9-telegram-docs.md`. Прод стенд
и прод-бот не тронуты: полный pytest выполнен на копии кода в `/tmp/veles-k9`
(`git archive HEAD`, HEAD = f5624bf), каталог удалён после прогона (проверено
`ls` - пусто); живой прогон моста НЕ делался (координатор сделает на деплое).

## Что сделано

- **`clients/telegram-bridge/documents.py`** (новый, стиль voice.py): клиент
  фасада - `submit_document` (multipart `POST /v1/documents`, поля title/group),
  `get_job` (`GET /v1/documents/jobs/{id}`), `list_groups` (`GET /v1/groups`);
  ошибки >=400 -> `FacadeError` с текстом `detail` (показывается пользователю
  как есть); `esc()` - экранирование Telegram-HTML (паттерн tg_format._esc);
  `TG_GETFILE_MAX_BYTES` = 20 МБ, `DOC_EXTENSIONS` = {txt, md, pdf}; state
  групп загрузок (load/save_doc_state, JSON рядом с bridge-state).
- **`bridge.py`**: `handle_document` - document-message от владельца (гейт в
  `handle` ДО любых сетевых действий): гейт расширения -> гейт 20 МБ
  (`file_size`, честное сообщение про лимит Telegram) -> резервация слота ->
  «📄 Принял <имя>, загружаю...» сразу -> скачивание (механизм getFile
  переиспользован у voice.download_tg_voice) -> POST фасада -> daemon-поток.
  `doc_watch_loop` (паттерн reminder_loop): поллинг джоба, завершение вторым
  сообщением («✅ Загружено: <title> (группа ...)» / «❌ Не удалось: <причина>»
  / честный таймаут 900 с); потеря джоба фасадом (рестарт, 404) - отказ, а не
  вечное молчание. Слоты одновременных ожиданий: `reserve_doc_slot`/
  `release_doc_slot`, лимит `BRIDGE_DOC_MAX_CONCURRENT` (5), исчерпан -
  отказ до скачивания. `/docgroup` - список из фасада + текущая; `/docgroup
  <id>` - валидность сверяется с `GET /v1/groups` (опечатку видно сразу),
  persist в `<stem pairing-state>-docgroups.json`. `send_html` - статусные
  сообщения с plain-fallback (без markdown-конвертации - она коверкала бы
  имена файлов); имена/ошибки экранируются `documents.esc`. help, /help-текст,
  KNOWN_COMMANDS, setMyCommands дополнены.
- **НЕ тронуто** (параллельная K10 правит `veles/api/documents.py` DELETE и
  store/overlay): фасад, tracker-команды, voice-поток, tg_format.

## Как работает флоу (итог)

Сотрудник кидает боту файл -> бот (только владелец/спаренный чат) проверяет
формат (txt/md/pdf) и размер (<= 20 МБ, лимит Telegram getFile) -> «Принял» ->
скачивание -> multipart POST в фасад с группой чата (/docgroup или личная
дефолтная) -> фоновый поток опрашивает `GET /v1/documents/jobs/{id}` -> второе
сообщение: загружено (title + группа) / причина отказа (текст фасада) /
таймаут ожидания моста. Отказы фасада (чужая группа, квота, формат) доходят
до пользователя текстом detail без приукрашивания.

## Проверки (команды + выводы)

1. **Локально (Windows, Git Bash, Python 3.11), новые тесты:**

   ```
   $ python -m pytest tests/test_bridge_documents.py -q
   17 passed in 0.40s   (после добавления test_auth_header_from_env: 18)
   ```

   Покрытие по брифу: чужой document - игнор (без скачивания); скачивание
   fail -> честное сообщение и слот освобождён; POST fail -> текст detail как
   есть; успех -> «Принял» + завершающее «✅ Загружено ...» из watcher-потока;
   /docgroup show/set/persist (файл state)/невалид (не сохранён)/ошибка
   фасада; экранирование HTML в имени (`a<b>&.md` -> `a&lt;b&gt;&amp;.md`);
   гейт 20 МБ; юнит на лимит: 5 резерваций ok, 6-я False, release возвращает
   слот; отказ при исчерпанных слотах не съедает слот; watch: failed/потеря
   джоба/таймаут; send_html plain-fallback.

2. **Локально, полный pytest:**

   ```
   $ python -m pytest tests -q
   1 failed, 199 passed, 1 skipped
   ```

   Единственный отказ - `tests/test_tracker.py::test_atomic_write_reader_
   sees_whole_file` - ПРЕДСУЩЕСТВУЮЩИЙ флейк на Windows (гонка читателя с
   `os.replace`, WinError-лотерея планировщика): на чистой базе 80a63e1 та
   же картина - 5 прогонов подряд: 5 failed / 1 passed; с изменениями - 5
   прогонов: 3 passed / 2 failed. K9 tracker не трогает. Авторитетный прогон
   - Linux стенд, там тест детерминированно зелёный (п. 3).
   Попутно пойман и устранён собственный грех: первая редакция теста ставила
   `os.environ.setdefault("VELES_API_TOKEN", ...)` на уровне модуля - это
   глобальная env процесса pytest и ломала identity-тесты фасада (33 failed);
   убрано, проверено: полный сет после удаления - 199 passed + тот флейк.

3. **стенд, полный pytest на `git archive HEAD` (f5624bf):**

   ```
   $ git archive HEAD | ssh стенд "rm -rf /tmp/veles-k9 && mkdir -p /tmp/veles-k9 && tar -x -C /tmp/veles-k9"
   $ cd /tmp/veles-k9 && CUDA_VISIBLE_DEVICES= <server-home>/veles-venv/bin/python -m pytest tests -q
   201 passed, 1 warning in 8.86s
   ```

   Прогон ПОВТОРЁН на финальном HEAD после коммита f5624bf (первый прогон был
   на 7700dfe - 201 passed, 1 warning in 8.90s). Арифметика: база 80a63e1 =
   183 (замер дан в задании) + 18 новых K9 = 201. (1 warning - SyntaxWarning
   в test_slice.py, pre-existing.)

4. **Чистка стенд и прод:**

   ```
   $ ssh стенд "rm -rf /tmp/veles-k9 && ls -d /tmp/veles-k9"
   ls: cannot access '/tmp/veles-k9': No such file or directory
   $ pgrep -af 'veles serve' | grep 8400
   364101 ... veles serve --host 0.0.0.0 --port 8400
   $ curl -s http://127.0.0.1:8400/healthz
   {"status":"ok","ts":...,"index_loaded":true}
   ```

   Прод-фасад (8400) не перезапускался и здоров. Процесс на порту 18450
   запущен не этой фазой - не тронут.

## Отступления от буквы брифа (зафиксировано)

- «Отвечает "Принял..." по завершении POST» реализовано как «Принял» СРАЗУ
  после гейтов, до скачивания (заказ в задании фазы: «статус "принял"
  сразу»): скачивание 20 МБ может занять секунды. При отказе скачивания/POST
  после «Принял» идёт честное сообщение об отказе - статусы не врут о приёме
  файла в работу, а не о результате.
- Валидность группы при `/docgroup <id>` сверяется с `GET /v1/groups` сразу
  (а не только при следующей загрузке): опечатка видна в момент выбора. Право
  на группу это не подменяет - отказ фасада на POST всё равно показывается
  как есть (бриф, п. 3).
- Гейт 20 МБ - по `file_size` из message-метаданных Telegram; если Telegram
  не прислал размер, честно отработает отказ getFile («Не смог скачать файл
  из Telegram»).

## Не проверено

- **Живой Telegram-путь (реальный бот, getFile, sendMessage, прокси)** -
  MANUAL для владельца: прод-бот по условию фазы не трогается; координатор
  сделает на деплое (переключение service-файла на новую версию моста).
  Покрыто фейками Telegram (FakeTg/FakeThread) и фейком фасада, как voice в
  K-голосовых.
- **Интеграционный POST из мост-клиента в живой фасад** - частично: клиент
  `documents.py` покрыт юнит-тестами с фейковым HTTP-слоем (FakeVeles,
  статус-коды, detail); живой multipart-POST в фасад не выполнялся из моста
  (фасад-сторона POST /v1/documents и job-статусов покрыта тестами фасада
  K4/K7 и не менялась).
- **Одновременные загрузки >1 в одном чате** - лимит проверен юнитом на
  слотах; реальная параллельность потоков под живым Telegram не
  воспроизводилась.
- **`BRIDGE_DOC_*` env-переопределения** (poll/timeout/лимит) - код читает,
  тесты гоняют дефолты и monkeypatch-значения; живой рестарт с чужими env не
  прогонялся.
