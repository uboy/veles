# Ревью безопасности K2 (трекер: атомарная запись, git-lock) — 29.09.2026

Объект: ветка `feature/corp-ai-kb-tracker-multiuser`, коммит `1f05937` (база
`b9d28af`) — атомарная запись карточек (tmp + os.replace), flock на `.git-lock`,
бэкап TRACKER_ROOT в ночной скрипт стенд, шаблоны lite (AGENTS.md, скиллы).
Проверялись `clients/telegram-bridge/tracker.py`, `bridge.py`, фасадный
`/tracker` (read-only потребитель), deploy-kit.

## Вердикт: PASS_С_ЗАМЕЧАНИЯМИ (P0 нет)

Фиксы — в этом же eval, ниже (раздел «Фиксы (тот же заход)»).

## Находки

- **[P1] Гонка порядкового номера в create_item.** `tracker.py:143-161`:
  `list_items` и подбор `n` выполняются ДО захвата `.git-lock` (лок на
  `tracker.py:163`). Два одновременных create_item одного chat_id читают
  одинаковый список, выбирают один item_id, и второй `os.replace` молча
  затирает карточку первого (данные потеряны, коммит-пустышка). Отдельно:
  файл без валидной шапки `list_items` не видит, но имя занять может —
  подбор номера по списку недостаточен, нужна перепроверка по существованию
  пути.
- **[P1] Напоминалка пишет мимо лока и по устаревшему тексту.**
  `bridge.py:515-528` (reminder_loop): текст читается до `send()` в Telegram
  (секунды), после отправки `p.write_text(...)` пишет прочитанный снимок —
  если в окне set_status сменил статус на done, напоминалка затирает
  изменение старым текстом; запись без `_atomic_write`/`_git_lock` — мимо
  атомарности и дисциплины коммитов, плюс конфликт index.lock с параллельным
  /task.
- **[P2] `_atomic_write` (tracker.py:34-48):** tmp-файл с предсказуемым именем
  создаётся обычным write (без O_EXCL) — symlink-гонка на общем каталоге;
  нет fsync перед os.replace — потеря карточки при сбое питания.
- **[P2] `_git_commit` (tracker.py:115-122):** `.gitignore` (`.git-lock`,
  `.*.tmp-*`, `chat-url.txt`) создаётся только у новых установок; существующие
  (напр. стенд) коммят служебные файлы.
- **[P2] `_git` (tracker.py:108-112):** stderr глушится (DEVNULL) — отказ git
  (index.lock, битый репо) невидим, «каждое изменение = коммит» не
  диагностируется.
- **[P2] Мёртвый код:** `_commit` (tracker.py:125-128) не имеет вызовов.
- **[P2] LockTimeout не доходит до пользователя:** `bridge.py:224, 439-444,
  454` — create_item/set_status без обработки `tracker.LockTimeout`, таймаут
  лока молча падает в log.exception, пользователь не получает ответа.
- **[P2] Висячая ссылка в ките:** `deploy/kit/README.md:46` ссылается на
  `deploy/стенд/veles-backup.sh`, который в кит не входит — корпоративный админ
  по документации получает несуществующий файл.
- **[P2] Документация reminded расходится с кодом:** шаблон
  `deploy/kit/templates/AGENTS.md:29` показывает `reminded:` полем
  frontmatter, тогда как напоминалка дописывает метку последней строкой
  файла.

## Признано корректным (обходов не найдено)

os.replace на одной ФС атомарен; tmp в том же каталоге, что цель (замена не
едет между ФС); flock-таймаут fail-closed с понятным текстом; Windows-fallback
без flock осознан (один пользователь, git сам вернёт index.lock); лок
в `.git-lock`, git-фаза только у держателя лока.

## Фиксы (тот же заход)

Ветка `feature/corp-ai-kb-tracker-multiuser`, коммиты по находкам; публичные
сигнатуры create_item/set_status/list_items не менялись. Прогон — стенд (Ubuntu,
`<server-home>/veles-venv`, python 3.12), код `git archive HEAD | ssh tar -x -C
/tmp/veles-k2-fix`, прод-стек стенд не затронут, каталог вычищен.

| Находка | Коммит | Чем доказано |
|---|---|---|
| P1-1 гонка NN-номера | `d077ae2` | `test_concurrent_create_two_threads` (барьер, 2 потока x 2 карточки одного пользователя: id все разные, 4 файла, 4 коммита, porcelain чист) и `test_create_skips_existing_target` (имя занято файлом без шапки — берётся следующий NN); на стенд PASSED |
| P1-2 напоминалка | `296068e` | новые публичные `reminder_due`/`mark_reminded`; `test_reminder_due_cases`, `test_mark_reminded_positive` (метка последней строкой тела + коммит), `test_mark_reminded_negative` (статус сменили на done — False, файл байт в байт); reminder_loop в bridge.py: send вне лока, метка только если перепроверка прошла |
| P2a O_EXCL+fsync | `bae9378` | `_atomic_write` через os.open(O_EXCL, 0600)+fdopen+fsync; регрессия `test_atomic_write_reader_sees_whole_file` PASSED на стенд |
| P2b .gitignore самохил | `bae9378` | создаётся при init и при любом коммите, если отсутствует; счётчики коммитов в конкурентных тестах не изменились (4 create = 4 коммита) |
| P2c stderr git | `bae9378` | `git ... stderr=PIPE`, при коде != 0 `log.warning` с хвостом stderr 200 Б (модульный logger) |
| P2d мёртвая `_commit` | `bae9378` | `grep -rn "_commit" --include="*.py"`: вызовов не было (только определение и его же вызов `_git_commit`), удалена |
| P2e LockTimeout | `031ba77` | create_item в голосовых, /task (короткий и транскрипция) и set_status в /done обёрнуты `except tracker.LockTimeout -> send(chat_id, str(e))`; тесты моста 26 passed |
| P2f бэкапы кита | `c6f7229` | ссылка на deploy/стенд/veles-backup.sh заменена inline-сниппетом (tar TRACKER_ROOT + retention 7 дней) |
| P2g reminded в AGENTS.md | `c6f7229` | метка документирована как последняя строка файла (не поле frontmatter), из примера шапки убрана |

Вывод полного сета (стенд, `/tmp/veles-k2-fix`):

```
cd /tmp/veles-k2-fix && <server-home>/veles-venv/bin/python -m pytest tests -q
93 passed, 1 warning in 6.73s        (было 88, +5 новых тестов)
```

Стабильность конкурентных тестов (3 прогона подряд, не флапают):

```
tests/test_tracker.py::test_concurrent_create_two_threads
tests/test_tracker.py::test_concurrent_set_status_two_threads
2 passed in 0.35s / 0.35s / 0.36s
```

Трекер-файл целиком на стенд: 13 passed (было 8, +5).

## Замечания вне фиксов

- Windows-хост (не канон): `test_atomic_write_reader_sees_whole_file` флапает
  PermissionError/WinError 5 — os.replace цели при открытом читателем файле;
  наблюдается и до, и после фиксов, на стенд/Linux не воспроизводится. Не чинилось
  (вне списка находок ревью).
- Текст напоминания теперь `⏰ Задача к сроку: <title>` без даты due (контракт
  `reminder_due -> title`; дата остаётся в карточке).

Дозакрытие координатора (b353f8b): деградация выше устранена — `reminder_due`
возвращает `(due, title)`, мост печатает исходный формат `⏰ Задача к сроку
<due>: <title>`; тесты переведены на кортеж. стенд: `pytest tests/test_tracker.py
-q` → 13 passed.

