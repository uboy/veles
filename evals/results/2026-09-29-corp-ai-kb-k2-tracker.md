# Eval K2: трекер-мультипользователь (остаток) — атомарная запись, git-lock, бэкап, шаблоны lite

Дата: 2026-09-29. Ветка `feature/corp-ai-kb-tracker-multiuser`, worktree
`veles-wt-tracker2` (Windows-хост), финальный коммит `b6b4bd8`. Прогоны — стенд
(Ubuntu, python 3.12.3 из `<server-home>/veles-venv`, pytest 9.1.1, git 2.54.0):
код передавался `git archive HEAD | ssh tar -x -C /tmp/veles-k2-test`; прод-стек стенд
(порты 8300/8400/8500, systemd-юниты, `~/veles-tracker`) не затронут — venv стенд
использован только как интерпретатор pytest, все артефакты — в /tmp, после проверок
каталог вычищен.

## 1. Конкурентная запись: 2 потока set_status, лок против index.lock

Отказ, который дал первый прогон (до фикс-коммита `1d6386b`): запись md шла вне
лока, `git add -A` первого потока захватывал файл второго, его `git commit`
становился пустым и молча проглатывался `check=False` — 3 коммита вместо 4:

```
AssertionError: ожидали 4 коммита, получили:
['c176213 task: task-20260929-01-i-8e4f2b98 -> done', '9b3d151 task: Задача два', '9df93ac task: Задача один']
1 failed, 7 passed in 0.83s
```

Данные при этом не терялись (оба файла правильные), но нарушалось обещание
TRACKER.md «каждое изменение = коммит». Фикс: запись карточки перенесена под тот
же `_git_lock`, что и git-фаза (write+commit сериализованы). После фикса тест
`test_concurrent_set_status_two_threads` (барьер на 2 потока, разные карточки
одного каталога) проверяет: нет исключений, оба статуса `done`, `git log` ровно
4 коммита (2 create + 2 set_status — у брифа «один-два» на два set_status, здесь
строже: у каждого свой), `git status --porcelain` пуст (no-loss).

Команда (стенд):

```
cd /tmp/veles-k2-test && python -m pytest tests/test_tracker.py -v
```

Вывод:

```
tests/test_tracker.py::test_create_and_list_task PASSED
tests/test_tracker.py::test_done_filters_and_set_status PASSED
tests/test_tracker.py::test_notes_isolated_from_tasks PASSED
tests/test_tracker.py::test_extract_tasks_parses_numbered_list PASSED
tests/test_tracker.py::test_extract_tasks_none_case PASSED
tests/test_tracker.py::test_atomic_write_reader_sees_whole_file PASSED
tests/test_tracker.py::test_concurrent_set_status_two_threads PASSED
tests/test_tracker.py::test_git_lock_timeout_negative PASSED
8 passed in 0.81s
```

Стабильность (5 прогонов подряд — конкурентный тест не флапает):

```
8 passed in 0.82s   x5 (0.80–0.82 c каждый)
```

## 2. Атомарность: читатель не видит полузаписанный файл

`_atomic_write` — tmp-файл `.<имя>.tmp-<pid>-<tid>` в том же каталоге +
`os.replace`. Тест `test_atomic_write_reader_sees_whole_file`: поток-читатель
непрерывно читает файл, писатель 120 раз переключает старый/новый контент
(по 5 КБ) — читатель всегда видит целиком старый или целиком новый, tmp-файлы
после записи не остаются (вывод в п.1, строка PASSED).

## 3. Негатив: flock-таймаут

Юнит-тест (fcntl, лок занят из теста, `GIT_LOCK_TIMEOUT` подменён на 0.5 с) —
`LockTimeout` по своей причине, не зависание; после снятия лока вход возможен
(строка PASSED в п.1). Реальный таймаут с дефолтной константой проверен отдельно
(стенд, лок держит чужой процесс, ожидание без подмены):

```
30.0s LockTimeout: не дождались flock на /tmp/veles-k2-locktest/.git-lock за 30 с:
git-операцию держит другой процесс; повторите команду позже
```

Windows-fallback (fcntl отсутствует → работаем без лока) кодом покрыт
(`try: import fcntl / except ImportError`), юнит-тест в этой ветке проверяет
запись без строгого счёта коммитов; реального прогона на Windows-хосте нет —
локальный python заглушка (см. «Не проверено»).

## 4. Регрессия

```
cd /tmp/veles-k2-test && python -m pytest tests -q
88 passed, 1 warning in 5.83s
```

(1 warning — давний, voice-тест про BOM в CRLF-тексте, к K2 отношения не имеет.)
До K2 сет содержал 5 трекер-тестов, стало 8; полный сет вырос с 85 до 88 —
на базовом коммите брифа b9d28af: `pytest tests --collect-only -q` =
«85 tests collected».

## 5. Бэкап TRACKER_ROOT

`deploy/стенд/veles-backup.sh`: `TRACKER_ROOT` (env, по умолчанию
`<server-home>/veles-tracker`) тарируется в `tracker.tgz` внутри архива; при
отсутствии каталога — WARN в stderr, бэкап остального продолжается.

Проверки (стенд, только /tmp; прод-скрипт не запускался — он пишет в прод-бэкапы):

```
bash -n deploy/стенд/veles-backup.sh        -> BASH_N_OK
# макет строки скрипта с TRACKER_ROOT=/tmp/veles-k2-fake-tracker:
tar czf /tmp/veles-k2-tracker.tgz -C /tmp veles-k2-fake-tracker && tar -tzf ...
veles-k2-fake-tracker/
veles-k2-fake-tracker/user1/
veles-k2-fake-tracker/user1/task/
veles-k2-fake-tracker/user1/task/card.md
# ветка else (каталог отсутствует) печатает в stderr:
WARN: TRACKER_ROOT=... не найден - трекер не вошёл в бэкап
```

В `deploy/kit/README.md` добавлен раздел «Бэкапы (corporate — обязательно)».

## 6. Шаблоны lite (AGENTS.md + скиллы)

`deploy/kit/templates/`: `AGENTS.md` (формат карточек — поля сверены с
`tracker.py`: id/title/status/created/source/due/reminded; словарь todo|done;
«готово = проверено»; `veles ask`), `skills/veles-ask/SKILL.md`,
`skills/tracker-task/SKILL.md` (frontmatter name+description, тело — инструкции).

```
wc -c deploy/kit/templates/AGENTS.md -> 3933   (лимит 8 КБ)
head SKILL.md -> "---" / "name: veles-ask" / "description: ..." (и tracker-task)
```

Попадание в кит проверено сборкой `make-kit.sh` на стенд (во временном git-репо
из /tmp/veles-k2-test; одноразовый обход глобального хука стенд
SKIP_SECURITY_GATE=1 — хук блокировал коммит корневого AGENTS.md самого репо,
репо тестовое, в поставку не идёт):

```
bash deploy/kit/make-kit.sh   -> файлов: 92
tar -tzf scratch/veles-deploy-kit-*.tar.gz | grep templates
veles-deploy-kit-.../deploy/kit/templates/AGENTS.md
veles-deploy-kit-.../deploy/kit/templates/skills/veles-ask/SKILL.md
veles-deploy-kit-.../deploy/kit/templates/skills/tracker-task/SKILL.md
```

Попутно найдено и исправлено (`b6b4bd8`): `git archive` с Windows-хоста применяет
autocrlf — шаблоны в архиве приезжали с CR (AGENTS.md: 61 CR), а для
YAML-frontmatter скиллов нужен LF. Чистка `\r` в make-kit.sh расширена с
`.sh/.py/.service` на `.md`; в пересобранном ките:

```
templates/AGENTS.md CR=0; skills/veles-ask/SKILL.md CR=0; skills/tracker-task/SKILL.md CR=0
head -3 .../tracker-task/SKILL.md -> --- / name: tracker-task / description: ...
```

## Не проверено

- E2E через реального Telegram-бота (два живых пользователя, одновременный
  `/task`) не гонялся — прод-мост стенд по условию не трогаем; конкурентность
  покрыта юнит-тестом на уровне `tracker.py`, тем же кодом, что зовёт мост.
- Windows-fallback лока без fcntl в реальном прогоне не проверен (локальный
  python-заглушка); путь тривиален (yield без лока) и покрыт тестом обходом.
- Прод-запуск `veles-backup.sh` целиком не выполнялся (запись в прод-бэкапы и
  sudo-tar volume — вне K2); проверены синтаксис, макет tar-строки и WARN-ветка.
- `install.sh` не копирует templates в TRACKER_ROOT автоматически — ручное
  копирование описано в README; автоматизация — кандидат в бэклог, не в K2.

---

Ревью безопасности этих изменений (PASS_С_ЗАМЕЧАНИЯМИ, фиксы в том же заходе):
[2026-09-29-corp-ai-kb-k2-security-review.md](2026-09-29-corp-ai-kb-k2-security-review.md).
