# Eval K5: deploy-kit (install/uninstall/make-kit) + вычистка машинных путей

Дата: 2026-09-28. Ветка: `feature/corp-ai-kb-deploy-kit`, worktree
`~/proj/veles-wt-deploy` (Windows), тяжёлые прогоны - стенд (Ubuntu 24.04, python 3.12.3,
docker compose v2.5.0). Архив финальной версии: `veles-deploy-kit-bd19c86.tar.gz`
(= git HEAD на момент окончания работ); полный цикл install → CLI → verify → uninstall
переигран именно на нём. E2E проводился в `/tmp/veles-kit-e2e.*` на стенд; прод-стек стенд
(порты 8300/8400/8500, юниты veles-*, `~/proj/veles-bench`, `<secrets-dir>`) не затронут -
lite-профиль портов не слушает, venv стенд использован только как интерпретатор для
pytest, pip install - только в tmp-venv из /tmp.

## 1. make-kit.sh собирает архив с COMMIT.txt

Команда (Windows, Git Bash):

```
bash deploy/kit/make-kit.sh
```

Вывод:

```
архив: /c/Users/<local-pc>/proj/veles-wt-deploy/scratch/veles-deploy-kit-1f81e95.tar.gz
  версия: 1f81e95
  файлов: 108
```

Содержимое COMMIT.txt (распаковка на стенд, `cat .../COMMIT.txt`; показан снятый при
промежуточной сборке 0407362 - формат одинаков для всех, финальная сборка 1f81e95):

```
kit_version: 0407362
commit: 04073624d5b61b2bd7a71116d1cc2c042071d1db
date: 2026-09-28T23:05:23+03:00
subject: fix(k5): REPO_DIR в install.sh/install.ps1 ...
built: 2026-09-28T23:05:24+03:00
```

Гигиена архива: git archive HEAD (без .git/data/venv); find по `*.env`/`secrets*` в
make-kit прерывает сборку (страховка). `.env.example` пришлось добавлять `git add -f`:
глобальный gitignore машины (`.env.*`) молча исключил файл из коммита и из архива -
исправлено коммитом, в финальном архиве файл есть (`tar tzf | grep -c .env.example` = 1).

## 2. E2E lite на стенд: install → CLI → verify → uninstall чисто

Команды (стенд, каталог /tmp/veles-kit-e2e.2170254, архив veles-deploy-kit-bd19c86.tar.gz):

```
tar xzf veles-deploy-kit-1f81e95.tar.gz
cd veles-deploy-kit-1f81e95/deploy/kit
bash install.sh --profile lite --yes
./.venv/bin/veles status
bash verify-install.sh
bash uninstall.sh --purge-data --yes
ls -A /tmp/veles-kit-e2e.2170254
```

Установка (фрагмент):

```
[ok] python: /usr/bin/python3.12 (Python 3.12.3)
[ok] openssl: есть (генерация секретов)
[ok] .env записан: .../deploy/kit/.env (права 600)
[ok] CLI установлен: .../deploy/kit/.venv/bin/veles
Итог: FAIL нет. Компоненты профиля lite работают (SKIP = осознанно не настроено).
```

CLI (вне окружения .env - дефолты config.py без машинных путей):

```
$ .venv/bin/veles status
корпус: <server-home>/veles-data/corpus          <- ~/veles-data/corpus, не хардкод
канон: 0 документов, 0 узлов (<kit>/data/canonical)
индекс: нет (<kit>/data/index)                  exit=0
$ .venv/bin/veles ask "тест"
индекс не собран - сначала `index`              exit=1 (ожидаемый отказ без индекса)
```

verify-install.sh: exit=0, таблица [OK] .env (600) / [OK] CLI / [OK] CLI: veles status.

Uninstall --purge-data --yes: exit=0; удалил .venv (101M), .env, .veles-kit и весь
распакованный кит (маркер COMMIT.txt); после снятия:

```
ls -A /tmp/veles-kit-e2e.2170254   -> (пусто)
каталог кита отсутствует - следов нет; порты не использовались; процессов нет
```

## 3. Негатив: preflight без python3.11 (PATH-обманка)

```
mkdir /tmp/veles-kit-fakebin; симлинки всего из /usr/bin,/bin кроме python*
env PATH=/tmp/veles-kit-fakebin bash .../deploy/kit/install.sh --profile lite --yes
```

Вывод:

```
[ошибка] python 3.11+ не найден в PATH. Установите python3.11+ (пакет python3.11/python3) и повторите
negative exit=1
```

## 4. docker compose config -q (стенд, docker 29.6.2 / compose v2.5.0)

```
cp deploy/kit/docker-compose.yml deploy/kit/.env.example <tmp>/; cp .env.example .env
sed -i "s/=GENERATE/=$(openssl rand -hex 32)/" .env   # только на проверочном .env в /tmp
docker compose -f docker-compose.yml config -q                      -> compose base: OK
docker compose -f docker-compose.yml --profile bridge config -q     -> compose bridge: OK
config --services: veles-facade, open-webui (+ telegram-bridge с профилем bridge)
```

`docker compose up` НЕ запускался (вне агента по брифу - следующий шаг на тестовом
сервере). Фасад в compose не публикует порт наружу (`expose 8400`), наружу - только
Open WebUI (решение ревью безопасности K1 от 28.09).

## 5. bash -n / PSParser

- `bash -n` install.sh, uninstall.sh, verify-install.sh, make-kit.sh, lib/common.sh -
  все OK (локально, Git Bash; shellcheck на обеих машинах не установлен - пропущен).
- PSParser (Windows PowerShell 5.1, после добавления UTF-8 BOM в .ps1 - без BOM
  кириллица читалась как ANSI и ломала парс):

```
PARSE OK: <local-pc>\proj\veles-wt-deploy\deploy\kit\windows\install.ps1
PARSE OK: <local-pc>\proj\veles-wt-deploy\deploy\kit\windows\uninstall.ps1
```

Полный E2E windows-профиля на Windows-ПК - вне агента (по брифу).

## 6. Машинных путей в коде нет (grep-доказательство)

Команда (корень репо, Git Bash; исключены docs/, deploy/стенд/, scratch/, .git/):

```
grep -rn "C:\\Users\\<local-pc>\|<server-home>\|10\.13\.105\.101" . \
  --exclude-dir=.git --exclude-dir=scratch --exclude-dir=docs --exclude-dir=стенд \
  | grep -v "^Binary"
```

Вывод: пусто, `grep exit=1` (совпадений нет).

Что именно вычищено:
- `veles/config.py`: CORPUS_ROOT без `r"<local-pc>\proj\knowledge-sources"` -
  дефолт `~/veles-data/corpus`; из комментария VELES_PUBLIC_URL убран IP стенд;
- `veles/cli.py` (cmd_ingest): без корпуса - понятная ошибка с подсказкой
  про `VELES_CORPUS_ROOT` (fail-closed), тест `test_ingest_fail_closed_without_corpus`;
- `clients/telegram-bridge/TRACKER.md`, `e2e/E2E-REPORT.md`: `<server-home>/...`
  заменены на `~/...` (та же семантика, user-relative).

Разрешённые остатки (по брифу): docs/ (DEPLOY-стенд.md - рунбук прод-топологии,
бриф K5) и deploy/стенд/ (живые юниты/скрипты прод-сервера стенд).

## 7. Регрессия pytest не сломана

Код на стенд перенесён `git archive HEAD | ssh tar -xf - -C /tmp/veles-kit-test`
(без .git/data); venv стенд `~/veles-venv` использован только как интерпретатор,
pip install не выполнялся:

```
cd /tmp/veles-kit-test && ~/veles-venv/bin/python -m pytest tests -q
81 passed in 6.75s
```

(1 warning SyntaxWarning `\u` в test_slice.py - существовавший ранее, не из этой фазы.)
/tmp/veles-kit-test после прогона удалён.

## Отступления от брифа (фактом)

1. В ките два файла сверх таблицы брифа: `deploy/kit/Dockerfile` и
   `Dockerfile.bridge` - без них compose corporate не собирает сервисы.
2. pyproject: fastapi/uvicorn/httpx добавлены в core-зависимости (фасад импортирует их
   без extras - pip install -e . раньше давал нерабочий CLI/serve) и добавлен
   консольный скрипт `veles` (`[project.scripts]`), на него опирается install.sh.
3. CORPUS_ROOT дефолт - `~/veles-data/corpus` (подкаталог корня данных, а не сам
   `~/veles-data`): корпус и рантайм-данные не смешиваются (инвариант стандарта).
4. По ревью безопасности K1 (28.09, после брифа): corporate - дефолт identity `jwt`;
   `headers` только через `--identity-headers` с предупреждением; фасад без наружного
   порта; fail-closed без identity+токена.
5. `.env.example` добавлен `git add -f` (глобальный gitignore `.env.*` исключал его
   молча - попал бы в каждую сборку кита как дыра).
