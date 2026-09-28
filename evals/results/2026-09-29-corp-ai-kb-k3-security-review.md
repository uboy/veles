# Ревью безопасности K3 (версии документов) — 29.09.2026

Объект: ветка `feature/corp-ai-kb-revisions`, HEAD 953ae7f (фаза K3: ревизии
документов - `veles/canonical/store.py` ingest_document/архивация в
`revision-r{n}/`, ревизионный фильтр в `veles/retrieval/retriever.py` рядом
с ACL). Метод: чтение кода шва ACL × revision и хранения канона; находки
воспроизводимы на tmp-структуре.

## Вердикт: PASS_С_ЗАМЕЧАНИЯМИ (P0 нет)

## Ключевое подтверждение: revision-фильтр не ослабляет ACL

`veles/retrieval/retriever.py`, `hybrid_search.authorized()`: ACL-конъюнкт
первый и безусловный - узел без пересечения групп с `allowed_groups` отсекается
ДО любого ревизионного ветвления; `revision="all"` достижим только после
пройденной ACL-проверки; ревизионный фильтр применён в том же предикате на
ОБОИХ каналах (dense и bm25) до RRF-слияния, т.е. узел, отсечённый ACL, не
может пройти ни одним каналом и не влияет на ранжирование. Запрос с правами на
документ получает архивную ревизию ровно на тех же группах, что и текущую:
расширения прав через `all`/`r<n>` нет.

## Находки

- **[P1]** `veles/canonical/store.py:194-199` - в `_archive_current_revision`
  manifest.json переезжает в каталог архива ПЕРВЫМ, nodes.jsonl вторым. Падение
  процесса между этими двумя move: `revision-rN/` остаётся с manifest без
  nodes.jsonl, а единственная копия узлов rN - в корне doc_dir. Последствия:
  (а) recovery-ветка `ingest_document` (manifest текущего нет, архивы есть)
  вызывает `save_document` и молча ПЕРЕЗАПИСЫВАЕТ корневой nodes.jsonl - узлы
  rN потеряны без следа; (б) `iter_documents(include_revisions=True)` и
  `_archived_revisions` считают архив полным по одному manifest.json →
  FileNotFoundError на build_index / cmd_status.
- **[P2-1]** `veles/canonical/store.py:176-194` - «самолечение» не покрывает
  липкий пустой каталог: если `rdir.mkdir()` выполнился и ни один move не дошёл,
  то на каждом следующем ингесте `rdir.exists()`=True, манифеста в архиве нет,
  sha-сверка невозможна → вечный RuntimeError «требуется ручной разбор»,
  документ заблокирован для обновления навсегда.
- **[P2-2]** `veles/canonical/store.py:242` - ложный no-op на пустых sha256:
  `if current.get("sha256") == doc.sha256` при legacy-манифесте без sha256
  («») и входящем `doc.sha256=""` даёт «без изменений» с молчаливым
  отбрасыванием загруженного документа (аудит не пишется).
- **[P2-3]** `veles/canonical/store.py:241-248` - revision_id из текущего
  манифеста не валидируется перед формированием архивного каталога:
  гипотетическое «abc» ушло бы в `revision-abc/`, невидимый glob
  `revision-r*` (не попадает в нумерацию `_archived_revisions` и в retrieval).

## Отложено (вне K3, не блокирует)

- кеш для `_resolve_revision` (`veles/api/app.py:112` - перебор node_meta на
  каждый запрос; производительность, не безопасность);
- revision-aware `/v1/source` и `/v1/asset` (ссылки на источники архивных
  ревизий) - открытый вопрос владельцу, см. eval K3.

## Фиксы (тот же заход)

Ветка `feature/corp-ai-kb-revisions`, коммиты 7c3a817 (фиксы store.py) и bbebd37
(тесты); eval-файл ревью - e314539.

- **P1** → 7c3a817. В `_archive_current_revision` nodes.jsonl и ассеты переезжают
  в каталог архива РАНЬШЕ manifest.json - manifest стал маркером завершённости
  архива, сбой между move не оставляет «полного» архива без единственной копии
  узлов. Перечисление архивов (`_archive_complete` в `_archived_revisions` и
  `iter_documents(include_revisions=True)`) пропускает revision-r* без
  nodes.jsonl с предупреждением в stderr вместо FileNotFoundError.
- **P2-1** → 7c3a817. Каталог ревизии без manifest.json трактуется как
  незавершённая архивация и дозавершается идемпотентно (`_move_into`: в приёмнике
  уже лежит обрывок - источник-дубль убирается, переехавшие узлы не
  перезаписываются). RuntimeError остался только для полного архива (manifest
  присутствует) с другим содержимым.
- **P2-2** → 7c3a817. `if current.get("sha256") and current.get("sha256") ==
  doc.sha256` - пустой sha256 legacy-манифеста больше не даёт ложного no-op.
- **P2-3** → 7c3a817. В `ingest_document` архивируемый revision_id обязан
  матчить `^r\d+$`, иначе RuntimeError с путём манифеста и ожидаемым форматом.
- **Уточнение сверх буквы ревью (почему):** первый срез фикса провалил новый
  тест `test_half_moved_archive_does_not_crash` - recovery-ветка
  `ingest_document` при обрыве «manifest уехал в архив, nodes остались в корне»
  считала архивов нет и создавала документ заново как r1, стирая корневой
  nodes.jsonl (это и есть сценарий потери из P1-(а), в другой ветке кода).
  Поэтому recovery-ветка расширена: обрывки (`_unfinished_archives`, manifest в
  архиве без nodes) входят в нумерацию восстановления, а
  `_archive_current_revision` при отсутствующем текущем манифесте дозавершает
  архив ДО записи новой ревизии - единственная копия узлов не стирается.

Покрытие (bbebd37, tests/test_revisions.py): главный security-комбайн
`test_archived_revision_invisible_cross_acl` (закрытый документ, чужая группа -
пусто в default/all/r1, в т.ч. через HTTP; entitled-группа видит обе ревизии в
all, дефолт - только текущую), сбой между архивацией и записью → r2 без потери,
полупереезжий архив → перечисление/индекс не падают (предупреждение) и ингест
дозавершивает без потери r1, липкий пустой revision-r1 → дозавершение и
продолжение нумерации r3, sha256='' + sha256='' → не no-op.

Доказательство (полный сет, код передан `git archive HEAD | ssh tar -x -C
/tmp/veles-k3-fix`, каталог вычищен после прогона):

```
$ cd /tmp/veles-k3-fix && CUDA_VISIBLE_DEVICES= <server-home>/veles-venv/bin/python -m pytest tests -q
116 passed, 1 warning in 5.38s     (стенд, python 3.12.3 venv; было 111)

$ python -m pytest tests -q        (локально, Windows, python 3.11)
116 passed in 19.31s
```

Регрессия: прежние 26 кейсов K3 зелёные (no-op по sha256, цепочка r1→r2→r3,
legacy-канон, AC6/AC7); negatives падают по своей причине
(`test_half_moved_archive_does_not_crash` на первом срезе фикса упал именно на
created-r1-с-потерей узлов).
