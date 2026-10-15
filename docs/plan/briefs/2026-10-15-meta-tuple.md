# Execution brief: дожатие компактной меты (схемы-кортежи, логические id в блобе)

Дата: 2026-10-15. Решение владельца: A (дожать), B (мета-SQLite) - потом
с конфиг-переключателем (фаза 9 плана).
Карточка: ai-tracker work/veles/task-veles-meta-compact.md.
Ветка: feature/meta-tuple (worktree C:\Users\devl\proj\veles-tuple).

## Факты после части 2 (main f0a9633)

A/B на bm1 (индекс 235 754 узла): plain-dict мета = 1065 МБ анонимки,
компактная = 1197 МБ. Компоненты компактной: blob текстов 76 МБ (сжатие
отличное), order 25 + pos 37, rest 620 МБ, таблицы/колонки ~40. Разбивка
rest: каркасы dict'ов ~410 МБ (dict на 10 полей ~1.8 КБ/узел), logical_node_id
62 МБ (строки уникальные), provenance 61 МБ (внутри: lines - списки из 1-3
int; adapter/source_id уже интернированы), heading_path 19 МБ (12 804
уникальных кортежа на 235 754 узлов), document_id/title/revision_id/status/
sha256_doc/access_groups/lost_force/asset - по 3-8 МБ.

## Что делаем (только veles/indexing/meta_compact.py + тесты)

1. rest-записи -> схемы-кортежи: «схема узла» = кортеж имён оставшихся полей;
   таблица уникальных схем (int32-колонка на узел); значения - кортеж в
   порядке схемы. Каркасы dict (~410 МБ) заменяются кортежами (~100-140 МБ).
   Материализация узла: dict собирается из схемы+кортежа как раньше.
2. logical_node_id: отдельный zlib-блоб + offsets/lengths (как text), колонка
   офсетов вместо строк.
3. heading_path: таблица уникальных кортежей (как access_groups) +
   int32-колонка.
4. provenance: lines - две int32-колонки (start/len) или int32-колонка длины
   + плоский массив (выбрать по замеру); прочие поля provenance - в схему
   кортежа.
5. Всё остальное без изменений: dict-семантика, setitem/_added/_patches
   (K4), to_plain_dict, iter_node_fields, patch_node_meta, MetaCompactError.

## Критерии готовности

- Полный pytest зелёный (тесты части 2 без правок обязаны пройти; новые:
  бит-в-бит доступа на профиле полей реального корпуса - heading_path,
  provenance.lines, logical_node_id, asset у подмножества узлов; save
  round-trip; K4).
- A/B на bm1 (тот же индекс 235 754 узла, /tmp/rss_ab.py): anon компактной
  <= 850 МБ при plain 1065 (цель -300..-450 МБ от текущих 1197).
- Latency гейт фазы 6 (< +5%) воспроизводится.
- Ревью независимого агента: approved.

## Результат (заполнить по завершении)

- Коммит(ы):
- A/B bm1:
- Ревью:
