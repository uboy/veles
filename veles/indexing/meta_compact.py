"""Компактное RAM-представление меты узлов индекса (фазы 5 и 8 эпика index-format).

Проблема: после json.loads(meta.json) node_meta держит полные тексты узлов
в питоновских dict'ах - на прода-корпусе 235 754 узла это ~1.3 ГиБ RAM
фасада, доминирующий потребитель (бриф
docs/plan/briefs/2026-10-15-meta-compact.md). Формат ДИСКА не меняется:
IndexBundle.save разворачивает компактную форму обратно (to_plain_dict).

Часть 2 (фаза 5) сжала тексты и повторяющиеся строки; замер на реальном
корпусе (бриф docs/plan/briefs/2026-10-15-meta-tuple.md) показал, что после
текстов главная масса - каркасы dict'ов rest-записей (~410 МБ), уникальные
logical_node_id (62 МБ), provenance (61 МБ, внутри lines-списки) и
heading_path (19 МБ). Фаза 8 дожимает их (CompactNodeMeta - MutableMapping):

- тексты: ОДИН bytes-блоб из per-node zlib-чанков (сжатие каждого текста
  отдельно => ленивая распаковка O(текста) без прохода по всему блобу);
  границы чанков - _off (int64, длина N+1);
- logical_node_id: СВОЙ zlib-блоб с офсетами _ln_off (int64, N+1) - строки
  уникальны, интернирование не жмёт; не-str значение (в т.ч. null) остаётся
  обычным полем узла;
- rest-записи узлов: СХЕМЫ-КОРТЕЖИ вместо dict. Схема - кортеж имён полей
  узла (в исходном порядке; строковый text в схему не входит - его место
  восстанавливается добавлением в конец, как в части 2). Таблица уникальных
  схем (у узлов одного документа наборы полей совпадают - схем немного) +
  int32-колонка _scol на узел; на узел - КОРТЕЖ ЗНАЧЕНИЙ в порядке схемы
  (на месте полей, вынесенных в колонки/таблицы/блобы, - заглушка _SKIP).
  Интернированные поля остаются в схеме именами - этим сохраняется исходный
  порядок ключей при материализации (равенство dict от порядка не зависит,
  но бит-в-бит доступ включает набор и значения всех полей);
  интернированные скаляры (document_id, title, revision_id, status,
  sha256_doc): таблицы уникальных значений + int32-колонки; access_groups и
  heading_path - таблицы КОРТЕЖЕЙ (у большинства узлов совпадают) +
  int32-колонки; неинтернируемые значения этих полей (null, не тот тип)
  остаются в кортеже узла как есть;
- provenance: строки adapter/source_id - в своих таблицах (_pcols);
  lines - пара int32-колонок start/len + ПЛОСКИЙ int32-массив значений
  (_pl_vals; len = -2 - явный null, -1 - поля нет); ОСТАЛЬНЫЕ поля
  provenance (page, t_start, ...) - в схему кортежа узла под именами
  «provenance.<поле>» (дети собираются обратно при материализации). Если
  форма не раскладывается (lines - не список int, в узле есть своё поле
  с именем «provenance.*», provenance - не dict) - provenance целиком
  остаётся значением кортежа, строки внутри - sentinel-маркерами, как в
  части 2. Порядок ключей материализованного provenance после разложения
  не сохраняется (равенство dict и json round-trip от него не зависят);
- новые узлы (очередь загрузок K4, __setitem__/update) - отдельный plain-dict
  поверх компактных структур, отдаётся приоритетно, живёт как обычный dict;
- точечные правки существующих узлов (superseded K3, tombstone K10, смена
  групп) - ТОЛЬКО через patch_node/patch_node_meta: присваивание в dict,
  возвращённый __getitem__, при следующем доступе потерялось бы (запись
  материализуется заново каждый раз).

Потокобезопасность - как у прежнего dict (K4/K21): структуры загрузки
строятся до публикации бандла; одиночные __setitem__/patch_node атомарны
под GIL; чтение существующих узлов из другого потока при setitem безопасно;
итерация во время расширения - под INDEX_LOCK, как раньше.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator, Mapping, MutableMapping, Sequence

import numpy as np

# zlib-уровень текстовых чанков: замер на синтетике 100к узлов
# (scratch/rss-meta-100k.py, 20к текстов ~1.2 КБ): level 1 -> 9.5 МБ блоба,
# level 2 -> 9.4 МБ (-1%), level 6 -> 9.1 МБ; скорость сжатия одинакова,
# распаковка в hot-path важнее размера - фиксируем 1. Тот же уровень - для
# чанков logical_node_id (те же короткие utf-8 строки).
_ZLIB_LEVEL = 1


class MetaCompactError(Exception):
    """Порча компактных структур меты (блоб/офсеты/кодировка) - внятная
    ошибка вместо молчаливого мусора в выдаче."""


class _Mark:
    """Sentinel интернированного поля ВЛОЖЕННОГО provenance (ветка, когда
    разложение невозможно): материализатор подменяет его значением из
    таблицы по int32-колонке. Сравнение по identity (наследуемый
    __eq__/__hash__ object) - значение из json этими объектами быть не может."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # диагностика дампов
        return f"<intern:{self.name}>"


_M_ADAPTER = _Mark("provenance.adapter")
_M_SOURCE_ID = _Mark("provenance.source_id")

# скалярные строки, интернируемые в таблицы уникальных значений
_SCALAR_FIELDS = ("document_id", "title", "revision_id", "status", "sha256_doc")
# строки внутри provenance, интернируемые так же
_PROV_FIELDS = ("adapter", "source_id")

# префикс имён полей provenance, разложенных в схему кортежа узла
_PROV_PREFIX = "provenance."
_PROV_PREFIX_LEN = len(_PROV_PREFIX)

# маркер «поля нет» (значение поля может быть и None - различаем)
_MISSING = object()

# заглушка в кортеже значений узла: поле вынесено в колонку/таблицу/блоб и
# восстанавливается по имени при материализации (identity-сравнение)
_SKIP = object()

# коды состояния provenance узла при сборке
_PROV_ABSENT = 0       # поля provenance нет
_PROV_DECOMPOSED = 1   # разложен: pcols + pl_* + дети в схеме
_PROV_VALUE = 2        # значение кортежа: raw (не-dict) или nested-копия


class _Schema:
    """Схема узла: имена всех полей (кроме строкового text) в исходном
    порядке + производные индексы. Объект один на уникальный набор полей;
    создаются только в from_dict."""

    __slots__ = ("names", "map", "prov_children")

    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        # имя -> позиция в кортеже значений (быстрый lookup для get_fields)
        self.map = {k: p for p, k in enumerate(names)}
        # разложенные поля provenance: (имя без префикса, позиция)
        self.prov_children = tuple(
            (n[_PROV_PREFIX_LEN:], p)
            for p, n in enumerate(names)
            if n.startswith(_PROV_PREFIX)
        )


class _StringTable:
    """Таблица уникальных значений: value -> индекс (int32-колонка узлов)."""

    __slots__ = ("values", "_index")

    def __init__(self) -> None:
        self.values: list = []
        self._index: dict = {}

    def intern(self, value) -> int:
        idx = self._index.get(value, -1)
        if idx < 0:
            idx = len(self.values)
            self._index[value] = idx
            self.values.append(value)
        return idx


class CompactNodeMeta(MutableMapping):
    """dict-совместимая компактная мета узлов (см. докстринг модуля).

    Живой цикл: json.loads(meta.json) -> CompactNodeMeta.from_dict() в
    load_index; очередь K4 добавляет узлы __setitem__/update (plain-dict
    поверх); правки статусов/групп - patch_node; save разворачивает
    to_plain_dict()."""

    def __init__(self) -> None:
        # компактная часть (заполняется только from_dict, дальше read-only):
        self._order: list[str] = []          # nid в исходном порядке
        self._pos: dict[str, int] = {}       # nid -> номер строки
        self._blob: bytes = b""              # конкатенация zlib-чанков текстов
        self._off = np.zeros(1, dtype=np.int64)  # границы чанков, len = N+1
        self._ln_blob: bytes = b""           # zlib-чанки logical_node_id
        self._ln_off = np.zeros(1, dtype=np.int64)
        self._schemas: list[_Schema] = []    # таблица уникальных схем
        self._scol = np.zeros(0, dtype=np.int32)  # nid -> индекс схемы (-1: raw)
        self._rows: list[tuple] = []         # кортеж значений на узел
        self._tables: dict[str, _StringTable] = {f: _StringTable() for f in _SCALAR_FIELDS}
        self._cols: dict[str, np.ndarray] = {}
        self._ptables: dict[str, _StringTable] = {f: _StringTable() for f in _PROV_FIELDS}
        self._pcols: dict[str, np.ndarray] = {}
        self._groups_tbl = _StringTable()    # кортежи групп
        self._groups_col = np.zeros(0, dtype=np.int32)
        self._hpath_tbl = _StringTable()     # кортежи heading_path
        self._hpath_col = np.zeros(0, dtype=np.int32)
        # provenance.lines: плоский int32-массив + колонки start/len
        # (len = -2: явный null; -1: поля нет)
        self._pl_start = np.zeros(0, dtype=np.int32)
        self._pl_len = np.full(0, -1, dtype=np.int32)
        self._pl_vals = np.zeros(0, dtype=np.int32)
        self._n_prov_decomposed = 0
        self._n_prov_nested = 0
        # поверх компактной части (мутируемая, K4):
        self._added: dict[str, dict] = {}    # узлы setitem/update - как есть
        self._patches: dict[str, dict] = {}  # точечные правки compact-узлов

    # ------------------------------------------------------------ построение

    @classmethod
    def from_dict(cls, data: Mapping[str, dict], zlib_level: int = _ZLIB_LEVEL) -> "CompactNodeMeta":
        """Сжатие payload'а meta["node_meta"] (результат json.loads) - единственный
        вход в компактную форму. Значения узлов не копируются, кроме вложенного
        provenance в ветке без разложения (уходят adapter/source_id в таблицы)."""
        obj = cls()
        n = len(data)
        obj._order = list(data.keys())
        obj._pos = {nid: i for i, nid in enumerate(obj._order)}
        obj._off = np.zeros(n + 1, dtype=np.int64)
        obj._ln_off = np.zeros(n + 1, dtype=np.int64)
        obj._scol = np.full(n, -1, dtype=np.int32)
        # -1 = поле у узла отсутствует (колонки читаются только при живой
        # заглушке в кортеже - там гарантированно проинтернированное значение)
        obj._cols = {f: np.full(n, -1, dtype=np.int32) for f in _SCALAR_FIELDS}
        obj._pcols = {f: np.full(n, -1, dtype=np.int32) for f in _PROV_FIELDS}
        obj._groups_col = np.full(n, -1, dtype=np.int32)
        obj._hpath_col = np.full(n, -1, dtype=np.int32)
        obj._pl_start = np.zeros(n, dtype=np.int32)
        obj._pl_len = np.full(n, -1, dtype=np.int32)
        tables, cols = obj._tables, obj._cols
        ptables, pcols = obj._ptables, obj._pcols
        groups_tbl, groups_col = obj._groups_tbl, obj._groups_col
        hpath_tbl, hpath_col = obj._hpath_tbl, obj._hpath_col
        schemas, schema_ids = obj._schemas, {}
        pl_start_col, pl_len_col = obj._pl_start, obj._pl_len

        chunks: list[bytes] = []
        ln_chunks: list[bytes] = []
        rows: list[tuple] = []
        pl_vals: list[int] = []
        n_decomposed = 0
        n_nested = 0
        off = 0
        ln_off = 0
        for i, (nid, node) in enumerate(data.items()):
            if not isinstance(node, dict):
                # нестандартное значение узла (чужой/ручной meta.json): храним
                # как есть (scol = -1), при доступе отдаём без материализации -
                # как отдал бы прежний dict
                chunks.append(b"")
                ln_chunks.append(b"")
                obj._off[i + 1] = off
                obj._ln_off[i + 1] = ln_off
                rows.append(node)
                continue
            # текст: не-str значение (в т.ч. null) остаётся обычным полем
            # кортежа; пустая строка даёт валидный непустой чанк
            text = node.get("text")
            if isinstance(text, str):
                raw = zlib.compress(text.encode("utf-8"), zlib_level)
                text_in_row = False
            else:
                raw = b""
                text_in_row = "text" in node
            chunks.append(raw)
            off += len(raw)
            obj._off[i + 1] = off
            # logical_node_id: строка - в свой zlib-блоб (строки уникальны -
            # интернирование не жмёт); не-str/отсутствие - обычное поле кортежа
            lnid = node.get("logical_node_id")
            if type(lnid) is str:
                ln_raw = zlib.compress(lnid.encode("utf-8"), zlib_level)
                lnid_in_row = False
            else:
                ln_raw = b""
                lnid_in_row = "logical_node_id" in node
            ln_chunks.append(ln_raw)
            ln_off += len(ln_raw)
            obj._ln_off[i + 1] = ln_off

            # provenance: разложить на колонки/плоский массив, если форма
            # позволяет; иначе - вложенное значение с sentinel-интернированием
            prov = node.get("provenance", _MISSING)
            prov_state = _PROV_ABSENT
            prov_value = None
            prov_children: dict[str, object] = {}
            if prov is not _MISSING and type(prov) is dict and not any(
                k != "provenance" and k.startswith(_PROV_PREFIX) for k in node
            ):
                lines = prov.get("lines", _MISSING)
                lines_ok = lines is _MISSING or lines is None or (
                    type(lines) is list
                    and all(
                        type(x) is int and -(1 << 31) <= x < (1 << 31) for x in lines
                    )
                )
                if lines_ok:
                    v = prov.get("adapter")
                    if type(v) is str:
                        pcols["adapter"][i] = ptables["adapter"].intern(v)
                    v = prov.get("source_id")
                    if type(v) is str:
                        pcols["source_id"][i] = ptables["source_id"].intern(v)
                    if lines is not _MISSING:
                        if lines is None:
                            pl_len_col[i] = -2
                        else:
                            pl_start_col[i] = len(pl_vals)
                            pl_vals.extend(lines)
                            pl_len_col[i] = len(lines)
                    for ck, cv in prov.items():
                        if ck == "lines":
                            continue
                        if (ck == "adapter" or ck == "source_id") and type(cv) is str:
                            continue  # уже в таблице ptables
                        prov_children[ck] = cv
                    prov_state = _PROV_DECOMPOSED
                    n_decomposed += 1
            if prov_state == _PROV_ABSENT and prov is not _MISSING:
                if type(prov) is dict:
                    # разложение невозможно (lines не ints, коллизия имён):
                    # shallow copy - таблицы держат строки, запись - свой словарь
                    prov2 = dict(prov)
                    v = prov2.get("adapter")
                    if type(v) is str:
                        pcols["adapter"][i] = ptables["adapter"].intern(v)
                        prov2["adapter"] = _M_ADAPTER
                    v = prov2.get("source_id")
                    if type(v) is str:
                        pcols["source_id"][i] = ptables["source_id"].intern(v)
                        prov2["source_id"] = _M_SOURCE_ID
                    prov_value = prov2
                    n_nested += 1
                else:
                    prov_value = prov  # не-dict (в т.ч. null) - как есть
                prov_state = _PROV_VALUE

            # схема: имена полей узла в исходном порядке (строковый text -
            # исключение: восстанавливается добавлением в конец) + разложенные
            # поля provenance; сортировка детей - стабильность схем
            names_list = [k for k in node if k != "text" or text_in_row]
            if prov_state == _PROV_DECOMPOSED:
                names_list.extend(
                    sorted(_PROV_PREFIX + c for c in prov_children)
                )
            names = tuple(names_list)
            si = schema_ids.get(names)
            if si is None:
                si = len(schemas)
                schema_ids[names] = si
                schemas.append(_Schema(names))
            obj._scol[i] = si
            # кортеж значений в порядке схемы; на месте вынесенных полей -
            # заглушка _SKIP (восстанавливается по имени)
            row: list = []
            for k in names:
                if k == "text":  # здесь только не-str text
                    row.append(node["text"])
                    continue
                if k == "logical_node_id":
                    row.append(_SKIP if type(node[k]) is str else node[k])
                    continue
                if k == "provenance":
                    row.append(_SKIP if prov_state == _PROV_DECOMPOSED else prov_value)
                    continue
                if k.startswith(_PROV_PREFIX) and prov_state == _PROV_DECOMPOSED:
                    # иначе это СВОЁ поле узла - обычное значение кортежа
                    row.append(prov_children[k[_PROV_PREFIX_LEN:]])
                    continue
                v = node[k]
                extracted = False
                col = cols.get(k)
                if col is not None and type(v) is str:
                    col[i] = tables[k].intern(v)
                    extracted = True
                elif k == "access_groups" and type(v) is list and all(
                    type(g) is str for g in v
                ):
                    groups_col[i] = groups_tbl.intern(tuple(v))
                    extracted = True
                elif k == "heading_path" and type(v) is list and all(
                    type(g) is str for g in v
                ):
                    hpath_col[i] = hpath_tbl.intern(tuple(v))
                    extracted = True
                row.append(_SKIP if extracted else v)
            rows.append(tuple(row))

        obj._blob = b"".join(chunks)
        obj._ln_blob = b"".join(ln_chunks)
        obj._rows = rows
        obj._pl_vals = np.array(pl_vals, dtype=np.int32) if pl_vals else np.zeros(0, dtype=np.int32)
        obj._n_prov_decomposed = n_decomposed
        obj._n_prov_nested = n_nested
        return obj

    # --------------------------------------------------- MutableMapping API

    def __getitem__(self, key: str) -> dict:
        added = self._added.get(key)
        if added is not None:
            return added
        i = self._pos.get(key)
        if i is None:
            raise KeyError(key)
        si = int(self._scol[i])
        if si < 0:
            # нестандартное значение узла (см. from_dict) - без материализации
            return self._rows[i]
        schema = self._schemas[si]
        row = self._rows[i]
        node: dict = {}
        cols = self._cols
        tables = self._tables
        # разложён ли provenance у этого узла (дети в кортеже - его части,
        # а не поля верхнего уровня; литеральные поля «provenance.*» узла
        # материализуются как есть)
        p_prov = schema.map.get("provenance", -1)
        prov_decomposed = p_prov >= 0 and row[p_prov] is _SKIP
        for p, k in enumerate(schema.names):
            v = row[p]
            if v is not _SKIP:
                if prov_decomposed and k.startswith(_PROV_PREFIX):
                    continue  # ребёнок provenance - уже внутри собираемого dict
                if k == "provenance" and type(v) is dict and (
                    v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
                ):
                    # adapter/source_id - sentinel'ы, достраиваем из таблиц
                    v = self._rebuild_provenance(v, i)
                node[k] = v
                continue
            # вынесенное поле: восстанавливаем по имени
            col = cols.get(k)
            if col is not None:  # интернированный скаляр
                node[k] = tables[k].values[int(col[i])]
            elif k == "access_groups":
                node[k] = list(self._groups_tbl.values[int(self._groups_col[i])])
            elif k == "heading_path":
                node[k] = list(self._hpath_tbl.values[int(self._hpath_col[i])])
            elif k == "logical_node_id":
                node[k] = self._node_lnid(i)
            else:  # provenance, разложенный по колонкам/плоскому массиву
                node[k] = self._rebuild_prov_decomposed(i, schema, row)
        text = self._node_text(i)
        if text is not None:
            # text стоял в исходном порядке ключей не последним; порядок ключей
            # dict ни на что не влияет (равенство dict, json.dumps/loads)
            node["text"] = text
        patch = self._patches.get(key)
        if patch:
            node.update(patch)
        return node

    def _rebuild_provenance(self, prov: dict, i: int) -> dict:
        """Ветка без разложения: вложенный dict с sentinel-адаптером/source_id."""
        out = dict(prov)
        a = out.get("adapter")
        if a is _M_ADAPTER:
            out["adapter"] = self._ptables["adapter"].values[int(self._pcols["adapter"][i])]
        s = out.get("source_id")
        if s is _M_SOURCE_ID:
            out["source_id"] = self._ptables["source_id"].values[int(self._pcols["source_id"][i])]
        return out

    def _rebuild_prov_decomposed(self, i: int, schema: _Schema, row: tuple) -> dict:
        """Сборка provenance из колонок/плоского массива/детей схемы."""
        prov: dict = {}
        a = self._pcols["adapter"][i]
        if a >= 0:
            prov["adapter"] = self._ptables["adapter"].values[int(a)]
        s = self._pcols["source_id"][i]
        if s >= 0:
            prov["source_id"] = self._ptables["source_id"].values[int(s)]
        for child, p in schema.prov_children:
            prov[child] = row[p]
        ln = int(self._pl_len[i])
        if ln >= 0:
            st = int(self._pl_start[i])
            prov["lines"] = self._pl_vals[st:st + ln].tolist()
        elif ln == -2:
            prov["lines"] = None
        return prov

    def __setitem__(self, key: str, value: dict) -> None:
        # K4: узлы очереди загрузок живут обычным dict поверх компактных
        # структур (их мало, сжатие не нужно); если id уже есть в компактной
        # части - эта запись затирает её для всех чтений (приоритет _added)
        self._added[key] = value

    def __delitem__(self, key: str) -> None:
        raise NotImplementedError(
            "CompactNodeMeta: удаление узлов не поддерживается (K10 - tombstone "
            "status=deleted через patch_node_meta, физика удаления - пересборка)"
        )

    def __iter__(self) -> Iterator[str]:
        added = self._added
        for nid in self._order:
            if nid not in added:
                yield nid
        yield from added

    def __len__(self) -> int:
        # setitem поверх существующего id затирает компактную строку (редко):
        # len = компактные + новые (не перекрывающиеся) added
        pos = self._pos
        return len(pos) + sum(1 for k in self._added if k not in pos)

    def __contains__(self, key: object) -> bool:
        # без материализации: наличие не обязано распаковывать текст
        return key in self._added or key in self._pos

    # -------------------------------------------------------- правки (K3/K10)

    def patch_node(self, nid: str, patch: Mapping) -> None:
        """Точечный патч полей узла (superseded/tombstone/группы). Единственный
        способ изменить существующий узел: присваивания в dict, возвращённый
        __getitem__, не сохраняются - при следующем доступе запись
        материализуется заново из компактных данных."""
        added = self._added.get(nid)
        if added is not None:
            added.update(patch)  # обычный dict - правим напрямую
            return
        cur = self._patches.get(nid)
        if cur is None:
            self._patches[nid] = dict(patch)
        else:
            cur.update(patch)

    def get_fields(self, nid: str, keys: Sequence[str]) -> dict:
        """Быстрое чтение нескольких полей узла БЕЗ материализации записи и
        распаковки текста: интернированные скаляры - из таблиц по колонкам,
        access_groups/heading_path - из таблиц кортежей, logical_node_id - из
        своего блоба, «text» - распаковка своего чанка, остальные - прямым
        чтением кортежа узла. Горячие циклы hybrid_search (authorized/бусты)
        читают 1-3 скаляра на каждого кандидата; полный __getitem__ с
        zlib-текстом (~10 мкс) на каждом из ~150-250 чтений запроса давал
        +250-313% latency (evals/results/2026-10-15-meta-compact.md).
        Значения те же, что отдаёт __getitem__ (с учётом _added/_patches);
        узла нет - пустой dict."""
        added = self._added.get(nid)
        if added is not None:
            return {k: added[k] for k in keys if k in added}
        i = self._pos.get(nid)
        if i is None:
            return {}
        out: dict = {}
        patch = self._patches.get(nid)
        si = int(self._scol[i])
        if si < 0:
            # нестандартное значение-не-dict: полей нет (только патчи)
            if patch is not None:
                for k in keys:
                    if k in patch:
                        out[k] = patch[k]
            return out
        schema = self._schemas[si]
        smap = schema.map
        row = self._rows[i]
        cols = self._cols
        tables = self._tables
        # разложён ли provenance (дети в кортеже - не поля верхнего уровня)
        p_prov = smap.get("provenance", -1)
        prov_decomposed = p_prov >= 0 and row[p_prov] is _SKIP
        for k in keys:
            if patch is not None and k in patch:
                out[k] = patch[k]
                continue
            p = smap.get(k, -1)
            if p < 0:
                # в схеме узла поля нет; «text» при этом может лежать в блобе
                if k == "text":
                    t = self._node_text(i)
                    if t is not None:
                        out[k] = t
                continue
            v = row[p]
            if v is not _SKIP:
                if prov_decomposed and k.startswith(_PROV_PREFIX):
                    continue  # ребёнок provenance - не поле верхнего уровня
                if k == "provenance" and type(v) is dict and (
                    v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
                ):
                    out[k] = self._rebuild_provenance(v, i)
                else:
                    out[k] = v
                continue
            col = cols.get(k)
            if col is not None:  # интернированный скаляр
                out[k] = tables[k].values[int(col[i])]
            elif k == "access_groups":
                out[k] = list(self._groups_tbl.values[int(self._groups_col[i])])
            elif k == "heading_path":
                out[k] = list(self._hpath_tbl.values[int(self._hpath_col[i])])
            elif k == "logical_node_id":
                out[k] = self._node_lnid(i)
            else:  # provenance, разложенный по колонкам/плоскому массиву
                out[k] = self._rebuild_prov_decomposed(i, schema, row)
        return out

    def iter_fields(self, *keys: str) -> Iterator[tuple[str, dict]]:
        """O(N)-скан выбранных полей по всем узлам БЕЗ материализации и
        распаковки текстов: values()/items() при компактной форме дают полный
        узел на каждый nid (zlib-текст ~10 мкс/узел - секундный обход и
        транзитный снапшот всех текстов на прода-корпусе), а сканам
        /toc-/stats, _resolve_revision, doc_titles, regroup, DELETE,
        supersede/tombstone и инвентарю картинок нужны 1-5 скаляров на узел.
        Ленивый генератор (nid, {поле: значение}); отсутствующие поля
        опускаются (потребительский .get с дефолтом работает как раньше);
        запрошенный 'text'/'logical_node_id' распаковывается - явный запрос.
        Значения те же, что отдаёт __getitem__ (с учётом _added/_patches);
        порядок - как в __iter__. Нестандартные значения-не-dict (см.
        from_dict) дают пустую запись - прежний код на них падал бы на .get,
        скан молча пропускает. Полные узлы - обычный доступ [nid]."""
        added = self._added
        patches = self._patches
        cols = self._cols
        tables = self._tables
        rows = self._rows
        schemas = self._schemas
        scol = self._scol
        # быстрый путь одиночного интернированного скаляра / групп (часть
        # сканов-потребителей): колонка разворачивается одним tolist(), без
        # per-node словарных поисков; узлы с неинтернируемым значением (null
        # и т.п.) уходят в lookup по схеме
        if len(keys) == 1:
            k = keys[0]
            col = cols.get(k)
            if col is not None:
                values = tables[k].values
                for i, (nid, c) in enumerate(zip(self._order, col.tolist())):
                    if nid in added:
                        continue
                    p = patches.get(nid)
                    if p is not None and k in p:
                        yield nid, {k: p[k]}
                    elif c >= 0:
                        yield nid, {k: values[c]}
                    else:
                        si = scol[i]
                        pp = schemas[si].map.get(k, -1) if si >= 0 else -1
                        yield nid, ({} if pp < 0 else {k: rows[i][pp]})
                for nid, node in added.items():
                    if k in node:
                        yield nid, {k: node[k]}
                    else:
                        yield nid, {}
                return
            if k == "access_groups":
                gvalues = self._groups_tbl.values
                for i, (nid, g) in enumerate(zip(self._order, self._groups_col.tolist())):
                    if nid in added:
                        continue
                    p = patches.get(nid)
                    if p is not None and k in p:
                        yield nid, {k: p[k]}
                    elif g >= 0:
                        yield nid, {k: list(gvalues[g])}
                    else:
                        si = scol[i]
                        pp = schemas[si].map.get(k, -1) if si >= 0 else -1
                        yield nid, ({} if pp < 0 else {k: rows[i][pp]})
                for nid, node in added.items():
                    if k in node:
                        yield nid, {k: node[k]}
                    else:
                        yield nid, {}
                return
        groups_col = self._groups_col
        groups_tbl = self._groups_tbl
        hpath_col = self._hpath_col
        hpath_tbl = self._hpath_tbl
        for i, nid in enumerate(self._order):
            if nid in added:
                continue  # затёрт setitem'ом - отдадется ниже из added
            out: dict = {}
            patch = patches.get(nid)
            si = scol[i]
            if si < 0:
                # нестандартное значение-не-dict: полей нет, патчи отдаются
                if patch is not None:
                    for k in keys:
                        if k in patch:
                            out[k] = patch[k]
                yield nid, out
                continue
            schema = schemas[si]
            smap = schema.map
            row = rows[i]
            # разложён ли provenance (дети в кортеже - не поля верхнего уровня)
            p_prov = smap.get("provenance", -1)
            prov_decomposed = p_prov >= 0 and row[p_prov] is _SKIP
            for k in keys:
                if patch is not None and k in patch:
                    out[k] = patch[k]
                    continue
                p = smap.get(k, -1)
                if p < 0:
                    if k == "text":
                        t = self._node_text(i)
                        if t is not None:
                            out[k] = t
                    continue
                v = row[p]
                if v is not _SKIP:
                    if prov_decomposed and k.startswith(_PROV_PREFIX):
                        continue  # ребёнок provenance - не поле верхнего уровня
                    if k == "provenance" and type(v) is dict and (
                        v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
                    ):
                        out[k] = self._rebuild_provenance(v, i)
                    else:
                        out[k] = v
                    continue
                col = cols.get(k)
                if col is not None:  # интернированный скаляр
                    out[k] = tables[k].values[int(col[i])]
                elif k == "access_groups":
                    out[k] = list(groups_tbl.values[int(groups_col[i])])
                elif k == "heading_path":
                    out[k] = list(hpath_tbl.values[int(hpath_col[i])])
                elif k == "logical_node_id":
                    out[k] = self._node_lnid(i)
                else:  # provenance, разложенный по колонкам/плоскому массиву
                    out[k] = self._rebuild_prov_decomposed(i, schema, row)
            yield nid, out
        for nid, node in added.items():
            yield nid, {k: node[k] for k in keys if k in node}

    # --------------------------------------------------------------- выгрузка

    def to_plain_dict(self) -> dict[str, dict]:
        """Полный payload для записи meta.json: все узлы материализованы,
        тексты распакованы. Только путь save - в живом фасаде не вызывается."""
        return {nid: self[nid] for nid in self}

    def compact_stats(self) -> dict:
        """Служебные цифры для eval/замеров (не контракт)."""
        return {
            "nodes": len(self._pos),
            "blob_bytes": len(self._blob),
            "lnid_blob_bytes": len(self._ln_blob),
            "unique_scalars": {name: len(t.values) for name, t in self._tables.items()},
            "unique_prov": {name: len(t.values) for name, t in self._ptables.items()},
            "unique_groups": len(self._groups_tbl.values),
            "unique_heading_paths": len(self._hpath_tbl.values),
            "schemas": len(self._schemas),
            "lines_values": int(self._pl_vals.size),
            "prov_decomposed": self._n_prov_decomposed,
            "prov_nested": self._n_prov_nested,
            "added": len(self._added),
            "patched": len(self._patches),
        }

    # ----------------------------------------------------------- внутреннее

    def _node_text(self, i: int) -> str | None:
        """Ленивая распаковка текста узла i; текста у узла нет - None."""
        start = int(self._off[i])
        end = int(self._off[i + 1])
        if end <= start:
            return None
        try:
            return zlib.decompress(self._blob[start:end]).decode("utf-8")
        except (zlib.error, UnicodeDecodeError) as e:
            raise MetaCompactError(
                f"компактная мета: не распакован текст узла {self._order[i]!r} "
                f"(чанк байт [{start}:{end}] при блобе {len(self._blob)} байт): {e}"
            ) from e

    def _node_lnid(self, i: int) -> str:
        """Ленивая распаковка logical_node_id узла i. Вызывается только когда
        заглушка в кортеже гарантирует строку, поэтому пустой чанк - порча."""
        start = int(self._ln_off[i])
        end = int(self._ln_off[i + 1])
        if end <= start:
            raise MetaCompactError(
                f"компактная мета: у узла {self._order[i]!r} ожидается "
                f"logical_node_id, но чанк пуст (офсеты [{start}:{end}], "
                f"блоб {len(self._ln_blob)} байт)"
            )
        try:
            return zlib.decompress(self._ln_blob[start:end]).decode("utf-8")
        except (zlib.error, UnicodeDecodeError) as e:
            raise MetaCompactError(
                f"компактная мета: не распакован logical_node_id узла "
                f"{self._order[i]!r} (чанк байт [{start}:{end}] при блобе "
                f"{len(self._ln_blob)} байт): {e}"
            ) from e


def patch_node_meta(node_meta, nid: str, patch: Mapping) -> None:
    """Единая точка правки меты узла для всех потребителей.

    Присваивания в dict, возвращённый node_meta[nid], при компактной форме
    молча терялись бы (материализация каждый доступ заново), поэтому tombstone
    (overlay), superseded (queue), смена групп (documents) идут через сюда:
    CompactNodeMeta - patch_node, обычный dict - прежний update на месте.
    """
    if isinstance(node_meta, CompactNodeMeta):
        node_meta.patch_node(nid, patch)
    else:
        target = node_meta.get(nid)
        if target is not None:
            target.update(patch)


def iter_node_fields(node_meta, *keys: str) -> Iterator[tuple[str, dict]]:
    """O(N)-скан выбранных полей для потребителей-итераторов (тот же dispatch,
    что get_fields в retriever): CompactNodeMeta - лёгкие записи только
    запрошенных полей (без распаковки текстов, кроме явно запрошенного
    'text'); обычный dict - полные узлы как есть, прежнее поведение без
    лишней работы. Потребители, которым нужен text конкретного узла, берут
    полный node_meta[nid] точечно."""
    scan = getattr(node_meta, "iter_fields", None)
    if scan is not None:
        return scan(*keys)
    return node_meta.items()
