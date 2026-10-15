"""Компактное RAM-представление меты узлов индекса (фаза 5 эпика index-format).

Проблема: после json.loads(meta.json) node_meta держит полные тексты узлов
в питоновских dict'ах - на прода-корпусе 235 754 узла это ~1.3 ГиБ RAM
фасада, доминирующий потребитель (бриф
docs/plan/briefs/2026-10-15-meta-compact.md). Формат ДИСКА не меняется:
IndexBundle.save разворачивает компактную форму обратно (to_plain_dict).

CompactNodeMeta - MutableMapping[str, dict] поверх:
- тексты: ОДИН bytes-блоб из per-node zlib-чанков (сжатие каждого текста
  отдельно => ленивая распаковка O(текста) без прохода по всему блобу);
  границы чанков - _off (int64, длина N+1);
- интернированные повторяющиеся строки: таблицы уникальных значений +
  int32-колонка на узел (document_id, title, revision_id, status,
  sha256_doc, provenance.adapter, provenance.source_id); access_groups -
  таблица КОРТЕЖЕЙ групп (у большинства узлов группы совпадают);
- прочие поля узла (asset, heading_path, logical_node_id, lost_force и любые
  неожиданные) - в per-node записи как есть, без интернирования: их доля в
  RAM на порядок меньше текстов. Interned-поля остаются в той же записи
  sentinel-маркерами - исходный порядок ключей сохраняется;
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
# распаковка в hot-path важнее размера - фиксируем 1.
_ZLIB_LEVEL = 1


class MetaCompactError(Exception):
    """Порча компактных структур меты (блоб/офсеты/кодировка) - внятная
    ошибка вместо молчаливого мусора в выдаче."""


class _Mark:
    """Sentinel интернированного поля в per-node записи: материализатор
    подменяет его значением из таблицы по int32-колонке. Сравнение по
    identity (наследуемый __eq__/__hash__ object) - значение из json этими
    объектами быть не может."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # диагностика дампов
        return f"<intern:{self.name}>"


_M_DOCUMENT_ID = _Mark("document_id")
_M_TITLE = _Mark("title")
_M_REVISION_ID = _Mark("revision_id")
_M_STATUS = _Mark("status")
_M_SHA256_DOC = _Mark("sha256_doc")
_M_ACCESS_GROUPS = _Mark("access_groups")
_M_ADAPTER = _Mark("provenance.adapter")
_M_SOURCE_ID = _Mark("provenance.source_id")

# скалярные строки, интернируемые в таблицы уникальных значений
_SCALAR_FIELDS = ("document_id", "title", "revision_id", "status", "sha256_doc")
# строки внутри provenance, интернируемые так же
_PROV_FIELDS = ("adapter", "source_id")

# маркер «поля нет» для rec.get в get_fields (значение поля может быть None)
_MISSING = object()


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
        self._rest: list[dict] = []          # per-node записи без text
        self._tables: dict[str, _StringTable] = {f: _StringTable() for f in _SCALAR_FIELDS}
        self._cols: dict[str, np.ndarray] = {}
        self._ptables: dict[str, _StringTable] = {f: _StringTable() for f in _PROV_FIELDS}
        self._pcols: dict[str, np.ndarray] = {}
        self._groups_tbl = _StringTable()    # кортежи групп
        self._groups_col = np.zeros(0, dtype=np.int32)
        # поверх компактной части (мутируемая, K4):
        self._added: dict[str, dict] = {}    # узлы setitem/update - как есть
        self._patches: dict[str, dict] = {}  # точечные правки compact-узлов
        # sentinel -> (таблица, колонка): один lookup вместо цепочки if;
        # заполняется в from_dict, когда колонки выделены
        self._mark_sources: dict[_Mark, tuple] = {}

    # ------------------------------------------------------------ построение

    @classmethod
    def from_dict(cls, data: Mapping[str, dict], zlib_level: int = _ZLIB_LEVEL) -> "CompactNodeMeta":
        """Сжатие payload'а meta["node_meta"] (результат json.loads) - единственный
        вход в компактную форму. Значения узлов не копируются, кроме shallow
        copy provenance (уходят adapter/source_id в таблицы)."""
        obj = cls()
        n = len(data)
        obj._order = list(data.keys())
        obj._pos = {nid: i for i, nid in enumerate(obj._order)}
        obj._off = np.zeros(n + 1, dtype=np.int64)
        # -1 = поле у узла отсутствует (читается только при живом sentinel)
        obj._cols = {f: np.full(n, -1, dtype=np.int32) for f in _SCALAR_FIELDS}
        obj._pcols = {f: np.full(n, -1, dtype=np.int32) for f in _PROV_FIELDS}
        obj._groups_col = np.full(n, -1, dtype=np.int32)
        tables, cols = obj._tables, obj._cols
        ptables, pcols = obj._ptables, obj._pcols
        groups_tbl, groups_col = obj._groups_tbl, obj._groups_col

        chunks: list[bytes] = []
        rest: list[dict] = []
        off = 0
        for i, (nid, node) in enumerate(data.items()):
            if not isinstance(node, dict):
                # нестандартное значение узла (чужой/ручной meta.json): храним
                # как есть, при доступе отдаём без материализации - как отдал бы
                # прежний dict
                chunks.append(b"")
                obj._off[i + 1] = off
                rest.append(node)
                continue
            # текст: не-str значение (в т.ч. null) остаётся в per-node записи
            # как есть; пустая строка даёт валидный непустой чанк
            text = node.get("text")
            if isinstance(text, str):
                raw = zlib.compress(text.encode("utf-8"), zlib_level)
                rec = {k: v for k, v in node.items() if k != "text"}
            else:
                raw = b""
                rec = dict(node)
            chunks.append(raw)
            off += len(raw)
            obj._off[i + 1] = off
            v = rec.get("document_id")
            if type(v) is str:
                cols["document_id"][i] = tables["document_id"].intern(v)
                rec["document_id"] = _M_DOCUMENT_ID
            v = rec.get("title")
            if type(v) is str:
                cols["title"][i] = tables["title"].intern(v)
                rec["title"] = _M_TITLE
            v = rec.get("revision_id")
            if type(v) is str:
                cols["revision_id"][i] = tables["revision_id"].intern(v)
                rec["revision_id"] = _M_REVISION_ID
            v = rec.get("status")
            if type(v) is str:
                cols["status"][i] = tables["status"].intern(v)
                rec["status"] = _M_STATUS
            v = rec.get("sha256_doc")
            if type(v) is str:
                cols["sha256_doc"][i] = tables["sha256_doc"].intern(v)
                rec["sha256_doc"] = _M_SHA256_DOC
            prov = rec.get("provenance")
            if type(prov) is dict:
                # shallow copy: таблицы держат строки, запись - свой словарь
                prov2 = dict(prov)
                v = prov2.get("adapter")
                if type(v) is str:
                    pcols["adapter"][i] = ptables["adapter"].intern(v)
                    prov2["adapter"] = _M_ADAPTER
                v = prov2.get("source_id")
                if type(v) is str:
                    pcols["source_id"][i] = ptables["source_id"].intern(v)
                    prov2["source_id"] = _M_SOURCE_ID
                rec["provenance"] = prov2
            v = rec.get("access_groups")
            if type(v) is list and all(type(g) is str for g in v):
                groups_col[i] = groups_tbl.intern(tuple(v))
                rec["access_groups"] = _M_ACCESS_GROUPS
            rest.append(rec)

        obj._blob = b"".join(chunks)
        obj._rest = rest
        obj._mark_sources = {
            _M_DOCUMENT_ID: (tables["document_id"], cols["document_id"]),
            _M_TITLE: (tables["title"], cols["title"]),
            _M_REVISION_ID: (tables["revision_id"], cols["revision_id"]),
            _M_STATUS: (tables["status"], cols["status"]),
            _M_SHA256_DOC: (tables["sha256_doc"], cols["sha256_doc"]),
        }
        return obj

    # --------------------------------------------------- MutableMapping API

    def __getitem__(self, key: str) -> dict:
        added = self._added.get(key)
        if added is not None:
            return added
        i = self._pos.get(key)
        if i is None:
            raise KeyError(key)
        node: dict = {}
        rec = self._rest[i]
        if type(rec) is not dict:
            # нестандартное значение узла (см. from_dict) - без материализации
            return rec
        if rec:
            mark_sources = self._mark_sources
            for k, v in rec.items():
                if type(v) is _Mark:
                    # интернированное поле: значение из таблицы по колонке
                    if v is _M_ACCESS_GROUPS:
                        node[k] = list(self._groups_tbl.values[int(self._groups_col[i])])
                    else:
                        tbl, col = mark_sources[v]
                        node[k] = tbl.values[int(col[i])]
                    continue
                if k == "provenance" and type(v) is dict and (
                    v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
                ):
                    # adapter/source_id - sentinel'ы, достраиваем из таблиц
                    v = self._rebuild_provenance(v, i)
                node[k] = v
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
        out = dict(prov)
        a = out.get("adapter")
        if a is _M_ADAPTER:
            out["adapter"] = self._ptables["adapter"].values[int(self._pcols["adapter"][i])]
        s = out.get("source_id")
        if s is _M_SOURCE_ID:
            out["source_id"] = self._ptables["source_id"].values[int(self._pcols["source_id"][i])]
        return out

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
        access_groups - из таблицы групп, «text» - распаковка своего чанка,
        остальные - прямым чтением per-node записи. Горячие циклы hybrid_search
        (authorized/бусты) читают 1-3 скаляра на каждого кандидата; полный
        __getitem__ с zlib-текстом (~10 мкс) на каждом из ~150-250 чтений
        запроса давал +250-313% latency (evals/results/2026-10-15-meta-compact.md).
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
        cols = self._cols
        tables = self._tables
        for k in keys:
            if patch is not None and k in patch:
                out[k] = patch[k]
                continue
            col = cols.get(k)
            if col is not None:  # интернированный скаляр
                c = col[i]
                if c >= 0:
                    out[k] = tables[k].values[int(c)]
                continue
            if k == "access_groups":
                g = self._groups_col[i]
                if g >= 0:
                    out[k] = list(self._groups_tbl.values[int(g)])
                continue
            rec = self._rest[i]
            if type(rec) is not dict:
                continue
            v = rec.get(k, _MISSING)
            if v is _MISSING:
                if k == "text":
                    t = self._node_text(i)
                    if t is not None:
                        out[k] = t
                continue
            if type(v) is _Mark:
                # ключ вне cols с sentinel-значением не встречается (sentinels
                # стоят только у интернированных полей) - но не мусор:
                out[k] = self[nid][k]
            elif k == "provenance" and (
                v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
            ):
                out[k] = self._rebuild_provenance(v, i)
            else:
                out[k] = v
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
        запрошенный 'text' распаковывается - явный запрос. Значения те же,
        что отдаёт __getitem__ (с учётом _added/_patches); порядок - как в
        __iter__. Нестандартные значения-не-dict (см. from_dict) дают пустую
        запись - прежний код на них падал бы на .get, скан молча пропускает.
        Полные узлы - обычный доступ [nid]."""
        added = self._added
        patches = self._patches
        cols = self._cols
        tables = self._tables
        # быстрый путь одиночного интернированного скаляра / групп (5 из 8
        # сканов-потребителей): вся колонка разворачивается одним tolist(),
        # без per-node словарных поисков cols/tables
        if len(keys) == 1:
            k = keys[0]
            col = cols.get(k)
            if col is not None:
                values = tables[k].values
                for nid, c in zip(self._order, col.tolist()):
                    if nid in added:
                        continue
                    p = patches.get(nid)
                    if p is not None and k in p:
                        yield nid, {k: p[k]}
                    elif c >= 0:
                        yield nid, {k: values[c]}
                    else:
                        yield nid, {}
                for nid, node in added.items():
                    if k in node:
                        yield nid, {k: node[k]}
                    else:
                        yield nid, {}
                return
            if k == "access_groups":
                gvalues = self._groups_tbl.values
                for nid, g in zip(self._order, self._groups_col.tolist()):
                    if nid in added:
                        continue
                    p = patches.get(nid)
                    if p is not None and k in p:
                        yield nid, {k: p[k]}
                    elif g >= 0:
                        yield nid, {k: list(gvalues[g])}
                    else:
                        yield nid, {}
                for nid, node in added.items():
                    if k in node:
                        yield nid, {k: node[k]}
                    else:
                        yield nid, {}
                return
        rest = self._rest
        groups_col = self._groups_col
        groups_tbl = self._groups_tbl
        for i, nid in enumerate(self._order):
            if nid in added:
                continue  # затёрт setitem'ом - отдадется ниже из added
            out: dict = {}
            patch = patches.get(nid)
            rec = rest[i]
            rec_ok = type(rec) is dict
            for k in keys:
                if patch is not None and k in patch:
                    out[k] = patch[k]
                    continue
                if not rec_ok:
                    continue
                col = cols.get(k)
                if col is not None:  # интернированный скаляр
                    c = col[i]
                    if c >= 0:
                        out[k] = tables[k].values[int(c)]
                    continue
                if k == "access_groups":
                    g = groups_col[i]
                    if g >= 0:
                        out[k] = list(groups_tbl.values[int(g)])
                    continue
                v = rec.get(k, _MISSING)
                if v is _MISSING:
                    if k == "text":
                        t = self._node_text(i)
                        if t is not None:
                            out[k] = t
                    continue
                if type(v) is _Mark:
                    out[k] = self[nid][k]
                elif k == "provenance" and (
                    v.get("adapter") is _M_ADAPTER or v.get("source_id") is _M_SOURCE_ID
                ):
                    out[k] = self._rebuild_provenance(v, i)
                else:
                    out[k] = v
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
            "unique_scalars": {name: len(t.values) for name, t in self._tables.items()},
            "unique_prov": {name: len(t.values) for name, t in self._ptables.items()},
            "unique_groups": len(self._groups_tbl.values),
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
