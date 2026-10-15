"""Компактная мета узлов (фаза 5 плана index-format, часть 2; бриф
docs/plan/briefs/2026-10-15-meta-compact.md).

Покрытие:
- бит-в-бит доступа: каждое поле каждого узла компактной формы == исходному
  dict (синтетика 5000+ узлов: русский/латиница, дублирующиеся document_id/
  title/revision_id, редкие и нестандартные поля);
- dict-семантика: len/contains/iter/keys/get/eq, values/items - ленивые
  представления, __getitem__ возвращает СВЕЖИЙ dict;
- setitem очереди загрузок (K4) из другого потока поверх компактной части +
  len/iter/values после add;
- правки существующих узлов только через patch_node_meta (superseded K3,
  tombstone K10, regroup групп) - компактная форма и прежний plain dict;
- save round-trip: load_index (мета компактна) -> bundle.save -> перечитанный
  meta.json по-ключевому равен исходному, тексты байт-в-байт;
- потребители на компактной мете: image_index.inventory_from_meta, junk-фильтр
  retriever (is_junk_node по meta.get("text"), как в hybrid_search);
- негатив: подрезанный блоб, битый байт чанка, сдвинутые офсеты ->
  MetaCompactError (внятная ошибка, не мусор).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import ItemsView, KeysView, ValuesView

import pytest

from veles.config import EMBED_DIM, EMBED_MODEL
from veles.canonical.store import _doc_dir
from veles.indexing.bm25 import Bm25Index
from veles.indexing.indexer import load_index
from veles.indexing.meta_compact import (
    CompactNodeMeta,
    MetaCompactError,
    iter_node_fields,
    patch_node_meta,
)
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import is_junk_node
from veles.indexing.image_index import inventory_from_meta

EMBED_META = {"model": EMBED_MODEL, "backend": "numpy"}


def make_nodes(n: int = 5000) -> dict[str, dict]:
    """Синтетика в форме прода-корпуса: дублирующиеся document_id/title/
    revision_id/status, тексты рус+латиница переменной длины с уникальным
    хвостом, редкие (asset) и нестандартные (page, prov без source_id) поля."""
    n_docs = max(1, n // 50)
    docs = [
        (
            f"doc-{d:03d}",
            f"Регламент №{d} о закупках (Reglement {d})",
            f"{d:064x}",
        )
        for d in range(n_docs)
    ]
    groups_pool = [["hr"], ["hr", "it"], ["*"], ["fin", "legal", "hr"]]
    adapters = ["md", "docling", "video"]
    sources = ["src-sharepoint", "src-1c", "src-wiki"]
    out: dict[str, dict] = {}
    for i in range(n):
        did, title, sha = docs[i % n_docs]
        node: dict = {
            "document_id": did,
            "title": title,
            "logical_node_id": f"ln-{i % 97}-{i}",
            "heading_path": [title, f"Глава {i % 12}", f"Статья {i}"],
            "text": (
                f"Порог единственной закупки составляет {i} рублей (threshold {i}). "
                "Общие положения о порядке осуществления закупки, требованиях "
                "к участникам и обосновании выбора поставщика. "
            ) * (1 + i % 3) + f"Уникальный хвост узла node-{i}.",
            "provenance": {
                "source_id": sources[i % len(sources)],
                "adapter": adapters[i % len(adapters)],
                "lines": [i, i + 7] if i % 2 else None,
            },
            "access_groups": groups_pool[i % len(groups_pool)],
            "sha256_doc": sha,
            "revision_id": f"r{1 + i % 3}",
            "status": ["current", "current", "superseded"][i % 3],
            "lost_force": i % 17 == 0,
        }
        if i % 5 == 0:
            node["asset"] = f"pic-{i:03d}.png"
        if i % 11 == 0:
            node["page"] = i % 300  # нестандартное поле верхнего уровня
        if i % 23 == 0:
            node["provenance"] = {"adapter": "unknown"}  # prov без source_id
        if i % 41 == 0:
            del node["access_groups"]  # узел без групп
        out[f"nv-{i:06d}"] = node
    return out


def _assert_nodes_equal(got: dict, expected: dict, where: str) -> None:
    """Каждое поле каждого узла - точное равенство с исходным dict."""
    assert set(got) == set(expected), f"{where}: набор полей"
    for field in expected:
        assert got[field] == expected[field], f"{where}: поле {field}"


# ------------------------------------------------------------- бит-в-бит


def test_bitwise_access_every_field_of_every_node():
    plain = make_nodes(5000)
    compact = CompactNodeMeta.from_dict(plain)
    assert len(compact) == len(plain)
    assert set(compact) == set(plain)
    for nid, expected in plain.items():
        _assert_nodes_equal(compact[nid], expected, nid)
    # повторный доступ (материализация заново) - тот же результат
    for nid in list(plain)[:100]:
        _assert_nodes_equal(compact[nid], plain[nid], f"повтор {nid}")


def test_getitem_returns_fresh_dict():
    plain = make_nodes(20)
    compact = CompactNodeMeta.from_dict(plain)
    nid = next(iter(plain))
    m = compact[nid]
    m["text"] = "испорчено"
    m["взломано"] = True
    _assert_nodes_equal(compact[nid], plain[nid], nid)
    assert "взломано" not in compact[nid]
    # два доступа - разные объекты dict (свежая материализация)
    assert compact[nid] is not compact[nid]


def test_dict_semantics_lazy_views():
    plain = make_nodes(100)
    compact = CompactNodeMeta.from_dict(plain)
    assert isinstance(compact.keys(), KeysView)
    assert isinstance(compact.values(), ValuesView)
    assert isinstance(compact.items(), ItemsView)
    # исходный порядок ключей сохранён
    assert list(compact.keys()) == list(plain.keys())
    assert list(iter(compact)) == list(plain)
    assert dict(compact.items()) == plain
    assert list(compact.values()) == [plain[k] for k in plain]
    assert compact.get("нет-такого") is None
    assert compact.get("nv-000001") == plain["nv-000001"]
    with pytest.raises(KeyError):
        compact["нет-такого"]
    assert "nv-000001" in compact
    assert "нет-такого" not in compact
    assert len(compact) == len(plain)
    assert compact == plain  # Mapping.__eq__ против dict


def test_degenerate_nodes_preserved():
    plain = {
        "a": {"document_id": "d1", "text": ""},          # пустой текст
        "b": {"title": "только заголовок"},               # без text
        "c": {"text": None, "payload": [1, 2]},           # text=null
        "d": "не dict вовсе",                             # нестандартное значение
    }
    compact = CompactNodeMeta.from_dict(plain)
    assert compact == plain
    assert compact["a"] == {"document_id": "d1", "text": ""}
    assert compact["c"] == {"text": None, "payload": [1, 2]}
    assert compact["d"] == "не dict вовсе"
    empty = CompactNodeMeta.from_dict({})
    assert len(empty) == 0
    assert empty.to_plain_dict() == {}
    assert empty.get("x") is None
    assert "x" not in empty


# ----------------------------------------------------- K4: setitem из потока


def test_setitem_from_another_thread_k4():
    plain = make_nodes(300)
    compact = CompactNodeMeta.from_dict(plain)
    new_nodes = {
        f"new-{i:03d}": {
            "document_id": "doc-new",
            "title": "Загруженный документ",
            "text": f"Текст нового узла {i}: положение о командировании сотрудников.",
            "provenance": {"source_id": "src-upload", "adapter": "md"},
            "access_groups": ["user:42"],
            "status": "current",
            "revision_id": "r1",
        }
        for i in range(50)
    }
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for nid, node in new_nodes.items():
                compact[nid] = node
                time.sleep(0.001)  # читатель успевает войти в цикл чтения
        except Exception as e:  # noqa: BLE001 - ошибка уходит в assertions
            errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    # чтение СУЩЕСТВУЮЩИХ узлов из главного потока во время setitem из другого
    checked = 0
    while t.is_alive():
        for i in range(50):
            _assert_nodes_equal(compact[f"nv-{i:06d}"], plain[f"nv-{i:06d}"], "k4")
            checked += 1
    t.join()
    assert not errors
    assert checked > 0
    # добавленное читается бит-в-бит из главного потока
    for nid, expected in new_nodes.items():
        _assert_nodes_equal(compact[nid], expected, nid)
    assert len(compact) == len(plain) + len(new_nodes)
    assert set(iter(compact)) == set(plain) | set(new_nodes)
    assert dict(compact.items()) == {**plain, **new_nodes}


def test_setitem_overrides_compact_node():
    plain = make_nodes(10)
    compact = CompactNodeMeta.from_dict(plain)
    replacement = {"document_id": "doc-over", "text": "запись поверх", "status": "current"}
    compact["nv-000003"] = replacement
    # затирающая запись отдаётся приоритетно, len/iter не задваивают узел
    assert compact["nv-000003"] == replacement
    assert len(compact) == 10
    assert list(iter(compact)).count("nv-000003") == 1
    patch_node_meta(compact, "nv-000003", {"status": "deleted"})
    assert compact["nv-000003"]["status"] == "deleted"  # патч правит _added
    assert replacement["status"] == "deleted"  # тот же dict - как у обычного dict


# ------------------------------------------------------- правки (K3/K10/группы)


def test_patch_node_meta_compact_and_plain():
    plain = make_nodes(50)
    compact = CompactNodeMeta.from_dict(plain)
    # superseded (K3, как queue.upsert_document)
    patch_node_meta(compact, "nv-000000", {"status": "superseded"})
    assert compact["nv-000000"]["status"] == "superseded"
    assert plain["nv-000000"]["status"] == "current"  # исходный payload не тронут
    # tombstone (K10, как overlay.tombstone_document)
    patch_node_meta(compact, "nv-000001", {"status": "deleted"})
    assert compact["nv-000001"]["status"] == "deleted"
    # regroup групп (documents): один объект списка на все узлы документа
    parsed = ["audit-2027"]
    for nid in ("nv-000002", "nv-000003"):
        patch_node_meta(compact, nid, {"access_groups": parsed})
    assert compact["nv-000002"]["access_groups"] == ["audit-2027"]
    assert compact["nv-000003"]["access_groups"] == ["audit-2027"]
    # patch накапливается на узле, to_plain_dict отдаёт пропатченное
    patch_node_meta(compact, "nv-000000", {"lost_force": True})
    rt = compact.to_plain_dict()
    assert rt["nv-000000"]["status"] == "superseded"
    assert rt["nv-000000"]["lost_force"] is True
    # патченный узел не попал в дубли: порядок ключей и длина целы
    assert len(rt) == len(plain)
    # несуществующий nid на компактной форме - KeyError-семантика dict.get
    # в потребителях не используется; patch отсутствующего молча создаёт патч,
    # чтение отсутствующего по-прежнему KeyError
    with pytest.raises(KeyError):
        compact["нет-такого"]
    # прежний plain dict через тот же хелпер - прежняя семантика
    d = {"x": {"status": "current"}}
    patch_node_meta(d, "x", {"status": "superseded"})
    assert d["x"]["status"] == "superseded"
    patch_node_meta(d, "нет-такого", {"status": "superseded"})  # молча, как .get
    assert "нет-такого" not in d


# ------------------------------------------------------- save round-trip


def _write_index_dir(dir_path, plain: dict) -> None:
    dir_path.mkdir(parents=True)
    (dir_path / "meta.json").write_text(
        json.dumps(
            {"node_meta": plain, "embed": EMBED_META, "identifiers": ["НК РФ", "350 000"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    Bm25Index().save(dir_path)
    NumpyFlatStore().save(dir_path)


def test_load_save_roundtrip_meta_bitwise(tmp_path):
    plain = make_nodes(500)
    src = tmp_path / "src"
    _write_index_dir(src, plain)
    bundle = load_index(src)
    # интеграция: load_index отдаёт компактную форму
    assert isinstance(bundle.node_meta, CompactNodeMeta)
    assert len(bundle.node_meta) == len(plain)
    out = tmp_path / "out"
    bundle.save(out, embed_meta=EMBED_META)
    reread = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    # identifiers при save сортируются (как и до правки) - сверка со sorted
    assert reread["identifiers"] == sorted(["НК РФ", "350 000"])
    assert reread["embed"] == EMBED_META
    # по-ключевое равенство с исходным payload, тексты байт-в-байт
    nodes = reread["node_meta"]
    assert nodes == plain
    for nid, expected in plain.items():
        _assert_nodes_equal(nodes[nid], expected, nid)
    # «рестарт»: перечитанный каталог снова компактный и бит-в-бит
    bundle2 = load_index(out)
    assert isinstance(bundle2.node_meta, CompactNodeMeta)
    assert bundle2.node_meta.to_plain_dict() == plain


# ------------------------------------------------- потребители на компактной мете


def test_get_fields_matches_getitem_without_materialization():
    plain = make_nodes(60)
    compact = CompactNodeMeta.from_dict(plain)
    keys = ("status", "access_groups", "revision_id", "title", "document_id", "asset")
    for nid, expected in plain.items():
        want = {k: expected[k] for k in keys if k in expected}
        assert compact.get_fields(nid, keys) == want, nid
    # не-интернированные поля, text и provenance (с sentinel-интернированием)
    nid = "nv-000001"
    assert compact.get_fields(nid, ("logical_node_id",)) == {
        "logical_node_id": plain[nid]["logical_node_id"]
    }
    assert compact.get_fields(nid, ("text",)) == {"text": plain[nid]["text"]}
    assert compact.get_fields(nid, ("provenance",))["provenance"] == plain[nid]["provenance"]
    # отсутствующий узел и отсутствующие поля
    assert compact.get_fields("нет-такого", ("status",)) == {}
    assert compact.get_fields(nid, ("нет-такого",)) == {}
    # узел без запрошенного поля не ломается (nv-000041 без access_groups)
    assert "access_groups" not in compact.get_fields("nv-000041", ("access_groups", "status"))
    # патчи (K3/K10) и setitem (K4) учитываются
    patch_node_meta(compact, "nv-000000", {"status": "superseded"})
    assert compact.get_fields("nv-000000", ("status",)) == {"status": "superseded"}
    compact["nv-new"] = {"title": "новый", "status": "current"}
    assert compact.get_fields("nv-new", ("status", "title")) == {
        "title": "новый", "status": "current"
    }


def test_retriever_junk_filter_on_compact_meta():
    plain = {
        "nv-junk": {
            "document_id": "d1",
            "title": "Кодекс",
            "text": "Утратил силу.",
            "status": "current",
        },
        "nv-real": {
            "document_id": "d1",
            "title": "Кодекс",
            "text": "Порог единственной закупки составляет 350 тысяч рублей.",
            "status": "current",
        },
    }
    compact = CompactNodeMeta.from_dict(plain)
    # тот же паттерн доступа, что в hybrid_search (retriever.py:309-330):
    # bundle.node_meta.get(nid) -> is_junk_node(meta.get("text", ""))
    survivors = []
    for nid in ("nv-junk", "nv-real", "нет-такого"):
        meta = compact.get(nid)
        if meta and not is_junk_node(meta.get("text", "")):
            survivors.append(nid)
    assert survivors == ["nv-real"]


def test_image_inventory_on_compact_meta(tmp_path):
    canonical = tmp_path / "canonical"
    doc_id = "doc-042"
    asset_dir = _doc_dir(doc_id, canonical) / "assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "pic-001.png").write_bytes(b"\x89PNG-fake")
    base = {
        "document_id": doc_id,
        "title": "Схема процесса",
        "text": "Подпись схемы: движение заявки",
        "provenance": {"source_id": "src-1", "adapter": "docling", "page": 2},
        "access_groups": ["hr"],
        "status": "current",
        "revision_id": "r1",
    }
    plain = {
        "nv-img": {**base, "asset": "pic-001.png"},
        "nv-old": {**base, "asset": "pic-001.png", "status": "superseded"},
        "nv-nofile": {**base, "asset": "pic-999.png"},
        "nv-notimg": {**base},  # без asset
    }
    compact = CompactNodeMeta.from_dict(plain)
    records, skipped = inventory_from_meta(compact, [canonical])
    assert len(records) == 1
    rec = records[0]
    assert rec.image_id == "nv-img"
    assert rec.document_id == doc_id
    assert rec.asset == "pic-001.png"
    assert rec.page == 2
    assert rec.access_groups == ["hr"]  # кортеж из таблицы групп -> list
    assert "Подпись схемы" in rec.caption
    assert {s["image_id"] for s in skipped} == {"nv-nofile"}
    assert {s["reason"] for s in skipped} == {"asset-file-missing"}
    # superseded-узел с asset пропускается молча (не в records, не в skipped)
    assert "nv-old" not in {r.image_id for r in records}


# ---------------------------------------------------------------- негативы


def test_truncated_blob_raises_meta_compact_error():
    plain = make_nodes(20)
    broken = CompactNodeMeta.from_dict(plain)
    broken._blob = broken._blob[:-3]  # хвост последнего чанка отрезан
    last = list(plain)[-1]
    with pytest.raises(MetaCompactError):
        broken[last]["text"]


def test_flipped_byte_in_blob_raises_meta_compact_error():
    plain = make_nodes(20)
    flipped = CompactNodeMeta.from_dict(plain)
    blob = bytearray(flipped._blob)
    blob[len(blob) // 2] ^= 0xFF  # битый байт в сжатом чанке
    flipped._blob = bytes(blob)
    raised = [nid for nid in plain if _raises_compact(flipped, nid)]
    assert raised, "битый байт обязан дать MetaCompactError хотя бы на одном узле"


def test_shifted_offsets_raise_meta_compact_error():
    plain = make_nodes(20)
    shifted = CompactNodeMeta.from_dict(plain)
    shifted._off[3] += 2  # граница чанков уехала внутрь соседнего потока
    with pytest.raises(MetaCompactError):
        shifted[list(plain)[3]]["text"]


def _raises_compact(meta: CompactNodeMeta, nid: str) -> bool:
    try:
        meta[nid]["text"]
    except MetaCompactError:
        return True
    return False


# ------------- фаза 8: схемы-кортежи, блоб logical_node_id, lines в int32 -------------
#
# Профиль полей реального корпуса (бриф docs/plan/briefs/2026-10-15-meta-tuple.md):
# у части узлов нет asset/heading_path/lines/logical_node_id; пустые списки,
# явные null, неинтернируемые типы, коллизии имён - всё обязано материализоваться
# бит-в-бит (набор полей, значения, отсутствие vs null).


def make_profile_nodes(n: int = 600) -> dict[str, dict]:
    """Синтетика реального профиля: индексы веток подобраны детерминированно
    (np-<i>), тесты ссылаются на конкретные узлы - не менять раскладку."""
    docs = [(f"doc-{d:03d}", f"Регламент №{d}", f"{d:064x}") for d in range(12)]
    out: dict[str, dict] = {}
    for i in range(n):
        did, title, sha = docs[i % 12]
        prov: dict = {
            "source_id": f"src-{i % 3}",
            "adapter": "md",
            "lines": [i, i + 7],
        }
        node: dict = {
            "document_id": did,
            "title": title,
            "sha256_doc": sha,
            "revision_id": "r1",
            "status": "current",
            "text": f"Текст узла {i}: положение о порядке действий. Хвост {i}.",
            "provenance": prov,
            "access_groups": ["hr"],
            "logical_node_id": f"ln-{i}",
            "heading_path": [title, f"Раздел {i % 9}"],
            "lost_force": False,
        }
        if i % 5:  # у части узлов asset нет
            node["asset"] = f"pic-{i:03d}.png"
        if i % 7 == 0:
            del node["heading_path"]             # поля нет
        elif i % 11 == 0:
            node["heading_path"] = []            # пустой список
        elif i % 13 == 0:
            node["heading_path"] = None          # явный null (не интернируется)
        if i % 4 == 0:
            node["provenance"]["page"] = i % 300  # ребёнок provenance -> схема
        if i % 9 == 0:
            prov["lines"] = []                   # пустой список lines
        elif i % 6 == 0:
            prov["lines"] = None                 # явный null
        elif i % 37 == 0:
            del prov["lines"]                    # lines нет
        if i % 41 == 0:
            node["provenance"] = {"adapter": "md", "lines": ["5-9"]}  # не-int
        if i % 43 == 0:
            node["provenance"] = None            # provenance - null
        if i % 47 == 0:
            node["provenance.page"] = 7          # коллизия имён верхнего уровня
        if i % 53 == 0:
            del node["logical_node_id"]          # поля нет
        elif i % 59 == 0:
            node["logical_node_id"] = None       # явный null
        if i % 61 == 0:
            node["status"] = None                # неинтернируемый скаляр
        if i % 67 == 0:
            node["title"] = None
        if i % 71 == 0:
            node["access_groups"] = None
        if i % 73 == 0:
            node["text"] = None                  # text - null (в кортеж)
        out[f"np-{i:06d}"] = node
    return out


def test_real_profile_bitwise_access_every_field():
    plain = make_profile_nodes(600)
    compact = CompactNodeMeta.from_dict(plain)
    assert len(compact) == len(plain)
    assert set(compact) == set(plain)
    for nid, expected in plain.items():
        _assert_nodes_equal(compact[nid], expected, nid)
    # повторный доступ (материализация заново) - тот же результат
    for nid in list(plain)[:80]:
        _assert_nodes_equal(compact[nid], plain[nid], f"повтор {nid}")


def test_real_profile_get_fields_and_scans_match_getitem():
    plain = make_profile_nodes(600)
    compact = CompactNodeMeta.from_dict(plain)
    keys = ("status", "access_groups", "document_id", "title", "revision_id",
            "heading_path", "provenance", "asset", "logical_node_id",
            "provenance.page", "lost_force", "sha256_doc")
    for nid, expected in plain.items():
        want = {k: expected[k] for k in keys if k in expected}
        assert compact.get_fields(nid, keys) == want, nid
    # одиночные сканы (быстрый путь по колонке) - те же значения, включая
    # null-значения и отсутствие поля
    for k in keys:
        scanned = dict(compact.iter_fields(k))
        assert list(scanned) == list(plain), k
        for nid, expected in plain.items():
            want = {k: expected[k]} if k in expected else {}
            assert scanned[nid] == want, f"{k}/{nid}"
    # общий путь скана (несколько полей за проход)
    scanned = dict(compact.iter_fields(*keys))
    for nid, expected in plain.items():
        want = {k: expected[k] for k in keys if k in expected}
        assert scanned[nid] == want, nid
    # скан text отдаёт и text=null из кортежа (не только блоб)
    scanned_text = dict(compact.iter_fields("text"))
    for nid, expected in plain.items():
        assert scanned_text[nid] == {"text": expected["text"]}, nid


def test_schema_tuples_dedup():
    plain = make_profile_nodes(600)
    compact = CompactNodeMeta.from_dict(plain)
    stats = compact.compact_stats()
    # каркасы dict заменены кортежами значений
    assert all(type(r) is tuple for r in compact._rows)
    # профиль разнообразный, но схем на порядок меньше, чем узлов
    assert stats["schemas"] < len(plain) // 4, (
        f"схем {stats['schemas']} на {len(plain)} узлов - дедупликация не работает"
    )
    # одинаковый набор полей всех узлов - ровно одна схема
    uniform = {
        f"u-{i:04d}": {
            "document_id": "d1",
            "title": "Т",
            "text": f"текст {i}",
            "provenance": {"source_id": "s", "adapter": "md", "lines": [i, i + 2]},
            "access_groups": ["hr"],
            "status": "current",
        }
        for i in range(100)
    }
    assert CompactNodeMeta.from_dict(uniform).compact_stats()["schemas"] == 1


def test_logical_node_id_blob_and_negatives():
    plain = make_profile_nodes(120)
    compact = CompactNodeMeta.from_dict(plain)
    stats = compact.compact_stats()
    assert stats["lnid_blob_bytes"] > 0
    # строковые id читаются из блоба бит-в-бит
    assert compact["np-000001"]["logical_node_id"] == plain["np-000001"]["logical_node_id"]
    # отсутствие и явный null различаются
    assert "logical_node_id" not in compact["np-000000"]
    assert compact["np-000059"]["logical_node_id"] is None
    # подрезанный блоб логических id - внятная ошибка, в т.ч. в get_fields
    broken = CompactNodeMeta.from_dict(plain)
    broken._ln_blob = broken._ln_blob[:-2]  # хвост последнего чанка отрезан
    with pytest.raises(MetaCompactError):
        broken["np-000119"]["logical_node_id"]
    with pytest.raises(MetaCompactError):
        broken.get_fields("np-000119", ("logical_node_id",))
    # сдвинутые офсеты - тоже порча
    shifted = CompactNodeMeta.from_dict(plain)
    shifted._ln_off[1] += 3
    with pytest.raises(MetaCompactError):
        shifted["np-000001"]["logical_node_id"]
    # «пропавший» чанк при живой заглушке в кортеже - порча, не пустая строка
    emptied = CompactNodeMeta.from_dict(plain)
    emptied._ln_off[2] = emptied._ln_off[1]
    with pytest.raises(MetaCompactError):
        emptied["np-000001"]["logical_node_id"]


def test_heading_path_table_of_tuples():
    # дублирующиеся пути - таблица уникальных кортежей
    shared = {
        f"s-{i:03d}": {
            "document_id": "d1",
            "title": f"т-{i}",
            "text": f"текст {i}",
            "heading_path": ["Общий", f"Раздел {i % 4}"],
            "status": "current",
        }
        for i in range(50)
    }
    compact = CompactNodeMeta.from_dict(shared)
    assert compact.compact_stats()["unique_heading_paths"] == 4
    for i in range(50):
        nid = f"s-{i:03d}"
        hp = compact[nid]["heading_path"]
        assert hp == shared[nid]["heading_path"], nid
        assert type(hp) is list
    # правка пути только через patch - как regroup
    patch_node_meta(compact, "s-000", {"heading_path": ["Новый", "Путь"]})
    assert compact["s-000"]["heading_path"] == ["Новый", "Путь"]
    # профиль реального корпуса: пустой список, null, отсутствие
    prof = CompactNodeMeta.from_dict(make_profile_nodes(200))
    assert prof["np-000011"]["heading_path"] == []      # 11 % 11 == 0
    assert prof["np-000013"]["heading_path"] is None    # 13 % 13 == 0
    assert "heading_path" not in prof["np-000007"]      # 7 % 7 == 0


def test_provenance_lines_columns_and_fallbacks():
    plain = make_profile_nodes(300)
    compact = CompactNodeMeta.from_dict(plain)
    stats = compact.compact_stats()
    assert stats["lines_values"] > 0
    assert stats["prov_decomposed"] > 0
    assert stats["prov_nested"] > 0
    # пустой список / явный null / отсутствие lines различаются бит-в-бит
    assert compact["np-000009"]["provenance"]["lines"] == []      # 9 % 9 == 0
    assert compact["np-000006"]["provenance"]["lines"] is None    # 6 % 6 == 0
    assert "lines" not in compact["np-000037"]["provenance"]      # 37 % 37 == 0
    # не-int lines: разложение невозможно, provenance вложенный - те же значения
    assert compact["np-000041"]["provenance"] == {"adapter": "md", "lines": ["5-9"]}
    # provenance = null
    assert compact["np-000043"]["provenance"] is None
    # коллизия: своё поле "provenance.page" и вложенный provenance соседствуют
    n47 = compact["np-000047"]
    assert n47["provenance.page"] == 7
    assert n47["provenance"]["lines"] == [47, 54]
    # ребёнок provenance (page) собирается из схемы кортежа
    assert compact["np-000004"]["provenance"] == {
        "source_id": "src-1",
        "adapter": "md",
        "page": 4,
        "lines": [4, 11],
    }
    # get_fields и скан отдают собранный provenance
    assert compact.get_fields("np-000004", ("provenance",)) == {
        "provenance": compact["np-000004"]["provenance"]
    }
    scanned = dict(compact.iter_fields("provenance"))
    for nid, expected in plain.items():
        want = {"provenance": expected["provenance"]} if "provenance" in expected else {}
        assert scanned[nid] == want, nid


def test_profile_save_roundtrip_via_json():
    plain = make_profile_nodes(200)
    compact = CompactNodeMeta.from_dict(plain)
    payload = json.loads(json.dumps(compact.to_plain_dict(), ensure_ascii=False))
    # по-ключевое равенство (порядок ключей provenance после разложения
    # может отличаться - равенство dict от него не зависит)
    assert payload == plain
    for nid, expected in plain.items():
        _assert_nodes_equal(payload[nid], expected, nid)
    # рестарт: перечитанный payload снова компактен и бит-в-бит
    c2 = CompactNodeMeta.from_dict(payload)
    assert c2.to_plain_dict() == plain


# ------------------------------- O(N)-сканы полей (ревью раунд 1) -------------------------------


def test_iter_fields_matches_full_access():
    """Скан отдаёт те же значения, что полный доступ (в т.ч. патчи, added,
    отсутствие поля, порядок __iter__) - смена значений сканов запрещена."""
    plain = make_nodes(200)
    compact = CompactNodeMeta.from_dict(plain)
    # поля всех сканов-потребителей: /toc-/stats, _resolve_revision, doc_titles,
    # regroup, DELETE, supersede/tombstone, инвентарь картинок
    keys = ("status", "access_groups", "document_id", "title", "heading_path",
            "revision_id", "asset", "provenance")
    scanned = dict(compact.iter_fields(*keys))
    assert list(scanned) == list(plain)  # порядок итерации сохранён
    for nid, light in scanned.items():
        want = {k: plain[nid][k] for k in keys if k in plain[nid]}
        assert light == want, nid
    # отсутствующее поле опускается (nv-000041 без access_groups)
    assert "access_groups" not in scanned["nv-000041"]
    # патчи видны скану
    patch_node_meta(compact, "nv-000000", {"status": "superseded"})
    patch_node_meta(compact, "nv-000001", {"access_groups": ["audit-2027"]})
    assert dict(compact.iter_fields("status"))["nv-000000"] == {"status": "superseded"}
    assert dict(compact.iter_fields("access_groups"))["nv-000001"] == {
        "access_groups": ["audit-2027"]
    }
    # added-узлы (K4) в скане, затёртые setitem'ом без дублей
    compact["nv-new"] = {"document_id": "doc-x", "text": "новый"}
    compact["nv-000002"] = {"document_id": "doc-y", "text": "поверх"}
    scanned2 = dict(compact.iter_fields("document_id"))
    assert scanned2["nv-new"] == {"document_id": "doc-x"}
    assert scanned2["nv-000002"] == {"document_id": "doc-y"}
    assert len(scanned2) == len(plain) + 1
    # явный запрос text распаковывается
    assert dict(compact.iter_fields("text"))["nv-000003"] == {"text": plain["nv-000003"]["text"]}


def test_iter_node_fields_plain_dict_yields_full_nodes():
    """Обычный dict через диспетчер - прежнее поведение: полные узлы, ноль
    копий (надмножество запрошенных полей)."""
    plain = make_nodes(10)
    pairs = list(iter_node_fields(plain, "document_id"))
    assert pairs == [(nid, node) for nid, node in plain.items()]


def test_scan_fields_does_not_decompress_text():
    """Структурная регрессия находки ревью: скан без 'text' не трогает блоб -
    на испорченном блобе проходит, полный доступ и скан с 'text' падают
    MetaCompactError."""
    plain = make_nodes(30)
    broken = CompactNodeMeta.from_dict(plain)
    broken._blob = broken._blob[:-3]
    # O(N)-сканы потребителей (/toc, _resolve_revision, doc_titles, regroup,
    # DELETE, supersede, tombstone) не читают text - работают на битом блобе
    scanned = dict(broken.iter_fields("status", "document_id", "title"))
    assert len(scanned) == len(plain)
    assert scanned["nv-000000"] == {
        "status": plain["nv-000000"]["status"],
        "document_id": plain["nv-000000"]["document_id"],
        "title": plain["nv-000000"]["title"],
    }
    # а чтение текста тем временем честно падает (подрезан последний чанк)
    with pytest.raises(MetaCompactError):
        broken[list(plain)[-1]]["text"]
    with pytest.raises(MetaCompactError):
        dict(broken.iter_fields("text"))


def test_stats_and_revision_scan_on_compact_meta():
    """Регрессия ревью на уровне API: /v1/knowledge/stats (O(N)-снапшот под
    INDEX_LOCK) и _resolve_revision (каждый revision-запрос) работают на
    компактной мете и дают тот же результат, что на dict-мете."""
    from fastapi.testclient import TestClient

    from tests.test_api import FakeEmbedder, make_bundle
    from veles.api.app import VelesPipeline, create_app

    def client_for(compact: bool):
        bundle = make_bundle()
        if compact:
            bundle.node_meta = CompactNodeMeta.from_dict(bundle.node_meta)
        return TestClient(create_app(pipeline=VelesPipeline(bundle, FakeEmbedder(), None)))

    d_plain = client_for(False).get("/v1/knowledge/stats").json()
    r = client_for(True).get("/v1/knowledge/stats")
    assert r.status_code == 200
    assert r.json() == d_plain
    assert d_plain["totals"]["nodes"] == 2

    c = client_for(True)
    # известная ревизия (в мете нет revision_id -> дефолт "r1") проходит,
    # неизвестная - 400: скан revision_id с компактной меты собрал набор
    assert c.post("/v1/knowledge/query",
                  json={"query": "какая норма управляемости", "revision": "r1"}).status_code == 200
    assert c.post("/v1/knowledge/query",
                  json={"query": "какая норма управляемости", "revision": "r99"}).status_code == 400
