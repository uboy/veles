"""Коннектор №1 (JTBD P0-1): `veles sync --root` - синхронизация каталога-
источника с каноном (бриф docs/plan/briefs/2026-10-11-connektor-papka.md).

Проверки плана фазы 1: дифф (новый/изменённый/удалённый/неизменённый файл),
симлинк/junction наружу игнорируются (побег за корень), dry-run и отказ без
--yes ничего не меняют, изменение файла - ревизия r2 (старая в архиве),
удаление файла - tombstone (канон в trash, документ вон из поиска),
идемпотентность (второй sync - ноль дельты), corpus-документы без маркера
владения sync не удаляются никогда, файловый лок от параллельного sync.
Без индекса sync пишет только канон (мета.json не создаётся). Эмбеддер -
детерминированный hashing (канон: тесты без загрузки моделей), dim = EMBED_DIM.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import pytest

from veles.canonical.store import _doc_dir, save_document, trash_root
from veles.config import EMBED_DIM, EMBED_MODEL
from veles.ingest.sync import (
    REASON_COLLISION,
    REASON_ENCODING,
    REASON_FORMAT,
    REASON_FOREIGN,
    REASON_LOOP,
    REASON_OUTSIDE,
    REASON_SYMLINK,
    build_plan,
    root_token,
    sync_lock,
    sync_marker,
)
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.indexing.indexer import build_index, load_index
from veles.indexing.vector_store import NumpyFlatStore
from veles.retrieval.retriever import hybrid_search

HASH_DIM = EMBED_DIM
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+")

SEED_DOC = """\
# Коучинг и мотивация

Коучинг работает с целями и мотивацией сотрудника.
"""
SEED_QUERY = "коучинг цели мотивация"
SEED_MARK = "работает с целями"

DOC_V1 = """\
# Инструкция

Регламент доступа формируется отделом безопасности ежеквартально.
"""
DOC_V2 = DOC_V1.replace("ежеквартально", "ежемесячно")
DOC_QUERY = "как часто формируется регламент доступа"
DOC_MARK_V1 = "ежеквартально"
DOC_MARK_V2 = "ежемесячно"

DOC_B = "Справка: командировочные расходы считаются по среднему за квартал.\n"
DOC_C = "Заметка: пароль от стенда хранится в сейфе, срок - 30 дней.\n"
DOC_D = "Памятка: новый сотрудник получает доступ на второй день.\n"

# ссылки-каталоги, созданные тестами (чистка после теста: junction на Windows
# нельзя удалять рекурсией - rmtree в 3.11 ушёл бы внутрь на цель)
_DIR_LINKS: list[object] = []


class HashingEmbedder:
    """Тот же стаб, что в test_cli_add/test_delete: sublinear tf по хеш-бакетам."""

    def __init__(self, dim: int = HASH_DIM):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tfs: dict[str, int] = {}
        for tok in _TOKEN_RE.findall(text.lower().replace("ё", "е")):
            tfs[tok] = tfs.get(tok, 0) + 1
        for tok, tf in tfs.items():
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dim
            vec[bucket] += np.float32(1.0 + np.log(tf))
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_passages(self, texts) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def make_dir_link(link, target) -> bool:
    """Ссылка на каталог: junction на Windows (не требует привилегии),
    symlink на POSIX. True - создана (регистрируется для чистки)."""
    _DIR_LINKS.append(link)
    if sys.platform == "win32":
        # вывод cmd - в OEM-кодировке: читаем бинарно, нужен только returncode
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return r.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except OSError:
        return False


def make_file_link(link, target) -> bool:
    """Симлинк на файл: на Windows требует привилегии - нет, False."""
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False


@pytest.fixture(autouse=True)
def _remove_dir_links():
    """Чистка ссылок-каталогов: rmdir (junction) / unlink (symlink), не рекурсия."""
    yield
    for link in _DIR_LINKS:
        if sys.platform == "win32":
            subprocess.run(["cmd", "/c", "rmdir", str(link)], capture_output=True)
        else:
            try:
                os.unlink(link)
            except OSError:
                pass
    _DIR_LINKS.clear()


@pytest.fixture()
def sync_env(tmp_path, monkeypatch):
    """tmp-окружение: VELES_DATA (канон/trash/лок), канон без индекса
    (INDEX_DIR указывает на несуществующий data/index - sync пишет только
    канон); корень источника - tmp «источник» (source строго read-only)."""
    data = tmp_path / "veles-data"
    canonical = data / "canonical"
    index_dir = data / "index"
    monkeypatch.setattr("veles.canonical.store.CANONICAL_DIR", canonical)
    monkeypatch.setattr("veles.cli.INDEX_DIR", index_dir)
    monkeypatch.setenv("VELES_DATA", str(data))
    root = tmp_path / "источник"
    root.mkdir()
    return {"data": data, "canonical": canonical, "index_dir": index_dir, "root": root}


@pytest.fixture()
def indexed_env(sync_env, tmp_path, monkeypatch):
    """sync_env + мини-индекс (настоящий build_index/load_index) с одним
    корпусным документом ВНЕ корня sync (derived_from onedev-vyshka: - проверка,
    что sync чужие документы не удаляет). Эмбеддер cmd_sync подменён стабом."""
    canonical, index_dir = sync_env["canonical"], sync_env["index_dir"]
    corpus = tmp_path / "corpus"
    seed = corpus / "05_vyshka_academic/lectures/коучинг.md"
    seed.parent.mkdir(parents=True)
    seed.write_text(SEED_DOC, encoding="utf-8", newline="\n")
    doc = TxtMdParserAdapter().parse_file(seed, corpus_root=corpus)
    save_document(doc, root=canonical)
    bundle = build_index(HashingEmbedder(), NumpyFlatStore(dim=HASH_DIM))
    index_dir.mkdir(parents=True)
    bundle.save(index_dir, embed_meta={"model": EMBED_MODEL, "backend": "numpy-flat"})
    monkeypatch.setattr("veles.cli.build_embedder", lambda: HashingEmbedder())
    return sync_env


def run_sync(monkeypatch, root, *flags) -> int:
    from veles.cli import main

    monkeypatch.setattr(sys, "argv", ["veles", "sync", "--root", str(root), *flags])
    return main()


def audit_events(env) -> list[dict]:
    path = env["canonical"] / "ingest-audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def search_texts(index_dir, q: str) -> list[str]:
    bundle = load_index(index_dir)
    qvec = HashingEmbedder().embed_query(q)
    return [e.text for e in hybrid_search(bundle, qvec, q, top_k=8)]


def manifest_of(env, doc_id: str) -> dict:
    return json.loads(
        (_doc_dir(doc_id, env["canonical"]) / "manifest.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------- первый sync


def test_initial_add_canon_only_without_index(sync_env, monkeypatch, capsys) -> None:
    """Без собранного индекса sync пишет только канон: оба файла в каноне с
    sha256 источника и маркером владения fsync:<корень>:<rel>, meta.json не
    создаётся, trash пуст."""
    root = sync_env["root"]
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (root / "b.txt").write_text(DOC_B, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "добавлено=2" in out
    assert "синхронизация завершена: добавлено=2" in out

    for doc_id, text in (("a.md", DOC_V1), ("b.txt", DOC_B)):
        manifest = manifest_of(sync_env, doc_id)
        assert manifest["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert manifest["derived_from"] == [sync_marker(root, doc_id)]
    # индекс не создаётся, trash пуст
    assert not sync_env["index_dir"].exists()
    assert not list(sync_env["data"].glob("trash*"))


def test_dry_run_changes_nothing(sync_env, monkeypatch, capsys) -> None:
    """--dry-run показывает дельту и не трогает ничего: канон/trash/индекс нет."""
    root = sync_env["root"]
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, root, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "добавлено=1" in out and "dry-run: ничего не изменено" in out
    assert not sync_env["canonical"].exists() or not any(sync_env["canonical"].iterdir())
    assert not sync_env["index_dir"].exists()
    assert not list(sync_env["data"].glob("trash*"))


def test_without_yes_only_plan(sync_env, monkeypatch, capsys) -> None:
    """Без --yes - только план и код 1 (как `veles delete`): канон не тронут."""
    root = sync_env["root"]
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, root) == 1
    out = capsys.readouterr().out
    assert "Подтвердите применение дельты: --yes" in out
    assert not sync_env["canonical"].exists() or not any(sync_env["canonical"].iterdir())


def test_root_missing(sync_env, monkeypatch, capsys) -> None:
    assert run_sync(monkeypatch, sync_env["root"] / "нет-такого", "--yes") == 1
    assert "не найден" in capsys.readouterr().out


# --------------------------------------------------------------------- дифф


def test_diff_added_updated_deleted_unchanged(sync_env, monkeypatch, capsys) -> None:
    """Полный дифф: после первого sync правим a.md, добавляем c.md, удаляем
    b.md, не трогаем d.md -> дельта обновлено=1/добавлено=1/удалено=1/
    без изменений=1; в каноне r2 с архивом r1, удалённый - в trash."""
    root, env = sync_env["root"], sync_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (root / "b.txt").write_text(DOC_B, encoding="utf-8", newline="\n")
    (root / "d.md").write_text(DOC_D, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()

    # второй заход: правка/добавление/удаление/неизменный
    (root / "a.md").write_text(DOC_V2, encoding="utf-8", newline="\n")
    (root / "c.md").write_text(DOC_C, encoding="utf-8", newline="\n")
    (root / "b.txt").unlink()

    assert run_sync(monkeypatch, root) == 1  # без --yes дельта не применяется
    assert "обновлено=1" in capsys.readouterr().out
    assert manifest_of(env, "a.md")["revision_id"] == "r1"  # ещё r1

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "обновлено=1, удалено=1" in out
    assert "без изменений=1" in out

    # a.md: ревизия r2, r1 в архиве; c.md добавлен; d.md без изменений;
    # b.txt удалён - в trash, в каноне его нет
    manifest = manifest_of(env, "a.md")
    assert manifest["revision_id"] == "r2" and manifest["previous_revision"] == "r1"
    assert manifest["sha256"] == hashlib.sha256(DOC_V2.encode("utf-8")).hexdigest()
    assert (_doc_dir("a.md", env["canonical"]) / "revision-r1" / "manifest.json").is_file()
    assert manifest_of(env, "c.md")["revision_id"] == "r1"
    assert not _doc_dir("b.txt", env["canonical"]).exists()
    trash_entries = list(trash_root().iterdir())
    assert len(trash_entries) == 1
    assert (trash_entries[0] / "manifest.json").is_file()
    deleted_manifest = json.loads((trash_entries[0] / "manifest.json").read_text(encoding="utf-8"))
    assert deleted_manifest["document_id"] == "b.txt"
    # аудит: событие delete
    del_events = [e for e in audit_events(env) if e.get("event") == "delete"]
    assert del_events and del_events[-1]["document_id"] == "b.txt"


def test_idempotent_second_sync(sync_env, monkeypatch, capsys) -> None:
    """Идемпотентность: второй sync без изменений - ноль дельты, ни одного
    нового события аудита, build_plan пуст."""
    root, env = sync_env["root"], sync_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (root / "b.txt").write_text(DOC_B, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()
    events_before = audit_events(env)

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "добавлено=0, обновлено=0, удалено=0" in out
    assert "ошибок=0" in out
    assert audit_events(env) == events_before
    assert not build_plan(root).has_delta


# ------------------------------------------------------------- симлинк-побеги


def test_symlink_file_outside_ignored(sync_env, monkeypatch, capsys) -> None:
    """Симлинк-файл наружу не ингестится: запись в отчёт (пропущено), в каноне
    только настоящий файл."""
    link = make_file_link(sync_env["root"] / "ссылка.md", sync_env["data"] / "тайное.md")
    if not link:
        pytest.skip("симлинки на файлах недоступны на этой машине (нет привилегии)")
    (sync_env["root"] / "настоящий.md").write_text(DOC_V1, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, sync_env["root"], "--yes") == 0
    out = capsys.readouterr().out
    assert f"пропущен ({REASON_SYMLINK}): ссылка.md" in out
    assert "пропущено=1" in out
    assert manifest_of(sync_env, "настоящий.md")
    assert not _doc_dir("ссылка.md", sync_env["canonical"]).exists()
    assert not _doc_dir("тайное.md", sync_env["canonical"]).exists()


def test_dir_escape_ignored(sync_env, monkeypatch, capsys) -> None:
    """Ссылка-каталог наружу (junction/symlink) не разворачивается: содержимое
    не ингестится, обход не уходит за корень и не циклится; соседний файл
    синхронизируется."""
    secret = sync_env["data"] / "секреты"
    secret.mkdir(parents=True)
    (secret / "пароли.md").write_text("root:toor\n", encoding="utf-8", newline="\n")
    link = sync_env["root"] / "снаружи"
    if not make_dir_link(link, secret):
        pytest.skip("ссылки на каталоги недоступны на этой машине")
    (sync_env["root"] / "настоящий.md").write_text(DOC_V1, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, sync_env["root"], "--yes") == 0
    out = capsys.readouterr().out
    reason = REASON_SYMLINK if link.is_symlink() else REASON_OUTSIDE
    assert f"пропущен ({reason}): снаружи" in out
    assert "пропущено=1" in out
    assert manifest_of(sync_env, "настоящий.md")
    assert not _doc_dir("снаружи/пароли.md", sync_env["canonical"]).exists()
    assert not _doc_dir("пароли.md", sync_env["canonical"]).exists()


def test_unsupported_extension_counted_as_skipped(sync_env, monkeypatch, capsys) -> None:
    """Файлы не-txt/md/pdf не ингестятся и попадают в «пропущено» по формату."""
    (sync_env["root"] / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (sync_env["root"] / "фото.png").write_bytes(b"\x89PNG-fake")

    assert run_sync(monkeypatch, sync_env["root"], "--yes") == 0
    out = capsys.readouterr().out
    assert "пропущены по формату: .png: 1" in out
    assert "пропущено=1" in out
    assert not _doc_dir("фото.png", sync_env["canonical"]).exists()


# ------------------------------------------------ живой индекс: ревизия/удаление


def test_revision_on_file_change(indexed_env, monkeypatch, capsys) -> None:
    """Изменение файла - ревизия r2 (K3), старая в архиве; живой индекс
    обновлён без пересборки: после «рестарта» (load_index) ищется новый текст,
    старый выведен (superseded)."""
    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()
    assert any(DOC_MARK_V1 in t for t in search_texts(env["index_dir"], DOC_QUERY))

    (root / "a.md").write_text(DOC_V2, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "обновлено=1" in out

    manifest = manifest_of(env, "a.md")
    assert manifest["revision_id"] == "r2" and manifest["previous_revision"] == "r1"
    assert (_doc_dir("a.md", env["canonical"]) / "revision-r1" / "nodes.jsonl").is_file()

    texts = search_texts(env["index_dir"], DOC_QUERY)
    assert any(DOC_MARK_V2 in t for t in texts), texts
    assert not any(DOC_MARK_V1 in t for t in texts), texts
    # корпусный сосед не задет
    assert any(SEED_MARK in t for t in search_texts(env["index_dir"], SEED_QUERY))


def test_tombstone_on_delete(indexed_env, monkeypatch, capsys) -> None:
    """Удаление файла - tombstone (K10): канон в trash, узлы выведены из поиска
    (сразу и после рестарта), аудит-событие delete; корпусный сосед жив."""
    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()
    assert any(DOC_MARK_V1 in t for t in search_texts(env["index_dir"], DOC_QUERY))

    (root / "a.md").unlink()
    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "удалено=1" in out and "узлов выведено из индекса" in out

    assert not _doc_dir("a.md", env["canonical"]).exists()
    trash_entries = list(trash_root().iterdir())
    assert len(trash_entries) == 1 and (trash_entries[0] / "nodes.jsonl").is_file()
    texts = search_texts(env["index_dir"], DOC_QUERY)
    assert not any(DOC_MARK_V1 in t for t in texts), texts
    del_events = [e for e in audit_events(env) if e.get("event") == "delete"]
    assert del_events and del_events[-1]["document_id"] == "a.md"
    assert any(SEED_MARK in t for t in search_texts(env["index_dir"], SEED_QUERY))


def test_unrelated_canon_docs_not_deleted(indexed_env, monkeypatch, capsys) -> None:
    """Безопасность удаления: corpus-документ (derived_from onedev-vyshka:, без
    маркера владения sync) не попадает в дельту удаления, хотя его пути нет в
    источнике; остаётся в каноне и в поиске."""
    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()

    plan = build_plan(root)
    assert "05_vyshka_academic/lectures/коучинг.md" not in plan.deleted
    assert plan.deleted == []
    assert manifest_of(env, "05_vyshka_academic/lectures/коучинг.md")
    assert any(SEED_MARK in t for t in search_texts(env["index_dir"], SEED_QUERY))


# --------------------------------------------------------------------- лок


def test_sync_lock_blocks_second_sync(sync_env, monkeypatch, capsys) -> None:
    """Файловый лок: пока один sync держит лок, второй честно отказывается
    (код 1), ничего не синхронизирует; занятый лок - отказ, не ожидание."""
    root = sync_env["root"]
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    with sync_lock():
        assert run_sync(monkeypatch, root, "--yes") == 1
        out = capsys.readouterr().out
        assert "уже выполняется" in out
        assert not sync_env["canonical"].exists() or not any(sync_env["canonical"].iterdir())
    # после освобождения лока sync проходит
    assert run_sync(monkeypatch, root, "--yes") == 0
    assert "добавлено=1" in capsys.readouterr().out


# ---------------------------------------- регрессии по ревью (раунд 1) ------


def test_foreign_document_never_updated_or_deleted(indexed_env, monkeypatch, capsys) -> None:
    """(ревью high) Файл источника с rel, совпадающим с document_id чужого
    (корпусного) документа: НЕ обновляется (иначе ingest_document архивировал
    бы корпусную ревизию и затирал derived_from, а последующее удаление файла
    унесло бы весь doc_dir чужих ревизий в trash) и НЕ удаляется после
    удаления файла. Ревизия r1, sha, derived_from и поиск нетронуты."""
    root, env = indexed_env["root"], indexed_env
    seed_id = "05_vyshka_academic/lectures/коучинг.md"
    (root / seed_id).parent.mkdir(parents=True)
    (root / seed_id).write_text(
        "Чужой контент клиента: коучинг тут ни при чём.\n",
        encoding="utf-8", newline="\n",
    )

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert f"пропущен ({REASON_FOREIGN}): {seed_id}" in out
    assert "пропущено=1" in out
    # манифест нетронут: r1, исходный sha и derived_from корпусного конвейера
    manifest = manifest_of(env, seed_id)
    assert manifest["revision_id"] == "r1"
    assert manifest["sha256"] == hashlib.sha256(SEED_DOC.encode("utf-8")).hexdigest()
    assert manifest["derived_from"] == [f"onedev-vyshka:{seed_id}"]
    assert any(SEED_MARK in t for t in search_texts(env["index_dir"], SEED_QUERY))

    # удаление файла источника тоже не трогает чужой документ
    (root / seed_id).unlink()
    assert run_sync(monkeypatch, root, "--yes") == 0
    assert manifest_of(env, seed_id)["revision_id"] == "r1"
    assert not trash_root().exists() or not list(trash_root().iterdir())


def test_doc_dir_collision_skipped(sync_env, monkeypatch, capsys) -> None:
    """(ревью) _doc_dir схлопывает 'a/b.md' и 'a_b.md' в один каталог канона:
    второй файл - skipped, а не молчаливая перезапись; повторные sync стабильны
    (без ревизий r2->r3->r4 с флипом document_id)."""
    root, env = sync_env["root"], sync_env
    (root / "a").mkdir()
    (root / "a" / "b.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (root / "a_b.md").write_text(DOC_B, encoding="utf-8", newline="\n")

    for _ in range(3):  # раньше: три sync давали r2->r3->r4
        assert run_sync(monkeypatch, root, "--yes") == 0
        out = capsys.readouterr().out
        assert f"пропущен ({REASON_COLLISION}" in out
    dirs = [p for p in env["canonical"].iterdir() if p.is_dir()]
    assert len(dirs) == 1
    manifest = manifest_of(env, "a/b.md")
    assert manifest["revision_id"] == "r1"  # осцилляции ревизий нет
    assert manifest["document_id"] == "a/b.md"
    assert manifest["sha256"] == hashlib.sha256(DOC_V1.encode("utf-8")).hexdigest()


@pytest.mark.skipif(sys.platform != "win32", reason="регистронезависимость - специфика NTFS")
def test_doc_dir_collision_by_case_win32(sync_env, monkeypatch, capsys) -> None:
    """(ревью) На NTFS каталоги канона 'a_b.md' (от a/b.md) и 'A_B.md'
    совпадают по регистру: второй файл пропускается с причиной коллизии.
    (Два файла В ОДНОМ каталоге, различающиеся только регистром, на NTFS
    невозможны - это один файл, поэтому различие создаём разделителем.)"""
    root = sync_env["root"]
    (root / "a").mkdir()
    (root / "a" / "b.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    (root / "A_B.md").write_text(DOC_B, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert "добавлено=1" in out
    assert f"пропущен ({REASON_COLLISION}" in out
    dirs = [p for p in sync_env["canonical"].iterdir() if p.is_dir()]
    assert len(dirs) == 1


def test_junction_loop_cut(sync_env, monkeypatch, capsys) -> None:
    """(ревью) Ссылка-каталог в УЖЕ ПРОЙДЕННЫЙ каталог (junction на сам корень)
    вырезается: обход не зацикливается и не плодит дубли одного файла с разных
    глубин."""
    root, env = sync_env["root"], sync_env
    (root / "настоящий.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    if not make_dir_link(root / "петля", root):
        pytest.skip("ссылки на каталоги недоступны на этой машине")

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert f"пропущен ({REASON_LOOP}): петля" in out
    assert "добавлено=1" in out
    dirs = [p for p in env["canonical"].iterdir() if p.is_dir()]
    assert len(dirs) == 1  # один документ, без дублей "из глубины петли"
    assert manifest_of(env, "настоящий.md")["document_id"] == "настоящий.md"


def test_junction_multi_loop_no_combinatorial_blowup(sync_env, monkeypatch, capsys) -> None:
    """(ревью, случай «не запускался») Несколько ссылок-петель на разной
    глубине: обход ограничен единственным визитом каждого резолва каталога -
    комбинаторного перебора путей до глубины 63 нет, каждый файл в каноне
    ровно один раз."""
    root, env = sync_env["root"], sync_env
    (root / "верхний.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    nested = root / "вложенный"
    nested.mkdir()
    (nested / "нижний.md").write_text(DOC_B, encoding="utf-8", newline="\n")
    created = (
        make_dir_link(root / "петля_верх", root)
        and make_dir_link(nested / "петля_глуб", root)
    )
    if not created:
        pytest.skip("ссылки на каталоги недоступны на этой машине")

    started = time.monotonic()
    assert run_sync(monkeypatch, root, "--yes") == 0
    assert time.monotonic() - started < 30  # перебор длился бы минуты
    out = capsys.readouterr().out
    assert f"пропущен ({REASON_LOOP}): петля_верх" in out
    assert f"пропущен ({REASON_LOOP}): вложенный/петля_глуб" in out
    assert "добавлено=2" in out
    dirs = [p for p in env["canonical"].iterdir() if p.is_dir()]
    assert len(dirs) == 2


def test_crash_between_canon_and_overlay_self_heals(indexed_env, monkeypatch, capsys) -> None:
    """(ревью) Падение между ingest_document и append_overlay: sync честно
    докладывает ошибку, канон r2 при устаревшем индексе; повторный sync (даже
    без дельты) доводит индекс по журналу намерений - и идемпотентен."""
    from veles.indexing import overlay as overlay_mod
    from veles.ingest import sync as sync_mod

    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()

    original = overlay_mod.append_overlay
    state = {"fail": True}

    def flaky(index_dir, new_nodes, patches, ids, matrix, bm25_pairs, timeout_s=120.0):
        if state["fail"]:
            raise RuntimeError("симуляция падения между каноном и overlay")
        return original(index_dir, new_nodes, patches, ids, matrix, bm25_pairs,
                        timeout_s=timeout_s)

    monkeypatch.setattr(overlay_mod, "append_overlay", flaky)

    (root / "a.md").write_text(DOC_V2, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 1  # ошибка доложена
    assert "индекс не обновлён" in capsys.readouterr().out
    assert manifest_of(env, "a.md")["revision_id"] == "r2"
    texts = search_texts(env["index_dir"], DOC_QUERY)
    assert any(DOC_MARK_V1 in t for t in texts)  # индекс устарел: старое живо
    assert not any(DOC_MARK_V2 in t for t in texts)
    assert ("a.md", "upsert") in sync_mod._pending_repairs(root)

    state["fail"] = False
    assert run_sync(monkeypatch, root, "--yes") == 0  # дельта 0 - ремонт догоняет
    out = capsys.readouterr().out
    assert "добавлено=0, обновлено=0, удалено=0" in out
    assert "доведено до индекса=1" in out
    texts = search_texts(env["index_dir"], DOC_QUERY)
    assert any(DOC_MARK_V2 in t for t in texts), texts
    assert not any(DOC_MARK_V1 in t for t in texts), texts
    assert sync_mod._pending_repairs(root) == {}


def test_repair_skips_readd_when_overlay_already_applied(indexed_env, monkeypatch, capsys) -> None:
    """(ревью Ф4, ветка идемпотентности ремонта) Процесс умер ПОСЛЕ
    append_overlay, но ДО закрытия намерения: повторный sync НЕ дублирует
    векторы - закрывает намерение как «узлы уже в индексе», документ ищется,
    каждый node_version_id в векторном хранилище ровно один раз."""
    from veles.ingest import sync as sync_mod

    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()

    # симуляция kill между overlay и done-меткой журнала
    sync_mod._append_repair("a.md", "upsert", False, root)
    assert ("a.md", "upsert") in sync_mod._pending_repairs(root)

    bundle_before = load_index(env["index_dir"])
    ids_before = {
        nid for nid, m in bundle_before.node_meta.items()
        if m["document_id"] == "a.md"
    }
    assert ids_before

    assert run_sync(monkeypatch, root, "--yes") == 0  # дельта 0 - ремонт
    out = capsys.readouterr().out
    assert "добавлено=0, обновлено=0, удалено=0" in out
    assert "доведено до индекса=0" in out  # векторы не переписывались
    assert "ошибок=0" in out
    assert sync_mod._pending_repairs(root) == {}  # намерение закрыто без пере-добавления

    bundle_after = load_index(env["index_dir"])
    ids_after = {
        nid for nid, m in bundle_after.node_meta.items()
        if m["document_id"] == "a.md"
    }
    assert ids_after == ids_before
    # дубликатов в векторном хранилище нет: каждый id ровно один раз
    stored = list(bundle_after.vector.ids)
    for nid in ids_before:
        assert stored.count(nid) == 1, nid
    assert any(DOC_MARK_V1 in t for t in search_texts(env["index_dir"], DOC_QUERY))


def test_crash_between_delete_and_tombstone_self_heals(indexed_env, monkeypatch, capsys) -> None:
    """(ревью) Падение между delete_document и tombstone_document: канон в
    trash, но документ ещё ищется; следующий sync доводит tombstone по журналу."""
    from veles.indexing import overlay as overlay_mod
    from veles.ingest import sync as sync_mod

    root, env = indexed_env["root"], indexed_env
    (root / "a.md").write_text(DOC_V1, encoding="utf-8", newline="\n")
    assert run_sync(monkeypatch, root, "--yes") == 0
    capsys.readouterr()

    original = overlay_mod.tombstone_document
    state = {"fail": True}

    def flaky(index_dir, document_id, bundle=None):
        if state["fail"]:
            raise RuntimeError("симуляция падения между delete и tombstone")
        return original(index_dir, document_id, bundle)

    monkeypatch.setattr(overlay_mod, "tombstone_document", flaky)

    (root / "a.md").unlink()
    assert run_sync(monkeypatch, root, "--yes") == 1
    assert "tombstone индекса не записан" in capsys.readouterr().out
    assert not _doc_dir("a.md", env["canonical"]).exists()  # канон в trash
    assert any(DOC_MARK_V1 in t for t in search_texts(env["index_dir"], DOC_QUERY))
    assert ("a.md", "tombstone") in sync_mod._pending_repairs(root)

    state["fail"] = False
    assert run_sync(monkeypatch, root, "--yes") == 0
    assert "доведено до индекса=1" in capsys.readouterr().out
    texts = search_texts(env["index_dir"], DOC_QUERY)
    assert not any(DOC_MARK_V1 in t for t in texts), texts
    assert sync_mod._pending_repairs(root) == {}


def test_overlay_cross_process_lock(indexed_env) -> None:
    """(ревью) append_overlay сериализован МЕЖДУ процессами (overlay.lock):
    чужой владелец лока - TimeoutError, а не молчаливая потеря дельты;
    освобождение - дельта пишется и применяется при рестарте."""
    from veles.indexing.overlay import OVERLAY_LOCK_NAME, append_overlay
    from veles.locks import file_lock

    index_dir = indexed_env["index_dir"]
    with file_lock(index_dir / OVERLAY_LOCK_NAME):
        with pytest.raises(TimeoutError):
            append_overlay(
                index_dir, {}, {}, [],
                np.zeros((0, 1), dtype=np.float32), [], timeout_s=0.2,
            )
    patch = {next(iter(load_index(index_dir).node_meta)): {"status": "superseded"}}
    append_overlay(index_dir, {}, patch, [], np.zeros((0, 1), dtype=np.float32), [])
    fresh = load_index(index_dir)  # patch-only оверлей применился к живому узлу
    nid = next(iter(patch))
    assert fresh.node_meta[nid]["status"] == "superseded"


def test_file_lock_cross_process(tmp_path) -> None:
    """veles/locks.file_lock: дочерний ПРОЦЕСС держит лок - родитель получает
    TimeoutError; после завершения ребёнка лок свободен."""
    from veles.locks import file_lock

    path = tmp_path / "cross.lock"
    child_code = (
        "import time\n"
        "from veles.locks import file_lock\n"
        f"with file_lock({str(path)!r}):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(3)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code], stdout=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    try:
        assert proc.stdout is not None and proc.stdout.readline().strip() == b"held"
        with pytest.raises(TimeoutError):
            with file_lock(path, timeout_s=0.5):
                pass
    finally:
        proc.wait(timeout=15)
    with file_lock(path, timeout_s=5):
        pass  # ребёнок завершился - лок освободился


def test_non_utf8_skipped(sync_env, monkeypatch, capsys) -> None:
    """(ревью) Файл в CP1251 не ингестится молча как mojibake: skipped с
    причиной в отчёте, документа в каноне нет; соседний UTF-8 файл обычный."""
    root = sync_env["root"]
    (root / "cp1251.txt").write_bytes(
        "Проверка кодировки: привет, мир.\n".encode("cp1251")
    )
    (root / "ok.md").write_text(DOC_V1, encoding="utf-8", newline="\n")

    assert run_sync(monkeypatch, root, "--yes") == 0
    out = capsys.readouterr().out
    assert f"пропущен ({REASON_ENCODING}): cp1251.txt" in out
    assert "пропущено=1" in out
    assert not _doc_dir("cp1251.txt", sync_env["canonical"]).exists()
    assert manifest_of(sync_env, "ok.md")["document_id"] == "ok.md"
