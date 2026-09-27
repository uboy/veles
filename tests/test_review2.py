"""Тесты ревью-2 фиксов и парсеров (без моделей)."""

from __future__ import annotations

import hashlib
import json

import pytest

from veles.canonical.model import CanonicalDocument, Provenance
from veles.canonical.store import load_document, save_document
from veles.ingest.txt_md import TxtMdParserAdapter
from veles.policy.identifiers import extract_identifiers, identifiers_missed


def test_sha256_is_over_raw_bytes_for_binary(tmp_path):
    """Ревью-2 P1: для бинарных/не-UTF8 данных хеш = хеш байтов, не декодированного
    текста (старый код терял байты на errors='replace')."""
    raw = bytes(range(256))  # любой бинарный мусор, включая невалидный UTF-8
    p = tmp_path / "bin.txt"
    p.write_bytes(raw)
    doc = TxtMdParserAdapter().parse_file(p, corpus_root=tmp_path)
    assert doc.sha256 == hashlib.sha256(raw).hexdigest()


def test_store_atomic_save_order(tmp_path):
    """Ревью-2 P2: nodes.jsonl пишется до manifest; после save оба на месте и
    согласованы (маркер готовности - manifest)."""
    doc = CanonicalDocument(
        document_id="x/doc.txt",
        title="X",
        parser_profile="txt-md:0.1",
        sha256="s",
    )
    doc.add_node(
        logical_node_id="x/doc.txt#p0001",
        text="тело",
        provenance=Provenance(adapter="txt-md:0.1", source_id="x/doc.txt", lines=[1, 1]),
    )
    d = save_document(doc, root=tmp_path)
    assert (d / "nodes.jsonl").exists() and (d / "manifest.json").exists()
    assert not (d / "nodes.jsonl.tmp").exists() and not (d / "manifest.tmp").exists()
    loaded = load_document(doc.document_id, root=tmp_path)
    assert loaded.nodes[0].text == "тело"


def test_identifiers_word_boundaries():
    """Ревью-2 P1: 'ЛАРД 12' не 'РД 12', 'ВСП 20.13330' не 'СП 20.13330'."""
    assert extract_identifiers("согласно ЛАРД 12 и ВСП 20.13330") == []
    assert "ГОСТ 17.2.3.01-86" in extract_identifiers("по ГОСТ 17.2.3.01-86")
    assert "МДС 81-28.2001" in extract_identifiers("МДС 81-28.2001")


def test_identifiers_gost_r_equivalence():
    """'ГОСТ Р X' и 'ГОСТ X' - одно и то же ядро; any-found семантика."""
    known = {"ГОСТ 17.2.3.01-86", "СП 51.13330"}
    # 'ГОСТ Р 17.2.3.01-86' в вопросе должен находить 'ГОСТ 17.2.3.01-86' в корпусе
    assert identifiers_missed(["ГОСТ Р 17.2.3.01-86"], known) is False
    # неизвестный номер - честный miss
    assert identifiers_missed(["ГОСТ 99.9.9-99"], known) is True
    # нет номеров в вопросе - miss нет
    assert identifiers_missed([], known) is False
    # один известный из двух - достаточно (any-found)
    assert identifiers_missed(["ГОСТ 17.2.3.01-86", "ГОСТ 99.9.9-99"], known) is False


def test_lcs_gold_small_sets():
    """LCS-gold: сет крошечный и находит юнит с дословным куском; ответ-парафраз
    без дословного куска >= min_chars не локализуется (в отличие от bag-метода)."""
    from bench.rrncb import Unit, lcs_gold, norm_text, rare_tokens

    units = [
        Unit("u1", "Высота ограды игровых площадок должна быть в пределах 1 - 1,2 метра."),
        Unit("u2", "СНиП 21-01-97 устанавливает требования пожарной безопасности зданий."),
        Unit("u3", "Норма управляемости руководителя обычно составляет от 5 до 9 пунктов."),
    ]
    norm_cache = [norm_text(u.text) for u in units]
    tok_index: dict[str, set[int]] = {}
    for i, u in enumerate(units):
        for w in rare_tokens(u.text):
            tok_index.setdefault(w, set()).add(i)

    verbatim = "Согласно МДС 35-7.2000 высота ограды игровых площадок должна быть в пределах 1 - 1,2 метра."
    gs = lcs_gold(units, verbatim, norm_cache, tok_index, min_chars=40)
    assert gs == {"u1"}
    # абстрактный пересказ без дословного куска - не локализуется (это фича, не баг)
    paraphrase = "Ограды игровых зон делают невысокими, примерно метр, из металла или дерева"
    assert lcs_gold(units, paraphrase, norm_cache, tok_index, min_chars=40) is None


def test_ingest_fail_closed_without_manifests(tmp_path, monkeypatch, capsys):
    """Ревью-2 P2: пустые манифесты при включённой сверке = ошибка, не молчаливый ingest."""
    import argparse

    from veles import cli

    monkeypatch.setattr(cli, "CORPUS_ROOT", tmp_path)
    monkeypatch.setattr(cli, "_manifest_known_relpaths", lambda: set())
    with pytest.raises(SystemExit):
        cli.cmd_ingest(argparse.Namespace(no_manifest_check=False))
