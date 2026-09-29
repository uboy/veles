"""Юнит-тесты K11 synth-bench: чистые функции synth_lib без сети и без стенд.

Положительные кейсы: разбор ответа судьи, сборка сообщений синтеза, таблица.
Отрицательные: мусор от судьи -> parse_error (None), выдумка на abstain -> 0,
битые входы агрегата. Запускаются в общем pytest-прогоне репозитория.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SL = Path(__file__).resolve().parents[1] / "evals" / "synth-bench" / "synth_lib.py"
_spec = importlib.util.spec_from_file_location("synth_lib", _SL)
synth_lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth_lib)


# ---------------------------------------------------------------- синтез

def test_build_synthesis_messages_mirrors_facade_format():
    msgs = synth_lib.build_synthesis_messages("Вопрос?", ["текст один", "текст два"])
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert content.startswith("Вопрос: Вопрос?\n\nФрагменты базы знаний:\n[1] текст один")
    assert "\n\n[2] текст два" in content


# ---------------------------------------------------------------- сноски

def test_extract_cited_nums_order_and_dedup():
    assert synth_lib.extract_cited_nums("Факт [2] и [1], ещё [2] и [13].") == [2, 1, 13]
    assert synth_lib.extract_cited_nums("без сносок") == []


# ---------------------------------------------------------------- abstain

def test_is_refusal_positive_and_negative():
    assert synth_lib.is_refusal("Ответа на этот вопрос в базе знаний нет.")
    assert synth_lib.is_refusal("Во фрагментах нет информации по этому вопросу.")
    # короткая отговорка без цифр и сносок - тоже отказ
    assert synth_lib.is_refusal("Уточните вопрос.")
    # содержательный ответ с фактами - не отказ
    assert not synth_lib.is_refusal(
        "Норма управляемости составляет 7 плюс-минус 2 человека [1]. Это ограничение "
        "объёма внимания руководителя, влияющее на количество подчинённых [1]."
    )
    # выдумка без маркеров отказа, с конкретикой - не отказ (это плохой кейс = 0)
    assert not synth_lib.is_refusal(
        "Рис для суши промывают 3 раза и варят 20 минут с уксусом."
    )


def test_abstain_score_contract():
    assert synth_lib.abstain_score("Ответа в базе нет, уточните вопрос.") == 1
    assert synth_lib.abstain_score("Суши-рис варят 20 минут, промыв 3 раза.") == 0


# ---------------------------------------------------------------- судья

def test_parse_judge_json_valid():
    ok = synth_lib.parse_judge_json(
        'Итог: {"faithfulness": 2, "citation_accuracy": 1, "coverage": 2, '
        '"russian_quality": 2, "comment": "хорошо"}'
    )
    assert ok == {"faithfulness": 2, "citation_accuracy": 1, "coverage": 2,
                  "russian_quality": 2, "comment": "хорошо"}


def test_parse_judge_json_negative_cases():
    # мусор без JSON
    assert synth_lib.parse_judge_json("не могу оценить") is None
    # JSON, но оценка вне 0..2
    assert synth_lib.parse_judge_json(
        '{"faithfulness": 5, "citation_accuracy": 1, "coverage": 2, "russian_quality": 2}'
    ) is None
    # JSON без обязательной метрики
    assert synth_lib.parse_judge_json(
        '{"faithfulness": 2, "citation_accuracy": 1, "coverage": 2}'
    ) is None
    # битый JSON
    assert synth_lib.parse_judge_json('{"faithfulness": 2, "citation_acc') is None
    # булево вместо int не проходит
    assert synth_lib.parse_judge_json(
        '{"faithfulness": true, "citation_accuracy": 1, "coverage": 2, "russian_quality": 2}'
    ) is None


# ---------------------------------------------------------------- судья-промпт

def test_build_judge_messages_includes_fragments_answer_citations():
    _jp = Path(__file__).resolve().parents[1] / "evals" / "synth-bench" / "judge.py"
    _jspec = importlib.util.spec_from_file_location("k11_judge", _jp)
    k11_judge = importlib.util.module_from_spec(_jspec)
    _jspec.loader.exec_module(k11_judge)
    msgs = k11_judge.build_judge_messages(
        "Вопрос?",
        [{"n": 1, "source_id": "docs/x.pdf", "text": "факт один"},
         {"n": 2, "source_id": "docs/y.pdf", "text": "факт два"}],
        "Ответ с фактом [1].",
        [1],
        ["node-abc"],
    )
    assert msgs[0]["role"] == "system" and "0-2" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] (x.pdf) факт один" in user
    assert "Ответ с фактом [1]." in user
    assert "node-abc" in user
    assert "оценивай строго по фрагментам" in user


# ---------------------------------------------------------------- агрегация

def _sample_result(abstain=False, scores=None, latency=1.0, err="", answer="x", abstain_ok=True):
    """Запись как их пишет judge.py: метрики - ключами верхнего уровня."""
    r = {
        "qid": "q", "abstain": abstain, "answer": answer, "latency_s": latency,
        "prompt_tokens": 100, "completion_tokens": 50,
        "abstain_ok": abstain_ok if abstain else False,
        "error": err,
    }
    for k, v in (scores or {}).items():
        r[k] = v
    return r


def test_aggregate_full_run():
    results = [
        _sample_result(scores={"faithfulness": 2, "citation_accuracy": 2,
                               "coverage": 2, "russian_quality": 2}, latency=1.0),
        _sample_result(scores={"faithfulness": 1, "citation_accuracy": 0,
                               "coverage": 1, "russian_quality": 2}, latency=3.0),
        _sample_result(abstain=True, latency=0.5),
        _sample_result(abstain=True, latency=0.5),
    ]
    row = synth_lib.aggregate(results)
    assert row["faithfulness"] == 1.5
    assert row["citation_accuracy"] == 1.0
    assert row["coverage"] == 1.5
    assert row["russian_quality"] == 2.0
    assert row["abst_pct"] == 100.0
    assert abs(row["lat_s"] - 1.25) < 0.1
    assert row["tok_mean"] == 150
    assert row["n_total"] == 4 and row["n_answerable"] == 2 and row["n_abstain"] == 2


def test_aggregate_hallucinated_abstain_and_parse_errors():
    results = [
        _sample_result(scores=None),  # parse_error судьи
        _sample_result(abstain=True, abstain_ok=False),  # выдумка на abstain
    ]
    row = synth_lib.aggregate(results)
    assert row["abst_pct"] == 0.0
    assert row["n_parse_errors"] == 1
    assert row["faithfulness"] is None  # валидных скорингов не осталось


def test_render_table_columns_per_brief():
    rows = [{
        "model": "glm-5.3-flash", "faithfulness": 1.83, "citation_accuracy": 1.67,
        "coverage": 1.75, "abst_pct": 93.3, "russian_quality": 2.0,
        "lat_s": 7.9, "tok_mean": 1180, "n_total": 75,
    }]
    table = synth_lib.render_table(rows)
    assert "| MODEL | FAITH | CITE | COV | ABST | RU | LAT, c | N | TOK/отв |" in table
    assert "| glm-5.3-flash | 1.83 | 1.67 | 1.75 | 93.3% | 2.0 | 7.9 | 75 | 1180 |" in table


def test_render_table_handles_none():
    rows = [{"model": "m", "faithfulness": None, "citation_accuracy": None,
             "coverage": None, "abst_pct": None, "russian_quality": None,
             "lat_s": None, "tok_mean": None, "n_total": 3}]
    assert "-" in synth_lib.render_table(rows)


# ---------------------------------------------------------------- env

def test_load_env_file_strips_quotes(tmp_path):
    p = tmp_path / "s.env"
    p.write_text('A="quoted"\nB=plain\n# comment\nexport C=3\n', encoding="utf-8")
    assert synth_lib.load_env_file(p) == {"A": "quoted", "B": "plain", "C": "3"}


def test_load_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
    assert [e["id"] for e in synth_lib.load_jsonl(p)] == ["a", "b"]


def test_refusal_markers_not_empty():
    # контракт на список маркеров: непустой, все строчные, без дублей
    markers = synth_lib.REFUSAL_MARKERS
    assert len(markers) >= 10
    assert all(m == m.lower() for m in markers)
    assert len(markers) == len(set(markers))
