"""Видео-конвейер (poisk-po-video): таймкоды в схеме/цитатах/рендерах,
адаптер видео с фейковым VoxFusion (без сети), маршрут очереди.

ffmpeg-тесты помечены skipif: у GitHub-раннеров и машин разработчиков ffmpeg
есть, у минимальных окружений - нет; логика без ffmpeg (слияние сегментов,
формат таймкода, схема) проверяется всегда.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from veles.canonical.model import Provenance
from veles.indexing.indexer import IndexBundle
from veles.ingest import queue as ingest_queue
from veles.ingest import voxfusion
from veles.ingest.video import VIDEO_EXTENSIONS, VideoParserAdapter, merge_segments
from veles.retrieval.retriever import Citation, format_timecode, hybrid_search

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg недоступен")


# ------------------------------------------------------------------- схема/цитаты


def test_provenance_timecodes_roundtrip() -> None:
    p = Provenance(adapter="video:0.1", source_id="встреча.mp4", t_start=1.0, t_end=75.5)
    assert p.t_start == 1.0 and p.t_end == 75.5
    assert p.model_dump()["t_start"] == 1.0
    # не-видео узел: поля None, ничего не ломают
    old = Provenance(adapter="txt-md:0.1", source_id="a.md", lines=[1, 2])
    assert old.t_start is None and old.t_end is None


def test_format_timecode() -> None:
    assert format_timecode(0) == "00:00:00"
    assert format_timecode(754.2) == "00:12:34"
    assert format_timecode(3661) == "01:01:01"


def test_citation_human_timecode() -> None:
    c = Citation(
        title="Встреча", document_id="в.mp4", heading_path=["Встреча"],
        source_id="в.mp4", lines=None, node_version_id="n1",
        t_start=10.0, t_end=80.0,
    )
    assert "00:00:10-00:01:20" in c.human()
    plain = Citation(
        title="Док", document_id="d.md", heading_path=[],
        source_id="d.md", lines=[3, 4], node_version_id="n2",
    )
    assert "00:" not in plain.human() and "строки 3-4" in plain.human()


def test_query_citations_carry_timecodes() -> None:
    """Цитата /v1/knowledge/query несёт t_start/t_end из provenance узла."""
    meta = {
        "n1": {
            "text": "Обсудили бюджет проекта.",
            "access_groups": ["personal"],
            "title": "Встреча правления",
            "document_id": "встреча.mp4",
            "heading_path": ["Встреча правления"],
            "provenance": {
                "adapter": "video:0.1", "source_id": "встреча.mp4",
                "t_start": 10.0, "t_end": 25.0,
            },
        },
        "n2": {
            "text": "Отвлечённая тема.",
            "access_groups": ["personal"],
            "title": "Другой документ",
            "document_id": "d.md",
            "heading_path": [],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "d.md", "lines": [1, 2]},
        },
    }
    bundle = IndexBundle(
        vector=_FixedVector([("n1", 0.9), ("n2", 0.1)]),
        bm25=_FixedBm25([("n1", 1.0), ("n2", 0.1)]),
        node_meta=meta,
    )
    ev = hybrid_search(bundle, np.zeros(4, dtype=np.float32), "бюджет проекта", top_k=2)
    assert ev[0].citations[0].t_start == 10.0 and ev[0].citations[0].t_end == 25.0
    assert ev[1].citations[0].t_start is None


# ------------------------------------------------------------- слияние сегментов


def test_merge_segments_greedy_and_tail() -> None:
    short = [
        {"start": 0.0, "end": 10.0, "text": "слово " * 40},
        {"start": 10.0, "end": 20.0, "text": "слово " * 40},
        {"start": 20.0, "end": 30.0, "text": "хвост"},
    ]
    merged = merge_segments(short)
    assert len(merged) == 1, "короткие сегменты сливаются до максимума узла"
    assert merged[0][0] == 0.0 and merged[0][1] == 30.0
    assert "хвост" in merged[0][2]

    big = [
        {"start": 0.0, "end": 5.0, "text": "а" * 1300},
        {"start": 5.0, "end": 8.0, "text": "б" * 1300},
    ]
    assert len(merge_segments(big)) == 2, "узел не раздувается сверх максимума"


# --------------------------------------------------------------- адаптер видео


class _Resp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class FakeVoxClient:
    """Клиент без сети: фиксирует submit, отдаёт готовые сегменты."""

    def __init__(self, segments: list[dict]) -> None:
        self.segments = segments
        self.submitted_data: dict | None = None

    def post(self, url, files=None, data=None, headers=None):
        self.submitted_data = data or {}
        return _Resp({"job_id": "job-1"})

    def get(self, url, headers=None):
        return _Resp({
            "status": "done",
            "segments": self.segments,
            "text": " ".join(s["text"] for s in self.segments),
        })


def _make_mp4(tmp_path: Path, seconds: float = 2.0) -> Path:
    out = tmp_path / "встреча.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc2=duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-shortest", "-pix_fmt", "yuv420p", str(out)],
        check=True, capture_output=True,
    )
    return out


@needs_ffmpeg
def test_video_adapter_nodes_timecodes_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXFUSION_API_TOKEN", "test-token")
    monkeypatch.setenv("VELES_VIDEO_FRAME_INTERVAL_S", "1")
    video = _make_mp4(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / video.name).write_bytes(video.read_bytes())

    segs = [
        {"start": 0.2, "end": 1.0, "text": "первая реплика встречи"},
        {"start": 1.0, "end": 1.8, "text": "вторая реплика"},
    ]
    doc = VideoParserAdapter().parse_file(
        corpus / video.name, corpus_root=corpus, vx_client=FakeVoxClient(segs),
        poll_timeout_s=30.0,
    )
    assert doc.document_type == "video"
    assert doc.parser_profile == "video:0.1"
    assert doc.derived_from == [f"video-audio:{video.name}"]
    seg_nodes = [n for n in doc.nodes if "#seg" in n.logical_node_id]
    assert len(seg_nodes) == 1, "короткие реплики сливаются в один узел"
    prov = seg_nodes[0].provenance
    assert prov.t_start == pytest.approx(0.2)
    assert prov.t_end == pytest.approx(1.8)
    assert "первая реплика" in seg_nodes[0].text and "вторая реплика" in seg_nodes[0].text
    # активы при парсе живут во временнике (конвенция docling); материализует их
    # save_document - проверяем именно продовый путь сохранения
    from veles.canonical.store import save_document

    persist = tmp_path / "persist"
    save_document(doc, persist, by="test")
    manifest = json.loads(
        (persist / doc.document_id / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.get("assets"), "ключевые кадры попадают в активы документа"
    for rel in manifest["assets"].values():
        assert (persist / doc.document_id / rel).exists()


@needs_ffmpeg
def test_video_adapter_silent_video_placeholder(tmp_path, monkeypatch) -> None:
    """Видео без распознанной речи - документ с узлом-заглушкой, не падение."""
    monkeypatch.setenv("VOXFUSION_API_TOKEN", "test-token")
    monkeypatch.setenv("VELES_VIDEO_FRAME_INTERVAL_S", "1")
    video = _make_mp4(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / video.name).write_bytes(video.read_bytes())
    doc = VideoParserAdapter().parse_file(
        corpus / video.name, corpus_root=corpus, vx_client=FakeVoxClient([]),
        poll_timeout_s=30.0,
    )
    assert any("#silent" in n.logical_node_id for n in doc.nodes)
    assert doc.assets, "кадры сохраняются и у видео без речи"

def test_video_adapter_without_token_fails_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VOXFUSION_API_TOKEN", raising=False)
    fake = tmp_path / "в.mp4"
    fake.write_bytes(b"not really a video")
    with pytest.raises(RuntimeError, match="VOXFUSION_API_TOKEN"):
        VideoParserAdapter().parse_file(fake, corpus_root=tmp_path, vx_client=FakeVoxClient([]))


def test_voxfusion_submit_requests_segments(tmp_path) -> None:
    """В VoxFusion уходит include_segments=true - иначе таймкодов не будет."""
    client = FakeVoxClient([])
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF-fake")
    voxfusion.submit(client, audio, language="ru")
    assert client.submitted_data.get("include_segments") == "true"
    assert client.submitted_data.get("language") == "ru"


# --------------------------------------------------------------------- очередь


def test_queue_allows_video_extensions() -> None:
    assert {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"} <= ingest_queue.ALLOWED_EXTENSIONS
    assert VIDEO_EXTENSIONS == {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}


@needs_ffmpeg
def test_queue_routes_video_to_video_adapter(tmp_path, monkeypatch) -> None:
    video = _make_mp4(tmp_path, seconds=0.5)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / video.name).write_bytes(video.read_bytes())
    monkeypatch.setenv("VOXFUSION_API_TOKEN", "test-token")
    monkeypatch.setenv("VELES_VIDEO_FRAME_INTERVAL_S", "1")
    # очередь создаёт адаптер сама (клиент не инъекцируется) - глушим сеть
    monkeypatch.setattr(
        voxfusion, "submit", lambda client, audio_path, language="ru": "job-x"
    )
    monkeypatch.setattr(
        voxfusion, "poll_segments",
        lambda client, job_id, timeout_s=60.0: (
            "done", [{"start": 0.05, "end": 0.4, "text": "реплика из очереди"}], "реплика",
        ),
    )

    called = {}
    real = VideoParserAdapter.parse_file

    def spy(self, path, corpus_root, **kwargs):
        called["hit"] = True
        return real(self, path, corpus_root, **kwargs)

    monkeypatch.setattr(VideoParserAdapter, "parse_file", spy)
    doc = ingest_queue.parse_document_file(corpus / video.name, corpus, ".mp4")
    assert called.get("hit") and doc.document_type == "video"


# --------------------------------------------------------- рендер chat-фасада


def _fixed_pipeline(meta: dict):
    from veles.api.app import VelesPipeline

    class _Zero:
        dim = 4

        def embed_query(self, text):
            return np.zeros(4, dtype=np.float32)

    bundle = IndexBundle(
        vector=_FixedVector([("n1", 0.9), ("n2", 0.1)]),
        bm25=_FixedBm25([("n1", 1.0), ("n2", 0.1)]),
        node_meta=meta,
    )
    return VelesPipeline(bundle, _Zero(), llm=None)


class _FixedVector:
    def __init__(self, ids) -> None:
        self.ids = ids

    def search(self, q, top_k=10):
        return self.ids


class _FixedBm25:
    def __init__(self, ids) -> None:
        self.ids = ids

    def search(self, q, top_k=10):
        return self.ids


def test_chat_render_shows_timecode() -> None:
    """Рендер chat/completions печатает таймкод перед текстом источника."""
    from veles.api.app import create_app, render_chat_content

    meta = {
        "n1": {
            "text": "Обсудили бюджет проекта.",
            "access_groups": ["personal"],
            "title": "Встреча правления",
            "document_id": "встреча.mp4",
            "heading_path": ["Встреча правления"],
            "provenance": {
                "adapter": "video:0.1", "source_id": "встреча.mp4",
                "t_start": 754.0, "t_end": 780.0,
            },
        },
        "n2": {
            "text": "Отвлечённая тема.",
            "access_groups": ["personal"],
            "title": "Другой документ",
            "document_id": "d.md",
            "heading_path": [],
            "provenance": {"adapter": "txt-md:0.1", "source_id": "d.md", "lines": [1, 2]},
        },
    }
    app = create_app(pipeline=_fixed_pipeline(meta))
    client = TestClient(app)
    r = client.post("/v1/knowledge/query", json={"query": "бюджет проекта", "top_k": 2})
    assert r.status_code == 200
    cit = r.json()["citations"][0]
    assert cit["t_start"] == 754.0 and cit["t_end"] == 780.0

    rendered = render_chat_content(
        {
            "decision": "answer",
            "diagnostics": {},
            "answer": "Обсудили бюджет проекта. [1]",
            "citations": r.json()["citations"],
            "images": [],
        },
        mint_token_links=False,
    )
    assert "00:12:34-00:13:00" in rendered
