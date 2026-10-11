"""Парсер-адаптер видео -> CanonicalDocument (poisk-po-video, video:0.1).

Поток: ffmpeg извлекает аудио (wav 16k mono) и ключевые кадры (активы
документа); речь транскрибирует VoxFusion (сегменты с таймкодами); сегменты
сливаются в узлы ~700-1200 знаков с локатором t_start/t_end (секунды от
начала файла). Расшифровки идут в общий гибридный индекс - это текст;
кадры - только активы (просмотр /v1/asset), CLIP-индексация кадров - вне
скоупа до включения канала изображений.

ffmpeg обязателен на машине ингеста (проверяется, отказ - с понятной
причиной). Токен VoxFusion не настроен - отказ джобы (fail-closed), тихой
деградации нет.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..canonical.model import CanonicalDocument, Provenance
from . import voxfusion

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
PROFILE = "video:0.1"

_MAX_NODE_CHARS = 1200  # верхняя граница узла - как у txt_md (~1200)
_MIN_NODE_CHARS = 700   # к сегменту короче - досыпаем следующий, если влезает


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _need_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg не найден в PATH - видео-конвейер невозможен. "
            "Установите ffmpeg и повторите загрузку."
        )
    return exe


def _run(exe: str, args: list[str]) -> dict:
    r = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg {' '.join(args[:2])}: {r.stderr.strip()[-300:]}")
    try:
        return json.loads(r.stdout) if r.stdout.strip() else {}
    except json.JSONDecodeError:
        return {}


def _duration(exe: str, video: Path) -> float:
    """Длительность в секундах: ffprobe (точный JSON), без него - stderr ffmpeg."""
    probe = shutil.which("ffprobe")
    if probe:
        r = subprocess.run(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(video)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        try:
            return float(json.loads(r.stdout)["format"]["duration"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    r = subprocess.run(
        [exe, "-hide_banner", "-i", str(video)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", r.stderr)
    if not m:
        return 0.0
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def _extract_audio(exe: str, video: Path, workdir: Path) -> Path:
    wav = workdir / "audio.wav"
    _run(exe, [
        "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ])
    return wav


def _extract_frames(exe: str, video: Path, assets_dir: Path) -> dict[str, str]:
    """Ключевые кадры как активы. Кадры - необязательная косметика: на видео
    короче интервала их ноль, и это не ошибка ингеста (возвращаем {})."""
    interval = _env_float("VELES_VIDEO_FRAME_INTERVAL_S", 30.0)
    r = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
         "-vf", f"fps=1/{max(interval, 1.0)},scale='min(1280,iw)':-2",
         "-frame_pts", "1", str(assets_dir / "frame-%06d.jpg")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assets: dict[str, str] = {}
    for p in sorted(assets_dir.glob("frame-*.jpg")):
        assets[p.name] = str(p)
    if r.returncode != 0 and not assets:
        return {}  # ни одного кадра - идём без активов, речь важнее
    return assets


def merge_segments(segments: list[dict]) -> list[tuple[float, float, str]]:
    """Сегменты речи -> узлы: копим, пока не выйдем за _MAX_NODE_CHARS;
    короткий хвост досыпается к предыдущему, если тот короче _MIN_NODE_CHARS.
    Возвращает (t_start, t_end, текст) слитых узлов."""
    merged: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if merged and len(merged[-1][2]) + len(text) + 1 <= _MAX_NODE_CHARS:
            s, e, t = merged[-1]
            merged[-1] = (s, float(seg["end"]), f"{t} {text}")
            continue
        merged.append((float(seg["start"]), float(seg["end"]), text))
    # хвост, короче минимума, приклеиваем к предыдущему узлу (таймкод растёт)
    if len(merged) >= 2 and len(merged[-1][2]) < _MIN_NODE_CHARS:
        s0, e0, t0 = merged[-2]
        s1, e1, t1 = merged[-1]
        merged[-2:] = [(s0, e1, f"{t0} {t1}")]
    return merged


class VideoParserAdapter:
    parser_profile = PROFILE

    def parse_file(
        self, path: Path, corpus_root: Path,
        *, vx_client=None, poll_timeout_s: float | None = None,
    ) -> CanonicalDocument:
        """vx_client - HTTP-клиент VoxFusion (инъекция для тестов без сети);
        None -> создаётся httpx.Client на время парса."""
        rel = path.relative_to(corpus_root).as_posix()
        raw_bytes = path.read_bytes()
        if not voxfusion.enabled():
            raise RuntimeError(
                "транскрибация видео недоступна: VOXFUSION_API_TOKEN не настроен "
                "(та же пара переменных, что у Telegram-моста)"
            )
        exe = _need_ffmpeg()

        # каталог кадров живёт до save_document (прецедент docling: mkdtemp без
        # очистки - save_document переносит файлы в канон, сироты вычищает
        # repo-hygiene); аудио-временник можно убирать сразу
        assets_dir = Path(tempfile.mkdtemp(prefix="veles-assets-"))
        with tempfile.TemporaryDirectory(prefix="veles-video-") as td:
            workdir = Path(td)
            wav = _extract_audio(exe, path, workdir)
            assets = _extract_frames(exe, path, assets_dir)

            timeout = poll_timeout_s or _env_float(
                "VELES_VIDEO_TRANSCRIBE_TIMEOUT_S", voxfusion.DEFAULT_TIMEOUT_S
            )
            close_client = vx_client is None
            if vx_client is None:
                import httpx

                vx_client = httpx.Client(timeout=60.0)
            try:
                job_id = voxfusion.submit(vx_client, wav)
                status, segments, info = voxfusion.poll_segments(
                    vx_client, job_id, timeout_s=timeout
                )
            finally:
                if close_client:
                    vx_client.close()
            if status != "done":
                raise RuntimeError(f"транскрибация не удалась: {info}")

        doc = CanonicalDocument(
            document_id=rel,
            title=path.stem,
            document_type="video",
            sha256=CanonicalDocument.compute_sha256(raw_bytes),
            parser_profile=self.parser_profile,
            # честно по инварианту 4: узлы - производные от аудио файла-источника
            derived_from=[f"video-audio:{rel}"],
            assets=assets,
        )
        doc.add_node(
            logical_node_id=f"{rel}#root",
            node_type="root",
            heading_path=[path.stem],
            text=path.stem,
            provenance=Provenance(adapter=self.parser_profile, source_id=rel),
        )
        duration = _duration(exe, path) if exe else 0.0
        if segments:
            idx = 0
            for t0, t1, text in merge_segments(segments):
                idx += 1
                doc.add_node(
                    logical_node_id=f"{rel}#seg{idx:04d}",
                    node_type="paragraph",
                    parent_id=f"{rel}#root",
                    heading_path=[path.stem],
                    text=text,
                    provenance=Provenance(
                        adapter=self.parser_profile, source_id=rel,
                        t_start=t0, t_end=t1,
                    ),
                )
        else:
            # звука с речью нет - документ с кадрами и метаданными, поиск
            # найдёт его по названию; тихой лжи (мнимых узлов) не создаём
            dur = f", длительность {format_timecode_plain(duration)}" if duration else ""
            doc.add_node(
                logical_node_id=f"{rel}#silent",
                node_type="paragraph",
                parent_id=f"{rel}#root",
                heading_path=[path.stem],
                text=f"Речь в видео не распознана{dur}; кадры приложены как активы.",
                provenance=Provenance(adapter=self.parser_profile, source_id=rel),
            )
        return doc


def format_timecode_plain(seconds: float) -> str:
    """Секунды -> «ЧЧ:ММ:СС» без зависимостей от retrieval (для сообщений парсера)."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
