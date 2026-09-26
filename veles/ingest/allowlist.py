"""Allowlist-конвейер (стандарт v3.0 §7): OneDrive -> scanner -> manifest -> corpus.

Оригиналы только читаются; в каталог источников кладутся ТОЛЬКО копии исходников.
Манифест (метаданные конвейера) пишется в репозиторий veles, не в corpus root.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from ..config import MANIFESTS_DIR

# Вышка, Модуль_1 (срез Ф1). Паттерны имён - из стандарта v3.0 §7.
VYSHKA_MODULE1 = {
    "name": "vyshka-module1-lectures",
    "root": Path(r"<local-pc>\OneDrive\MyDocs\diffdocs\Вышка\02_Модули\1_курс\Модуль_1"),
    "dest": Path("05_vyshka_academic/lectures"),
    "dir_include": re.compile(r"лекция", re.IGNORECASE),  # только каталоги лекций
    "file_include": re.compile(r"(транскрипция|пересказ)", re.IGNORECASE),
    "file_ext": {".txt"},
    # Чёрный список стандарта: ДЗ/оценки/ведомости/админ/диссертация - в любом сегменте пути
    "path_exclude": re.compile(r"ДЗ|Оценк|ведомост|00_Админ|01_Оргинфо|Диссертация", re.IGNORECASE),
}


class ManifestEntry(BaseModel):
    source_path: str
    dest_rel: str
    sha256: str
    size_bytes: int
    copied: bool


class CorpusManifest(BaseModel):
    name: str
    created_at: str
    root: str  # корень сканирования (для аудита; в канон не попадает)
    included: list[ManifestEntry] = Field(default_factory=list)
    excluded: list[dict] = Field(default_factory=list)  # {path, reason}
    stats: dict = Field(default_factory=dict)


def scan_and_copy(
    spec: Optional[dict] = None,
    corpus_root: Optional[Path] = None,
    copy: bool = True,
    manifest_dir: Optional[Path] = None,
) -> CorpusManifest:
    """Скан по строгому allowlist, копирование в корпус, запись манифеста в veles."""
    spec = spec or VYSHKA_MODULE1
    corpus_root = corpus_root or Path(r"<local-pc>\proj\knowledge-sources")
    root: Path = spec["root"]
    manifest = CorpusManifest(
        name=spec["name"],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        root=str(root),
    )

    if not root.exists():
        raise FileNotFoundError(f"allowlist root не найден: {root}")

    # Кандидаты = расширение + паттерн имени + каталог-лекция; чёрный список режет
    # уже из кандидатов (иначе excluded завышается файлами, отсечёнными dir_include).
    candidates = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in spec["file_ext"]
        and spec["file_include"].search(p.name)
        and spec["dir_include"].search(str(p.parent))
    )
    files = [p for p in candidates if not spec["path_exclude"].search(str(p.relative_to(root)))]

    for src in files:
        rel = src.relative_to(root)
        dest = corpus_root / spec["dest"] / rel
        data = src.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        copied = False
        if copy:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != sha:
                shutil.copy2(src, dest)
                copied = True
        manifest.included.append(
            ManifestEntry(
                source_path=str(src),
                dest_rel=(Path(spec["dest"]) / rel).as_posix(),
                sha256=sha,
                size_bytes=len(data),
                copied=copied,
            )
        )

    manifest.excluded = [
        {"path": str(p), "reason": "blacklist-term"}
        for p in candidates
        if spec["path_exclude"].search(str(p.relative_to(root)))
    ]

    manifest.stats = {
        "included_files": len(manifest.included),
        "included_bytes": sum(e.size_bytes for e in manifest.included),
        "copied_now": sum(1 for e in manifest.included if e.copied),
        "excluded_by_blacklist": len(manifest.excluded),
        "outside_allowlist": sum(
            1
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in spec["file_ext"] and p not in candidates
        ),
    }

    out_dir = manifest_dir or MANIFESTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec['name']}-{manifest.created_at[:10]}.json"
    out.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
