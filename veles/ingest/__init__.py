"""Ingestion: allowlist-конвейер и парсер-адаптеры."""

from __future__ import annotations

from .allowlist import scan_and_copy
from .txt_md import TxtMdParserAdapter

__all__ = ["scan_and_copy", "TxtMdParserAdapter"]
