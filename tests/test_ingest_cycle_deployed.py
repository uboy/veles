"""K13: интеграционный прогон e2e-цикла сохранения (e2e/run_ingest_cycle.py).

По умолчанию SKIP: полный цикл поднимает фасад с настоящим эмбеддером (минуты)
и выполняется на стенд отдельной командой (см. evals/results/2026-09-29-k13-*).
Включается переменной окружения VELES_INGEST_E2E=1; на Windows пропускается
всегда (парсинг/индексация - не для хоста владельца, канон AGENTS.md).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "e2e" / "run_ingest_cycle.py"


@pytest.mark.skipif(os.name == "nt",
                    reason="e2e-цикл с фасадом и эмбеддером - на выделенном хосте (стенд)")
@pytest.mark.skipif(not os.environ.get("VELES_INGEST_E2E"),
                    reason="включается VELES_INGEST_E2E=1 (тяжёлый цикл, минуты)")
def test_ingest_cycle_deployed():
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(ROOT)],
        capture_output=True, text=True, timeout=2400, env=env,
    )
    assert r.returncode == 0, (
        "e2e-цикл сохранения упал:\n"
        f"{r.stdout[-3000:]}\n{r.stderr[-1500:]}"
    )
    assert "ВЕРДИКТ: ЗЕЛЁНЫЙ" in r.stdout
