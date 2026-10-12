"""Межпроцессные файловые блокировки (msvcrt.locking / fcntl.flock).

Зачем: threading.Lock защищает потоки одного процесса, но не процессы.
Писатели overlay и параллельные sync живут в разных процессах (фасад,
`veles add`, `veles sync`) - им нужен лок уровня ОС. Блокировка привязана
к открытому дескриптору: при смерти процесса ОС снимает её сама,
stale-локов не остаётся (тот же механизм, что у sync_lock).
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path


@contextlib.contextmanager
def file_lock(path: Path, timeout_s: float = 120.0):
    """Эксклюзивный межпроцессный лок на файле path.

    Блокирует вызывающего до освобождения лока, но не дольше timeout_s
    (0 - одна неблокирующая попытка); истечение - TimeoutError. Кнопка
    «занято» и ожидание разделены на уровне вызывающего: критично-частые
    операции берут короткий таймаут и честно отказываются, фоновые - ждут."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    f = os.fdopen(fd, "r+b")
    acquired = False
    deadline = time.monotonic() + max(0.0, timeout_s)
    try:
        while True:
            f.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"файловый лок занят дольше {timeout_s:g}с: {path}"
                    ) from None
                time.sleep(0.05)
        yield True
    finally:
        if acquired:
            try:
                f.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:  # noqa: BLE001 - разблокировка не маскирует ошибки кода
                pass
        f.close()
