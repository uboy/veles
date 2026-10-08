"""Точка входа veles.exe: без аргументов - сервер, с аргументами - CLI veles.

Запуск veles.exe: подхватывает .env из своего каталога и текущего (это делает
veles.config), поднимает сервер на 127.0.0.1:8400. «veles.exe add файл.md» и
прочие команды работают как обычный CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    # .env рядом с exe - удобное место для настроек частного пользователя
    exe_dir = Path(getattr(sys, "frozen", False) and sys.executable or __file__).parent
    os.chdir(exe_dir)
    from veles.cli import main as cli_main

    if len(sys.argv) == 1:
        sys.argv += ["serve", "--host", "127.0.0.1", "--port", "8400"]
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
